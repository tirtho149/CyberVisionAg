# Disease Registry Pipeline

Automated pipeline that builds evidence-backed crop disease registries. Every piece of information is traceable to a specific source (web URL or PDF page) with a verbatim quote.

## How It Works

Traditional approaches ask an LLM "tell me about disease X", making it impossible to distinguish sourced facts from confabulation. This pipeline flips that:

1. **Fetch real documents first** (web pages or PDF pages)
2. **Extract only what's explicitly written** with verbatim quotes
3. **Merge across sources** while tracking provenance per field

Disease names always come from **image folder names** — this is the single source of truth for which diseases to include.

## Two Tracks

### 1. Local track (PDF → symptoms)

Given a reference PDF and folder names, the system:
- Reads the PDF page by page using the Anthropic API (native PDF support)
- Extracts visual symptom descriptions for each disease with verbatim quotes from the PDF
- Uses LLM-based name matching to map PDF disease names to folder names
- Outputs an xlsx with a "Visual Description" column

### 2. Internet track (web → full registry)

Given folder names, the system runs automated internet research:
- **Discovery**: One web search per disease (parallel, 4 at a time) for authoritative sources (university extension pages, APS compendia)
- **Extraction**: Each source URL is fetched and an LLM extracts disease data (pathogen, affected parts, symptoms) with verbatim quotes from the page text
- **Reconciliation**: Records from multiple sources are merged into one canonical entry per disease, keeping per-field citations
- LLM-based name matching maps web-extracted names back to folder names
- Outputs a complete xlsx with all columns filled from web sources

Both tracks always output **all folder diseases** — with data where found, nulls where not.

## Provenance: How Information Is Traced Back

Every field in the registry JSON is stored as `{value, url, quote}`:

```json
{
  "disease_name": "Charcoal Rot",
  "pathogen_scientific_name": {
    "value": "Macrophomina phaseolina",
    "url": "https://cropprotectionnetwork.org/encyclopedia/charcoal-rot-of-soybean",
    "quote": "Charcoal rot is caused by the soilborne fungus Macrophomina phaseolina"
  },
  "visual_symptoms": {
    "summary": {
      "value": "Lower stem and taproots show charcoal-like gray discoloration...",
      "url": "https://cropprotectionnetwork.org/encyclopedia/charcoal-rot-of-soybean",
      "quote": "The lower stem and taproots have a light gray discoloration..."
    }
  }
}
```

- **value**: The extracted fact
- **url**: The web page or `pdf://filename.pdf` where it came from
- **quote**: The exact sentence copied from the source that supports the value

In the markdown output files, each cell is a clickable hyperlink to its source URL. In the xlsx, the "Visual Description" column combines the symptom summary, diagnostic features, and look-alikes into one readable text.

The raw extraction files (`raw_extractions.json`, `pdf_extractions.json`) preserve the full per-source records before merging, so you can trace any field back through the pipeline.

## Quick Start

```bash
# From the AgCrawler/ directory:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning

# Local: PDF -> symptom extraction
python -m disease_registry.pipeline --crop soybean --track local \
  --pdf CyberVisionAg/knowledge_docs/soybean-compressed.pdf \
  --disease-dir "Disease Images for prof Arti"

# Internet: web discovery -> extraction -> reconciliation
python -m disease_registry.pipeline --crop soybean --track internet \
  --disease-dir "Disease Images for prof Arti"

# Both tracks + cross-comparison
python -m disease_registry.pipeline --crop soybean --track both \
  --pdf CyberVisionAg/knowledge_docs/soybean-compressed.pdf \
  --disease-dir "Disease Images for prof Arti"

# Quick mode (fewer sources, shorter timeouts)
python -m disease_registry.pipeline --crop soybean --track internet --quick

# Resume internet pipeline from a specific stage
python -m disease_registry.pipeline --crop soybean --track internet --resume-from reconciliation
```

## CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--crop` | Yes | Crop name (e.g., soybean, corn, wheat) |
| `--track` | No | Which pipeline: `local`, `internet`, or `both` (default: `both`) |
| `--disease-dir` | Yes | Path to image directory with disease folders (provides disease name list) |
| `--pdf` | Local | Path to reference PDF (required for local track) |
| `--quick` | No | Quick mode for testing (fewer sources, shorter timeouts) |
| `--resume-from` | No | Resume internet track from stage: `discovery`, `extraction`, `reconciliation` |

## Output Files

All outputs go to `disease_registry/outputs/{Crop}/` (one folder per crop, e.g., `outputs/Soybean/`):

| File | Track | Description |
|------|-------|-------------|
| `local.xlsx` | Local | Disease registry with Visual Description from PDF |
| `local_registry.json` | Local | Full registry with per-field provenance (`pdf://` sources) |
| `local_registry.md` | Local | Markdown table with hyperlinked cells |
| `pdf_extractions.json` | Local | Raw per-chunk PDF extraction records |
| `internet.xlsx` | Internet | Complete sheet built from web sources |
| `final_registry.json` | Internet | Full registry with per-field provenance (web URL sources) |
| `registry.md` | Internet | Markdown table with hyperlinked cells |
| `discovery_results.json` | Internet | Candidate source URLs found during discovery |
| `raw_extractions.json` | Internet | Raw per-source extraction records (before merging) |
| `discrepancy_report.md` | Both | Comparison of local vs internet registries |

## Prerequisites

- **Python 3.10+** (conda env: `vl-reasoning`)
- **Anthropic API key** in `.env` file (`ANTHROPIC_API_KEY=...`)
- **Claude Code CLI** installed and authenticated (used by internet track for web search)
- **openpyxl** (xlsx reading/writing)
- **PyMuPDF** (`pip install pymupdf`) -- optional, for page-by-page PDF chunking

## File Structure

```
disease_registry/
├── pipeline.py              # CLI + dispatcher (--track local|internet|both)
├── shared.py                # Shared infra: API client, claude -p wrapper, LLM name matching
├── local_pipeline.py        # Local track: folder names + PDF extraction
├── internet_pipeline.py     # Internet track: discovery -> extraction -> reconciliation
├── utils.py                 # fetch_page(), registry_to_markdown(), write_enriched_xlsx()
├── config.py                # All tunable parameters (batch sizes, timeouts, model, etc.)
├── prompts/                 # All LLM prompts and JSON schemas
│   ├── discovery.py         # Web discovery prompts (free-form + targeted)
│   ├── extraction.py        # Web + PDF extraction prompts
│   ├── reconciliation.py    # Merging + name normalization prompts
│   └── comparison.py        # Local vs internet comparison prompt
└── outputs/                 # All generated results
```

## Inference Evaluation in CyberVisionAg/

The disease registry outputs (`Soybean_local.xlsx`, `Soybean_internet.xlsx`) feed into the inference pipeline in `CyberVisionAg/`. An A/B evaluation compares how curated symptom sources affect classification accuracy.

### Evaluation Design

Three runs on the **Soybean** dataset:

| Run | Approach | Knowledge Source | k meaning |
|-----|----------|-----------------|-----------|
| **Few-shot baseline** | k random labeled train images in-context, single API call | Training images from `Curated_Local_Dataset/train/Soybean_Diseases/` | k = number of example images |
| **Agentic + Default KB** | Agent with tool calls | `disease_symptoms_crop_wise.md` (OpenAI-generated) | k = max reference image views |
| **Agentic + Local KB** | Agent with tool calls | `Soybean_local.xlsx` visual descriptions | k = max reference image views |
| **Agentic + Internet KB** | Agent with tool calls | `Soybean_internet.xlsx` visual descriptions | k = max reference image views |

- **Model**: `claude-sonnet-4-6` for all runs
- **Train split** → example/reference images; **Test split** → inference targets
- **k parameter**: Unified across paradigms. For few-shot, k = number of context images. For agentic, k = max `get_reference_image` calls (symptom reads, class listing, etc. are unlimited)

### Integrity Checklist

- [x] **No filename leakage**: Test image filenames uploaded with neutral name `test_image.jpg` (fixed in `upload_image()`)
- [ ] **No path leakage**: Ground truth folder path must not appear in prompt content
- [x] **Few-shot labels are intentional**: Example image labels are provided (that's the point), but test image labels are hidden

### Run Commands

```bash
# Setup
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
cd /path/to/AgCrawler
set -a && source .env && set +a

# Quick smoke test (2 classes, 1 image each) — default KB
PYTHONUNBUFFERED=1 python -m CyberVisionAg.agent \
  --symptom-source default --quick-test 2

# Agentic + local KB (disease_registry PDF extraction)
PYTHONUNBUFFERED=1 python -m CyberVisionAg.agent \
  --symptom-source local --quick-test 2

# Agentic + internet KB (disease_registry web extraction)
PYTHONUNBUFFERED=1 python -m CyberVisionAg.agent \
  --symptom-source internet --quick-test 2

# Full run (all 32 Soybean classes, all test images)
PYTHONUNBUFFERED=1 python -m CyberVisionAg.agent \
  --symptom-source default --dataset Soybean_Diseases

# With k budget (limit reference image views)
PYTHONUNBUFFERED=1 python -m CyberVisionAg.agent \
  --symptom-source local --dataset Soybean_Diseases --k 3 --quick-test 5
```

Results go to `CyberVisionAg/results/agent/{symptom_source}/logs/{dataset}/`.

#### A/B comparison (all 3 KB sources in parallel)

Runs default, local, and internet sources concurrently (parallel=12 per source, seed=42). Results go to `/tmp/agent_{source}.log`.

```bash
bash CyberVisionAg/run_eval.sh
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symptom-source` | `default` | KB source: `default` (markdown), `local` (PDF xlsx), `internet` (web xlsx) |
| `--dataset` | `Soybean_Diseases` | Dataset folder name under `Curated_Local_Dataset/` |
| `--num-classes` | all | Limit to N random classes |
| `--images-per-class` | all | Test images per class |
| `--quick-test N` | — | Shortcut: sets num-classes=N, images-per-class=1 |
| `--k` | unlimited | Max `get_reference_image` calls per image (other tools unlimited) |
| `--parallel` | `1` | Number of images to classify concurrently |
| `--seed` | `42` | Random seed for reproducible class selection |

### Implementation Progress

- [x] Agentic pipeline works end-to-end with default KB
- [x] Argparse CLI added to `agent.py` (`--symptom-source`, `--dataset`, `--quick-test`, `--k`, etc.)
- [x] `load_symptoms_from_xlsx()` — reads Disease + Visual Description from xlsx, converts to markdown format
- [x] `--symptom-source` wired to select KB loader (default/local/internet)
- [x] `--k` wired as reference image budget in `execute_tool()`
- [x] Results dirs separated by source (`results/agent/{source}/logs/`)
- [x] KB coverage logged at startup (e.g., "KB coverage: 32/32 diseases have symptom text")
- [x] Smoke tested all 3 sources (default, local, internet) — 2 classes, 1 image each
- [x] `--parallel N` for concurrent image classification via ThreadPoolExecutor
- [x] `--seed` flag (default=42) for reproducible random class selection
- [x] Dynamic diagnostic prompt with budget awareness and verify-before-submit phase
- [x] Removed Files API — always inline base64 (fixes parallel errors)
- [x] Disabled Judge (Stage 3) — calibration eval skipped for now
- [x] A/B eval verified: 5 classes × 5 images, parallel=12, seed=42, zero errors
- [ ] Few-shot baseline (`few_shot_eval.py`) — future, separate file
- [ ] `eval_compare.py` — compare results across runs — future

