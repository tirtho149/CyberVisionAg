# Paper Storyline & Writing Plan

## Title (working)

**SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis**

---

## Big Picture

We present a large-scale effort in plant disease diagnosis that combines data collection, automated knowledge base generation, and an explainable agentic diagnostic system. We have collected ~670,000 images across 56 crops (478 disease classes) and built automated pipelines that generate structured, source-cited disease knowledge for any crop. We demonstrate and evaluate the full system on three crops (Soybean 27 classes, Corn 24 classes, Mango 7 classes) — chosen to represent different scales of difficulty. Evaluation on additional crops is planned but limited by the compute-intensive nature of agentic inference.

The system is designed around **explainability**: every diagnosis produces a full reasoning trace showing which references were examined, what visual features were observed, and how alternatives were ruled out. In agricultural diagnostics, knowing *why* matters for treatment decisions and for the farmer or extension agent to visually verify the diagnosis.

---

## Contributions

### 1. Large-scale multi-crop image dataset
- ~670,000 images across 56 crops, 478 disease classes, multi-organ (leaf, stem, root, seed, ear, head)
- Data distribution: `CyberVisionAg/open_agentic/all-crops.xlsx` (3 sheets: By Crop & Disease, Summary, Image Sources)
- Sources: Internal (expert-curated), PlantVillage, LeafNet, CDDM, New Plant Diseases, and others
- Curated train/test splits for the three evaluation crops
- Will be made publicly available

### 2. Source-first disease registry pipeline
- Automated KB generation (`disease_registry/`): given a crop name, produces structured disease knowledge with **per-field provenance** — every field carries a `{value, url, quote}` triple traced to a real web source
- Two tracks: local (PDF extraction via Anthropic API native PDF support) and internet (web discovery + extraction + reconciliation)
- KBs and pipeline will be contributed as a public resource

### 3. Agentic diagnostic method
- An autonomous reasoning agent that receives a test image, labeled reference images, and optionally a structured symptom KB
- Selectively chooses which references to examine, reasons step-by-step in natural language, uses KB symptoms to guide visual feature identification
- Produces a structured prediction with full diagnostic reasoning trace — every step is visible and auditable
- Explainability is the design goal, not a side effect

### 4. Systematic evaluation
- Zero-shot baseline establishes what the model knows from pretraining
- Ablation across reference budget (k), KB source (none/local/internet), and model quality (haiku/sonnet/opus)
- Cross-crop evaluation on datasets of varying difficulty

---

## Key Findings (brief)

- Zero-shot is non-trivial — the model has pretraining knowledge of plant diseases
- Adding references with deliberative reasoning provides substantial lift over zero-shot
- Structured symptom KB contributes meaningful improvement, particularly at lower reference budgets
- Model quality scales consistently across all crops
- Full reasoning traces provide transparency into every diagnostic decision

---

## Paper Structure (synced with main.tex)

### Section 1: Introduction (`sec:intro`)
- Problem: plant disease diagnosis at scale needs transparency and verifiable knowledge grounding
- Gap: existing datasets are leaf-only, no symptom metadata, no provenance; existing classifiers are black boxes
- Contributions (four-fold, as listed above)

### Section 2: Related Work (`sec:related`)
- [ ] Plant disease classification (CNN/ViT fine-tuning)
- [ ] Few-shot learning in agriculture
- [ ] VLMs for visual reasoning and classification
- [ ] Agentic AI systems and tool-use
- [ ] Explainability / interpretability in agricultural AI
- [ ] Disease knowledge bases and symptom databases
- *User will provide abstracts and references.bib files; section will be written from those and iterated*

### Section 3: Dataset (`sec:dataset`)
- Image sources: expert-curated across ~350 crops, multi-organ coverage
- Evaluation subset: Soybean (27 classes), Corn (24 classes), Mango (7 classes)
- Train/test splits and curation details
- Registry schema table: crop, disease, pathogen, type, affected organs, visual symptoms — each with `{value, url, quote}`

### Section 4: Disease Registry Pipeline (`sec:registry`)
- Source-first principle: fetch real documents, extract only what's written, never generate from model knowledge
- Three stages: Discovery → Extraction → Reconciliation (pipeline figure already in main.tex)
- Local track: PDF page-by-page extraction
- Internet track: parallel per-disease web discovery
- Per-field provenance on all output fields
- Implementation: `disease_registry/` directory (TODO: inspect pipeline code in detail to write this section thoroughly)

### Section 5: Agentic Diagnostic Pipeline (`sec:agent`)
- System architecture: eval harness spawns parallel `claude -p` agents
- Each agent receives: test image, reference image paths (collages), class list, optional KB text
- Multi-turn reasoning: observe test image → form hypotheses → select reference → compare → articulate → decide
- Reference budget k controls how many collages the agent can view
- Prompt design and reasoning structure

### Section 6: Experiments (`sec:experiments`)
- Datasets and setup
- Zero-shot baseline
- Ablations: k values, KB source, model scaling
- Metrics: accuracy, per-class accuracy (mean + std across classes), cost, avg turns, refs viewed
- Per-class std captures instability / variance across disease classes
- Results and analysis
- Qualitative: example reasoning traces

### Section 7: Discussion (`sec:discussion`)
- Few-shot vs agentic reasoning: different inference paradigms. Few-shot processes all references in a single forward pass with no structured comparison — it is a black box. Our approach is deliberative and sequential, optimizing for explainability. There is a cost to this (more compute per image), but in agricultural diagnostics, transparency and verifiability are essential.
- When KB helps and when it doesn't
- Failure analysis: visually ambiguous diseases
- Cost-accuracy tradeoffs
- Limitations

### Section 8: Conclusion (`sec:conclusion`)

### Abstract (write last)

---

## Figures Plan

All figures follow `plotting_instructions.md` (Arial font, rcParams, dpi=300, PDF output).

**Data-driven principle**: All plots read directly from raw data files (summary JSONs, `all-crops.xlsx`). If data is updated or experiments are rerun, plots can be regenerated automatically without manual edits.

**Inspect protocol**: When reviewing any figure, generate 4 quadrant crops (top-left, top-right, bottom-left, bottom-right) as separate images to check for label clipping, text overlap, rotation issues, and readability at print scale. Trigger with the word "inspect".

**Figure generation principles**:
- Research standard approaches online before hand-rolling complex visualizations.
- Use the right tool for the job — don't force one library to do everything.
- For overlaid text on charts, add a semi-transparent background for readability.
- Merge long tails into "+N more" entries to avoid visual noise.
- Always export at high resolution for publication quality.

**Artifact regeneration commands** (run from `AgCrawler/` root):
```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
set -a && source .env && set +a

# Regenerate all figures (outputs to writing/.../figures/)
python -m CyberVisionAg.open_agentic.generate_figures

# Regenerate all traces (outputs to writing/.../traces/)
python -m CyberVisionAg.open_agentic.trace_to_tex

# Regenerate all tables (outputs to writing/.../tables/)
python -m CyberVisionAg.open_agentic.generate_tables

# Recompile paper
cd writing/69aae430e8bdcbd9056bf911 && pdflatex -interaction=nonstopmode main.tex

# Print results tables
cd CyberVisionAg && bash open_agentic/run_sweeps.sh results soybean
bash open_agentic/run_sweeps.sh results corn
bash open_agentic/run_sweeps.sh results mango
```
Every generated artifact (figures, traces, tables) has a single command that reproduces it from raw data. If experiments are rerun or data changes, only the relevant command needs to be re-executed.

### Figure 1: System Overview (already in main.tex as `fig:overview`)
- Block diagram: Crop name → Registry pipeline → Disease Registry → Agent + Images → Prediction
- Already implemented in TikZ

### Figure 2: Registry Pipeline (already in main.tex as `fig:pipeline`)
- Discovery → Extraction → Reconciliation with per-field citation example
- Already implemented in TikZ

### Figure 3: Dataset Scale + Agentic Flow (combined multi-panel)
- **Type**: Multi-panel figure combining data overview and system flow
- **Panel A**: Bar chart of top crops by image count (from `all-crops.xlsx`)
- **Panel B**: Distribution of diseases per crop or images per disease (histogram)
- **Panel C**: Agentic prediction flow diagram — test image → agent reasoning steps → prediction (simplified visual)
- **Config**: `figsize=(13, 5)` or similar dense layout
- **Data source**: Reads directly from `all-crops.xlsx` — updates automatically if data changes
- **Purpose**: Show scale of data and how the system works in one dense figure

### Figure 4: Main Results Panel (dense, multi-dimensional)
- **Type**: 1x3 line plot panel (one per crop)
- **Config**: `figsize=(13, 4)`, large rcParams
- **Lines**: Agent no-KB, Agent+internet KB (+ Agent+local KB for soybean)
- **Horizontal dashed line**: Zero-shot floor
- **X-axis**: k (1, 4, 8, 16)
- **Y-axis**: Accuracy (%)
- **Include**: Error bars or shaded bands showing per-class std
- **Purpose**: Show lift over zero-shot, scaling with k, KB contribution, and variance — all in one dense figure

### Figure 5: Model Scaling + Zero-shot Comparison (combined dense figure)
- **Type**: 1x2 panel or similar dense layout
- **Left panel**: Model ablation (haiku/sonnet/opus) across all 3 crops — grouped bars with std error bars
- **Right panel**: Zero-shot vs Agent no-KB vs Agent+KB at a fixed k, all 3 crops
- **Config**: `figsize=(11, 4)` or similar — dense, one figure showing multiple dimensions
- **Purpose**: Two findings in one figure — model quality scaling and KB effect

### Figure 6: Per-class Accuracy Heatmap
- **Type**: Heatmap for one crop (corn or soybean)
- **Rows**: Disease classes
- **Columns**: Zero-shot, Agent no-KB, Agent+KB
- **Config**: `figsize=(6, 8)` or as needed
- **Purpose**: Show where the system helps, where diseases are genuinely hard

### Figure 7+: Reasoning Trace Examples (LaTeX, not matplotlib)
- **Script**: `CyberVisionAg/open_agentic/trace_to_tex.py` (code lives with the pipeline)
- **Output directory**: `writing/69aae430e8bdcbd9056bf911/traces/` (outputs live with the paper)
- **main.tex** uses `\input{traces/trace_example_1.tex}` — script changes go to trace files, main.tex just references them
- **Each trace `.tex` file clearly shows**: k value, model, KB source (internet/local/none), then step-by-step reasoning (which refs viewed, visual observations, differential reasoning, final prediction, ground truth)
- **Start small**: 1-2 traces (one correct, one incorrect)
- **Build up**: Eventually many traces covering different crops, KB sources, k values, models, correct/incorrect cases
- **This will be a long set of trace documents** — organized and indexed systematically

### Table 1: Dataset Summary
- Crops, number of classes, train/test image counts, KB coverage

### Table 2: Main Results (in-text, only table)
- Method × k with accuracy % and delta from zero-shot
- Auto-generated via `generate_tables.py` from summary JSONs
- Bold best per crop-k, caption notes 3 images/class
- Model ablation shown in Figure 5 (no separate table needed)

---

## Step-by-Step Execution Plan

### Phase 1: Lock down results
- [x] Run all sweeps (soybean, corn, mango)
- [x] Fix corn k=8 contamination
- [x] Run zero-shot baseline for all 3 crops
- [x] Verify all numbers are clean and consistent (cross-check all 54 summary JSONs)
- [x] Commit final results

### Phase 2: Generate figures
- [x] Create `CyberVisionAg/open_agentic/generate_figures.py` (reads directly from summary JSONs + `all-crops.xlsx`)
- [x] Figure 3: Sunburst — crop-disease hierarchy (Plotly, exported via kaleido)
- [x] Figure 4: Accuracy vs k line plots (1x3 panel) with zero-shot floor
- [x] Figure 5: Model scaling + KB comparison (1x2 combined panel)
- [ ] Figure 6: Per-class accuracy heatmap (deferred)
- [x] All plots read from raw data files — no hardcoded numbers
- [x] Saved as PDF in `writing/69aae430e8bdcbd9056bf911/figures/`
- [x] Integrated into main.tex at correct positions

### Phase 3: Reasoning trace pipeline
- [x] Write `CyberVisionAg/open_agentic/trace_to_tex.py` — converts JSON trace → formatted `.tex` file
- [x] Each `.tex` file displays: config (k, model, KB source), test image thumbnail, step-by-step reasoning, prediction, ground truth
- [x] Output to `writing/69aae430e8bdcbd9056bf911/traces/` with images in `traces/images/`
- [x] main.tex uses `\input{traces/trace_intext.tex}` for in-text example (soybean, internet KB, k=4)
- [x] 1 in-text trace + 15 appendix traces (no duplication)
- [x] Appendix included via `\input{traces/traces_appendix.tex}`
- [ ] Expand later: more crops, KB sources, k values, models as needed

### Phase 4: Write the paper (in main.tex)
- [x] Section 1: Introduction (data-first framing, four contributions, citations grounded)
- [x] Section 2: Related Work (flowing prose, no subsections — datasets, VLMs, CoT/agentic)
- [x] Section 3.1: Dataset — Image Sources (four source categories with citations)
- [ ] Section 3.2: Dataset — Splits (still bullet points)
- [ ] Section 3.3: Dataset — Registry Schema (table exists, needs prose)
- [x] Section 4: Disease Registry Pipeline (source-first principle, three stages in flowing prose)
  - **Future note**: Reconciliation prompt includes 6-tier source authority ranking (NCBI > APS > CABI > peer-reviewed > extension > industry). Confirmed in `prompts/reconciliation.py:15-22`. Could be mentioned in a revision if reviewers ask about conflict resolution.
- [x] Section 5: Agentic Diagnostic Pipeline (3 paragraphs: inputs, reasoning process, trace/explainability)
  - **Future**: Add appendix section before traces with: (1) exact system prompt, (2) example user message structure, (3) one complete registry schema entry. All auto-generated from source files via script.
- [x] Section 6: Experiments (setup, results with inline numbers verified against data, figures/tables/trace referenced)
- [ ] Section 7: Discussion (still bullet points — few-shot vs agentic, limitations, cost-explainability)
- [ ] Section 8: Conclusion (empty)
- [ ] Abstract (write last)

### Phase 5: Literature review
- [x] Receive abstracts and references.bib files from user
- [x] Added 20+ references: 6 published papers, 14 Kaggle datasets, 6 additional papers (ChatLeafDisease, PDD-AGENT, CDDM, WDLM, AgroGPT, Agri-LLaVA, AgReason, AgEval)
- [x] All bib entries verified (URLs, authors, titles) via parallel agents + WebFetch
- [x] Write Section 2: Related Work from provided references
- [x] References integrated naturally into Introduction and Related Work
- [ ] Iterate with user on coverage and framing (ongoing)

### Phase 6: Polish
- [ ] Cross-check all numbers: text vs figures vs tables vs summary JSONs
- [ ] Ensure figure/table references in text are correct
- [ ] Expand reasoning trace collection (more crops, more cases)
- [ ] Proofread for consistency and clarity
- [ ] Format for target venue
