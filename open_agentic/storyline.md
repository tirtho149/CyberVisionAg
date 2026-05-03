# Paper Storyline & Writing Plan

> **How this file works**: This is the single source of truth for paper writing. Major steps are marked `[x]` (done) or `[ ]` (pending). When working on a step, add small sub-items below it with specific instructions (e.g., "remove collage mention at line 191"). Remove these sub-items once done. The major step markers stay permanently as a record of what was completed.

## Title (working)

**SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis**

---

## One-liner

Plant disease diagnosis at scale is bottlenecked by the lack of datasets that cover enough crops and diseases. We compile one of the largest disease image datasets (~1.1M images, 53 crops, 259 classes) from multiple sources and complement it with automatically generated, source-cited symptom descriptions. We demonstrate how this combination of visual and structured knowledge enables an agentic reasoning system to diagnose diseases across crops with full transparency and no task-specific training.

## Big Picture

We present a large-scale effort in plant disease diagnosis that combines data collection, automated knowledge base generation, and an explainable agentic diagnostic system. We have collected ~1.1M images across 53 crops (259 disease classes) and built automated pipelines that generate structured, source-cited disease knowledge for any crop. We demonstrate and evaluate the full system on three crops (Soybean 25 classes, Corn 30 classes, Mango 4 classes), chosen to represent different scales of difficulty.

The system is designed around **explainability**: every diagnosis produces a full reasoning trace showing which references were examined, what visual features were observed, and how alternatives were ruled out. In agricultural diagnostics, knowing *why* matters for treatment decisions and for the farmer or extension agent to visually verify the diagnosis.

**Framing notes** (for writing):
- **Knowledge-augmented visual reasoning**: The KB provides structured symptom descriptions and anatomical context (which organs each disease affects). This tells the agent what visual signals to look for and narrows the candidate set via an anatomical index.
- **Guided chain of thought**: Given this knowledge, the agent follows a structured but open-ended reasoning sequence: review symptoms, narrow by anatomical context, then sequentially compare reference images before deciding. Every step is visible in the trace.

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
- KB provides symptom descriptions and anatomical context; the agent uses this to know what to look for and to narrow candidates
- Guided chain of thought: review symptoms, narrow by anatomy, sequentially compare reference images, decide
- Produces a structured prediction with full diagnostic reasoning trace
- Explainability is the design goal, not a side effect

### 4. Systematic evaluation
- Zero-shot baseline establishes what the model knows from pretraining
- Ablation across reference budget (k), KB source (none/internet), and model quality (haiku/sonnet/opus)
- Cross-crop evaluation on datasets of varying difficulty (4, 25, 30 classes)

---

## Key Findings (brief)

- Higher k (more reference images) consistently improves accuracy across all crops and configurations
- KB improves accuracy across all crops: +7.9 to +13.6pp on Corn, +2.7 to +13.5pp on Soybean. Mean improvement +15.2pp at k=16
- Stronger models perform better: Opus > Sonnet > Haiku on all crops (Opus reaches 62.2% on Soybean, 61.4% on Corn)
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
- Evaluation subset: Soybean (25 classes, 74 test), Corn (30 classes, 88 test), Mango (4 classes, 40 test)
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
- Each agent receives: test image, individual reference image paths, class list, optional KB text + anatomical index
- Guided chain of thought: observe test image → identify anatomical context → narrow candidates via anatomical index → review KB symptoms → sequentially compare reference images → decide
- Reference budget k controls how many images the agent can view
- Prompt design: k-adaptive (at high k, visual comparison takes priority over KB text)
- [ ] **TODO**: Remove collage and calibration/attractor guide references. Add anatomical context description.

### Section 6: Experiments (`sec:experiments`)
- Datasets and setup (KB-filtered prepared datasets, anatomical index)
- Ablations: k values (0,1,4,8,16), KB source (none/internet), model scaling (haiku/sonnet/opus)
- Metrics: accuracy, per-class accuracy, cost, avg turns, refs viewed
- Results and analysis
- Qualitative: example reasoning traces
- [ ] **TODO**: Update inline numbers to match new sweep results
- [ ] **TODO**: Remove attractor guide/calibration subsection and confusion matrices
- [ ] **TODO**: Add discussion of anatomical context narrowing
- [ ] **TODO**: Update or regenerate reasoning trace examples from new results

### Section 7: Discussion (`sec:discussion`)
- Few-shot vs agentic reasoning: different inference paradigms. Few-shot processes all references in a single forward pass with no structured comparison. Our approach is deliberative and sequential, optimizing for explainability.
- KB helps consistently across crops, especially at lower k
- Failure analysis: visually ambiguous diseases (trace analysis shows agent views GT class but picks wrong match)
- Cost-accuracy tradeoffs
- Limitations
- [ ] **TODO**: Update with new results and remove calibration framing

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

```bash
# Confusion matrix plots + per-class change analysis (for updating main.tex examples)
python -m CyberVisionAg.open_agentic.plot_confusion_matrix \
  --results-dir CyberVisionAg/results/open_agentic/{Dataset}/internet/opus/k8_cg \
  --output writing/.../figures/confusion_matrix_opus_attractor.png \
  --title "+ Attractor Guide" \
  --compare-with CyberVisionAg/results/open_agentic/{Dataset}/internet/opus/k8
# Prints per-class accuracy changes and over-prediction shifts. Use these to update specific examples cited in main.tex.
```

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
- **No dashed line**: DC baseline removed; k=0 is the leftmost point on the curve
- **X-axis**: k (0, 1, 4, 8, 16)
- **Y-axis**: Accuracy (%)
- **Include**: Error bars or shaded bands showing per-class std
- **Purpose**: Show lift over zero-shot, scaling with k, KB contribution, and variance — all in one dense figure

### Figure 5: Model Scaling + Zero-shot Comparison (combined dense figure)
- **Type**: 1x2 panel or similar dense layout
- **Left panel**: Model ablation (haiku/sonnet/opus) across all 3 crops — grouped bars with std error bars
- **Right panel**: Baseline (k=0, no KB) vs Agent no-KB vs Agent+KB at k=4, all 3 crops
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
- Method × k with accuracy % and delta from baseline (Agent no KB, k=0)
- Auto-generated via `generate_tables.py` from summary JSONs
- Bold best per crop-k, caption notes 3 images/class
- Model ablation shown in Figure 5 (no separate table needed)
- [x] **DONE**: Main results table has Mean delta rows for Agent(no KB) and Agent+internet KB.
- [x] **DONE**: Few-shot comparison table with Mean delta row.

---

## Step-by-Step Execution Plan

### Phase 1: Lock down results
- [x] Run all sweeps with prepared datasets (soybean 25 cls, corn 30 cls, mango 4 cls)
- [x] All 17 configs per crop complete (none/internet x k=0,1,4,8,16 + model ablation + few-shot)
- [x] `generate_tables.py` and `generate_figures.py` updated and regenerated

### Phase 2: Update main.tex

- [x] **Fix class counts** — Updated to Soybean (25), Corn (30), Mango (4). Removed local KB mention.
- [x] **Rewrite Section 5** — Removed collage, added anatomical context/index, guided chain of thought, sequential comparison.
- [x] **Rewrite Section 6 intro** — Fixed inline numbers (Soybean +13.5pp, Corn +9.1pp), updated class counts, removed local KB.
- [x] **Replace Section 6.1** — Removed calibration subsection. TODO placeholder for new confusion matrices (k=0/none vs k=16/internet).
- [x] **Update Section 7** — Updated Mango to 4 classes, removed calibration paragraph, updated KB contribution with new numbers.
- [x] **Update abstract** — New numbers (+15.2pp mean), removed calibration sentence, added guided chain of thought.
- [x] **Update appendix** — Removed old attractor confusion matrices. TODO placeholder for new ones.
- [x] **Sunburst** — Updated to new CSV (1.1M images, 53 crops, 259 unique diseases)
- [x] **Dataset numbers** — All 6 locations updated (613K→1.1M, 55→53, 477→259)
- [x] **Section 3.2** — Expanded to describe KB-guided image filtering, anatomical tagging, clean splits
- [x] **Figure 2** — Extended with second row: Registry → Image Filtering → Ref set + Test set + Anatomical index

### Phase 3: Regenerate traces
- [x] In-text trace: auto-selected first correct from Soybean/internet/sonnet/k4
- [x] 15 appendix traces across all crops/configs
- [x] Confusion matrices: baseline (k=0/none) vs best (k=16/internet) for all 3 crops
- [x] Soybean highlight: Sudden_death_syndrome over-prediction (14 FP → 3 FP)

### Phase 4: Polish
- [x] Grepped main.tex: no remaining old references (collage, attractor, calibration, old class counts)
- [x] Paper compiles (23 pages, no errors)
- [x] Redesigned Figure 1 right panel (standard vs ours comparison + example entry with organs)
- [x] Redesigned Figure 2 (Curation + Inference groups, fontawesome icons, sample images, open-ended reasoning box, input tags)
- [x] Renamed Discussion → Discussion and Conclusion, added closing paragraph
- [x] Fixed em dash in Related Work (line 121)
- [x] Strengthened Section 3.2 with anatomical index reference to Figure 2
- [x] Moved dataset license table to top of appendix, restyled to match paper conventions
- [x] Added cost analysis table (auto-generated via generate_tables.py from summary JSONs)
- [x] Figure 2 v3: rebalanced toward dataset/curation. **Two expert phases now split across the two curation rows**: (1) `Expert Dedupe` between `Raw images` and `Image Filtering` (cross-source class-name dedupe), (2) `Expert Audit` between `Reconciliation` and `Disease KB` (per-field verbatim-evidence audit). Annotated `Raw images` with `$\sim$840K, multi-source` and `Disease KB` with `335 crops · 1{,}251 diseases` (numbers from Table 1). Renamed inference group to `Demonstrated Agentic Evaluation`. Added visible open-ended loop back-arrow `Compare → KB Lookup` with `repeat until k refs viewed`. Swapped test image to mango Anthracnose, fixed prediction text to `Anthracnose`. Added trace strip (Step 6/7/8) using verbatim Step 7 quote from `traces/trace_appendix_13.tex`. Section 4 prose updated to describe ONLY the evidence audit (no naming-dedupe claim). Section 3.2 prose updated to describe the cross-source class-name dedupe step before image filtering.
- [ ] Cross-check all inline numbers against summary JSONs
- [ ] Update open_agentic/README.md with final results table
- [ ] Commit and push all changes

### Phase 5: Future TODOs
- [ ] **Registry provenance example in appendix**: Re-run soybean registry pipeline to get a fresh `final_registry.json` with full provenance (value, url, quote triples). Currently the pipeline overwrites the JSON per crop; fix to save per-crop JSONs (like xlsx). Then add a worked example entry to Appendix B showing per-field provenance visually.
- [ ] **Per-crop registry JSON persistence**: Modify `pipeline.py` to save `{Crop}_final_registry.json` alongside `{Crop}_{track}.xlsx` so provenance is always available.
