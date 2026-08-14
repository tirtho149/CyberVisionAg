# Reproducing SAGE (WACV 2027)

Code + instructions to reproduce **Table 2/3** (open-model scaled study, Qwen3-VL-32B, 49 crops) and **Table 4** (in-depth agentic study, Claude, 7 crops).

The evaluation data (test + reference images, per-crop KB, anatomical part-index) is published on the Hugging Face Hub — the code here consumes it directly:

> **Dataset:** https://huggingface.co/datasets/tirtho149/sage-wacv
> - `table4_dataset/` — 7-crop agentic study (Claude): `<Crop>/{test,reference}/<Class>/*.jpg` + `<Crop>/kb/`
> - `table2_dataset/` — 49-crop scaled study (Qwen): flat `images/` + per-crop `manifest/`, `kb/`, `part_index/`

## 1. Get the data
```bash
pip install "huggingface_hub"
hf download tirtho149/sage-wacv --repo-type dataset --local-dir sage-wacv-data
```

## 2. Environment
```bash
conda create -n vl-reasoning python=3.10 -y && conda activate vl-reasoning
pip install -r disease_registry/requirements.txt
# Claude runs: Claude Code CLI installed + authenticated (ANTHROPIC_API_KEY)
# Qwen runs:  vLLM serving Qwen/Qwen3-VL-32B-Instruct  (--tool-call-parser hermes)
```

## 3. Table 4 — in-depth agentic study (Claude, 7 crops)
Harness: **`open_agentic/eval.py`** — spawns one autonomous `claude -p` agent per test image; the agent views the test image + reference images and consults the crop's symptom KB, then submits a structured prediction. Metrics = top-1 accuracy per crop.

```bash
# per crop (Soybean/Corn/Tomato/Mango/Rice/Banana/Potato), per KB condition, per k:
python -m open_agentic.eval \
  --dataset sage-wacv-data/table4_dataset/Soybean \
  --symptom-source internet \     # {none|internet}  -> toggles the KB (kb/internet.xlsx + part_index)
  --model claude-sonnet \         # tiers: claude-haiku / claude-sonnet / claude-opus
  --k 8 --parallel 12 --seed 42
```
- `--k {0,1,4,8}` = reference budget; **few-shot** baseline via `open_agentic/few_shot.py`.
- Build the paper table: `python -m open_agentic.generate_tables` (figures: `generate_figures.py`).
- Exact flags & options: see [`open_agentic/README.md`](open_agentic/README.md).

## 4. Table 2/3 — open-model scaled study (Qwen3-VL-32B, 49 crops)
Serve Qwen with vLLM, then run the same harness over `table2_dataset` (resolve images via `manifest/<crop>.json`: `images/<hash>+ext`). Methods: `zero_shot`(k0), `agentic`(k4), `few_shot`/static(k4) × {`none`,`internet`}. Top-1 accuracy per crop → **Table 2**; the KB-hurts subset → **Table 3**.

## 5. Knowledge base (already shipped in the dataset)
Each crop's `kb/internet.xlsx` (source-grounded visual-symptom descriptions with citations) and `part_index` (organ→disease prior) are included in the HF dataset. To regenerate from web sources: see [`disease_registry/README.md`](disease_registry/README.md) (`python -m disease_registry ...`).

## Code map
| Path | Role |
|---|---|
| `open_agentic/eval.py` | agentic eval harness (Claude/Gemini via headless `-p`) |
| `open_agentic/few_shot.py` | few-shot / static-reference baseline |
| `open_agentic/build_part_index.py` | anatomical organ → disease-class index |
| `open_agentic/prepare_dataset.py` | reference/test split preparation |
| `open_agentic/generate_tables.py`, `generate_figures.py` | paper tables & figures |
| `disease_registry/` | source-grounded KB generation (`internet.xlsx`) |
| `agent.py`, `run_eval.sh` | legacy fixed-pipeline agent (superseded by `open_agentic/`) |
| `performance_analysis.py`, `data_loader.py` | metrics + loaders |

See `README.md` and `open_agentic/README.md` for full architecture.
