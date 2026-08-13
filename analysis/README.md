# SAGE — reproducible analysis package

Every table, figure, and statistic in the paper's *Statistical Reliability,
Calibration, and Error Analysis* section is regenerated here **from the saved
per-image predictions only** — no model inference, no GPU, no API keys.

## Quick start

```bash
pip install -r analysis/requirements.txt
python analysis/reproduce_all.py        # rebuilds predictions.csv + runs A1–A10
```

Outputs are written to `analysis/out/`.

## What gets built

| Script | Produces | Paper item |
|--------|----------|-----------|
| `scripts/build_predictions_csv.py` | `analysis/predictions.csv` (one row per prediction) | foundation |
| `analysis/a1_bootstrap.py` | `out/a1_kb_effect.csv` + macro/headline CIs | paired-bootstrap significance |
| `analysis/a2_kcurve.py` | `out/a2_kcurve.csv` | non-monotonic reference budget |
| `analysis/a3_calibration.py` | `out/a3_calibration.txt`, `out/a3_reliability.png` | calibration (ECE/AUROC), reliability fig |
| `analysis/a6_error_taxonomy.py` | `out/a6_error_taxonomy.txt` | organ-aware error taxonomy |
| `analysis/a7_cost_pareto.py` | `out/a7_cost_pareto.csv` | cost/latency/Pareto (k=4 knee) |
| `analysis/a8_kb_quality.py` | `out/a8_kb_quality.txt` | KB-quality vs accuracy |
| `analysis/a9_dataset_stats.py` | `out/a9_dataset_stats.txt` | dataset characterization |
| `analysis/a10_stage_retention.py` | `out/a10_stage_retention.txt` | stage-wise candidate retention |

## `predictions.csv` schema (released for independent re-analysis)

`source, crop, condition, model, k, image_id, disease_true, disease_pred,
correct, confidence, num_turns, cost_usd, duration_ms, refs_viewed, refs_n,
has_error, error`

- The clean main-study slice used in the paper is
  `source=SAGE, model∈{opus,sonnet,haiku}, has_error=False`.
- `condition ∈ {none, internet, few_shot}`; `k ∈ {0,1,4,8,16}`.
- Stale/duplicate run dirs (`*_old`, `*.pre_*`, `*_pre<digits>`) are excluded by
  the builder; the `spark` source rows are the separate open-model (Qwen)
  reproduction — filter to `source=SAGE` for the Claude study.

## Notes / caveats
- Bootstrap uses a fixed seed (42), 10,000 resamples, and pairs on the shared
  image ids between the two conditions (per-condition n can differ).
- `a9` reads the provenance sheet, which sits at Excel's 1,048,575-row cap and
  may be truncated — those counts are a lower bound (flagged in its output).
- The deep expert-audit disagreement re-analysis is **not** reproducible here:
  the raw per-field verdicts behind Appendix Fig. `field_verdicts` were not
  retained in the repo (see `REVISION_PLAN.md`).
