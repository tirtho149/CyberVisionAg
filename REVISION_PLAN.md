# SAGE — Reviewer Response & Revision Plan

**Goal:** answer nearly every *answerable* reviewer criticism **without new model inference**, using already-saved predictions, confidences, traces, token/cost logs, KB registry, and dataset provenance. Be transparent about the 2 criticisms that genuinely need new experiments.

**Central reframing (do this first, everything else supports it):**
Stop positioning the paper as *"our agent is better at classification."*
Reposition as *"SAGE is a large, provenance-aware, multi-organ crop-disease evaluation **resource**, plus an agentic **baseline** demonstrating one way to exploit source-grounded symptom knowledge."*
All three reviewers independently praise the dataset (scale, multi-organ, per-field provenance) — lean into that.

---

## Step 0 — Artifact inventory (DONE)

| # | Artifact | Verdict | Location / notes |
|---|----------|---------|------------------|
| 1 | Per-image predictions | ✅ YES (~12k + 17.8k JSON) | `SAGE/results/open_agentic/`, `spark/upstream/results/`; fields: `ground_truth, prediction, correct, confidence, reasoning, num_turns, cost_usd, duration_ms, refs_viewed` |
| 2 | Confidence scores | ✅ YES (100%) | `confidence` 0–1 on every prediction → **calibration is a go** |
| 3 | Reasoning traces | ✅ YES (rich) | `results-report/agent/logs/.../*_log.json`: `trace[]` per tool call, `refs_viewed`, `symptoms_read`, `comparisons_made`, `vision_features`, plus a pre-computed `judge` calibration verdict |
| 4 | Token/cost/latency | ✅ YES | `cost_usd, duration_ms, num_turns` per image; aggregated in `summary.json` |
| 5 | Intermediate candidate lists | 🟡 PARTIAL | not stored explicitly; **reconstructable** by parsing `trace[]` (list_classes→read_symptom→compare→submit) |
| 6 | KB expert-audit records | 🔴 **BLOCKED** | see Investigation B below — number exists only in figure/text, no raw file |
| 7 | Dataset provenance | ✅ YES | `CyAg/crop_disease_registry_sheet3_image_sources.csv` (image→crop→disease→source, 1.5M+ rows); organ/symptom in registry JSON |
| 8 | Baseline ref sampling | ✅ CONFIRMED | `run_agent.py:484` = **first file alphabetically** per class, deterministic, NOT organ-matched |

### Investigation A — can we expand beyond 4 crops with existing runs? → 🟡 PARTIAL, heterogeneous
295 `summary.json` files exist across **Banana, Cauliflower, Coffee, Corn, Mango, Orange, Soybean, Sugarcane, Tomato** (+ stale `.pre_*`/`_old` dirs). **But the models are mixed** and NOT a uniform grid:
- Claude Haiku/Sonnet/Opus grid: mainly **Corn, Soybean, Tomato, Mango** (the "main study" crops).
- Extra crops (Banana, Coffee, Sugarcane, Cauliflower, Orange) are largely **gemini-flash / gemini-\*** or single-model **claude-sonnet-4-6**, with error-heavy cells (Sugarcane 118 errors, Coffee 40, Soybean 18).
- Duplicate/stale dirs (`Sugarcane_Diseases_old`, `*.pre_2026*`, `*_pre1776974939`) must be excluded.

**Implication:** expansion is *possible but not a clean drop-in*. Do NOT claim a uniform 9-crop Claude evaluation. Options: (i) report the extra crops as a **secondary open-model (Gemini/Sonnet) generalization slice** with its own protocol note, or (ii) restrict the clean comparative tables to the crops with the full Claude grid + low errors. **TODO: per-crop completeness vetting** (same model set / seed / k-grid / errors≈0) before folding any crop in.

### Investigation B — where is the 69–73% expert audit? → 🔴 NEGATIVE (no raw data)
The agronomist audit (5 crops: Soybean, Mango, Banana, Sugarcane, Corn; 5 fields) numbers appear **only** in:
- Figure `overleaf-6a62569c/figures/field_verdicts.png` — Affected Parts **69/14/17**, Diagnostic Features **73/23/4**, Disease Type **91/3/6**, Pathogen **71/17/11**, Symptom Desc **91/0/9** (agree/neutral/disagree %).
- `main.tex` lines ~531–536 prose.

**No per-disease/per-field verdict file, no agreement-computing script, exists in the repo.** Likely a local (uncommitted) spreadsheet on the first author's machine.
**Consequence:** Analysis 5 (deep disagreement re-analysis) is **blocked until that spreadsheet is recovered.** Fallbacks: (a) recover the sheet → then the deep breakdown becomes possible; (b) if unrecoverable, reframe: report the existing 3-way verdicts per field and state a finer breakdown was not retained. Analysis 6 can instead use the registry's self-reported `confidence: high/med/low` + `num_sources` as the KB-quality proxy.

---

## The plan (26 tracked tasks)

Legend — feasibility: ✅ ready · 🟡 needs parsing/vetting · 🔴 blocked/needs external input · ✍️ writing only

### Step 0 (foundation)
- **T1** ✅ Inventory saved artifacts — *DONE (this doc).*
- **T2** ✅ Build canonical `predictions.csv` (`image_id, crop, disease_true, disease_pred, k, condition, model, confidence, cost_usd, tokens, calls, latency_ms, trace_id`). Unblocks T3–T9, T22–23.

### P0 — Quantitative analyses (no new inference)
- **T3** ✅ Analysis 1 — paired bootstrap Δ + 95% CI + P(Δ>0) per crop×k; Wilson/binomial CIs on raw acc; Holm across k.
- **T4** ✅ Analysis 2 — k-curve uncertainty (paired bootstrap on 1→2→4→8); reframe non-monotonicity as a finding.
- **T5** ✅ Analysis 3 — calibration: ECE, Brier, reliability diagram, AUROC(conf→correct), P(correct|c≥.9) vs P(correct|c<.7). (bonus: pre-computed `judge` verdicts).
- **T6** ✅ Analysis 4 — error taxonomy (same-organ look-alike / stage / symptom / KB mismatch / visual quality / reference confusion / anatomical mislocalization).
- **T7** 🔴 Analysis 5 — deep expert-audit disagreement breakdown — **blocked on Investigation B spreadsheet**.
- **T8** 🟡 Analysis 6 — KB-quality → accuracy (use registry `confidence` high/med/low as proxy; report descriptively if overlap small).
- **T9** ✅ Analysis 7 — cost/token/latency/Pareto per k; ΔAcc/ΔCost; sweet spot (likely k=4).
- **T10** 🟡 Analysis 8 — stage-wise candidate retention (Recall@M through initial→anatomy→symptom→reference) by **parsing traces**; call it retention, not full ablation.

### P0 — Dataset characterization (no model calls)
- **T11** ✅ Analysis 9 — full-dataset stats (images/crop, diseases/crop, long-tail, sparsity, organ dist, source diversity, citation coverage, Gini/entropy).
- **T12** ✅ Analysis 10 — Evaluation Coverage Matrix (full 335/1,251/~839K vs evaluated subset) + restriction rationale.

### P1 — Claim reframing / writing (✍️)
- **T13** Reposition paper as evaluation resource + baseline (abstract/intro/conclusion).
- **T14** Separate dataset coverage from generalization claims (4-crop = procedure demo, not universal).
- **T15** Soften headline "+16.2 pp" → observed mean + bootstrap uncertainty (tune to T3).
- **T16** Explainability = auditability, NOT calibrated correctness (tie to T5).
- **T17** KB-leakage construction audit (flowchart raw→filter→organ→KB→eval; concede selection effects).
- **T18** Foundation-model prior confound — drop causal "new knowledge"; acquisition vs latent-activation undistinguished.
- **T19** Baseline fairness / rename → "first-reference few-shot VLM"; gain = KB + anatomy + selection + sequential reasoning (T8 confirmed first-file sampling).
- **T20** Terminology reconciliation methods clarification (no invented majority vote).
- **T21** Scaling/complexity analysis for large D_c (retrieval sublinear if r≪D_c); label computational.
- **T22** Strong limitations: long-tail crops, matched supervised baseline, multi-stress — explicit out-of-scope.
- **T23** Manuscript editing (repeated organization / cutoff sentence).

### P2 — Artifacts & packaging
- **T24** Fix HF schema — expose organ/symptoms/diagnostic_features/affected_parts/source_url/source_quote/reference_image_ids (or relational parquets w/ keys).
- **T25** Fix GitHub — README, env file, `reproduce_table2.py` on saved predictions; release `predictions.csv`.
- **T26** Provenance/validation schema — per-field Source + Validation table; label auto-annotated vs expert-verified.

---

## Cannot resolve without new experiments → reframe only
- Long-tail crop evaluation (→ T14, T22)
- Matched simple supervised baseline (→ T22)
- Full component **predictive** ablation (→ T10 retention as partial substitute)
- Multi-stress / multi-pathogen robustness (→ T22)
- KB-placeholder prior-knowledge test (→ T18)

## Suggested sequence
1. **T2** (predictions.csv) → **T3–T12** analyses/stats in parallel.
2. Recover the audit spreadsheet (unblock **T7**); vet extra crops (Investigation A) before any expansion claim.
3. **T13–T23** rewrite around the new numbers.
4. **T24–T26** artifacts (parallelizable).

## Two things that need YOU
1. **Locate the expert-audit spreadsheet** (the 5-crop × 5-field agronomist verdicts behind `field_verdicts.png`) — unblocks T7 and strengthens the KB story.
2. **Decide the expansion stance** — present extra crops as a secondary Gemini/Sonnet slice, or keep clean tables to the full-Claude-grid crops only.
