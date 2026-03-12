#!/usr/bin/env python3
from __future__ import annotations
"""
performance_analysis.py
=======================
Downstream analysis of agent.py prediction logs.

Log schema (written by agent.py → run_agent_on_crop)
─────────────────────────────────────────────────────
  Log path : results/agent/logs/<crop>/<ablation>/<image>_log.json
  Fields   :
    image_name, crop, ground_truth, prediction, correct, confidence
    ablation, tool_call_count
    refs_viewed        — list[str]  (class names viewed)
    symptoms_read      — list[str]  (disease names read)
    comparisons_made   — list[str]  (winner strings from compare_candidates)
    vision_features    — { primary_organ_affected, approximate_growth_stage, … }
    reasoning_summary  — str
    trace              — list[dict]
    judge              — { verdict, calibration_score, reasoning_consistency,
                           judge_notes, raw }
    num_turns, success, error, timestamp

Breakdown dimensions
────────────────────
  • Pathogen / Host / TypeOfDisease  (joined from database.xlsx)
  • Crop                             (from log "crop" field / dir structure)
  • Ablation mode  local / web
  • Vision organ   (primary_organ_affected from vision agent)
  • Growth stage   (approximate_growth_stage from vision agent)

Outputs
───────
  performance_by_pathogen.csv
  performance_by_type_of_disease.csv
  performance_by_host.csv
  performance_by_crop.csv
  ablation_comparison.csv
  confusion_matrix_<crop>.csv   (one per crop)
  confusion_matrix_overall.csv
  calibration_breakdown.csv
  tool_usage_stats.csv
  vision_organ_breakdown.csv    ← NEW (from vision_features)
  growth_stage_breakdown.csv    ← NEW (from vision_features)
  all_predictions_enriched.csv
  performance_report.tex  →  performance_report.pdf

Usage
─────
  python performance_analysis.py
  python performance_analysis.py --logs results/agent/logs \\
                                 --db /path/to/database.xlsx
  python performance_analysis.py --ablation web   # restrict to one mode
"""

import json
import math
import shutil
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required.  pip install pandas openpyxl")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (mirror agent.py)
# ══════════════════════════════════════════════════════════════════════════════

VALID_ABLATION_MODES = {"local", "web"}  # must match agent.py

CALIBRATION_VERDICTS = [
    "WELL_CALIBRATED", "OVERCONFIDENT", "UNDERCONFIDENT", "INCONSISTENT", "UNKNOWN"
]

ABLATION_LABELS = {
    "local": "Local_Registry (doc extraction)",
    "web":   "Registry (web-search enriched)",
}

# agent.py log path layout:  logs/<crop>/<ablation>/<stem>_log.json
# Both crop and ablation are inferred from the directory hierarchy.


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Normalise a string for fuzzy matching."""
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _pct(num: int, den: int) -> float:
    return round(num / den * 100, 1) if den else float("nan")


def _mean(lst: list) -> float:
    return round(sum(lst) / len(lst), 3) if lst else float("nan")


def _fmt(v, spec=".3f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return format(v, spec)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE  (database.xlsx)
# ══════════════════════════════════════════════════════════════════════════════

def load_database(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    df = pd.read_excel(db_path, engine="openpyxl")

    col_map: dict = {}
    for col in df.columns:
        n = _norm(str(col))
        if n in ("disease", "disease name"):
            col_map["Disease"] = col
        elif n in ("pathogen", "pathogen name"):
            col_map["Pathogen"] = col
        elif n in ("host",):
            col_map["Host"] = col
        elif n in ("type of disease", "type", "disease type"):
            col_map["TypeOfDisease"] = col
        elif n in ("crop",):
            col_map["Crop"] = col

    missing = [k for k in ("Disease", "Pathogen", "Host", "TypeOfDisease")
               if k not in col_map]
    if missing:
        print(f"  [WARN] DB columns not found: {missing}")
        print(f"         Available: {list(df.columns)}")

    df = df.rename(columns={v: k for k, v in col_map.items()})
    if "Disease" in df.columns:
        df["_norm"] = df["Disease"].astype(str).map(_norm)
    return df


def match_disease(norm_name: str, db: pd.DataFrame) -> dict:
    empty = {"Pathogen": "Unknown", "Host": "Unknown", "TypeOfDisease": "Unknown"}
    if "_norm" not in db.columns:
        return empty
    rows = db[db["_norm"] == norm_name]
    if rows.empty:
        rows = db[db["_norm"].apply(lambda x: norm_name in x or x in norm_name)]
    if rows.empty:
        return empty
    row = rows.iloc[0]
    return {
        "Pathogen":      str(row.get("Pathogen",      "Unknown")).strip(),
        "Host":          str(row.get("Host",          "Unknown")).strip(),
        "TypeOfDisease": str(row.get("TypeOfDisease", "Unknown")).strip(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LOG COLLECTION  — mirrors agent.py path:  logs/<crop>/<ablation>/*_log.json
# ══════════════════════════════════════════════════════════════════════════════

def collect_logs(logs_dir: Path, ablation_filter: str = None) -> list[dict]:
    """
    Collect all *_log.json files under logs_dir.

    Path structure expected (written by agent.py):
        logs/<crop>/<ablation>/<image>_log.json

    Inferred fields added to each log dict:
        _crop     — from log["crop"] field, falling back to dir level -2
        _ablation — from log["ablation"] field, falling back to dir level -1
        _log_path — absolute path string
    """
    logs: list[dict] = []
    for p in sorted(logs_dir.rglob("*_log.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Cannot read {p}: {e}")
            continue

        data["_log_path"] = str(p)

        # Infer crop and ablation from path if not stored in the log itself
        rel = p.relative_to(logs_dir)
        parts = rel.parts  # e.g. ("Soybean", "ABC", "img_log.json")

        data["_crop"] = (
            str(data.get("crop", "")).strip()
            or (parts[-3] if len(parts) >= 3 else "unknown")
        )
        data["_ablation"] = (
            str(data.get("ablation", "")).lower().strip()
            or (parts[-2].lower() if len(parts) >= 2 else "web")
        )
        logs.append(data)

    if ablation_filter:
        af = ablation_filter.lower().strip()
        before = len(logs)
        logs = [l for l in logs if l["_ablation"] == af]
        print(f"  Ablation filter '{af}': {before} → {len(logs)} logs retained.")

    return logs


# ══════════════════════════════════════════════════════════════════════════════
#  RECORD ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_records(logs: list[dict], db: pd.DataFrame) -> list[dict]:
    """
    Produce one flat dict per log that carries every field needed for
    all downstream analyses.  Matched against database.xlsx by ground truth.
    """
    records: list[dict] = []
    unmatched: list[str] = []

    for log in logs:
        gt         = str(log.get("ground_truth", "")).strip()
        pred       = str(log.get("prediction",   "")).strip()
        correct    = bool(log.get("correct", pred == gt))
        confidence = float(log.get("confidence", 0.0))
        crop       = log["_crop"]
        ablation   = log["_ablation"]

        # ── Judge / calibration ───────────────────────────────────────────
        # agent.py stores:  judge.verdict  (not judge.calibration_verdict)
        judge_block = log.get("judge", {})
        verdict = str(
            judge_block.get("verdict",
            judge_block.get("calibration_verdict", "UNKNOWN"))
        ).upper().strip()
        if verdict not in CALIBRATION_VERDICTS:
            verdict = "UNKNOWN"
        cal_score = float(judge_block.get("calibration_score", 0.0))

        # ── Tool usage ────────────────────────────────────────────────────
        tool_calls = int(log.get("tool_call_count", 0))

        # refs_viewed / symptoms_read are list[str] in agent.py
        rv = log.get("refs_viewed", [])
        sr = log.get("symptoms_read", [])
        # comparisons_made is list[str] (winner strings)
        cm = log.get("comparisons_made", log.get("comparisons", []))

        n_refs  = len(rv) if isinstance(rv, list) else 0
        n_syms  = len(sr) if isinstance(sr, list) else 0
        n_comps = len(cm) if isinstance(cm, list) else 0

        # Count discriminator calls from trace
        trace = log.get("trace", [])
        n_disc = sum(1 for e in trace
                     if e.get("tool") == "get_disease_discriminators")

        # ── Vision features ───────────────────────────────────────────────
        vf    = log.get("vision_features", {}) or {}
        organ = str(vf.get("primary_organ_affected",   "unknown")).strip().lower() or "unknown"
        stage = str(vf.get("approximate_growth_stage", "unknown")).strip().lower() or "unknown"

        # ── Database metadata (keyed to ground truth) ─────────────────────
        meta = match_disease(_norm(gt), db)
        if meta["Pathogen"] == "Unknown":
            unmatched.append(gt)

        records.append({
            # identity
            "image_name":          log.get("image_name", ""),
            "crop":                crop,
            "ground_truth":        gt,
            "prediction":          pred,
            "correct":             correct,
            "confidence":          confidence,
            "ablation":            ablation,
            # calibration
            "calibration_verdict": verdict,
            "calibration_score":   cal_score,
            # tool usage
            "tool_call_count":     tool_calls,
            "refs_viewed":         n_refs,
            "symptoms_read":       n_syms,
            "comparisons_made":    n_comps,
            "discriminator_calls": n_disc,
            # vision agent
            "organ_affected":      organ,
            "growth_stage":        stage,
            # db metadata
            "Pathogen":            meta["Pathogen"],
            "Host":                meta["Host"],
            "TypeOfDisease":       meta["TypeOfDisease"],
            # internal — stripped from CSV
            "_refs_list":  rv if isinstance(rv, list) else [],
            "_syms_list":  sr if isinstance(sr, list) else [],
            "_trace":      trace,
            "_log_path":   log.get("_log_path", ""),
        })

    if unmatched:
        uniq = sorted(set(unmatched))
        print(f"  [WARN] {len(unmatched)} record(s) unmatched in DB "
              f"({len(uniq)} unique):")
        for d in uniq[:10]:
            print(f"         • {d}")
        if len(uniq) > 10:
            print(f"         … and {len(uniq) - 10} more")

    return records


# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC DIMENSION TABLE  (works for any string grouping key)
# ══════════════════════════════════════════════════════════════════════════════

def build_dimension_table(records: list[dict], dim: str) -> pd.DataFrame:
    """
    Group by ``dim``, compute accuracy + confidence stats.
    dim can be any key present in every record dict.
    """
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r[dim]].append(r)

    rows = []
    for key, items in sorted(groups.items()):
        total   = len(items)
        correct = sum(1 for i in items if i["correct"])
        confs   = [i["confidence"] for i in items]
        c_confs = [i["confidence"] for i in items if i["correct"]]
        w_confs = [i["confidence"] for i in items if not i["correct"]]

        rows.append({
            dim:                 key,
            "total":             total,
            "correct":           correct,
            "accuracy_%":        _pct(correct, total),
            "avg_confidence":    _mean(confs),
            "avg_conf_correct":  _mean(c_confs),
            "avg_conf_wrong":    _mean(w_confs),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_ablation_comparison(records: list[dict]) -> pd.DataFrame:
    """
    Side-by-side accuracy, calibration and tool-usage per ablation mode.
    """
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["ablation"]].append(r)

    order     = ["local", "web"]
    sort_key  = lambda x: order.index(x) if x in order else 99

    rows = []
    for abl in sorted(groups, key=sort_key):
        items   = groups[abl]
        total   = len(items)
        correct = sum(1 for i in items if i["correct"])

        def _avg(f): return round(sum(i[f] for i in items) / total, 2) if total else 0.0

        cal_cnt = defaultdict(int)
        for i in items:
            cal_cnt[i["calibration_verdict"]] += 1

        rows.append({
            "ablation":          abl,
            "description":       ABLATION_LABELS.get(abl, abl),
            "total":             total,
            "correct":           correct,
            "accuracy_%":        _pct(correct, total),
            "avg_confidence":    _mean([i["confidence"] for i in items]),
            "avg_cal_score":     _mean([i["calibration_score"] for i in items]),
            "WELL_CALIBRATED":   cal_cnt["WELL_CALIBRATED"],
            "OVERCONFIDENT":     cal_cnt["OVERCONFIDENT"],
            "UNDERCONFIDENT":    cal_cnt["UNDERCONFIDENT"],
            "INCONSISTENT":      cal_cnt["INCONSISTENT"],
            "avg_tool_calls":    _avg("tool_call_count"),
            "avg_refs_viewed":   _avg("refs_viewed"),
            "avg_syms_read":     _avg("symptoms_read"),
            "avg_disc_calls":    _avg("discriminator_calls"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CROP × ABLATION BREAKDOWN  (primary structural dimension in agent.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_crop_table(records: list[dict]) -> pd.DataFrame:
    """
    Per-crop × ablation: accuracy, calibration, tool-usage.
    This mirrors the run_agent_on_crop() unit of work in agent.py.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["crop"], r["ablation"])].append(r)

    rows = []
    for (crop, abl), items in sorted(groups.items()):
        total   = len(items)
        correct = sum(1 for i in items if i["correct"])

        def _avg(f): return round(sum(i[f] for i in items) / total, 2) if total else 0.0

        cal_cnt = defaultdict(int)
        for i in items:
            cal_cnt[i["calibration_verdict"]] += 1

        rows.append({
            "crop":              crop,
            "ablation":          abl,
            "total":             total,
            "correct":           correct,
            "accuracy_%":        _pct(correct, total),
            "avg_confidence":    _mean([i["confidence"] for i in items]),
            "avg_cal_score":     _mean([i["calibration_score"] for i in items]),
            "WELL_CALIBRATED":   cal_cnt["WELL_CALIBRATED"],
            "OVERCONFIDENT":     cal_cnt["OVERCONFIDENT"],
            "UNDERCONFIDENT":    cal_cnt["UNDERCONFIDENT"],
            "INCONSISTENT":      cal_cnt["INCONSISTENT"],
            "avg_tool_calls":    _avg("tool_call_count"),
            "avg_refs_viewed":   _avg("refs_viewed"),
            "avg_syms_read":     _avg("symptoms_read"),
            "avg_disc_calls":    _avg("discriminator_calls"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def build_calibration_breakdown(records: list[dict]) -> pd.DataFrame:
    """Verdict counts + percentages per crop × ablation."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["crop"], r["ablation"])].append(r)

    rows = []
    for (crop, abl), items in sorted(groups.items()):
        total   = len(items)
        cal_cnt = defaultdict(int)
        for i in items:
            cal_cnt[i["calibration_verdict"]] += 1

        row: dict = {
            "crop":          crop,
            "ablation":      abl,
            "total":         total,
            "avg_cal_score": _mean([i["calibration_score"] for i in items]),
        }
        for v in CALIBRATION_VERDICTS:
            row[v]         = cal_cnt[v]
            row[f"{v}_%"]  = round(_pct(cal_cnt[v], total), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL USAGE STATS
# ══════════════════════════════════════════════════════════════════════════════

def build_tool_usage_stats(records: list[dict]) -> pd.DataFrame:
    """
    Per-crop × ablation tool-usage averages, split correct vs wrong.
    Reflects agent.py tool names: refs_viewed, symptoms_read,
    comparisons_made, discriminator_calls.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["crop"], r["ablation"])].append(r)

    rows = []
    for (crop, abl), items in sorted(groups.items()):
        n = len(items)
        ok  = [i for i in items if i["correct"]]
        bad = [i for i in items if not i["correct"]]

        def _avg(lst, f):
            return round(sum(i[f] for i in lst) / len(lst), 2) if lst else float("nan")

        rows.append({
            "crop":                    crop,
            "ablation":                abl,
            "n":                       n,
            # tool calls
            "avg_tool_calls":          _avg(items, "tool_call_count"),
            "avg_tool_calls_correct":  _avg(ok,    "tool_call_count"),
            "avg_tool_calls_wrong":    _avg(bad,   "tool_call_count"),
            # refs_viewed  (list of class names in agent.py → stored as count)
            "avg_refs_viewed":         _avg(items, "refs_viewed"),
            "avg_refs_correct":        _avg(ok,    "refs_viewed"),
            "avg_refs_wrong":          _avg(bad,   "refs_viewed"),
            # symptoms_read
            "avg_syms_read":           _avg(items, "symptoms_read"),
            "avg_syms_correct":        _avg(ok,    "symptoms_read"),
            "avg_syms_wrong":          _avg(bad,   "symptoms_read"),
            # compare_candidates calls  (comparisons_made in agent.py)
            "avg_comparisons":         _avg(items, "comparisons_made"),
            # discriminator calls (counted from trace)
            "avg_disc_calls":          _avg(items, "discriminator_calls"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  VISION AGENT BREAKDOWNS  (new — from vision_features in each log)
# ══════════════════════════════════════════════════════════════════════════════

def build_vision_organ_breakdown(records: list[dict]) -> pd.DataFrame:
    """Accuracy broken down by primary_organ_affected reported by vision agent."""
    return build_dimension_table(records, "organ_affected")


def build_growth_stage_breakdown(records: list[dict]) -> pd.DataFrame:
    """Accuracy broken down by approximate_growth_stage reported by vision agent."""
    return build_dimension_table(records, "growth_stage")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_confusion_matrix(records: list[dict],
                            crop_filter: str = None) -> pd.DataFrame:
    """
    GT × Prediction confusion matrix.
    Optionally restricted to a single crop.
    """
    subset = ([r for r in records if r["crop"] == crop_filter]
              if crop_filter else records)
    if not subset:
        return pd.DataFrame()

    classes = sorted(set(r["ground_truth"] for r in subset)
                     | set(r["prediction"]  for r in subset))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in subset:
        counts[r["ground_truth"]][r["prediction"]] += 1

    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for gt, preds in counts.items():
        for p, n in preds.items():
            matrix.at[gt, p] = n
    matrix.index.name   = "GT \\ Pred"
    matrix.columns.name = "Prediction"
    return matrix


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE PRINTERS
# ══════════════════════════════════════════════════════════════════════════════

def _dim_table_str(df: pd.DataFrame, dim: str, title: str = None) -> str:
    title = title or f"Performance by {dim}"
    lines = [f"\n{'─'*74}", f"  {title}", f"{'─'*74}"]
    if df.empty:
        lines.append("  (no data)"); return "\n".join(lines)

    lines.append(
        f"  {'':42s}  {'N':>5}  {'Corr':>5}  {'Acc%':>6}  "
        f"{'AvgConf':>8}  {'ConfOK':>7}  {'ConfBAD':>7}"
    )
    lines.append("  " + "─" * 70)
    for _, row in df.iterrows():
        lines.append(
            f"  {str(row[dim])[:42]:<42}  {row['total']:>5}  {row['correct']:>5}  "
            f"{_fmt(row['accuracy_%'], '.1f'):>5}%  "
            f"{_fmt(row['avg_confidence']):>8}  "
            f"{_fmt(row['avg_conf_correct']):>7}  "
            f"{_fmt(row['avg_conf_wrong']):>7}"
        )
    return "\n".join(lines)


def _ablation_table_str(df: pd.DataFrame) -> str:
    lines = [f"\n{'─'*100}", "  ABLATION COMPARISON", f"{'─'*100}"]
    if df.empty:
        lines.append("  (no data)"); return "\n".join(lines)

    lines.append(
        f"  {'Mode':<5} {'Description':<28} {'N':>5} {'Corr':>5} "
        f"{'Acc%':>6} {'AvgConf':>8} {'CalSc':>6} "
        f"{'WELL':>5} {'OVER':>5} {'UNDR':>5} {'INCO':>5} "
        f"{'Tools':>6} {'Refs':>5} {'Syms':>5} {'Disc':>5}"
    )
    lines.append("  " + "─" * 96)
    for _, row in df.iterrows():
        lines.append(
            f"  {row['ablation']:<5} {str(row['description'])[:28]:<28} "
            f"{row['total']:>5} {row['correct']:>5} "
            f"{_fmt(row['accuracy_%'], '.1f'):>5}% "
            f"{_fmt(row['avg_confidence']):>8} "
            f"{_fmt(row['avg_cal_score']):>6} "
            f"{row['WELL_CALIBRATED']:>5} {row['OVERCONFIDENT']:>5} "
            f"{row['UNDERCONFIDENT']:>5} {row['INCONSISTENT']:>5} "
            f"{row['avg_tool_calls']:>6.2f} {row['avg_refs_viewed']:>5.2f} "
            f"{row['avg_syms_read']:>5.2f} {row['avg_disc_calls']:>5.2f}"
        )
    return "\n".join(lines)


def _crop_table_str(df: pd.DataFrame) -> str:
    lines = [f"\n{'─'*88}", "  PERFORMANCE BY CROP × ABLATION", f"{'─'*88}"]
    if df.empty:
        lines.append("  (no data)"); return "\n".join(lines)

    lines.append(
        f"  {'Crop':<35} {'Abl':<5} {'N':>5} {'Corr':>5} "
        f"{'Acc%':>6} {'AvgConf':>8} "
        f"{'WELL':>5} {'OVER':>5} {'UNDR':>5} {'INCO':>5} "
        f"{'Tools':>6} {'Refs':>5} {'Syms':>5}"
    )
    lines.append("  " + "─" * 84)
    for _, row in df.iterrows():
        lines.append(
            f"  {str(row['crop'])[:35]:<35} {row['ablation']:<5} "
            f"{row['total']:>5} {row['correct']:>5} "
            f"{_fmt(row['accuracy_%'], '.1f'):>5}% "
            f"{_fmt(row['avg_confidence']):>8}  "
            f"{row['WELL_CALIBRATED']:>4} {row['OVERCONFIDENT']:>5} "
            f"{row['UNDERCONFIDENT']:>5} {row['INCONSISTENT']:>5} "
            f"{row['avg_tool_calls']:>6.2f} {row['avg_refs_viewed']:>5.2f} "
            f"{row['avg_syms_read']:>5.2f}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _tex(s: str) -> str:
    for char, rep in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$",  r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}",  r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(char, rep)
    return s


def _longtable(col_fmt: str, caption: str,
               header: str, rows: list[str], n: int,
               small: bool = True) -> str:
    cont = r"\midrule \multicolumn{" + str(n) + r"}{r}{\textit{cont'd\ldots}} \\"
    size_cmd = r"\small" if small else ""
    return "\n".join([
        r"\begin{adjustbox}{max width=\textwidth}",
        size_cmd,
        r"\begin{longtable}{" + col_fmt + "}",
        r"\caption{" + caption + r"} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        cont, r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ] + rows + [r"\end{longtable}", r"\end{adjustbox}"])


def _dim_to_latex(df: pd.DataFrame, dim: str, caption: str) -> str:
    if df.empty:
        return r"\textit{No data.}"
    dim_label = dim.replace("_", " ").title()
    col_labels = {
        dim: dim_label, "total": "N", "correct": "Corr",
        "accuracy_%": r"Acc (\%)", "avg_confidence": "AvgConf",
        "avg_conf_correct": "Conf OK", "avg_conf_wrong": "Conf Bad",
    }
    cols   = list(col_labels.keys())
    header = " & ".join(f"\\textbf{{{col_labels[c]}}}" for c in cols) + r" \\"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == dim:              cells.append(_tex(str(v)))
            elif c == "accuracy_%":  cells.append(_fmt(v, ".1f"))
            else:                    cells.append(_fmt(v, ".3f"))
        rows.append(" & ".join(cells) + r" \\")
    return _longtable("l" + "r"*(len(cols)-1), caption, header, rows, len(cols))


def _ablation_to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return r"\textit{No ablation data.}"
    col_labels = {
        "ablation": "Mode", "description": "Description",
        "total": "N", "correct": "Corr", "accuracy_%": r"Acc (\%)",
        "avg_confidence": "AvgConf", "avg_cal_score": "CalScore",
        "WELL_CALIBRATED": "WELL", "OVERCONFIDENT": "OVER",
        "UNDERCONFIDENT": "UNDER", "INCONSISTENT": "INCON",
        "avg_tool_calls": "Tools", "avg_refs_viewed": "Refs",
        "avg_syms_read": "Syms", "avg_disc_calls": "Disc",
    }
    cols   = list(col_labels.keys())
    header = " & ".join(f"\\textbf{{{col_labels[c]}}}" for c in cols) + r" \\"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in ("ablation", "description"):
                cells.append(_tex(str(v)))
            elif c == "accuracy_%":
                cells.append(_fmt(v, ".1f"))
            elif c in ("avg_confidence", "avg_cal_score"):
                cells.append(_fmt(v, ".3f"))
            elif c.startswith("avg_"):
                cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
            else:
                cells.append(str(v))
        rows.append(" & ".join(cells) + r" \\")
    return _longtable("ll" + "r"*(len(cols)-2),
                      "Ablation mode comparison --- accuracy, calibration, tool usage",
                      header, rows, len(cols))


def _crop_to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return r"\textit{No crop data.}"
    col_labels = {
        "crop": "Crop", "ablation": "Abl",
        "total": "N", "correct": "Corr", "accuracy_%": r"Acc (\%)",
        "avg_confidence": "AvgConf", "avg_cal_score": "CalScore",
        "WELL_CALIBRATED": "WELL", "OVERCONFIDENT": "OVER",
        "UNDERCONFIDENT": "UNDER", "INCONSISTENT": "INCON",
        "avg_tool_calls": "Tools", "avg_refs_viewed": "Refs",
        "avg_syms_read": "Syms", "avg_disc_calls": "Disc",
    }
    cols   = list(col_labels.keys())
    header = " & ".join(f"\\textbf{{{col_labels[c]}}}" for c in cols) + r" \\"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in ("crop", "ablation"):
                cells.append(_tex(str(v)))
            elif c == "accuracy_%":
                cells.append(_fmt(v, ".1f"))
            elif c in ("avg_confidence", "avg_cal_score"):
                cells.append(_fmt(v, ".3f"))
            elif c.startswith("avg_"):
                cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
            else:
                cells.append(str(v))
        rows.append(" & ".join(cells) + r" \\")
    return _longtable("ll" + "r"*(len(cols)-2),
                      r"Per-crop $\times$ ablation breakdown",
                      header, rows, len(cols))


def _calibration_to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return r"\textit{No calibration data.}"
    base = ["crop", "ablation", "total", "avg_cal_score"]
    vcols = []
    for v in ["WELL_CALIBRATED", "OVERCONFIDENT", "UNDERCONFIDENT", "INCONSISTENT"]:
        vcols += [v, f"{v}_%"]
    cols = [c for c in base + vcols if c in df.columns]
    labels = {
        "crop": "Crop", "ablation": "Abl", "total": "N",
        "avg_cal_score": "CalScore",
        "WELL_CALIBRATED": "WELL", "WELL_CALIBRATED_%": r"WELL\%",
        "OVERCONFIDENT":   "OVER", "OVERCONFIDENT_%":   r"OVER\%",
        "UNDERCONFIDENT":  "UNDR", "UNDERCONFIDENT_%":  r"UNDR\%",
        "INCONSISTENT":    "INCO", "INCONSISTENT_%":    r"INCO\%",
    }
    header = " & ".join(f"\\textbf{{{labels.get(c, c)}}}" for c in cols) + r" \\"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in ("crop", "ablation"): cells.append(_tex(str(v)))
            elif c == "avg_cal_score":    cells.append(_fmt(v, ".3f"))
            else:                         cells.append(str(v))
        rows.append(" & ".join(cells) + r" \\")
    return _longtable("ll" + "r"*(len(cols)-2),
                      "Calibration verdict breakdown by crop and ablation",
                      header, rows, len(cols))


def _confusion_to_latex(matrix: pd.DataFrame, title: str) -> str:
    if matrix.empty:
        return r"\textit{No data.}"
    classes = list(matrix.index)
    n = len(classes)
    if n > 30:
        return (r"\textit{Matrix too large for inline display "
                f"({n} classes). See CSV.}}")
    col_fmt = "l" + "r" * n
    rot     = " & ".join(
        r"\rotatebox{75}{\small " + _tex(c) + "}" for c in classes
    )
    header  = r"\textbf{GT $\backslash$ Pred} & " + rot + r" \\"
    rows = []
    for gt in classes:
        cells = [_tex(gt)]
        for p in classes:
            v = int(matrix.at[gt, p])
            if gt == p and v > 0:
                cells.append(r"\textbf{" + str(v) + "}")
            elif v > 0:
                cells.append(r"\textcolor{red}{" + str(v) + "}")
            else:
                cells.append("0")
        rows.append(" & ".join(cells) + r" \\")
    lines = [
        r"\begin{longtable}{" + col_fmt + "}",
        r"\caption{" + _tex(title) + r"} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endlastfoot",
    ] + rows + [r"\end{longtable}"]
    return "\n".join(lines)


def _detail_tables_to_latex(records: list[dict], db: pd.DataFrame) -> str:
    """
    Three landscape longtables (Pathogen / Host / TypeOfDisease) showing
    every prediction alongside its GT and Pred DB metadata.
    Includes ablation column.  Wrong names / mismatched dims shown in red.
    """
    enriched = []
    for r in records:
        gt, pred = r["ground_truth"], r["prediction"]
        gt_meta   = match_disease(_norm(gt),   db)
        pred_meta = match_disease(_norm(pred),  db)
        ok = (r"\textcolor{green!50!black}{\textbf{\checkmark}}"
              if r["correct"]
              else r"\textcolor{red}{\textbf{$\times$}}")
        enriched.append({
            "img":       _tex(str(r.get("image_name", ""))),
            "gt":        _tex(gt),
            "pred":      _tex(pred),
            "gt_raw":    gt,
            "pred_raw":  pred,
            "abl":       _tex(r.get("ablation", "")),
            "organ":     _tex(r.get("organ_affected", "")),
            "gt_meta":   gt_meta,
            "pred_meta": pred_meta,
            "ok":        ok,
            "conf":      f"{r['confidence']:.2f}",
        })

    def _dim_cell(gt_v, pred_v):
        v = _tex(pred_v)
        return (r"\textcolor{red}{" + v + "}") if pred_v != gt_v else v

    def _pred_cell(gt_r, pred_r):
        v = _tex(pred_r)
        return (r"\textcolor{red}{" + v + "}") if pred_r != gt_r else v

    def _match(gt_v, pred_v):
        return (r"\textcolor{green!50!black}{=}"
                if gt_v == pred_v else r"\textcolor{red}{$\neq$}")

    col_fmt = r"p{2.5cm}p{2.8cm}p{3.2cm}p{2.8cm}p{3.2cm}ccp{0.6cm}p{0.6cm}p{0.9cm}"

    sections = []
    for dim_key, dim_label, letter in [
        ("Pathogen",      "Pathogen",        "A"),
        ("Host",          "Host",            "B"),
        ("TypeOfDisease", "Type of Disease", "C"),
    ]:
        h_cells = [
            "Image", "GT", f"GT {dim_label}",
            "Prediction", f"Pred {dim_label}",
            "Match", r"\checkmark", "Conf", "Abl", "Organ",
        ]
        header = " & ".join(f"\\textbf{{{h}}}" for h in h_cells) + r" \\"
        rows = []
        for e in enriched:
            gv = e["gt_meta"][dim_key]
            pv = e["pred_meta"][dim_key]
            rows.append(" & ".join([
                e["img"], e["gt"], _tex(gv),
                _pred_cell(e["gt_raw"], e["pred_raw"]),
                _dim_cell(gv, pv),
                _match(gv, pv),
                e["ok"], e["conf"], e["abl"], e["organ"],
            ]) + r" \\")
        sections.append(
            f"\\subsection*{{Table {letter} — {_tex(dim_label)}}}\n\n"
            + _longtable(col_fmt,
                         f"GT vs Prediction: {_tex(dim_label)}",
                         header, rows, len(h_cells))
        )
    return "\n\n\\bigskip\n\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT EXTRACTION  — reads live source files, never hardcoded
# ══════════════════════════════════════════════════════════════════════════════

# Ordered list of (variable_name, human_label, source_file_stem)
_PROMPT_REGISTRY = [
    # generate_symptoms.py
    ("_SYSTEM_A",       r"Registry-A System Prompt (doc extraction --- Local\_Registry)",  "generate_symptoms"),
    ("_SYSTEM_B",       "Registry-B System Prompt (web-enriched --- Registry)",           "generate_symptoms"),
    ("_PROMPT_B",       "Registry-B User Prompt template",                              "generate_symptoms"),
    ("_SYSTEM_REG_A",   "Registry-A Row-Builder System Prompt",                         "generate_symptoms"),
    ("_SYSTEM_REG_B",   "Registry-B Row-Builder System Prompt",                         "generate_symptoms"),
    ("_PROMPT_REG_B",   "Registry-B Row-Builder User Prompt template",                  "generate_symptoms"),
    # agent.py
    ("VISION_AGENT_PROMPT",      "Vision Agent System Prompt",         "agent"),
    ("DIAGNOSTIC_AGENT_PROMPT",  "Diagnostic Agent System Prompt",     "agent"),
    ("JUDGE_SYSTEM_PROMPT",      "Judge System Prompt",                "agent"),
    ("_DISCRIMINATOR_SYSTEM",    "Discriminator System Prompt",        "agent"),
]


def collect_prompts(base_dir: Path) -> list[dict]:
    """
    Dynamically import generate_symptoms.py and agent.py from base_dir,
    extract every prompt string listed in _PROMPT_REGISTRY, and return
    a list of {name, label, source, text} dicts.

    Falls back to reading the raw source via regex if import fails.
    """
    import importlib.util, textwrap as tw, re as _re

    def _load_module(path: Path):
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            return None

    def _regex_extract(path: Path) -> dict:
        """Fallback: pull triple-quoted strings from raw source."""
        try:
            src = path.read_text(encoding="utf-8")
        except Exception:
            return {}
        out = {}
        pat = _re.compile(
            r'^([A-Z_][A-Z0-9_]*)\s*=\s*(?:textwrap\.dedent\()?\s*"""(.*?)"""',
            _re.MULTILINE | _re.DOTALL,
        )
        for m in pat.finditer(src):
            out[m.group(1)] = tw.dedent(m.group(2)).strip()
        return out

    modules  = {}
    fallback = {}
    for stem in ("generate_symptoms", "agent"):
        p = base_dir / f"{stem}.py"
        if p.exists():
            mod = _load_module(p)
            if mod:
                modules[stem] = mod
            else:
                fallback[stem] = _regex_extract(p)
        else:
            print(f"  [WARN] {p} not found — prompts from '{stem}' will be missing.")

    results = []
    for varname, label, stem in _PROMPT_REGISTRY:
        text = None
        if stem in modules:
            text = getattr(modules[stem], varname, None)
        if text is None and stem in fallback:
            text = fallback[stem].get(varname)
        if text is None:
            text = f"(prompt '{varname}' not found in {stem}.py)"
        results.append({"name": varname, "label": label, "source": stem, "text": str(text).strip()})
    return results


def _prompts_to_latex(prompts: list[dict]) -> str:
    """Render collected prompts as a LaTeX appendix section."""
    if not prompts:
        return r"\section{Prompts}\textit{No prompts collected.}"

    lines = [r"\section{Prompts}", ""]
    lines.append(
        r"All prompts are extracted at report-generation time directly from "
        r"\texttt{generate\_symptoms.py} and \texttt{agent.py}. "
        r"Nothing is hardcoded in this report."
    )
    lines.append("")

    current_source = None
    for p in prompts:
        if p["source"] != current_source:
            current_source = p["source"]
            lines.append(r"\subsection*{\texttt{" + _tex(current_source + ".py") + "}}")
            lines.append("")

        lines.append(r"\subsubsection*{" + p["label"] + "}")
        lines.append(r"\begin{quote}\ttfamily\small\obeylines\obeyspaces")
        # Escape the prompt text for LaTeX verbatim-like display
        escaped = (p["text"]
                   .replace("\\", r"\textbackslash{}")
                   .replace("{",  r"\{")
                   .replace("}",  r"\}")
                   .replace("&",  r"\&")
                   .replace("%",  r"\%")
                   .replace("$",  r"\$")
                   .replace("#",  r"\#")
                   .replace("~",  r"\textasciitilde{}")
                   .replace("^",  r"\textasciicircum{}")
                   .replace("_",  r"\_"))
        lines.append(escaped)
        lines.append(r"\end{quote}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  LATEX REPORT WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_latex_report(
        tex_path: Path,
        tables_dim:       dict[str, pd.DataFrame],   # Pathogen / Host / TypeOfDisease
        ablation_df:      pd.DataFrame,
        crop_df:          pd.DataFrame,
        calibration_df:   pd.DataFrame,
        tool_df:          pd.DataFrame,
        organ_df:         pd.DataFrame,
        stage_df:         pd.DataFrame,
        confusion_mats:   dict[str, pd.DataFrame],
        records:          list[dict],
        db:               pd.DataFrame,
        total:            int,
        correct:          int,
        acc:              float,
        avg_conf:         float,
        ablation_modes:   list[str],
        crops_present:    list[str],
        prompts:          list[dict] = None,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── §1  Standard dimension tables ────────────────────────────────────────
    dim_meta = {
        "Pathogen":      ("Performance by Pathogen",
                          "Accuracy broken down by causative pathogen."),
        "TypeOfDisease": ("Performance by Type of Disease",
                          "Accuracy broken down by disease type (Fungal, Bacterial…)."),
        "Host":          ("Performance by Host / Organ",
                          "Accuracy broken down by host organ affected."),
    }
    sec_dims = []
    for dim, (title, desc) in dim_meta.items():
        df = tables_dim.get(dim, pd.DataFrame())
        if not df.empty:
            sec_dims.append(
                f"\\section{{{_tex(title)}}}\n{_tex(desc)}\n\n"
                + _dim_to_latex(df, dim, _tex(title))
            )

    # ── §2  Ablation comparison ───────────────────────────────────────────────
    sec_abl = (
        r"\section{Ablation Mode Comparison}" + "\n"
        r"Accuracy, calibration and tool-usage statistics per KB source combination. "
        r"local\,=\,Local\_Registry (doc extraction), web\,=\,Registry (web-enriched)." + "\n\n"
        + _ablation_to_latex(ablation_df)
    )

    # ── §3  Crop × ablation ───────────────────────────────────────────────────
    sec_crop = (
        r"\section{Per-Crop Breakdown}" + "\n"
        r"Accuracy and calibration for each crop$\,\times\,$ablation combination. "
        r"Matches the \texttt{run\_agent\_on\_crop()} unit in \texttt{agent.py}." + "\n\n"
        + _crop_to_latex(crop_df)
    )

    # ── §4  Calibration ───────────────────────────────────────────────────────
    sec_cal = (
        r"\section{Calibration Breakdown}" + "\n"
        r"Judge verdict counts and percentages per crop $\times$ ablation." + "\n\n"
        + _calibration_to_latex(calibration_df)
    )

    # ── §5  Vision agent breakdowns ───────────────────────────────────────────
    sec_vision = (
        r"\section{Vision Agent Breakdowns}" + "\n"
        r"These breakdowns are derived from the \texttt{vision\_features} "
        r"dict stored in each log by \texttt{run\_vision\_agent()}." + "\n\n"
        r"\subsection*{By Primary Organ Affected}" + "\n"
        + _dim_to_latex(organ_df, "organ_affected",
                        "Accuracy by primary organ affected") + "\n\n"
        r"\subsection*{By Approximate Growth Stage}" + "\n"
        + _dim_to_latex(stage_df, "growth_stage",
                        "Accuracy by approximate growth stage")
    )

    # ── §6  Confusion matrices ────────────────────────────────────────────────
    cm_parts = [
        r"\section{Confusion Matrices}" + "\n"
        r"Diagonal = correct (\textbf{bold}); off-diagonal in "
        r"\textcolor{red}{red}."
    ]
    for name, mat in confusion_mats.items():
        cm_parts.append(
            f"\\subsection*{{{_tex(name)}}}\n"
            + _confusion_to_latex(mat, f"Confusion matrix: {name}")
        )
    sec_cm = "\n\n".join(cm_parts)

    # ── §7  All predictions detail ────────────────────────────────────────────
    sec_detail = (
        r"\section{All Predictions vs Validator}" + "\n"
        r"Every prediction showing GT and Pred alongside Pathogen, Host, "
        r"and Type of Disease. "
        r"\textcolor{red}{Red} = mismatch. "
        r"``Abl'' = ablation mode; ``Organ'' = vision-agent organ assessment." + "\n\n"
        r"\begin{landscape}\footnotesize" + "\n"
        + _detail_tables_to_latex(records, db)
        + "\n" + r"\normalsize\end{landscape}"
    )

    # ── §8  Prompts appendix ──────────────────────────────────────────────────
    sec_prompts = _prompts_to_latex(prompts or [])

    body = "\n\n".join([
        "\n\n".join(sec_dims),
        sec_abl, sec_crop, sec_cal, sec_vision, sec_cm, sec_detail,
        r"\appendix", sec_prompts,
    ])

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{booktabs,longtable,array,xcolor,hyperref,microtype,amssymb,pdflscape,adjustbox}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black}

\title{\textbf{Plant Disease Agent --- Performance Report}\\[0.4em]
       \large Breakdown by Pathogen, Type, Host, Crop, Ablation \& Vision Features}
\author{Auto-generated by \texttt{performance\_analysis.py}}
\date{""" + _tex(now) + r"""}

\begin{document}
\maketitle\tableofcontents\newpage

\section*{Overall Summary}
\begin{tabular}{ll}\toprule
\textbf{Metric} & \textbf{Value} \\\midrule
Total images        & """ + str(total)     + r""" \\
Correct             & """ + str(correct)   + r""" \\
Accuracy            & """ + f"{acc:.1f}\\%" + r""" \\
Avg confidence      & """ + f"{avg_conf:.3f}" + r""" \\
Crops               & """ + _tex(", ".join(crops_present))  + r""" \\
Ablation modes      & """ + _tex(", ".join(ablation_modes)) + r""" \\
\bottomrule\end{tabular}

""" + body + r"""
\end{document}
"""
    tex_path.write_text(tex, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  DEBUG SAMPLES
# ══════════════════════════════════════════════════════════════════════════════

def _print_debug_samples(logs: list[dict], db: pd.DataFrame, n: int = 5) -> None:
    print("\n" + "=" * 76)
    print(f"  DEBUG — first {n} logs: GT vs Pred DB metadata")
    print("=" * 76)
    for i, log in enumerate(logs[:n]):
        gt   = str(log.get("ground_truth", "")).strip()
        pred = str(log.get("prediction",   "")).strip()
        conf = float(log.get("confidence", 0.0))
        abl  = log.get("_ablation", "?")
        crop = log.get("_crop", "?")
        ok   = "✓" if log.get("correct", pred == gt) else "✗"

        gt_meta   = match_disease(_norm(gt),   db)
        pred_meta = match_disease(_norm(pred),  db)

        trace = log.get("trace", [])
        final = next((e for e in reversed(trace)
                      if e.get("phase") == "final_prediction"), {})
        reason = final.get("parsed", {}).get("reasoning", "")[:180]

        vf    = log.get("vision_features", {}) or {}
        organ = vf.get("primary_organ_affected", "?")
        stage = vf.get("approximate_growth_stage", "?")

        print(f"\n  [{i+1}] {log.get('image_name','')}  crop={crop}  "
              f"abl={abl}  {ok}  conf={conf:.2f}")
        print(f"       organ={organ}  stage={stage}")
        print(f"  {'─'*72}")
        print(f"  {'Field':<20}  {'GROUND TRUTH  (' + gt[:20] + ')':<32}  "
              f"{'PREDICTION  (' + pred[:20] + ')':<32}")
        print(f"  {'─'*72}")
        for field in ("Pathogen", "Host", "TypeOfDisease"):
            gv = gt_meta.get(field, "?")
            pv = pred_meta.get(field, "?")
            print(f"  {field:<20}  {gv:<32}  {pv:<32}  "
                  f"{'==' if gv == pv else '!='}")
        if reason:
            print(f"\n  Reason: {reason}")
    print("\n" + "=" * 76 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _safe_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(); return ""


def main() -> None:
    # Default paths — same layout as agent.py
    logs_dir = Path(__file__).parent / "results" / "agent" / "logs"
    out_dir  = Path(__file__).parent / "results" / "agent"
    db_path  = Path(__file__).parent / "crop_disease_registry_updated.xlsx"

    print("=" * 76)
    print("  AGENT PERFORMANCE ANALYSIS")
    print("  Log layout: results/agent/logs/<crop>/<ablation>/*_log.json")
    print("=" * 76)

    raw = _safe_input(f"\n  Logs dir  [{logs_dir}]\n  Press Enter to accept or type a new path: ")
    if raw: logs_dir = Path(raw)
    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}\nRun agent.py first.")

    raw = _safe_input(f"\n  Database  [{db_path}]\n  Press Enter to accept or type a new path: ")
    if raw: db_path = Path(raw)

    raw = _safe_input(f"\n  Output dir  [{out_dir}]\n  Press Enter to accept or type a new path: ")
    if raw: out_dir = Path(raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Ablation modes: {sorted(VALID_ABLATION_MODES)}")
    raw = _safe_input("  Filter to one ablation? [Enter = all / local / web]: ")
    abl_filter = raw.lower().strip() if raw else None
    if abl_filter and abl_filter not in VALID_ABLATION_MODES:
        raise SystemExit(f"Invalid ablation '{abl_filter}'. Valid: {sorted(VALID_ABLATION_MODES)}")

    raw = _safe_input("\n  Compile PDF report? [Enter = yes / n = skip]: ")
    compile_pdf = raw.lower() not in ("n", "no")

    print()
    print("=" * 76)
    print(f"  Logs dir      : {logs_dir}")
    print(f"  Database      : {db_path}")
    print(f"  Output dir    : {out_dir}")
    if abl_filter:
        print(f"  Ablation      : {abl_filter}")
    print("=" * 76)

    # ── 1. Database ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading validation database …")
    try:
        db = load_database(db_path)
        print(f"  {len(db)} rows loaded. "
              f"Columns: {[col for col in db.columns if not col.startswith('_')]}")
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        print("  Continuing without metadata — dimensions will be 'Unknown'.")
        db = pd.DataFrame(columns=["Disease", "_norm", "Pathogen", "Host", "TypeOfDisease"])

    # ── 2. Collect logs ───────────────────────────────────────────────────────
    print("\n[2/6] Collecting log files …")
    logs = collect_logs(logs_dir, ablation_filter=abl_filter)
    if not logs:
        raise SystemExit(f"No *_log.json files found under {logs_dir}")
    print(f"  {len(logs)} log file(s) collected.")
    ablation_modes = sorted(set(l["_ablation"] for l in logs))
    crops_present  = sorted(set(l["_crop"]     for l in logs))
    print(f"  Crops          : {crops_present}")
    print(f"  Ablation modes : {ablation_modes}")
    _print_debug_samples(logs, db)

    # ── 3. Enrich records ─────────────────────────────────────────────────────
    print("\n[3/6] Enriching records …")
    records = enrich_records(logs, db)
    print(f"  {len(records)} records enriched.")

    # ── 4. Build tables + save CSVs ───────────────────────────────────────────
    print("\n[4/6] Building analysis tables …")

    def _save(df: pd.DataFrame, name: str) -> None:
        p = out_dir / name
        df.to_csv(p, index=False)
        print(f"  Saved: {p}")

    tables_dim: dict[str, pd.DataFrame] = {}
    for dim, fname in [
        ("Pathogen",      "performance_by_pathogen.csv"),
        ("TypeOfDisease", "performance_by_type_of_disease.csv"),
        ("Host",          "performance_by_host.csv"),
    ]:
        df = build_dimension_table(records, dim)
        tables_dim[dim] = df
        _save(df, fname)

    ablation_df    = build_ablation_comparison(records)
    _save(ablation_df, "ablation_comparison.csv")

    crop_df = build_crop_table(records)
    _save(crop_df, "performance_by_crop.csv")

    calibration_df = build_calibration_breakdown(records)
    _save(calibration_df, "calibration_breakdown.csv")

    tool_df = build_tool_usage_stats(records)
    _save(tool_df, "tool_usage_stats.csv")

    organ_df = build_vision_organ_breakdown(records)
    _save(organ_df, "vision_organ_breakdown.csv")

    stage_df = build_growth_stage_breakdown(records)
    _save(stage_df, "growth_stage_breakdown.csv")

    confusion_mats: dict[str, pd.DataFrame] = {}
    for crop in crops_present:
        cm = build_confusion_matrix(records, crop_filter=crop)
        if not cm.empty:
            _save(cm, f"confusion_matrix_{crop}.csv")
            confusion_mats[crop] = cm
    overall_cm = build_confusion_matrix(records)
    if not overall_cm.empty:
        _save(overall_cm, "confusion_matrix_overall.csv")
        confusion_mats["Overall"] = overall_cm

    enriched_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in records
    ])
    _save(enriched_df, "all_predictions_enriched.csv")

    # ── 5. Console summary ────────────────────────────────────────────────────
    total    = len(records)
    correct  = sum(1 for r in records if r["correct"])
    acc      = _pct(correct, total)
    avg_conf = _mean([r["confidence"] for r in records])

    print(f"\n{'='*76}")
    print(f"  OVERALL  {correct}/{total}  ({acc:.1f}%)  avg_conf={_fmt(avg_conf)}")
    print(f"{'='*76}")
    print(_ablation_table_str(ablation_df))
    print(_crop_table_str(crop_df))
    for dim in ("Pathogen", "TypeOfDisease", "Host"):
        print(_dim_table_str(tables_dim[dim], dim))
    print(_dim_table_str(organ_df, "organ_affected", "Vision organ breakdown"))
    print(_dim_table_str(stage_df, "growth_stage",   "Growth stage breakdown"))

    # ── 6. LaTeX + optional PDF ───────────────────────────────────────────────
    print("\n[6/6] Writing LaTeX report …")
    tex_path = out_dir / "performance_report.tex"

    print("  Collecting prompts from source files …", end=" ", flush=True)
    prompts = collect_prompts(Path(__file__).parent)
    print(f"{len(prompts)} prompt(s) collected.")

    write_latex_report(
        tex_path       = tex_path,
        tables_dim     = tables_dim,
        ablation_df    = ablation_df,
        crop_df        = crop_df,
        calibration_df = calibration_df,
        tool_df        = tool_df,
        organ_df       = organ_df,
        stage_df       = stage_df,
        confusion_mats = confusion_mats,
        records        = records,
        db             = db,
        total          = total,
        correct        = correct,
        acc            = acc,
        avg_conf       = avg_conf,
        ablation_modes = ablation_modes,
        crops_present  = crops_present,
        prompts        = prompts,
    )
    print(f"  Saved: {tex_path}")

    if compile_pdf:
        if shutil.which("pdflatex"):
            print("  Compiling PDF …", end=" ", flush=True)
            try:
                for _ in range(2):   # double-pass for longtable page numbers
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode",
                         "-output-directory", str(out_dir), str(tex_path)],
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, cwd=str(out_dir),
                    )
                print(f"done  →  {tex_path.with_suffix('.pdf')}")
            except subprocess.CalledProcessError as e:
                print(f"FAILED (exit {e.returncode})")
                print(f"  Compile manually:  pdflatex {tex_path}")
        else:
            print("  [WARN] pdflatex not on PATH — skipping PDF compilation.")
            print(f"         Run manually:  pdflatex {tex_path}")

    print(f"\n{'='*76}")
    print(f"  Done.  All outputs → {out_dir}/")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()