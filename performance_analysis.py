#!/usr/bin/env python3
from __future__ import annotations
"""
performance_analysis.py
=======================
Downstream analysis of agent prediction logs, broken down by:
  • Pathogen
  • Type of Disease
  • Host

Reads:
  1. Agent log JSON files  — results/agent/logs/<dataset>/<image>_log.json
  2. Validation database   — database.xlsx (columns: Disease, Pathogen,
                             Host, Type of Disease, etc.)

Writes:
  • results/agent/performance_by_pathogen.csv
  • results/agent/performance_by_type_of_disease.csv
  • results/agent/performance_by_host.csv
  • results/agent/performance_summary.txt   (human-readable report)

Usage:
  python performance_analysis.py
  python performance_analysis.py --logs results/agent/logs --db /path/to/database.xlsx
"""

import os
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required.  Run: pip install pandas openpyxl")


# ── helpers ────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fuzzy matching."""
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_database(db_path: Path) -> pd.DataFrame:
    """Load the validation database and return a normalised DataFrame."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    df = pd.read_excel(db_path, engine="openpyxl")

    # Flexible column detection — accept any capitalisation / spacing
    col_map = {}
    for col in df.columns:
        norm = _normalise(str(col))
        if norm in ("disease", "disease name"):
            col_map["Disease"] = col
        elif norm in ("pathogen", "pathogen name"):
            col_map["Pathogen"] = col
        elif norm in ("host",):
            col_map["Host"] = col
        elif norm in ("type of disease", "type", "disease type"):
            col_map["TypeOfDisease"] = col
        elif norm in ("crop",):
            col_map["Crop"] = col

    missing = [k for k in ("Disease", "Pathogen", "Host", "TypeOfDisease")
               if k not in col_map]
    if missing:
        print(f"  [WARN] Columns not found in database: {missing}")
        print(f"         Available columns: {list(df.columns)}")

    rename = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename)

    # Build a normalised disease key for matching
    if "Disease" in df.columns:
        df["_disease_norm"] = df["Disease"].astype(str).map(_normalise)

    return df


def collect_logs(logs_dir: Path) -> list[dict]:
    """Recursively collect all *_log.json files under logs_dir."""
    logs = []
    for p in sorted(logs_dir.rglob("*_log.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_log_path"] = str(p)
            logs.append(data)
        except Exception as e:
            print(f"  [WARN] Could not read {p}: {e}")
    return logs


def match_disease(norm_name: str, db: pd.DataFrame) -> dict:
    """Return metadata dict for the best-matching disease row in db."""
    empty = {"Pathogen": "Unknown", "Host": "Unknown", "TypeOfDisease": "Unknown"}
    if "_disease_norm" not in db.columns:
        return empty

    # Exact match first
    rows = db[db["_disease_norm"] == norm_name]
    if rows.empty:
        # Partial match — disease name is contained in the db entry or vice-versa
        rows = db[
            db["_disease_norm"].apply(
                lambda x: norm_name in x or x in norm_name
            )
        ]
    if rows.empty:
        return empty

    row = rows.iloc[0]
    return {
        "Pathogen":      str(row.get("Pathogen",      "Unknown")).strip(),
        "Host":          str(row.get("Host",          "Unknown")).strip(),
        "TypeOfDisease": str(row.get("TypeOfDisease", "Unknown")).strip(),
    }


# ── aggregation ────────────────────────────────────────────────────────────────

def build_dimension_table(records: list[dict], dim: str) -> pd.DataFrame:
    """
    Given records with keys [dim, correct, confidence], return a DataFrame:
        dim | total | correct | accuracy | avg_confidence | avg_correct_conf | avg_wrong_conf
    """
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r[dim]].append(r)

    rows = []
    for key, items in sorted(groups.items()):
        total   = len(items)
        correct = sum(1 for i in items if i["correct"])
        acc     = correct / total * 100

        all_conf          = [i["confidence"] for i in items]
        correct_conf_vals = [i["confidence"] for i in items if i["correct"]]
        wrong_conf_vals   = [i["confidence"] for i in items if not i["correct"]]

        avg_conf          = sum(all_conf) / len(all_conf)
        avg_correct_conf  = (sum(correct_conf_vals) / len(correct_conf_vals)
                             if correct_conf_vals else float("nan"))
        avg_wrong_conf    = (sum(wrong_conf_vals)   / len(wrong_conf_vals)
                             if wrong_conf_vals else float("nan"))

        rows.append({
            dim:                  key,
            "total":              total,
            "correct":            correct,
            "accuracy_%":         round(acc, 1),
            "avg_confidence":     round(avg_conf, 3),
            "avg_conf_correct":   round(avg_correct_conf, 3) if correct_conf_vals else "–",
            "avg_conf_wrong":     round(avg_wrong_conf,   3) if wrong_conf_vals   else "–",
        })

    return pd.DataFrame(rows)


# ── report ─────────────────────────────────────────────────────────────────────

def _table_str(df: pd.DataFrame, dim: str) -> str:
    lines = [f"\n{'─'*72}", f"  Performance by {dim}", f"{'─'*72}"]
    if df.empty:
        lines.append("  (no data)")
        return "\n".join(lines)

    header = (f"  {'':40s}  {'N':>5}  {'Correct':>7}  {'Acc%':>6}  "
              f"{'AvgConf':>8}  {'ConfOK':>7}  {'ConfBAD':>7}")
    lines.append(header)
    lines.append("  " + "-" * 68)

    for _, row in df.iterrows():
        key = str(row[dim])[:40]
        lines.append(
            f"  {key:<40}  {row['total']:>5}  {row['correct']:>7}  "
            f"{row['accuracy_%']:>5.1f}%  {row['avg_confidence']:>8.3f}  "
            f"{str(row['avg_conf_correct']):>7}  {str(row['avg_conf_wrong']):>7}"
        )
    return "\n".join(lines)


# ── LaTeX report writer ────────────────────────────────────────────────────────

def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for char, replacement in replacements:
        s = s.replace(char, replacement)
    return s


def _df_to_latex_table(df: pd.DataFrame, dim: str, caption: str) -> str:
    """Render a performance DataFrame as a LaTeX longtable."""
    col_labels = {
        dim:               dim,
        "total":           "N",
        "correct":         "Correct",
        "accuracy_%":      "Acc (\\%)",
        "avg_confidence":  "Avg Conf",
        "avg_conf_correct":"Conf (OK)",
        "avg_conf_wrong":  "Conf (Bad)",
    }
    cols = list(col_labels.keys())
    header = " & ".join(col_labels[c] for c in cols) + r" \\"

    rows_tex = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if c == dim:
                cells.append(_tex_escape(str(val)))
            elif c == "accuracy_%":
                cells.append(f"{val:.1f}")
            elif c == "avg_confidence":
                cells.append(f"{val:.3f}")
            elif c in ("avg_conf_correct", "avg_conf_wrong"):
                cells.append(str(val) if val == "–" else f"{float(val):.3f}")
            else:
                cells.append(str(val))
        rows_tex.append(" & ".join(cells) + r" \\")

    n_cols  = len(cols)
    col_fmt = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{longtable}{" + col_fmt + "}",
        r"\caption{" + caption + r"} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{" + str(n_cols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines += rows_tex
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _make_longtable(col_fmt, caption, header, rows_tex, n_cols):
    """Generic longtable builder."""
    lines = [
        r"\begin{longtable}{" + col_fmt + "}",
        r"\caption{" + caption + r"} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{" + str(n_cols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines += rows_tex
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _records_to_detail_tables(records, db):
    """Three longtables: one each for Pathogen, Host, TypeOfDisease."""
    enriched = []
    for r in records:
        gt   = r["ground_truth"]
        pred = r["prediction"]
        gt_meta   = match_disease(_normalise(gt),   db)
        pred_meta = match_disease(_normalise(pred),  db)
        ok_sym = (r"\textcolor{green!50!black}{\textbf{\checkmark}}"
                  if r["correct"]
                  else r"\textcolor{red}{\textbf{$\times$}}")
        enriched.append({
            "image":      _tex_escape(str(r.get("image_name", ""))),
            "gt":         _tex_escape(gt),
            "pred":       _tex_escape(pred),
            "gt_raw":     gt,
            "pred_raw":   pred,
            "gt_meta":    gt_meta,
            "pred_meta":  pred_meta,
            "ok_sym":     ok_sym,
            "conf":       f"{r['confidence']:.2f}",
        })

    def dim_cell(gt_val, pred_val):
        v = _tex_escape(pred_val)
        return (r"\textcolor{red}{" + v + "}") if pred_val != gt_val else v

    def pred_name_cell(gt_raw, pred_raw):
        v = _tex_escape(pred_raw)
        return (r"\textcolor{red}{" + v + "}") if pred_raw != gt_raw else v

    def match_cell(gt_val, pred_val):
        return (r"\textcolor{green!50!black}{=}"
                if gt_val == pred_val
                else r"\textcolor{red}{$\neq$}")

    # Landscape usable ~24.7cm
    col_fmt = r"p{3.8cm}p{3.8cm}p{4.0cm}p{3.8cm}p{4.0cm}ccp{0.9cm}"

    dim_configs = [
        ("Pathogen",      "Pathogen",       "A"),
        ("Host",          "Host",           "B"),
        ("TypeOfDisease", "Type of Disease","C"),
    ]

    sections = []
    for dim_key, dim_label, letter in dim_configs:
        h_cells = [
            "Image", "Ground Truth", f"GT {dim_label}",
            "Prediction", f"Pred {dim_label}", "Match",
            r"\checkmark", "Conf"
        ]
        header = " & ".join(f"\\textbf{{{h}}}" for h in h_cells) + r" \\"
        rows = []
        for e in enriched:
            gt_v   = e["gt_meta"][dim_key]
            pred_v = e["pred_meta"][dim_key]
            rows.append(" & ".join([
                e["image"],
                e["gt"],
                _tex_escape(gt_v),
                pred_name_cell(e["gt_raw"], e["pred_raw"]),
                dim_cell(gt_v, pred_v),
                match_cell(gt_v, pred_v),
                e["ok_sym"],
                e["conf"],
            ]) + r" \\")
        sections.append(
            f"\\subsection*{{Table {letter} --- {_tex_escape(dim_label)}}}\n\n"
            + _make_longtable(col_fmt, f"GT vs Prediction: {dim_label}",
                              header, rows, len(h_cells))
        )

    return "\n\n\\bigskip\n\n".join(sections)


def write_latex_report(tex_path: Path, tables: dict, records: list,
                       db: pd.DataFrame,
                       total: int, correct: int,
                       acc: float, avg_conf: float) -> None:
    """Write the full LaTeX report to tex_path."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    dim_meta = {
        "Pathogen":      ("Performance by Pathogen",        "Accuracy broken down by causative pathogen."),
        "TypeOfDisease": ("Performance by Type of Disease",  "Accuracy broken down by disease type (Fungal, Bacterial, etc.)."),
        "Host":          ("Performance by Host / Organ",     "Accuracy broken down by host organ affected."),
    }

    table_blocks = []
    for dim, (title, desc) in dim_meta.items():
        df = tables.get(dim, pd.DataFrame())
        if df.empty:
            continue
        caption = _tex_escape(title)
        table_blocks.append(
            f"\\section{{{_tex_escape(title)}}}\n"
            f"{_tex_escape(desc)}\n\n"
            + _df_to_latex_table(df, dim, caption)
        )

    # Full detail section
    detail_section = (
        r"\section{All Predictions vs Validator}" + "\n"
        r"Every image showing Ground Truth and Prediction with their Pathogen, Host, "
        r"and Type of Disease from the validation database. "
        r"\textcolor{red}{Red cells} indicate a mismatch between GT and Prediction." + "\n\n"
        r"\begin{landscape}\footnotesize" + "\n"
        + _records_to_detail_tables(records, db)
        + "\n" + r"\normalsize\end{landscape}"
    )

    body = "\n\n".join(table_blocks) + "\n\n" + detail_section

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{amssymb}
\usepackage{pdflscape}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!60!black,
  urlcolor=blue!60!black,
}

\title{\textbf{Plant Disease Agent --- Performance Report}\\[0.4em]
       \large Breakdown by Pathogen, Type of Disease, and Host}
\author{Auto-generated by \texttt{performance\_analysis.py}}
\date{""" + _tex_escape(now) + r"""}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section*{Overall Summary}
\begin{tabular}{ll}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total images evaluated & """ + str(total) + r""" \\
Correct predictions    & """ + str(correct) + r""" \\
Overall accuracy       & """ + f"{acc:.1f}\\%" + r""" \\
Average confidence     & """ + f"{avg_conf:.3f}" + r""" \\
\bottomrule
\end{tabular}

""" + body + r"""

\end{document}
"""

    tex_path.write_text(tex, encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse agent performance by Pathogen / Host / Type of Disease."
    )
    parser.add_argument(
        "--logs",
        default=str(Path(__file__).parent / "results" / "agent" / "logs"),
        help="Root directory containing *_log.json files (searched recursively).",
    )
    parser.add_argument(
        "--db",
        default=str(Path("/home/user/Desktop/Disease Images for Arti/database.xlsx")),
        help="Path to the validation database Excel file.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results" / "agent"),
        help="Output directory for CSV and summary files.",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs)
    db_path  = Path(args.db)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  AGENT PERFORMANCE ANALYSIS — by Pathogen / Host / Type of Disease")
    print("=" * 72)
    print(f"  Log directory : {logs_dir}")
    print(f"  Database      : {db_path}")
    print(f"  Output dir    : {out_dir}")

    # ── 1. Load database ──────────────────────────────────────────────────────
    print("\n[1/4] Loading validation database …")
    try:
        db = load_database(db_path)
        print(f"  Loaded {len(db)} rows.  "
              f"Columns: {[c for c in db.columns if not c.startswith('_')]}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Proceeding without metadata — all dimensions will be 'Unknown'.")
        db = pd.DataFrame(columns=["Disease", "_disease_norm",
                                   "Pathogen", "Host", "TypeOfDisease"])

    # ── 2. Collect log files ──────────────────────────────────────────────────
    print("\n[2/4] Collecting agent log files …")
    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}\n"
                         "Run agent.py first to generate prediction logs.")

    logs = collect_logs(logs_dir)
    if not logs:
        raise SystemExit(f"No *_log.json files found under {logs_dir}")

    print(f"  Found {len(logs)} log file(s).")



    # ── DEBUG: 5-sample comparison — DB metadata for GT vs DB metadata for Pred ──
    print("\n" + "=" * 72)
    print("  DEBUG — 5 samples: DB metadata for Ground Truth vs Prediction")
    print("  Checks if Pathogen / Host / TypeOfDisease match between GT and Pred")
    print("=" * 72)

    for i, log in enumerate(logs[:5]):
        gt   = str(log.get("ground_truth", "")).strip()
        pred = str(log.get("prediction",   "")).strip()
        conf = float(log.get("confidence", 0.0))
        ok   = "\u2713 CORRECT" if log.get("correct", pred == gt) else "\u2717 WRONG"

        # Look up DB metadata for BOTH ground truth and prediction
        gt_meta   = match_disease(_normalise(gt),   db)
        pred_meta = match_disease(_normalise(pred),  db)

        # Pull reasoning snippet from trace
        trace = log.get("trace", [])
        final_entry = next(
            (e for e in reversed(trace) if e.get("phase") == "final_prediction"), {}
        )
        reasoning = final_entry.get("parsed", {}).get("reasoning", "")[:200]

        print(f"\n  [{i+1}] {log.get('image_name', '')}  →  {ok}  (conf={conf:.2f})")
        print(f"  {'─'*68}")
        print(f"  {'Field':<20}  {'GROUND TRUTH (' + gt + ')':<30}  {'PREDICTION (' + pred + ')':<30}")
        print(f"  {'─'*68}")

        for field in ("Pathogen", "Host", "TypeOfDisease"):
            gt_val   = gt_meta.get(field,   "?")
            pred_val = pred_meta.get(field, "?")
            match_sym = "==" if gt_val == pred_val else "!="
            print(f"  {field:<20}  {gt_val:<30}  {pred_val:<30}  {match_sym}")

        if reasoning:
            print(f"\n  Agent reasoning: {reasoning}")

    print("\n" + "=" * 72 + "\n")

    # ── 3. Build enriched records ─────────────────────────────────────────────
    print("\n[3/4] Joining log predictions with database metadata …")
    records = []
    unmatched = []

    for log in logs:
        gt         = str(log.get("ground_truth", "")).strip()
        pred       = str(log.get("prediction",   "")).strip()
        correct    = bool(log.get("correct", pred == gt))
        confidence = float(log.get("confidence", 0.0))

        norm_gt   = _normalise(gt)
        meta      = match_disease(norm_gt, db)

        if meta["Pathogen"] == "Unknown":
            unmatched.append(gt)

        records.append({
            "image_name":    log.get("image_name", ""),
            "ground_truth":  gt,
            "prediction":    pred,
            "correct":       correct,
            "confidence":    confidence,
            "Pathogen":      meta["Pathogen"],
            "Host":          meta["Host"],
            "TypeOfDisease": meta["TypeOfDisease"],
        })

    if unmatched:
        unique_unmatched = sorted(set(unmatched))
        print(f"  [WARN] {len(unmatched)} record(s) had no database match "
              f"({len(unique_unmatched)} unique diseases):")
        for d in unique_unmatched[:10]:
            print(f"         • {d}")
        if len(unique_unmatched) > 10:
            print(f"         … and {len(unique_unmatched) - 10} more")

    # ── 4. Aggregate and write ────────────────────────────────────────────────
    print("\n[4/4] Aggregating and writing results …")

    dims = [
        ("Pathogen",      "performance_by_pathogen.csv"),
        ("TypeOfDisease", "performance_by_type_of_disease.csv"),
        ("Host",          "performance_by_host.csv"),
    ]

    report_sections = []
    tables: dict[str, pd.DataFrame] = {}

    for dim, fname in dims:
        df = build_dimension_table(records, dim)
        tables[dim] = df
        out_csv = out_dir / fname
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")
        report_sections.append(_table_str(df, dim))

    # ── overall summary ───────────────────────────────────────────────────────
    total   = len(records)
    correct = sum(1 for r in records if r["correct"])
    acc     = correct / total * 100 if total else 0
    avg_conf = sum(r["confidence"] for r in records) / total if total else 0

    # ── print to console ──────────────────────────────────────────────────────
    print(f"\n  Overall accuracy : {correct}/{total} ({acc:.1f}%)")
    print(f"  Avg confidence   : {avg_conf:.3f}")

    # ── also save a full enriched CSV ─────────────────────────────────────────
    enriched_df = pd.DataFrame(records)
    enriched_path = out_dir / "all_predictions_enriched.csv"
    enriched_df.to_csv(enriched_path, index=False)
    print(f"  Saved full enriched log: {enriched_path}")

    # ── write LaTeX report ────────────────────────────────────────────────────
    tex_path = out_dir / "performance_report.tex"
    write_latex_report(
        tex_path=tex_path,
        tables=tables,
        records=records,
        db=db,
        total=total,
        correct=correct,
        acc=acc,
        avg_conf=avg_conf,
    )
    print(f"  Saved LaTeX report : {tex_path}")

    # ── compile to PDF ────────────────────────────────────────────────────────
    import shutil, subprocess
    pdf_path = tex_path.with_suffix(".pdf")

    if shutil.which("pdflatex"):
        print("  Compiling to PDF via pdflatex …", end=" ", flush=True)
        try:
            # Run twice so longtable page numbers and TOC are correct
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode",
                     "-output-directory", str(out_dir),
                     str(tex_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(out_dir),
                )
            print(f"done")
            print(f"  Saved PDF report   : {pdf_path}")
        except subprocess.CalledProcessError as e:
            print(f"FAILED (pdflatex exited with error {e.returncode})")
            print(f"  The .tex file is still valid — compile manually:")
            print(f"    pdflatex {tex_path}")
    else:
        print("  [WARN] pdflatex not found on PATH — skipping PDF compilation.")
        print(f"  Install TeX Live / MiKTeX and run:")
        print(f"    pdflatex {tex_path}")


if __name__ == "__main__":
    main()