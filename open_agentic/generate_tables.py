"""Generate LaTeX tables from summary JSONs.

Reads all results and produces .tex table files that can be
\\input{} from main.tex. Tables auto-update when experiments are rerun.

Usage:
    python -m CyberVisionAg.open_agentic.generate_tables
"""

import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "open_agentic"
TABLES_OUT = Path(__file__).resolve().parents[2] / "writing" / "69aae430e8bdcbd9056bf911" / "tables"

# Configs to exclude from tables (experimental/visual KB artifacts)
EXCLUDE_SOURCES = {"Corn_Diseases"}  # the doubled-path visual KB result


def _load(crop, source, model, k):
    """Load accuracy from a summary.json. Returns accuracy % or None."""
    path = RESULTS_DIR / crop / source / model / f"k{k}" / "summary.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d["metrics"]["accuracy"]


def _fmt(acc, bold=False):
    """Format accuracy as 'X.Y%' or '—'."""
    if acc is None:
        return "—"
    s = f"{acc:.1f}\\%"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def _fmt_with_delta(acc, baseline, bold=False):
    """Format as 'X.Y% (+D.D)' showing both accuracy and delta from baseline."""
    if acc is None:
        return "—"
    s = f"{acc:.1f}\\%"
    if baseline is not None:
        delta = acc - baseline
        sign = "+" if delta >= 0 else ""
        s += f" ({sign}{delta:.1f})"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def _fmt_delta(acc, baseline, bold=False):
    """Format as delta from baseline: '+X.Y' or '—'."""
    if acc is None or baseline is None:
        return "—"
    delta = acc - baseline
    sign = "+" if delta >= 0 else ""
    s = f"{sign}{delta:.1f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


# ── Table 2: Main results (method × k, sonnet) ───────────────────────────────

def table_main_results():
    """Compact table: zero-shot + agent methods, accuracy % only."""
    crops = [
        ("Soybean_Diseases", "Soybean (27)", ["none", "local", "internet"]),
        ("Corn_Diseases", "Corn (24)", ["none", "internet"]),
        ("Mango_Leaf_Disease", "Mango (7)", ["none", "internet"]),
    ]
    ks = [1, 4, 8, 16]
    method_labels = {
        "none": "Agent (no KB)",
        "local": "Agent + local KB",
        "internet": "Agent + internet KB",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Diagnostic accuracy across crops, methods, and reference budgets $k$ (Sonnet model, 3 test images per class). Parentheses in the Crop column denote number of disease classes. Values show accuracy\,\% with improvement over zero-shot in parentheses. Best per crop--$k$ in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{ll" + "r" * len(ks) + "}")
    lines.append(r"\toprule")
    lines.append(r"Crop & Method & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
    lines.append(r"\midrule")

    for crop_id, crop_label, sources in crops:
        zs = _load(crop_id, "few_shot", "sonnet", 0)

        # Collect all agent results to find best per k
        all_results = {}
        for source in sources:
            label = method_labels[source]
            all_results[label] = {k: _load(crop_id, source, "sonnet", k) for k in ks}

        best_per_k = {}
        for k in ks:
            vals = [all_results[m][k] for m in all_results if all_results[m][k] is not None]
            best_per_k[k] = max(vals) if vals else -1

        # Zero-shot row
        zs_str = _fmt(zs)
        lines.append(f"{crop_label} & Zero-shot & " + " & ".join([zs_str] * len(ks)) + r" \\")

        # Agent rows with delta from zero-shot
        for source in sources:
            label = method_labels[source]
            cells = []
            for k in ks:
                acc = all_results[label][k]
                is_best = acc is not None and abs(acc - best_per_k[k]) < 0.01
                cells.append(_fmt_with_delta(acc, zs, bold=is_best))
            lines.append(f" & {label} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Table 3: Model ablation ──────────────────────────────────────────────────

def table_model_ablation():
    """Model ablation: haiku/sonnet/opus at internet KB, k=8."""
    crops = [
        ("Soybean_Diseases", "Soybean"),
        ("Corn_Diseases", "Corn"),
        ("Mango_Leaf_Disease", "Mango"),
    ]
    models = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus")]

    # Find best per crop for bolding
    best_per_crop = {}
    for crop_id, _ in crops:
        vals = [_load(crop_id, "internet", m, 8) for m, _ in models]
        best_per_crop[crop_id] = max(v for v in vals if v is not None)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model scaling: accuracy (\%) with internet KB at $k=8$, 3 test images per class. Best per crop in \textbf{bold}.}")
    lines.append(r"\label{tab:model_ablation}")
    lines.append(r"\begin{tabular}{l" + "r" * len(crops) + "}")
    lines.append(r"\toprule")
    lines.append(r"Model & " + " & ".join(label for _, label in crops) + r" \\")
    lines.append(r"\midrule")

    for model_id, model_label in models:
        cells = []
        for crop_id, _ in crops:
            acc = _load(crop_id, "internet", model_id, 8)
            is_best = acc is not None and abs(acc - best_per_crop[crop_id]) < 0.01
            cells.append(_fmt(acc, bold=is_best))
        lines.append(f"{model_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Table A1 (Appendix): Delta from zero-shot ────────────────────────────────

def table_appendix_delta():
    """Appendix table showing improvement over zero-shot for every config."""
    crops = [
        ("Soybean_Diseases", "Soybean"),
        ("Corn_Diseases", "Corn"),
        ("Mango_Leaf_Disease", "Mango"),
    ]
    ks = [1, 4, 8, 16]

    # Define the configs to show (skip few_shot k>0 and experimental ones)
    agent_configs = [
        ("none", "sonnet", "Agent, no KB (Sonnet)"),
        ("local", "sonnet", "Agent + local KB (Sonnet)"),
        ("internet", "sonnet", "Agent + internet KB (Sonnet)"),
        ("internet", "haiku", "Agent + internet KB (Haiku)"),
        ("internet", "opus", "Agent + internet KB (Opus)"),
    ]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Improvement over zero-shot baseline (percentage points) across all configurations. 3 test images per class. Best improvement per crop--$k$ in \textbf{bold}.}")
    lines.append(r"\label{tab:delta_results}")

    for crop_id, crop_label in crops:
        zs = _load(crop_id, "few_shot", "sonnet", 0)

        lines.append(f"\\vspace{{4pt}}")
        lines.append(f"\\textbf{{{crop_label}}} (zero-shot: {_fmt(zs)})\\\\[2pt]")
        lines.append(r"\begin{tabular}{l" + "r" * len(ks) + "}")
        lines.append(r"\toprule")
        lines.append(r"Configuration & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
        lines.append(r"\midrule")

        # Find best delta per k
        best_delta = {}
        for k in ks:
            deltas = []
            for source, model, _ in agent_configs:
                acc = _load(crop_id, source, model, k)
                if acc is not None and zs is not None:
                    deltas.append(acc - zs)
            best_delta[k] = max(deltas) if deltas else -999

        for source, model, label in agent_configs:
            cells = []
            for k in ks:
                acc = _load(crop_id, source, model, k)
                if acc is not None and zs is not None:
                    delta = acc - zs
                    is_best = abs(delta - best_delta[k]) < 0.01
                    cells.append(_fmt_delta(acc, zs, bold=is_best))
                else:
                    cells.append("—")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    print("GENERATE TABLES")
    print(f"  Output: {TABLES_OUT}")
    print()

    tex = table_main_results()
    out = TABLES_OUT / "table_main_results.tex"
    out.write_text(tex)
    print(f"  Main results → {out.name}")

    tex = table_model_ablation()
    out = TABLES_OUT / "table_model_ablation.tex"
    out.write_text(tex)
    print(f"  Model ablation → {out.name}")

    # Appendix table removed — main text table with deltas covers it


if __name__ == "__main__":
    main()
