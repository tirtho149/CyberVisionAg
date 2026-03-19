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
    """Compact table: agent methods across k, delta from Agent(no KB, k=0) baseline."""
    crops = [
        ("Soybean_Diseases", "Soybean (27)", ["none", "local", "internet"]),
        ("Corn_Diseases", "Corn (24)", ["none", "internet"]),
        ("Mango_Leaf_Disease", "Mango (7)", ["none", "internet"]),
    ]
    ks = [0, 1, 4, 8, 16]
    method_labels = {
        "none": "Agent (no KB)",
        "local": "Agent + local KB",
        "internet": "Agent + internet KB",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Diagnostic accuracy across crops, methods, and reference budgets $k$ (Sonnet model, 3 test images per class). Parentheses in the Crop column denote number of disease classes. The baseline is Agent (no KB) at $k{=}0$ (model receives only the test image and class names). Values show accuracy\,\% with improvement over baseline in parentheses. Best per crop--$k$ in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "r" * len(ks) + "}")
    lines.append(r"\toprule")
    lines.append(r"Crop & Method & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
    lines.append(r"\midrule")

    for crop_id, crop_label, sources in crops:
        # Baseline: Agent (no KB) at k=0
        baseline = _load(crop_id, "none", "sonnet", 0)

        # Collect all agent results to find best per k
        all_results = {}
        for source in sources:
            label = method_labels[source]
            all_results[label] = {k: _load(crop_id, source, "sonnet", k) for k in ks}

        best_per_k = {}
        for k in ks:
            vals = [all_results[m][k] for m in all_results if all_results[m][k] is not None]
            best_per_k[k] = max(vals) if vals else -1

        first_row = True
        for source in sources:
            label = method_labels[source]
            cells = []
            for k in ks:
                acc = all_results[label][k]
                is_best = acc is not None and abs(acc - best_per_k[k]) < 0.01
                cells.append(_fmt_with_delta(acc, baseline, bold=is_best))
            prefix = crop_label if first_row else ""
            lines.append(f"{prefix} & {label} & " + " & ".join(cells) + r" \\")
            first_row = False

        lines.append(r"\midrule")

    # Average improvement (pp) over each crop's own baseline, averaged across crops
    # Only for methods present in all 3 crops: "none" and "internet"
    avg_methods = [
        ("none", "Agent (no KB)"),
        ("internet", "Agent + internet KB"),
    ]
    crop_baselines = {
        crop_id: _load(crop_id, "none", "sonnet", 0)
        for crop_id, _, _ in crops
    }

    lines[-1] = r"\midrule"
    first_avg = True
    for source, label in avg_methods:
        cells = []
        for k in ks:
            deltas = []
            for crop_id, _, _ in crops:
                acc = _load(crop_id, source, "sonnet", k)
                bl = crop_baselines[crop_id]
                if acc is not None and bl is not None:
                    deltas.append(acc - bl)
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                sign = "+" if avg_delta >= 0 else ""
                cells.append(f"{sign}{avg_delta:.1f}")
            else:
                cells.append("—")
        prefix = r"\textit{Mean $\Delta$ (pp)}" if first_avg else ""
        lines.append(f"{prefix} & \\textit{{{label}}} & " + " & ".join(cells) + r" \\")
        first_avg = False

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}%end resizebox")
    lines.append(r"\end{table*}")

    # Update caption to mention the mean row
    old_caption_end = r"Best per crop--$k$ in \textbf{bold}.}"
    new_caption_end = r"Best per crop--$k$ in \textbf{bold}. Bottom rows show mean improvement (pp) over baseline, averaged across crops.}"
    result = "\n".join(lines)
    result = result.replace(old_caption_end, new_caption_end)

    return result


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


# ── Table A1 (Appendix): Delta from direct classification ────────────────────────────────

def table_appendix_delta():
    """Appendix table showing improvement over Agent(no KB, k=0) baseline for every config."""
    crops = [
        ("Soybean_Diseases", "Soybean"),
        ("Corn_Diseases", "Corn"),
        ("Mango_Leaf_Disease", "Mango"),
    ]
    ks = [0, 1, 4, 8, 16]

    # Define the configs to show
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
    lines.append(r"\caption{Improvement over baseline (percentage points) across all configurations. Baseline is Agent (no KB) at $k{=}0$. 3 test images per class. Best improvement per crop--$k$ in \textbf{bold}.}")
    lines.append(r"\label{tab:delta_results}")

    for crop_id, crop_label in crops:
        baseline = _load(crop_id, "none", "sonnet", 0)

        lines.append(f"\\vspace{{4pt}}")
        lines.append(f"\\textbf{{{crop_label}}} (baseline: {_fmt(baseline)})\\\\[2pt]")
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
                if acc is not None and baseline is not None:
                    deltas.append(acc - baseline)
            best_delta[k] = max(deltas) if deltas else -999

        for source, model, label in agent_configs:
            cells = []
            for k in ks:
                acc = _load(crop_id, source, model, k)
                if acc is not None and baseline is not None:
                    delta = acc - baseline
                    is_best = abs(delta - best_delta[k]) < 0.01
                    cells.append(_fmt_delta(acc, baseline, bold=is_best))
                else:
                    cells.append("—")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Table: Few-shot comparison (appendix) ─────────────────────────────────────

def table_fewshot_comparison():
    """Appendix table: few-shot accuracy with delta from baseline (Agent no KB, k=0)."""
    crops = [
        ("Soybean_Diseases", "Soybean"),
        ("Corn_Diseases", "Corn"),
        ("Mango_Leaf_Disease", "Mango"),
    ]
    ks = [0, 1, 4, 8, 16]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Single-pass few-shot classification accuracy. The model receives $k$ randomly sampled labeled images in a single API call with no reasoning loop (Sonnet, 3 test images per class). Baseline is Agent (no KB) at $k{=}0$. Best per crop--$k$ in \textbf{bold}.}")
    lines.append(r"\label{tab:fewshot}")
    lines.append(r"\begin{tabular}{l" + "r" * len(ks) + "}")
    lines.append(r"\toprule")
    lines.append(r"Crop & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
    lines.append(r"\midrule")

    all_deltas_per_k = {k: [] for k in ks}

    for crop_id, crop_label in crops:
        baseline = _load(crop_id, "none", "sonnet", 0)
        fs_accs = [_load(crop_id, "few_shot", "sonnet", k) for k in ks]

        # Best per k for bolding
        best = max((a for a in fs_accs if a is not None), default=-1)

        cells = []
        for i, k in enumerate(ks):
            acc = fs_accs[i]
            is_best = acc is not None and abs(acc - best) < 0.01
            cells.append(_fmt_with_delta(acc, baseline, bold=is_best))
            if acc is not None and baseline is not None:
                all_deltas_per_k[k].append(acc - baseline)
        lines.append(f"{crop_label} & " + " & ".join(cells) + r" \\")

    # Mean delta row
    lines.append(r"\midrule")
    cells = []
    for k in ks:
        deltas = all_deltas_per_k[k]
        if deltas:
            avg = sum(deltas) / len(deltas)
            sign = "+" if avg >= 0 else ""
            cells.append(f"\\textit{{{sign}{avg:.1f}}}")
        else:
            cells.append("—")
    lines.append(r"\textit{Mean $\Delta$ (pp)} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Table: Attractor guide results ────────────────────────────────────────────

def table_attractor_guide():
    """Table comparing Opus baseline vs Opus + attractor guide at k=8."""
    crops = [
        ("Soybean_Diseases", "Soybean (27)"),
        ("Corn_Diseases", "Corn (31)"),
        ("Mango_Leaf_Disease", "Mango (7)"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Self-correction via attractor guide. Opus model, internet KB, $k=8$, 3 test images per class. The attractor guide identifies over-predicted classes and prompts the agent to reconsider before confirming.}")
    lines.append(r"\label{tab:attractor}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Crop & Baseline & + Attractor & $\Delta$ \\")
    lines.append(r"\midrule")

    all_deltas = []
    for crop_id, crop_label in crops:
        base = _load(crop_id, "internet", "opus", 8)
        # Attractor guide results use k8_cg directory
        ag_path = RESULTS_DIR / crop_id / "internet" / "opus" / "k8_cg" / "summary.json"
        ag = None
        if ag_path.exists():
            ag = json.loads(ag_path.read_text())["metrics"]["accuracy"]

        if base is not None and ag is not None:
            delta = ag - base
            sign = "+" if delta >= 0 else ""
            bold = delta > 0
            ag_str = _fmt(ag, bold=bold)
            lines.append(f"{crop_label} & {_fmt(base)} & {ag_str} & {sign}{delta:.1f} \\\\")
            all_deltas.append(delta)
        else:
            lines.append(f"{crop_label} & {_fmt(base)} & — & — \\\\")

    # Mean delta row
    if all_deltas:
        mean_delta = sum(all_deltas) / len(all_deltas)
        sign = "+" if mean_delta >= 0 else ""
        lines.append(r"\midrule")
        lines.append(f"\\textit{{Mean}} & & & \\textit{{{sign}{mean_delta:.1f}}} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

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

    tex = table_fewshot_comparison()
    out = TABLES_OUT / "table_fewshot_comparison.tex"
    out.write_text(tex)
    print(f"  Few-shot comparison → {out.name}")

    tex = table_attractor_guide()
    out = TABLES_OUT / "table_attractor_guide.tex"
    out.write_text(tex)
    print(f"  Attractor guide → {out.name}")


if __name__ == "__main__":
    main()
