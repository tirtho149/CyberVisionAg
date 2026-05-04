#!/usr/bin/env python3
"""Generate LaTeX tables for wheat disease evaluation results."""

import json
from pathlib import Path

RESULTS_DIR = Path("results/open_agentic/Wheat_Diseases")


def fmt(acc, bold=False):
    """Format accuracy as percentage with optional bold."""
    if acc is None:
        return "—"
    # acc is already in range 0-1 (like 0.193 for 19.3%)
    s = f"{acc*100:.1f}\\%"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def fmt_with_delta(acc, baseline, bold=False):
    """Format accuracy with delta from baseline."""
    if acc is None:
        return "—"
    # acc and baseline are both in range 0-1
    s = f"{acc*100:.1f}\\%"
    if baseline is not None and acc is not None:
        delta = (acc - baseline) * 100  # Convert to percentage points
        sign = "+" if delta >= 0 else ""
        s += f" ({sign}{delta:.1f}pp)"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def load_accuracy(src, model, k):
    """Load accuracy from summary.json."""
    path = RESULTS_DIR / src / model / f"k{k}" / "summary.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data["metrics"]["accuracy"]
    except:
        return None


def load_metrics(src, model, k):
    """Load all metrics from summary.json."""
    path = RESULTS_DIR / src / model / f"k{k}" / "summary.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data["metrics"]
    except:
        return None


def table_wheat_main():
    """Main results table: k-sweep for Sonnet, all sources."""
    ks = [0, 1, 4, 8, 16]
    sources = ["none", "internet"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Wheat disease classification: diagnostic accuracy across reference budgets $k$ and knowledge sources. Sonnet model, 3 test images per class (135 total). Baseline is Agent (no KB) at $k{=}0$. Values show accuracy\,\% with improvement over baseline in parentheses. Best per $k$ in \textbf{bold}.}")
    lines.append(r"\label{tab:wheat_main}")
    lines.append(r"\begin{tabular}{lr" + "r" * len(ks) + "}")
    lines.append(r"\toprule")
    lines.append(r"Method & & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
    lines.append(r"\midrule")

    # Baseline for comparison (already in range 0-1)
    baseline = load_accuracy("none", "sonnet", 0)

    # Few-shot baseline
    fs_accs = [load_accuracy("few_shot", "sonnet", k) for k in ks]
    best_fs = max([a for a in fs_accs if a is not None], default=0)

    fs_cells = []
    for acc in fs_accs:
        is_best = acc is not None and abs(acc - best_fs) < 0.01
        fs_cells.append(fmt_with_delta(acc, baseline, bold=is_best))
    lines.append(r"Few-shot baseline & & " + " & ".join(fs_cells) + r" \\")

    # Agentic results
    for source in sources:
        label = "No KB" if source == "none" else "Internet KB"
        accs = [load_accuracy(source, "sonnet", k) for k in ks]
        best_acc = max([a for a in accs if a is not None], default=0)

        cells = []
        for acc in accs:
            is_best = acc is not None and abs(acc - best_acc) < 0.01
            cells.append(fmt_with_delta(acc, baseline, bold=is_best))

        lines.append(r"Agent & " + label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def table_wheat_cost():
    """Cost analysis table."""
    ks = [0, 1, 4, 8, 16]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Cost and efficiency metrics per image (Sonnet, internet KB). Cost in USD per classification, Refs = average references viewed.}")
    lines.append(r"\label{tab:wheat_cost}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"$k$ & Cost (\$) & Turns & Refs \\")
    lines.append(r"\midrule")

    for k in ks:
        m = load_metrics("internet", "sonnet", k)
        if m is None:
            continue
        cost = f"{m.get('avg_cost_usd', 0):.5f}"
        turns = f"{m.get('avg_turns', 0):.1f}"
        refs = f"{m.get('avg_refs_viewed', 0):.2f}"
        lines.append(f"{k} & {cost} & {turns} & {refs} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def table_wheat_model_ablation():
    """Model ablation at k=8 with internet KB."""
    models = ["haiku", "sonnet", "opus", "gemini-flash", "gemini-pro"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model scaling with internet KB at $k=8$ (3 test images per class). Best in \textbf{bold}.}")
    lines.append(r"\label{tab:wheat_model_ablation}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Accuracy \\")
    lines.append(r"\midrule")

    accs = {}
    for model in models:
        acc = load_accuracy("internet", model, 8)
        if acc is not None:
            accs[model] = acc

    best = max(accs.values()) if accs else 0

    for model in models:
        if model in accs:
            acc = accs[model]
            is_best = abs(acc - best) < 0.01
            label = model.capitalize()
            lines.append(f"{label} & {fmt(acc, bold=is_best)} \\\\")
        else:
            lines.append(f"{model.capitalize()} & — \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def table_wheat_summary():
    """Summary comparison table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Wheat disease classification summary (k=0). 45 disease classes, 3 test images per class (135 total).}")
    lines.append(r"\label{tab:wheat_summary}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Accuracy \\")
    lines.append(r"\midrule")

    fewshot_k0 = load_accuracy("few_shot", "sonnet", 0)
    none_k0 = load_accuracy("none", "sonnet", 0)
    internet_k0 = load_accuracy("internet", "sonnet", 0)

    if fewshot_k0:
        lines.append(f"Few-shot baseline (k=0) & {fmt(fewshot_k0)} \\\\")
    if none_k0:
        lines.append(f"Agent (no KB, k=0) & {fmt(none_k0)} \\\\")
    if internet_k0:
        is_best = True
        lines.append(f"Agent + Internet KB (k=0) & {fmt(internet_k0, bold=is_best)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    """Generate all wheat LaTeX tables."""
    tables = {
        "table_wheat_main.tex": table_wheat_main(),
        "table_wheat_cost.tex": table_wheat_cost(),
        "table_wheat_model_ablation.tex": table_wheat_model_ablation(),
        "table_wheat_summary.tex": table_wheat_summary(),
    }

    for filename, content in tables.items():
        path = Path(filename)
        path.write_text(content)
        print(f"✓ Generated: {filename}")

    print("\n📋 LaTeX table files created. Use in document with \\input{table_wheat_*.tex}")


if __name__ == "__main__":
    main()
= Path(filename)
        path.write_text(content)
        print(f"✓ Generated: {filename}")

    print("\n📋 LaTeX table files created. Use in document with \\input{table_wheat_*.tex}")


if __name__ == "__main__":
    main()
ain()

    main()
ain()
