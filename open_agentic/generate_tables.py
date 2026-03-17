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


def _load(crop, source, model, k):
    """Load accuracy from a summary.json. Returns (correct, total, accuracy) or None."""
    path = RESULTS_DIR / crop / source / model / f"k{k}" / "summary.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    m = d["metrics"]
    return m["correct"], m["total"], m["accuracy"]


def _fmt(result, bold=False):
    """Format a result as 'correct/total (acc%)' or '—'."""
    if result is None:
        return "—"
    correct, total, acc = result
    s = f"{correct}/{total} ({acc:.1f}\\%)"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


# ── Table 2: Main results (method × k, sonnet) ───────────────────────────────

def table_main_results():
    """Compact table: zero-shot + agent methods across k values, per crop."""
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
    lines.append(r"\caption{Diagnostic accuracy across crops, methods, and reference budgets $k$ (Sonnet model). Zero-shot uses no reference images. Best result per crop--$k$ combination is in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{ll" + "r" * len(ks) + "}")
    lines.append(r"\toprule")
    lines.append(r"Crop & Method & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
    lines.append(r"\midrule")

    for crop_id, crop_label, sources in crops:
        # Zero-shot row
        zs = _load(crop_id, "few_shot", "sonnet", 0)
        zs_str = _fmt(zs) if zs else "—"
        lines.append(f"{crop_label} & Zero-shot & " + " & ".join([zs_str] * len(ks)) + r" \\")

        # Agent rows
        for source in sources:
            label = method_labels[source]
            results = [_load(crop_id, source, "sonnet", k) for k in ks]

            # Find best per k across methods for this crop
            # (we'll do bolding after collecting all methods)
            row = f" & {label} & " + " & ".join(_fmt(r) for r in results) + r" \\"
            lines.append(row)

        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Table 3: Model ablation ──────────────────────────────────────────────────

def table_model_ablation():
    """Model ablation table: haiku/sonnet/opus at internet KB, k=8."""
    crops = [
        ("Soybean_Diseases", "Soybean"),
        ("Corn_Diseases", "Corn"),
        ("Mango_Leaf_Disease", "Mango"),
    ]
    models = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus")]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model scaling: accuracy with internet KB at $k=8$.}")
    lines.append(r"\label{tab:model_ablation}")
    lines.append(r"\begin{tabular}{l" + "r" * len(crops) + "}")
    lines.append(r"\toprule")
    lines.append(r"Model & " + " & ".join(label for _, label in crops) + r" \\")
    lines.append(r"\midrule")

    for model_id, model_label in models:
        results = [_load(crop_id, "internet", model_id, 8) for crop_id, _ in crops]
        row = f"{model_label} & " + " & ".join(_fmt(r) for r in results) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Table A1 (Appendix): Full results ─────────────────────────────────────────

def table_appendix_full():
    """Complete results table with every configuration."""
    crops = [
        ("Soybean_Diseases", "Soybean"),
        ("Corn_Diseases", "Corn"),
        ("Mango_Leaf_Disease", "Mango"),
    ]
    ks = [0, 1, 4, 8, 16]

    # Collect all (source, model) combos that exist
    configs = []
    for crop_id, _ in crops:
        for source_dir in sorted((RESULTS_DIR / crop_id).iterdir()):
            if not source_dir.is_dir():
                continue
            source = source_dir.name
            for model_dir in sorted(source_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                key = (source, model)
                if key not in configs:
                    configs.append(key)

    # Sort configs logically
    source_order = {"few_shot": 0, "none": 1, "local": 2, "internet": 3}
    model_order = {"haiku": 0, "sonnet": 1, "opus": 2}
    configs.sort(key=lambda x: (source_order.get(x[0], 9), model_order.get(x[1], 9)))

    def _config_label(source, model):
        if source == "few_shot":
            return f"Few-shot ({model})"
        source_label = {"none": "no KB", "local": "local KB", "internet": "internet KB"}.get(source, source)
        return f"Agent + {source_label} ({model})"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{Complete results across all configurations. $k=0$ denotes zero-shot (no reference images).}")
    lines.append(r"\label{tab:full_results}")

    for crop_id, crop_label in crops:
        lines.append(f"\\vspace{{4pt}}")
        lines.append(f"\\textbf{{{crop_label}}}\\\\[2pt]")
        lines.append(r"\begin{tabular}{l" + "r" * len(ks) + "}")
        lines.append(r"\toprule")
        lines.append(r"Configuration & " + " & ".join(f"$k={k}$" for k in ks) + r" \\")
        lines.append(r"\midrule")

        # Find best accuracy per k for bolding
        best_per_k = {}
        for k in ks:
            best_acc = -1
            for source, model in configs:
                r = _load(crop_id, source, model, k)
                if r and r[2] > best_acc:
                    best_acc = r[2]
            best_per_k[k] = best_acc

        for source, model in configs:
            label = _config_label(source, model)
            cells = []
            for k in ks:
                r = _load(crop_id, source, model, k)
                is_best = r is not None and abs(r[2] - best_per_k[k]) < 0.01
                cells.append(_fmt(r, bold=is_best))
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

    # Table 2: Main results
    tex = table_main_results()
    out = TABLES_OUT / "table_main_results.tex"
    out.write_text(tex)
    print(f"  Main results table → {out.name}")

    # Table 3: Model ablation
    tex = table_model_ablation()
    out = TABLES_OUT / "table_model_ablation.tex"
    out.write_text(tex)
    print(f"  Model ablation table → {out.name}")

    # Table A1: Full appendix
    tex = table_appendix_full()
    out = TABLES_OUT / "table_appendix_full.tex"
    out.write_text(tex)
    print(f"  Full appendix table → {out.name}")


if __name__ == "__main__":
    main()
