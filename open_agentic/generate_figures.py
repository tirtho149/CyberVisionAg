"""Generate all paper figures from raw data files.

All plots read directly from summary JSONs and all-crops.xlsx.
No hardcoded numbers — if data changes, rerun this script.

Usage:
    python -m CyberVisionAg.open_agentic.generate_figures
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "open_agentic"
CSV_PATH = SCRIPT_DIR / "all-crops-v2.csv"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "writing" / "69aae430e8bdcbd9056bf911" / "figures"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_crop_data():
    """Load crop/disease/image data from all-crops-v2.csv."""
    import csv
    crops = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            crop = row["crop"]
            disease = row["disease"]
            images = int(row["images"])
            crops[crop].append((disease, images))
    return dict(crops)


def load_summary(crop, kb_source, model, k):
    """Load a single summary.json."""
    path = RESULTS_DIR / crop / kb_source / model / f"k{k}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_accuracy_and_std(crop, kb_source, model, k):
    """Get accuracy and per-class std from a summary."""
    d = load_summary(crop, kb_source, model, k)
    if not d:
        return None, None
    acc = d["metrics"]["accuracy"]
    if "per_class" in d:
        vals = list(d["per_class"].values())
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    else:
        std = 0.0
    return acc, std


# ── Figure 3: Sunburst (Plotly) ───────────────────────────────────────────────

def figure_3_sunburst():
    """Sunburst chart using Plotly: crop → disease hierarchy."""
    import plotly.graph_objects as go

    data = load_crop_data()
    crop_totals = {c: sum(img for _, img in ds) for c, ds in data.items()}
    sorted_crops = sorted(crop_totals.items(), key=lambda x: -x[1])
    total_images = sum(crop_totals.values())
    total_diseases = len({d for ds in data.values() for d, _ in ds})

    # Build hierarchical data for plotly sunburst
    # Structure: root → crops → diseases
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    # Color palette for top crops
    crop_colors = [
        '#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#7f8c8d', '#e84393', '#00cec9',
        '#6c5ce7', '#fdcb6e', '#d35400', '#2c3e50', '#c0392b',
    ]

    TOP_N = 15
    top = sorted_crops[:TOP_N]
    others_crops = sorted_crops[TOP_N:]
    others_total = sum(n for _, n in others_crops)
    others_diseases_count = sum(len(data[c]) for c, _ in others_crops)

    # Add top crops
    for i, (crop, total) in enumerate(top):
        crop_id = f"crop-{crop}"
        ids.append(crop_id)
        labels.append(crop)
        parents.append("")
        values.append(total)
        colors.append(crop_colors[i % len(crop_colors)])

        # Add diseases for this crop (top 10 + merge rest)
        diseases = sorted(data[crop], key=lambda x: -x[1])
        MAX_SHOW = 10
        shown = diseases[:MAX_SHOW]
        merged = diseases[MAX_SHOW:]

        for disease, n in shown:
            d_id = f"{crop}-{disease}"
            ids.append(d_id)
            labels.append(disease)
            parents.append(crop_id)
            values.append(n)
            colors.append(crop_colors[i % len(crop_colors)])

        if merged:
            merged_total = sum(n for _, n in merged)
            ids.append(f"{crop}-others")
            labels.append(f"+{len(merged)} more")
            parents.append(crop_id)
            values.append(merged_total)
            colors.append(crop_colors[i % len(crop_colors)])

    # Add "Others" as a single crop with aggregated diseases
    ids.append("crop-Others")
    labels.append(f"Others ({len(others_crops)})")
    parents.append("")
    values.append(others_total)
    colors.append('#bdc3c7')

    # Add top 5 crops from "Others" as sub-entries
    others_sorted = sorted(others_crops, key=lambda x: -x[1])
    shown_others = others_sorted[:5]
    remaining_others = others_sorted[5:]
    for c, n in shown_others:
        ids.append(f"others-{c}")
        labels.append(c)
        parents.append("crop-Others")
        values.append(n)
        colors.append('#d5dbdb')

    if remaining_others:
        remaining_total = sum(n for _, n in remaining_others)
        ids.append("others-rest")
        labels.append(f"+{len(remaining_others)} more")
        parents.append("crop-Others")
        values.append(remaining_total)
        colors.append('#eaeded')

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(width=1, color='white')),
        branchvalues="total",
        textinfo="label",
        insidetextorientation='radial',
        maxdepth=2,
        textfont=dict(size=11, family="Arial"),
    ))

    # Add a white circle shape behind center text for readability
    fig.update_layout(
        width=550,
        height=550,
        margin=dict(t=30, l=10, r=10, b=10),
        font=dict(family="Arial"),
        shapes=[
            dict(
                type="circle",
                xref="paper", yref="paper",
                x0=0.39, y0=0.39, x1=0.61, y1=0.61,
                fillcolor="rgba(255,255,255,0.85)",
                line=dict(width=0),
                layer="above",
            )
        ],
        annotations=[
            dict(
                text=f"<b>{total_images / 1_000_000:.1f}M</b><br>Images<br><span style='font-size:9px;color:#7f8c8d'>{len(data)} crops · {total_diseases} diseases</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, family="Arial", color="#2c3e50"),
                xref="paper", yref="paper",
            )
        ],
    )

    out = FIGURES_DIR / "fig3_sunburst.pdf"
    fig.write_image(str(out), scale=3)
    print(f"  Saved: {out}")

    # Also save as PNG for quick inspection
    out_png = FIGURES_DIR / "fig3_sunburst.png"
    fig.write_image(str(out_png), scale=3)
    print(f"  Saved: {out_png}")


# ── Figure 4: Accuracy vs k (1x3 panel) ──────────────────────────────────────

def figure_4_accuracy_vs_k():
    """Line plots: accuracy vs reference budget k, one panel per crop."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
    })

    crops = [
        ("Soybean_Diseases", "Soybean (25 classes)", ["none", "internet"]),
        ("Corn_Diseases", "Corn (30 classes)", ["none", "internet"]),
        ("Mango_Leaf_Disease", "Mango (4 classes)", ["none", "internet"]),
    ]
    ks = [0, 1, 4, 8, 16]

    method_styles = {
        "none":     {"label": "Agent (no KB)",       "color": "#3498db", "marker": "o", "ls": "-"},
        "internet": {"label": "Agent + internet KB", "color": "#e74c3c", "marker": "s", "ls": "-"},
        "local":    {"label": "Agent + local KB",    "color": "#2ecc71", "marker": "^", "ls": "-"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, (crop, title, sources) in zip(axes, crops):
        # Track all accuracy values for dynamic y-axis
        all_accs = []

        # Plot each method
        for source in sources:
            style = method_styles[source]
            accs = []
            stds = []
            valid_ks = []
            for k in ks:
                acc, std = get_accuracy_and_std(crop, source, "sonnet", k)
                if acc is not None:
                    accs.append(acc)
                    stds.append(std)
                    valid_ks.append(k)

            if not accs:
                continue

            accs = np.array(accs)
            all_accs.extend(accs.tolist())

            # Plot line with markers
            ax.plot(valid_ks, accs, marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], linewidth=2, markersize=6,
                    label=style["label"], zorder=3)

        # Add haiku and opus markers at k=8 (internet KB, model ablation)
        model_markers = [
            ("haiku", "v", "#95a5a6", "Haiku"),
            ("opus", "*", "#8e44ad", "Opus"),
        ]
        for model, marker, color, label in model_markers:
            acc, _ = get_accuracy_and_std(crop, "internet", model, 8)
            if acc is not None:
                ax.scatter([8], [acc], marker=marker, color=color, s=80,
                           zorder=5, edgecolors='black', linewidths=0.5,
                           label=f"{label} (k=8)" if ax == axes[0] else None)
                all_accs.append(acc)
                # Annotate with accuracy value
                offset = 6 if model == "opus" else -10
                ax.annotate(f"{acc:.1f}", (8, acc), textcoords="offset points",
                            xytext=(8, offset), fontsize=8, color=color, fontweight='bold')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Reference budget (k)")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.grid(True, linestyle='--', alpha=0.4)

        # Dynamic y-axis: min/max of data with 15% padding
        if all_accs:
            y_min = min(all_accs)
            y_max = max(all_accs)
            y_range = y_max - y_min
            pad = max(y_range * 0.15, 5)  # at least 5pp padding
            ax.set_ylim(max(0, y_min - pad), min(100, y_max + pad))

    axes[0].set_ylabel("Accuracy (%)")

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect all unique handles across panels
    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)

    fig.legend(all_handles, all_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=len(all_labels), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    out = FIGURES_DIR / "fig4_accuracy_vs_k.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 5: Model scaling + KB comparison ───────────────────────────────────

def _load_per_image_results(crop, kb_source, model, k):
    """Load all per-image result JSONs for a configuration."""
    import glob
    result_dir = RESULTS_DIR / crop / kb_source / model / f"k{k}"
    results = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        d = json.loads(path.read_text())
        if d.get("error"):
            continue
        results.append(d)
    return results


def figure_5_cost_accuracy():
    """Cost-accuracy tradeoff: single panel, accuracy averaged across Soybean + Corn.

    Per-image costs from both crops pooled as jittered dots at the mean accuracy level.
    No connecting line. Aggregate means as large bubbles with k labels.
    Haiku/Opus at k=8 labeled. Mango omitted (saturates >90%).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Match Figure 4 rcParams
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
    })

    crop_ids = ["Soybean_Diseases", "Corn_Diseases", "Mango_Leaf_Disease"]
    model_colors = {
        "haiku": "#f39c12",
        "sonnet": "#3498db",
        "opus": "#e74c3c",
    }
    model_labels = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus"}

    configs = [
        ("sonnet", 0), ("sonnet", 1), ("sonnet", 4), ("sonnet", 8), ("sonnet", 16),
        ("haiku", 8), ("opus", 8),
    ]

    # Bubble size scaling: k → area (more distinct sizes)
    def k_to_size(k):
        return 120 + k * 30

    fig, ax = plt.subplots(1, 1, figsize=(13, 4))

    rng = np.random.default_rng(42)

    # Per-image jitter clouds for each config
    for model, k in configs:
        accs = []
        all_costs = []
        for crop_id in crop_ids:
            d = load_summary(crop_id, "internet", model, k)
            if d:
                accs.append(d["metrics"]["accuracy"])
            results = _load_per_image_results(crop_id, "internet", model, k)
            all_costs.extend(r["cost_usd"] for r in results)

        if not accs or not all_costs:
            continue
        mean_acc = sum(accs) / len(accs)

        jitter = rng.normal(0, 0.5, len(all_costs))

        ax.scatter(
            all_costs, mean_acc + jitter,
            s=18, c=model_colors[model],
            alpha=0.2, edgecolors='none', zorder=2,
        )

    # Sonnet aggregate bubbles (no line, just bubbles with k-based sizes)
    for k in [0, 1, 4, 8, 16]:
        crop_costs, crop_accs = [], []
        for crop_id in crop_ids:
            d = load_summary(crop_id, "internet", "sonnet", k)
            if d:
                crop_costs.append(d["metrics"]["avg_cost_usd"])
                crop_accs.append(d["metrics"]["accuracy"])
        if crop_costs:
            mean_cost = sum(crop_costs) / len(crop_costs)
            mean_acc = sum(crop_accs) / len(crop_accs)
            ax.scatter(mean_cost, mean_acc, s=k_to_size(k),
                       c=model_colors["sonnet"],
                       edgecolors='black', linewidth=0.8, zorder=6,
                       marker='o', alpha=0.85)
            ax.annotate(f'k={k}', (mean_cost, mean_acc),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=9, color='black', fontweight='bold')

    # Haiku/Opus aggregate bubbles at k=8
    for model in ["haiku", "opus"]:
        crop_costs, crop_accs = [], []
        for crop_id in crop_ids:
            d = load_summary(crop_id, "internet", model, 8)
            if d:
                crop_costs.append(d["metrics"]["avg_cost_usd"])
                crop_accs.append(d["metrics"]["accuracy"])
        if crop_costs:
            mean_cost = sum(crop_costs) / len(crop_costs)
            mean_acc = sum(crop_accs) / len(crop_accs)
            ax.scatter(mean_cost, mean_acc, s=k_to_size(8),
                       c=model_colors[model],
                       edgecolors='black', linewidth=0.8, zorder=6,
                       marker='o', alpha=0.85)
            label = f'{model_labels[model]} (k=8)'
            ax.annotate(label, (mean_cost, mean_acc),
                        textcoords="offset points", xytext=(10, -8),
                        fontsize=9, color=model_colors[model], fontweight='bold')

    ax.set_xlabel("Cost per image (USD)")
    ax.set_ylabel("Mean accuracy (%)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(left=-0.02)

    # Legend: model colors + bubble size
    model_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[m],
               markersize=9, markeredgecolor='black', markeredgewidth=0.5,
               label=model_labels[m])
        for m in ["haiku", "sonnet", "opus"]
    ]
    scatter_handle = Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='lightgray', markersize=4,
                            alpha=0.4, label='Per-image cost')
    ax.legend(handles=model_handles + [scatter_handle],
              loc='lower right', framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig5_model_and_kb.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    print("\nFigure 3: Sunburst")
    figure_3_sunburst()
    print("\nFigure 4: Accuracy vs k")
    figure_4_accuracy_vs_k()
    print("\nFigure 5: Model scaling + KB comparison")
    figure_5_cost_accuracy()
    print("\nDone.")
