"""Sankey-style diagram: flows from crops (left) to diseases (right).
Crops stacked on the left, diseases on the right. Each flow has thickness
= 1 (each CORRECT crop-disease pair counts equally).

To keep it readable we take top-N crops by disease count and only
diseases that appear on ≥ MIN_OCCURRENCES crops.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz_common import BG, CROP_COLORS, FONT, INPUT_JSON, disease_crop_map, load_json, save_all

TOP_N_CROPS     = 20
MIN_OCCURRENCES = 2   # disease must span ≥ N crops


def draw(json_path, out_dir):
    entries = load_json(json_path)
    dm = disease_crop_map(entries)
    crop_to_d = {e["crop"]: list(e["diseases"]) for e in entries}

    top_crops = sorted(crop_to_d, key=lambda c: -len(crop_to_d[c]))[:TOP_N_CROPS]
    dis_set   = {d for d, cs in dm.items() if len(cs) >= MIN_OCCURRENCES}

    # Build flow list filtered to subset
    flows = []
    for c in top_crops:
        for d in crop_to_d[c]:
            if d in dis_set:
                flows.append((c, d))
    if not flows:
        print("No flows matching filters.")
        return

    # Count per side for sizing
    from collections import Counter
    src_counts = Counter(c for c, _ in flows)
    dst_counts = Counter(d for _, d in flows)

    # Order destinations by count descending
    dst_order = [d for d, _ in dst_counts.most_common()]
    src_order = [c for c in top_crops if c in src_counts]  # preserve top order

    # Assign y positions proportional to counts
    def stacked_positions(order, counts, gap_frac=0.02):
        total = sum(counts[k] for k in order) + gap_frac * max(0, len(order) - 1)
        pos = {}
        y = 0
        for k in order:
            h = counts[k]
            pos[k] = (y, y + h)
            y += h + gap_frac
        return pos, y  # total height

    src_pos, src_h = stacked_positions(src_order, src_counts)
    dst_pos, dst_h = stacked_positions(dst_order, dst_counts)
    total_h = max(src_h, dst_h)

    fig, ax = plt.subplots(figsize=(16, max(10, 0.35 * max(len(src_order), len(dst_order)))),
                           facecolor=BG)
    ax.set_facecolor(BG)

    colors = {c: CROP_COLORS[i % len(CROP_COLORS)] for i, c in enumerate(src_order)}

    # Draw crop boxes (left)
    bar_w = 0.06
    for c in src_order:
        y0, y1 = src_pos[c]
        ax.add_patch(plt.Rectangle((0, y0), bar_w, y1 - y0,
                                    facecolor=colors[c], edgecolor="#333333",
                                    linewidth=0.6))
        ax.text(-0.01, (y0 + y1)/2, f"{c} ({src_counts[c]})",
                ha="right", va="center", fontsize=8.5, fontfamily=FONT,
                color="#111111")

    # Draw disease boxes (right)
    x_right = 1.0
    for d in dst_order:
        y0, y1 = dst_pos[d]
        ax.add_patch(plt.Rectangle((x_right, y0), bar_w, y1 - y0,
                                    facecolor="#4D96FF", edgecolor="#333333",
                                    linewidth=0.4, alpha=0.85))
        label = d if len(d) <= 28 else d[:27] + "…"
        ax.text(x_right + bar_w + 0.01, (y0 + y1)/2,
                f"{label} ({dst_counts[d]})", ha="left", va="center",
                fontsize=7.5, fontfamily=FONT, color="#111111")

    # Draw flows (bezier)
    # Track consumed height on each side
    src_used = {c: src_pos[c][0] for c in src_order}
    dst_used = {d: dst_pos[d][0] for d in dst_order}
    n = 60
    t = np.linspace(0, 1, n)
    for c, d in flows:
        y_src_a = src_used[c]; y_src_b = y_src_a + 1
        y_dst_a = dst_used[d]; y_dst_b = y_dst_a + 1
        src_used[c] = y_src_b
        dst_used[d] = y_dst_b

        # Top curve
        x0, x1 = bar_w, x_right
        def bez(y_left, y_right):
            cx = (x0 + x1) / 2
            xs = (1-t)**3 * x0 + 3*(1-t)**2*t * cx + 3*(1-t)*t**2 * cx + t**3 * x1
            ys = (1-t)**3 * y_left + 3*(1-t)**2*t * y_left + 3*(1-t)*t**2 * y_right + t**3 * y_right
            return xs, ys
        xs_top, ys_top = bez(y_src_b, y_dst_b)
        xs_bot, ys_bot = bez(y_src_a, y_dst_a)

        xs = np.concatenate([xs_top, xs_bot[::-1]])
        ys = np.concatenate([ys_top, ys_bot[::-1]])

        ax.fill(xs, ys, color=colors[c], alpha=0.22, linewidth=0)

    ax.set_xlim(-0.25, x_right + bar_w + 0.45)
    ax.set_ylim(-0.5, total_h + 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"Sankey — top {len(src_order)} crops → diseases with ≥{MIN_OCCURRENCES} crops "
        f"({len(dst_order)} diseases, {len(flows)} flows)",
        fontsize=14, fontfamily=FONT, fontweight="bold", pad=14,
    )

    save_all(fig, out_dir, "viz_sankey")
    plt.close(fig)


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
