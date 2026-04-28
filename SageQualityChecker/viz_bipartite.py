"""Bipartite network graph: crops on the left, top cross-crop diseases on
the right. Edge = this crop has this disease labeled CORRECT. Useful for
spotting hub diseases that connect many crops.

To keep it readable, we keep the top-N crops by disease count and the
top-M cross-crop diseases. Edges only drawn within this subset.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from viz_common import BG, CROP_COLORS, FONT, INPUT_JSON, disease_crop_map, load_json, save_all

TOP_N_CROPS = 25
TOP_N_DIS   = 30


def draw(json_path, out_dir):
    entries = load_json(json_path)
    crop_to_diseases = {e["crop"]: set(e["diseases"]) for e in entries}

    # Pick top crops by disease count
    top_crops = sorted(crop_to_diseases, key=lambda c: -len(crop_to_diseases[c]))[:TOP_N_CROPS]
    # Pick top cross-crop diseases
    dm = disease_crop_map(entries)
    top_dis = sorted(dm, key=lambda d: -len(dm[d]))[:TOP_N_DIS]
    top_dis_set = set(top_dis)

    # Keep only edges within subset
    edges = [(c, d) for c in top_crops for d in crop_to_diseases[c] if d in top_dis_set]

    fig, ax = plt.subplots(figsize=(14, max(10, max(len(top_crops), len(top_dis)) * 0.36)),
                           facecolor=BG)
    ax.set_facecolor(BG)

    # Layout: crops left at x=0, diseases right at x=1, evenly spaced y
    crop_y = {c: i for i, c in enumerate(reversed(top_crops))}
    dis_y  = {d: i for i, d in enumerate(reversed(top_dis))}
    max_h  = max(len(top_crops), len(top_dis)) - 1

    # Scale y to [0, max_h] for both
    def y_crop(c): return crop_y[c] * (max_h / max(1, len(top_crops)-1))
    def y_dis(d):  return dis_y[d]  * (max_h / max(1, len(top_dis)-1))

    # Edges (behind nodes)
    for c, d in edges:
        ax.plot([0, 1], [y_crop(c), y_dis(d)],
                color="#999999", alpha=0.35, linewidth=0.8, zorder=1)

    # Crop nodes
    for i, c in enumerate(top_crops):
        color = CROP_COLORS[i % len(CROP_COLORS)]
        y = y_crop(c)
        ax.scatter([0], [y], s=260, color=color, edgecolor="#333333",
                   linewidth=0.8, zorder=3)
        ax.text(-0.02, y, c, ha="right", va="center", fontsize=9,
                fontfamily=FONT, color="#111111")

    # Disease nodes — sized by # crops linked
    for i, d in enumerate(top_dis):
        y = y_dis(d)
        size = 60 + 30 * min(10, len(dm[d]))
        ax.scatter([1], [y], s=size, color="#4D96FF",
                   edgecolor="#333333", linewidth=0.8, zorder=3, alpha=0.85)
        label = f"{d} ({len(dm[d])})"
        ax.text(1.02, y, label, ha="left", va="center", fontsize=9,
                fontfamily=FONT, color="#111111")

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-1, max_h + 1)
    ax.axis("off")
    ax.set_title(
        f"Bipartite graph — top {len(top_crops)} crops ↔ top {len(top_dis)} cross-crop diseases",
        fontsize=14, fontfamily=FONT, fontweight="bold", pad=14,
    )

    save_all(fig, out_dir, "viz_bipartite")
    plt.close(fig)


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
