"""Horizontal bar chart: top-N most cross-crop diseases (appearing across
the most crops). Surfaces generic diseases like Powdery Mildew, Anthracnose.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from viz_common import BG, CROP_COLORS, FONT, INPUT_JSON, disease_crop_map, load_json, save_all

TOP_N = 40


def draw(json_path, out_dir):
    entries = load_json(json_path)
    m = disease_crop_map(entries)

    items = sorted(((d, len(crops)) for d, crops in m.items()),
                   key=lambda x: -x[1])[:TOP_N]
    items.reverse()

    labels = [d for d, _ in items]
    counts = [c for _, c in items]

    fig, ax = plt.subplots(figsize=(12, max(6, len(items) * 0.32)),
                           facecolor=BG)
    ax.set_facecolor(BG)

    colors = [CROP_COLORS[i % len(CROP_COLORS)] for i in range(len(items))]
    bars = ax.barh(labels, counts, color=colors,
                   edgecolor="#333333", linewidth=0.4)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                str(c), va="center", fontsize=9, fontfamily=FONT,
                color="#333333")

    ax.set_xlabel("# of crops that list this disease (CORRECT)",
                  fontsize=11, fontfamily=FONT)
    ax.set_title(f"Top {len(items)} cross-crop diseases",
                 fontsize=14, fontfamily=FONT, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_all(fig, out_dir, "viz_bar_diseases")
    plt.close(fig)


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
