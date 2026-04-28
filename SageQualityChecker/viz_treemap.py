"""Treemap: crops as outer rects, diseases nested inside.

Each crop rect is sized by its number of CORRECT diseases. Each disease
is an equal-weight child tile. Labels fit much better than the sunburst.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import squarify

from viz_common import (BG, CROP_COLORS, FONT, INPUT_JSON, load_json, save_all)

TOP_N_CROPS = 20  # show top-N crops individually, rest go into "Others"


def draw(json_path, out_dir):
    entries = load_json(json_path)
    entries = sorted(entries, key=lambda e: -len(e.get("diseases", [])))

    top = entries[:TOP_N_CROPS]
    others = entries[TOP_N_CROPS:]

    crops = [{"crop": e["crop"], "diseases": e["diseases"]} for e in top]
    if others:
        merged = []
        seen = set()
        for e in others:
            for d in e["diseases"]:
                if d not in seen:
                    merged.append(d); seen.add(d)
        crops.append({"crop": f"Others ({len(others)} crops)", "diseases": merged})

    # Outer layout: crops sized by disease count
    sizes = [len(c["diseases"]) for c in crops]
    colors = [CROP_COLORS[i % len(CROP_COLORS)] for i in range(len(crops))]

    fig, ax = plt.subplots(figsize=(18, 12), facecolor=BG)
    ax.set_facecolor(BG)

    rects = squarify.squarify(
        squarify.normalize_sizes(sizes, 100, 100), 0, 0, 100, 100
    )

    for crop_info, rect, color in zip(crops, rects, colors):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        ax.add_patch(plt.Rectangle((x, y), dx, dy, facecolor=color,
                                    edgecolor="white", linewidth=2))
        # Crop label centred on outer rect — top strip
        label = f"{crop_info['crop']}\n({len(crop_info['diseases'])})"
        ax.text(x + dx/2, y + dy - 1.5, label,
                ha="center", va="top", fontsize=max(7, min(14, dx/6)),
                fontweight="bold", color="#111111", fontfamily=FONT)

        # Nested disease tiles within this crop rect
        n = len(crop_info["diseases"])
        if n == 0:
            continue
        d_sizes = [1] * n
        d_rects = squarify.squarify(
            squarify.normalize_sizes(d_sizes, dx - 2, dy - 6),
            x + 1, y + 1, dx - 2, dy - 6,
        )
        for disease, dr in zip(crop_info["diseases"], d_rects):
            dx2, dy2 = dr["dx"], dr["dy"]
            # Lighten the crop color
            f = 0.55
            h = color.lstrip("#")
            rgb = [int(h[i:i+2], 16)/255 for i in (0,2,4)]
            rgb = [c + (1-c)*f for c in rgb]
            tile_color = "#{:02X}{:02X}{:02X}".format(*(int(c*255) for c in rgb))
            ax.add_patch(plt.Rectangle((dr["x"], dr["y"]), dx2, dy2,
                                        facecolor=tile_color,
                                        edgecolor="white", linewidth=0.6))
            if dx2 > 3.5 and dy2 > 1.5:
                fs = max(5, min(8, 0.9 * min(dx2, dy2)))
                label = disease if len(disease) <= 22 else disease[:21] + "…"
                ax.text(dr["x"] + dx2/2, dr["y"] + dy2/2, label,
                        ha="center", va="center", fontsize=fs,
                        color="#222222", fontfamily=FONT)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.invert_yaxis()
    ax.axis("off")

    total_pairs = sum(len(e.get("diseases", [])) for e in entries)
    fig.suptitle(f"Crop × CORRECT disease treemap — {len(entries)} crops · "
                 f"{total_pairs} pairs",
                 fontsize=16, fontfamily=FONT, fontweight="bold",
                 color="#111111", y=0.98)

    save_all(fig, out_dir, "viz_treemap")
    plt.close(fig)


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
