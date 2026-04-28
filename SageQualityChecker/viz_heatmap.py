"""Heatmap: crops (rows) × diseases (columns). Cell filled = crop has
this disease labeled CORRECT. With 228×462 this is wide; we restrict to
top crops and diseases that appear >= MIN_OCCURRENCES times.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz_common import BG, FONT, INPUT_JSON, disease_crop_map, load_json, save_all

TOP_N_CROPS       = 40    # rows
MIN_OCCURRENCES   = 3     # include only diseases on >= N crops


def draw(json_path, out_dir):
    entries = load_json(json_path)
    dm = disease_crop_map(entries)
    crop_to_d = {e["crop"]: set(e["diseases"]) for e in entries}

    # Select rows / cols
    top_crops = sorted(crop_to_d, key=lambda c: -len(crop_to_d[c]))[:TOP_N_CROPS]
    diseases  = [d for d, crops in dm.items() if len(crops) >= MIN_OCCURRENCES]
    diseases.sort(key=lambda d: -len(dm[d]))

    if not diseases:
        print("No diseases meet MIN_OCCURRENCES threshold — lower it.")
        return

    # Build matrix
    mat = np.zeros((len(top_crops), len(diseases)), dtype=int)
    for i, c in enumerate(top_crops):
        for j, d in enumerate(diseases):
            if d in crop_to_d[c]:
                mat[i, j] = 1

    # Cluster-ish sort: rows by sum desc, cols by sum desc (already sorted)
    row_order = np.argsort(-mat.sum(axis=1))
    mat = mat[row_order]
    top_crops = [top_crops[i] for i in row_order]

    fig_w = max(14, 0.25 * len(diseases))
    fig_h = max(10, 0.32 * len(top_crops))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    cmap = plt.get_cmap("YlOrRd")
    # Show presence weighted by how "hub-like" the disease is
    weights = np.array([len(dm[d]) for d in diseases], dtype=float)
    weights /= weights.max() if weights.max() else 1
    display = mat * weights  # 0..1
    ax.imshow(display, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(diseases)))
    ax.set_xticklabels(diseases, rotation=90, fontsize=7, fontfamily=FONT)
    ax.set_yticks(range(len(top_crops)))
    ax.set_yticklabels(top_crops, fontsize=8, fontfamily=FONT)

    ax.set_title(
        f"Presence heatmap — top {len(top_crops)} crops × diseases with ≥{MIN_OCCURRENCES} crops "
        f"({len(diseases)} diseases)",
        fontsize=13, fontfamily=FONT, fontweight="bold", pad=14,
    )
    ax.grid(False)

    save_all(fig, out_dir, "viz_heatmap")
    plt.close(fig)


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
