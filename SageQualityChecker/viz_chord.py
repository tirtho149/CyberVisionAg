"""Chord diagram: crops as arcs around a circle, bezier ribbons between
pairs that share diseases. Ribbon thickness = count of shared diseases.

We show only the top-N crops (by # CORRECT diseases) to keep legibility.
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz_common import BG, CROP_COLORS, FONT, INPUT_JSON, load_json, save_all

TOP_N = 25
MIN_SHARED = 2  # only draw ribbons for pairs sharing ≥ this many diseases


def bezier_ribbon(ax, t1_a, t1_b, t2_a, t2_b, r, color, alpha=0.25):
    """Draw a ribbon between arc [t1_a, t1_b] and arc [t2_a, t2_b]."""
    n = 60
    # Two outer edges meeting at center with quadratic bezier via midpoint 0
    def side(t_a, t_b):
        # start point on arc end of source
        x1, y1 = r * math.cos(t_a), r * math.sin(t_a)
        x2, y2 = r * math.cos(t_b), r * math.sin(t_b)
        return (x1, y1), (x2, y2)

    (ax_, ay_), (bx, by) = side(t1_a, t1_b)
    (cx, cy), (dx, dy)   = side(t2_a, t2_b)

    def bez(p0, p1, p2):
        ts = np.linspace(0, 1, n)
        xs = (1-ts)**2 * p0[0] + 2*(1-ts)*ts*p1[0] + ts**2 * p2[0]
        ys = (1-ts)**2 * p0[1] + 2*(1-ts)*ts*p1[1] + ts**2 * p2[1]
        return xs, ys

    # Top curve: b -> d via (0,0)
    xs1, ys1 = bez((bx, by), (0, 0), (cx, cy))
    # Bottom curve: d -> b via (0,0)  (reversed)
    xs2, ys2 = bez((dx, dy), (0, 0), (ax_, ay_))

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    ax.fill(xs, ys, color=color, alpha=alpha, linewidth=0, zorder=2)


def draw(json_path, out_dir):
    entries = load_json(json_path)
    crop_to_d = {e["crop"]: set(e["diseases"]) for e in entries}
    top_crops = sorted(crop_to_d, key=lambda c: -len(crop_to_d[c]))[:TOP_N]

    # Pairwise shared-disease count
    shared = defaultdict(int)
    for i, a in enumerate(top_crops):
        for b in top_crops[i+1:]:
            n = len(crop_to_d[a] & crop_to_d[b])
            if n >= MIN_SHARED:
                shared[(a, b)] = n

    # Arc allocation: proportional to # diseases
    sizes = [len(crop_to_d[c]) for c in top_crops]
    total = sum(sizes)
    gap = 0.015
    usable = 2*math.pi - gap * len(top_crops)
    start = math.pi / 2
    arcs = {}
    for c, s in zip(top_crops, sizes):
        span = usable * s / total
        end = start - span
        arcs[c] = (end, start)
        start = end - gap

    fig, ax = plt.subplots(figsize=(14, 14), facecolor=BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.axis("off")
    ax.set_facecolor(BG)

    # Outer crop arcs
    R_IN, R_OUT = 0.95, 1.00
    colors = {c: CROP_COLORS[i % len(CROP_COLORS)] for i, c in enumerate(top_crops)}
    for c in top_crops:
        t1, t2 = arcs[c]
        th = np.linspace(t1, t2, 80)
        xs = np.concatenate([R_OUT*np.cos(th), R_IN*np.cos(th[::-1])])
        ys = np.concatenate([R_OUT*np.sin(th), R_IN*np.sin(th[::-1])])
        ax.add_patch(plt.Polygon(list(zip(xs, ys)), facecolor=colors[c],
                                  edgecolor="white", linewidth=1.0, zorder=5))
        # Label
        theta_mid = (t1 + t2) / 2
        lx, ly = 1.06 * math.cos(theta_mid), 1.06 * math.sin(theta_mid)
        rot = math.degrees(theta_mid)
        if 90 < rot % 360 < 270:
            rot += 180
        ax.text(lx, ly, c, ha="center", va="center", rotation=rot,
                rotation_mode="anchor", fontsize=9, fontfamily=FONT,
                color="#111111")

    # Ribbons. Use slice of each arc sized by # shared diseases; place them
    # side by side starting from the counter-clockwise end.
    allocated = {c: arcs[c][1] for c in top_crops}  # current "free" theta
    max_shared = max(shared.values()) if shared else 1
    for (a, b), n in sorted(shared.items(), key=lambda x: -x[1]):
        span_a = (arcs[a][1] - arcs[a][0]) * n / len(crop_to_d[a])
        span_b = (arcs[b][1] - arcs[b][0]) * n / len(crop_to_d[b])
        t1a = allocated[a]; t1b = t1a - span_a
        t2a = allocated[b]; t2b = t2a - span_b
        allocated[a] = t1b; allocated[b] = t2b
        alpha = 0.15 + 0.55 * (n / max_shared)
        bezier_ribbon(ax, t1a, t1b, t2a, t2b, R_IN, colors[a], alpha=alpha)

    ax.set_title(
        f"Chord — top {len(top_crops)} crops, ribbons = shared CORRECT diseases "
        f"(min shared = {MIN_SHARED})",
        fontsize=14, fontfamily=FONT, fontweight="bold", pad=14,
    )

    save_all(fig, out_dir, "viz_chord")
    plt.close(fig)


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
