

import json, math, os, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

# Input: filtered JSON of VALID crops × CORRECT disease labels.
# Each disease counts as 1 unit (no image-count weighting).
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "disease_label_correct_only.json"
OUT_PNG    = "sunburst_diagram.png"
OUT_PDF    = "sunburst_diagram.pdf"
OUT_SVG    = "sunburst_diagram.svg"

FIG_SIZE    = (16, 16)
DPI         = 300
BG          = "#FFFDF7"
import matplotlib.font_manager as fm

# Use Times New Roman if installed, otherwise DejaVu Serif
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Times New Roman" if "Times New Roman" in _available else "DejaVu Serif"

# Ring geometry
R_CENTER    = 0.18
R_CROP_IN   = 0.20
R_CROP_OUT  = 0.48
R_DIS_IN    = 0.50
R_DIS_OUT   = 1.01

GAP_CROP    = 0.010
GAP_DIS     = 0.0008

# How many top crops to show individually; rest → "Others"
TOP_N_CROPS = 40

# How many diseases to show individually per crop; rest → "N others"
TOP_N_DISEASES_PER_CROP = 3

# Minimum disease slice angle (radians) — tiny slices still get a sliver
MIN_DIS_SPAN = 0.0018

# Reserved minimum arc (radians) for each NAMED disease slice so its label
# always fits. "+N others" gets whatever remains.
MIN_NAMED_DIS_SPAN = 0.028

# Only draw disease-ring text labels for crops ranked at or above this crop
# (by # of CORRECT diseases). Crops ranked below still get colored slices
# but no text. Set to None to label every crop.
DISEASE_LABEL_CUTOFF_CROP = "Sugarcane"

# Bright, cheerful, light & happy colour palette — one per crop.
# Last entry is reserved for the "Others" bucket.
CROP_COLORS = [
    "#FF6B6B",   # coral red
    "#FFD93D",   # sunny yellow
    "#6BCB77",   # fresh green
    "#4D96FF",   # sky blue
    "#FF922B",   # warm orange
    "#CC5DE8",   # lilac purple
    "#20C997",   # mint teal
    "#F06595",   # hot pink
    "#74C0FC",   # baby blue
    "#A9E34B",   # lime green
    "#FFA94D",   # peach orange
    "#DA77F2",   # lavender
    "#63E6BE",   # aqua
    "#FF8787",   # light red
    "#66D9E8",   # cyan
    "#B197FC",   # periwinkle
    "#FCC419",   # amber
    "#51CF66",   # emerald
    "#FF6B9D",   # rose pink
    "#5C7CFA",   # indigo
    "#E599F7",   # orchid
    "#8CE99A",   # light green
    "#FFB86B",   # apricot
    "#69DB7C",   # leaf green
    "#FFA8A8",   # salmon
    "#CED4DA",   # Others — light grey
]

# Lighter versions for disease ring (auto-computed)
def lighten(hex_color, f=0.38):
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i+2], 16)/255 for i in (0,2,4)]
    rgb = [c + (1-c)*f for c in rgb]
    return "#{:02X}{:02X}{:02X}".format(*(int(c*255) for c in rgb))

def darken(hex_color, f=0.18):
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i+2], 16)/255 for i in (0,2,4)]
    rgb = [c*(1-f) for c in rgb]
    return "#{:02X}{:02X}{:02X}".format(*(int(c*255) for c in rgb))


# ── JSON loading ──────────────────────────────────────────────────────────────

def load_json(path):
    """Load VALID crops × CORRECT diseases from disease_label_correct_only.json.

    Each disease is weighted as 1 unit. Crop total = number of CORRECT diseases
    for that crop.
    """
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    crop_totals   = {}
    crop_diseases = {}
    for e in entries:
        crop = e["crop"].strip()
        diseases = [d.strip() for d in e.get("diseases", []) if d and d.strip()]
        if not diseases:
            continue
        crop_totals[crop]   = len(diseases)
        crop_diseases[crop] = {d: 1 for d in diseases}
    return crop_totals, crop_diseases


# ── Drawing primitives ────────────────────────────────────────────────────────

def arc_poly(ax, t1, t2, r_in, r_out, color, edge="#FFFFFF", lw=0.5):
    n  = 200
    th = np.linspace(t1, t2, n)
    xs = list(r_out*np.cos(th)) + list(r_in*np.cos(th[::-1]))
    ys = list(r_out*np.sin(th)) + list(r_in*np.sin(th[::-1]))
    ax.add_patch(plt.Polygon(list(zip(xs,ys)), closed=True,
                             facecolor=color, edgecolor=edge, linewidth=lw))


def arc_label(ax, text, r_mid, theta_mid, size, weight="normal",
              color="#111111", max_chars=22):
    if not text:
        return
    x = r_mid * math.cos(theta_mid)
    y = r_mid * math.sin(theta_mid)
    rot = math.degrees(theta_mid)
    if 90 < rot % 360 < 270:
        rot += 180
    if len(text) > max_chars:
        text = text[:max_chars-1] + "…"
    ax.text(x, y, text, ha="center", va="center",
            rotation=rot, rotation_mode="anchor",
            fontsize=size, fontfamily=FONT, fontweight=weight,
            color=color, clip_on=True, zorder=10)


def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


# ── Main ──────────────────────────────────────────────────────────────────────

def draw(json_path, out_dir):
    crop_totals, crop_diseases = load_json(json_path)
    all_crops   = sorted(crop_totals, key=lambda c: -crop_totals[c])
    grand_total = sum(crop_totals.values())
    total_pairs = sum(len(v) for v in crop_diseases.values())

    # Cross-crop disease stats — used for middle disc + sort key
    disease_crop_count = defaultdict(int)
    for d_map in crop_diseases.values():
        for d in d_map:
            disease_crop_count[d] += 1
    unique_diseases = len(disease_crop_count)

    # Collapse diseases per crop: keep TOP_N (by cross-crop popularity), rest → "N others"
    def collapse(d_map):
        items = sorted(d_map.items(),
                        key=lambda kv: (-disease_crop_count[kv[0]], kv[0]))
        keep = items[:TOP_N_DISEASES_PER_CROP]
        rest = items[TOP_N_DISEASES_PER_CROP:]
        out = {name: w for name, w in keep}
        if rest:
            others_weight = sum(w for _, w in rest)
            out[f"+{len(rest)} others"] = others_weight
        return out

    crop_diseases = {c: collapse(m) for c, m in crop_diseases.items()}

    # Determine which crops get disease-ring text labels
    if DISEASE_LABEL_CUTOFF_CROP and DISEASE_LABEL_CUTOFF_CROP in all_crops:
        cutoff_rank = all_crops.index(DISEASE_LABEL_CUTOFF_CROP)
        labeled_crops = set(all_crops[:cutoff_rank + 1])
    else:
        labeled_crops = set(all_crops)

    # Split into top-N and "Others"
    top_crops   = all_crops[:TOP_N_CROPS]
    other_crops = all_crops[TOP_N_CROPS:]

    # Build ordered list: top crops + Others bucket
    ordered = top_crops[:]
    others_total    = sum(crop_totals[c] for c in other_crops)
    others_diseases = defaultdict(int)
    for c in other_crops:
        for d, n in crop_diseases[c].items():
            others_diseases[d] += n
    # Collapse the combined Others bucket too
    others_diseases = dict(others_diseases)
    if others_diseases:
        items = sorted(others_diseases.items(), key=lambda kv: -kv[1])
        keep = items[:TOP_N_DISEASES_PER_CROP]
        rest = items[TOP_N_DISEASES_PER_CROP:]
        others_diseases = {n: w for n, w in keep}
        if rest:
            others_diseases[f"+{len(rest)} others"] = sum(w for _, w in rest)

    has_others = bool(other_crops)
    color_map  = {}
    for i, c in enumerate(top_crops):
        color_map[c] = CROP_COLORS[i % (len(CROP_COLORS)-1)]
    if has_others:
        ordered.append("__OTHERS__")
        color_map["__OTHERS__"] = CROP_COLORS[-1]

    # ── Canvas ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-1.10, 1.10)
    ax.axis("off")
    ax.set_facecolor(BG)

    # ── Crop ring ─────────────────────────────────────────────────────────────
    total_entries   = len(ordered)
    total_gap_crop  = GAP_CROP * total_entries
    usable_crop     = 2*math.pi - total_gap_crop
    start           = math.pi/2      # 12 o'clock, clockwise

    crop_angles = {}

    for label in ordered:
        if label == "__OTHERS__":
            images = others_total
        else:
            images = crop_totals[label]

        frac  = images / grand_total
        span  = usable_crop * frac
        end   = start - span
        crop_angles[label] = (end, start)
        color = color_map[label]

        arc_poly(ax, end, start, R_CROP_IN, R_CROP_OUT, color, lw=0.6)

        theta_mid = (start + end) / 2
        arc_len   = abs(span) * ((R_CROP_IN + R_CROP_OUT)/2)

        display = f"Others ({len(other_crops)})" if label == "__OTHERS__" else label

        if arc_len > 0.07:
            arc_label(ax, display, (R_CROP_IN+R_CROP_OUT)/2, theta_mid,
                      size=14, weight="bold", color="#111111")
        elif arc_len > 0.04:
            arc_label(ax, display, (R_CROP_IN+R_CROP_OUT)/2, theta_mid,
                      size=11, weight="bold", color="#111111", max_chars=12)
        elif arc_len > 0.022:
            arc_label(ax, display, (R_CROP_IN+R_CROP_OUT)/2, theta_mid,
                      size=8.5, weight="bold", color="#111111", max_chars=9)

        start = end - GAP_CROP

    # ── Disease ring ──────────────────────────────────────────────────────────
    for label in ordered:
        t1, t2 = crop_angles[label]

        if label == "__OTHERS__":
            diseases   = dict(sorted(others_diseases.items(), key=lambda x: -x[1]))
            crop_sum   = others_total
        else:
            diseases   = dict(sorted(crop_diseases[label].items(), key=lambda x: -x[1]))
            crop_sum   = crop_totals[label]

        n_dis      = len(diseases)
        if n_dis == 0:
            continue

        total_gap_dis = GAP_DIS * n_dis
        usable_dis    = (t2 - t1) - total_gap_dis
        if usable_dis <= 0:
            continue

        # Reserve a minimum arc for each NAMED disease so its label fits.
        # "+N others" takes whatever remains (proportional to its weight, but
        # floored at MIN_DIS_SPAN).
        items = list(diseases.items())
        named_idx   = [i for i, (n, _) in enumerate(items) if not n.startswith("+")]
        others_idx  = [i for i, (n, _) in enumerate(items) if n.startswith("+")]

        reserved = min(MIN_NAMED_DIS_SPAN, usable_dis / max(1, n_dis))
        reserved_total = reserved * len(named_idx)
        leftover = max(0.0, usable_dis - reserved_total)
        leftover_weight = sum(items[i][1] for i in others_idx) or 1

        spans = [0.0] * n_dis
        for i in named_idx:
            spans[i] = reserved
        for i in others_idx:
            w = items[i][1]
            spans[i] = max(MIN_DIS_SPAN, leftover * w / leftover_weight)

        # Rescale if we overshot
        total_span = sum(spans)
        if total_span > usable_dis and total_span > 0:
            s = usable_dis / total_span
            spans = [x * s for x in spans]

        d_start = t2
        base_rgb = [int(color_map[label].lstrip("#")[i:i+2], 16)/255 for i in (0,2,4)]
        draw_disease_text = label != "__OTHERS__" and label in labeled_crops

        for idx, ((disease, count), span_d) in enumerate(zip(items, spans)):
            d_end = d_start - span_d

            # Very pastel shades for disease ring — light and airy
            f = 0.55 + 0.18 * (idx % 2)
            rgb = [c + (1-c)*f for c in base_rgb]
            color_slice = "#{:02X}{:02X}{:02X}".format(*(int(c*255) for c in rgb))

            arc_poly(ax, d_end, d_start, R_DIS_IN, R_DIS_OUT,
                     color_slice, lw=0.4)

            theta_mid = (d_start + d_end) / 2
            arc_len   = abs(span_d) * ((R_DIS_IN + R_DIS_OUT)/2)
            is_named  = not disease.startswith("+")

            if draw_disease_text:
                if arc_len > 0.048:
                    arc_label(ax, disease, (R_DIS_IN+R_DIS_OUT)/2, theta_mid,
                              size=14.0, color="#111111", max_chars=26)
                elif arc_len > 0.026:
                    arc_label(ax, disease, (R_DIS_IN+R_DIS_OUT)/2, theta_mid,
                              size=11.5, color="#111111", max_chars=22)
                elif arc_len > 0.014:
                    arc_label(ax, disease, (R_DIS_IN+R_DIS_OUT)/2, theta_mid,
                              size=9.0, color="#111111", max_chars=16)
                elif is_named:
                    # Force-draw named-disease label even when the slice is thin
                    arc_label(ax, disease, (R_DIS_IN+R_DIS_OUT)/2, theta_mid,
                              size=7.0, color="#111111", max_chars=14)

            d_start = d_end - GAP_DIS

    # ── Centre disc ────────────────────────────────────────────────────────────
    # Shadow rings
    for r, alpha in [(R_CENTER+0.030, 0.06), (R_CENTER+0.018, 0.10), (R_CENTER+0.008, 0.14)]:
        shadow = plt.Circle((0,0), r, facecolor="#CCCCCC", edgecolor="none",
                            alpha=alpha, zorder=4)
        ax.add_patch(shadow)

    disc = plt.Circle((0,0), R_CENTER,
                      facecolor="#FFFFFF", edgecolor="#E0E0E0",
                      linewidth=2.0, zorder=5)
    ax.add_patch(disc)

    ax.text(0,  0.070, fmt(unique_diseases),
            ha="center", va="center",
            fontsize=30, fontfamily=FONT, fontweight="bold",
            color="#111111", zorder=6)
    ax.text(0,  0.008, "unique diseases",
            ha="center", va="center",
            fontsize=12, fontfamily=FONT, fontweight="normal",
            color="#333333", zorder=6)
    ax.text(0, -0.042,
            f"{total_pairs} crop-disease pairs",
            ha="center", va="center",
            fontsize=10, fontfamily=FONT, fontweight="bold",
            color="#333333", zorder=6)
    ax.text(0, -0.080,
            f"{len(all_crops)} crops",
            ha="center", va="center",
            fontsize=8, fontfamily=FONT,
            color="#555555", zorder=6)

    # ── Export ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    for fname, fmt_str in [(OUT_PNG,"png"),(OUT_PDF,"pdf"),(OUT_SVG,"svg")]:
        path = os.path.join(out_dir, fname)
        fig.savefig(path, format=fmt_str, dpi=DPI,
                    bbox_inches="tight", facecolor=BG)
        print(f"[✓] {fmt_str.upper():4s} → {path}")
    plt.close(fig)
    print("Done.")


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"Reading: {json_path}")
    draw(json_path, out_dir)

if __name__ == "__main__":
    main()