"""Shared helpers for the visualization scripts."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_JSON = SCRIPT_DIR / "disease_label_correct_only.json"

BG = "#FFFDF7"

_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Times New Roman" if "Times New Roman" in _available else "DejaVu Serif"

CROP_COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF922B",
    "#CC5DE8", "#20C997", "#F06595", "#74C0FC", "#A9E34B",
    "#FFA94D", "#DA77F2", "#63E6BE", "#FF8787", "#66D9E8",
    "#B197FC", "#FCC419", "#51CF66", "#FF6B9D", "#5C7CFA",
]


def load_json(path=INPUT_JSON):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def crop_disease_pairs(entries):
    """Return a list of (crop, disease) tuples."""
    return [(e["crop"], d) for e in entries for d in e.get("diseases", [])]


def disease_crop_map(entries):
    """Return dict: disease -> list of crops that have it (CORRECT)."""
    m = {}
    for e in entries:
        for d in e.get("diseases", []):
            m.setdefault(d, []).append(e["crop"])
    return m


def save_all(fig, out_dir, basename, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi, bbox_inches="tight", facecolor=BG)
        print(f"[✓] {ext.upper():4s} → {path}")
