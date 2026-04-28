#!/usr/bin/env python3
"""
Emit a comma-separated list of disease-class folder names to exclude,
based on SageQualityChecker's two-layer judge verdicts.

Reads: CyberVisionAg/SageQualityChecker/disease_label_subset_report.json
Drops: every disease labeled INCORRECT or QUESTIONABLE for the given crop.

Usage:
    python3 quality_exclude.py --crop Sugarcane
    python3 quality_exclude.py --crop Soybean --existing "A,B,C"

Output: one line, comma-separated class names (spaces -> underscores),
merged with --existing and deduped. Prints just the --existing string
(possibly empty) if the report is missing, the crop is absent, or the
crop's Layer 2 check was skipped. Never raises.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_REPORT = (
    Path(__file__).resolve().parent.parent
    / "SageQualityChecker"
    / "disease_label_subset_report.json"
)

DROP_VERDICTS = {"INCORRECT", "QUESTIONABLE"}


def load_bad_labels(report_path: Path, crop: str) -> list[str]:
    if not report_path.exists():
        return []
    try:
        records = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    crop_lc = crop.strip().lower()
    for rec in records:
        if rec.get("crop", "").strip().lower() != crop_lc:
            continue
        dc = rec.get("disease_check", {}) or {}
        if dc.get("skipped"):
            return []
        out: list[str] = []
        for v in dc.get("disease_verdicts", []) or []:
            if v.get("verdict") in DROP_VERDICTS:
                name = (v.get("disease") or "").strip()
                if name:
                    out.append(name.replace(" ", "_"))
        return out
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True,
                    help="Crop name as it appears in the quality report (case-insensitive)")
    ap.add_argument("--existing", default="",
                    help="Existing comma-separated exclude list to merge with")
    ap.add_argument("--report", default=str(DEFAULT_REPORT),
                    help="Path to disease_label_subset_report.json")
    args = ap.parse_args()

    bad = load_bad_labels(Path(args.report), args.crop)

    existing = [x.strip() for x in args.existing.split(",") if x.strip()]
    seen: set[str] = set()
    merged: list[str] = []
    for item in existing + bad:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    print(",".join(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
