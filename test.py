"""
For each (crop, disease) combination, list how many distinct sources contributed
images and which sources they are (with per-source image counts).

Layout assumed:
    <DATASET_ROOT>/<Crop>_<split>/<class>/<source>_<id>.<ext>

Output CSV columns:
    crop, class, split, num_sources, total_images, sources
        - sources is "Source1:count1; Source2:count2" sorted by count desc
"""

import csv
import re
from collections import defaultdict, Counter
from pathlib import Path

DATASET_ROOT = Path("/Users/tirthoroy/Desktop/CyberVisionAg/Prepared_Dataset")
OUTPUT_CSV = Path("crop_disease_sources.csv")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLIT_SUFFIX_RE = re.compile(r"_(train|test|val|valid|validation)$", re.IGNORECASE)
TRAILING_ID_RE = re.compile(r"_\d+$")


def parse_crop_split(folder: str) -> tuple[str, str]:
    m = SPLIT_SUFFIX_RE.search(folder)
    return (folder[: m.start()], m.group(1).lower()) if m else (folder, "")


def extract_source(stem: str) -> str:
    return TRAILING_ID_RE.sub("", stem)


def main() -> None:
    # (crop, class, split) -> Counter of source -> count
    bucket: dict[tuple[str, str, str], Counter] = defaultdict(Counter)

    for path in DATASET_ROOT.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        rel = path.relative_to(DATASET_ROOT).parts
        if len(rel) < 3:
            continue
        crop, split = parse_crop_split(rel[0])
        class_name = rel[1]
        source = extract_source(path.stem)
        bucket[(crop, class_name, split)][source] += 1

    rows = []
    for (crop, class_name, split), source_counts in sorted(bucket.items()):
        sources_str = "; ".join(
            f"{src}:{n}" for src, n in source_counts.most_common()
        )
        rows.append({
            "crop": crop,
            "class": class_name,
            "split": split,
            "num_sources": len(source_counts),
            "total_images": sum(source_counts.values()),
            "sources": sources_str,
        })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["crop", "class", "split", "num_sources", "total_images", "sources"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} (crop, class, split) rows to {OUTPUT_CSV}\n")
    for r in rows:
        print(
            f"  {r['crop']:<12} {r['class']:<22} {r['split']:<6} "
            f"{r['num_sources']} sources, {r['total_images']} images  "
            f"[{r['sources']}]"
        )


if __name__ == "__main__":
    main()