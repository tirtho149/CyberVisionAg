"""Build part_index.md from a prepared dataset's class/part folder layout.

Output format matches every existing part_index.md in the repo (plain
headers, no '##', no '- ' bullets):

    # Organ Part Index — <Crop>

    Use this to narrow candidates based on the plant part visible in the test image.

    leaf (N classes)
    Class1
    Class2
    ...

    stem (M classes)
    ...

Usage:
    python -m open_agentic.build_part_index \\
        --ref-dir CyberVisionAg/Prepared_Dataset/Tomato \\
        --exclude "ClassA,ClassB"

The script writes <ref-dir>/part_index.md (overwrites). Excluded classes
are skipped so the file always reflects the post-EXCLUDE class set the
eval will actually use.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

PARTS = ("leaf", "stem", "root", "pod", "seed", "whole_plant")
SKIP_DIRS = {"rejected", "merged", "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref-dir", required=True, type=Path,
                        help="Prepared dataset root (with <Class>/<part>/ subfolders).")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated class names to skip.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <ref-dir>/part_index.md).")
    args = parser.parse_args()

    ref_dir: Path = args.ref_dir
    if not ref_dir.is_dir():
        raise SystemExit(f"ERROR: --ref-dir does not exist: {ref_dir}")

    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}
    out = args.out or (ref_dir / "part_index.md")
    crop = ref_dir.name

    part_to_classes: dict[str, list[str]] = defaultdict(list)
    skipped = []
    for cls_dir in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if cls in exclude:
            skipped.append(cls)
            continue
        for part in PARTS:
            part_dir = cls_dir / part
            if part_dir.is_dir() and any(
                p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                for p in part_dir.iterdir()
            ):
                part_to_classes[part].append(cls)

    lines = [
        f"# Organ Part Index — {crop}",
        "",
        "Use this to narrow candidates based on the plant part visible in the test image.",
        "",
    ]
    for part in PARTS:
        cs = part_to_classes.get(part, [])
        if not cs:
            continue
        lines.append(f"{part} ({len(cs)} classes)")
        lines.extend(cs)
        lines.append("")

    out.write_text("\n".join(lines))

    n_classes = sum(len(v) for v in part_to_classes.values())
    n_unique = len({c for v in part_to_classes.values() for c in v})
    print(f"Wrote {out}")
    print(f"  unique classes: {n_unique}, part-class entries: {n_classes}, "
          f"skipped (excluded): {len(skipped)}")


if __name__ == "__main__":
    main()
