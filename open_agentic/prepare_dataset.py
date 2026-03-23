"""Filter and tag dataset images using KB descriptions.

For each image, one API call checks if it matches the KB description
and tags the affected plant part. Images are copied into part-based
subfolders. Non-matching images go to rejected/.

Usage:
    python -m CyberVisionAg.open_agentic.prepare_dataset \
      --input-dir CyberVisionAg/Curated_Local_Dataset/train/Soybean_Diseases \
      --output-dir CyberVisionAg/Prepared_Dataset/Soybean_Diseases \
      --max-per-part 5 --seed 42 --parallel 12
"""

import argparse
import base64
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openpyxl

SCRIPT_DIR = Path(__file__).resolve().parent
CYBERVISION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CYBERVISION_DIR.parent
REGISTRY_OUTPUTS = PROJECT_ROOT / "disease_registry" / "outputs"

EXCLUDE = {
    ".DS_Store", "Diaporthe_2015_Kanawha", "Green_stem",
    "Fusarium_healthy_vs_infected", "Stem_Canker", "Top_Dieback",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Does the image match the disease description?",
        },
        "part": {
            "type": "string",
            "enum": ["leaf", "stem", "root", "pod", "seed", "whole_plant"],
            "description": "The primary plant part shown in the image.",
        },
        "reason": {
            "type": "string",
            "description": "One sentence explaining why match is yes or no.",
        },
    },
    "required": ["match", "part", "reason"],
}


def load_kb(dataset: str) -> dict[str, str]:
    """Load KB descriptions from internet xlsx."""
    crop = dataset.replace("_Diseases", "").replace("_Disease", "").replace("_Leaf", "")
    for name in [crop, crop.capitalize(), crop.lower()]:
        path = REGISTRY_OUTPUTS / f"{name}_internet.xlsx"
        if path.exists():
            wb = openpyxl.load_workbook(path, read_only=True)
            ws = wb.active
            kb = {}
            for row in ws.iter_rows(min_row=2, values_only=True):
                disease = str(row[0]).strip() if row[0] else ""
                visual = str(row[4]).strip() if len(row) > 4 and row[4] else ""
                parts = str(row[3]).strip() if len(row) > 3 and row[3] else ""
                if disease:
                    entry = ""
                    if parts:
                        entry += f"Affected parts: {parts}\n"
                    if visual:
                        entry += f"Visual symptoms: {visual}"
                    kb[disease] = entry.strip() if entry.strip() else "No description available"
            wb.close()
            return kb
    return {}


def load_image_b64(path: str) -> str:
    """Load image as base64, resize if over 3.7MB."""
    data = Path(path).read_bytes()
    if len(data) <= 3_700_000:
        return base64.standard_b64encode(data).decode("utf-8")
    from PIL import Image
    import io
    img = Image.open(path)
    for max_dim in [2048, 1568, 1200, 800]:
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            resized = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )
        else:
            resized = img
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        if buf.tell() <= 3_700_000:
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=60)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def tag_image(
    image_path: str,
    class_name: str,
    kb_description: str,
) -> dict:
    """Tag a single image via Anthropic API with structured output."""
    from disease_registry.shared import api_query, parse_json_result

    img_b64 = load_image_b64(image_path)
    ext = Path(image_path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"

    prompt = (
        f"This image is labeled as **{class_name}** in a plant disease dataset.\n\n"
        f"## Disease Description (from knowledge base)\n{kb_description}\n\n"
        f"## Task\n"
        f"1. Does this image actually show symptoms consistent with {class_name} "
        f"as described above? Answer 'yes' or 'no'.\n"
        f"2. What is the primary plant part shown? Choose one: "
        f"leaf, stem, root, pod, seed, whole_plant.\n"
        f"3. One sentence explaining your match decision."
    )

    content_blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img_b64,
            },
        }
    ]

    raw = api_query(
        prompt=prompt,
        system_prompt="You are a plant pathology expert. Analyze the image and respond with structured JSON.",
        json_schema=TAG_SCHEMA,
        content_blocks=content_blocks,
        max_tokens=200,
    )

    result = parse_json_result(raw, "tag_image")
    return {
        "match": result.get("match", "no"),
        "part": result.get("part", "whole_plant"),
        "reason": result.get("reason", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Filter and tag dataset images using KB")
    parser.add_argument("--input-dir", required=True, help="Source folder with class subfolders")
    parser.add_argument("--output-dir", required=True, help="Destination folder")
    parser.add_argument("--max-per-part", type=int, default=5, help="Max images per class/part")
    parser.add_argument("--max-inspect-per-class", type=int, default=None,
                        help="Max images to inspect per class (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel API calls")
    parser.add_argument("--filename-prefix", default=None,
                        help="Only process files starting with this prefix")
    parser.add_argument("--exclude", default=None,
                        help="Comma-separated list of class names to skip")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    dataset = input_dir.name

    # Discover classes
    user_exclude = set(args.exclude.split(",")) if args.exclude else set()
    all_exclude = EXCLUDE | user_exclude
    classes = sorted(
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name not in all_exclude
    )

    # Load KB
    kb = load_kb(dataset)
    print(f"Dataset: {dataset}")
    print(f"Classes: {len(classes)}")
    print(f"KB entries: {len(kb)}")
    print(f"Max per part: {args.max_per_part}")
    print(f"Max inspect per class: {args.max_inspect_per_class or 'all'}")
    print(f"Seed: {args.seed}")
    print(f"Parallel: {args.parallel}")
    print()

    # Track quotas: {class: {part: count}}
    quotas = defaultdict(lambda: defaultdict(int))
    total_matched = 0
    total_rejected = 0
    total_skipped = 0
    total_inspected = 0
    errors = []
    all_tags = []  # (class, filename, match, part, reason)

    for cls in classes:
        cls_dir = input_dir / cls
        imgs = sorted(
            str(p) for p in cls_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
            and (not args.filename_prefix or p.name.startswith(args.filename_prefix))
        )

        # Shuffle deterministically
        rng = random.Random(args.seed)
        rng.shuffle(imgs)

        # Cap images to inspect
        if args.max_inspect_per_class and len(imgs) > args.max_inspect_per_class:
            imgs = imgs[:args.max_inspect_per_class]

        kb_desc = kb.get(cls, "No description available")
        if kb_desc == "No description available":
            print(f"  WARNING: {cls} has no KB entry")

        print(f"  {cls}: inspecting {len(imgs)} images, KB={'yes' if cls in kb else 'NO'}")

        def _process(img_path):
            try:
                tag = tag_image(img_path, cls, kb_desc)
                return img_path, tag, None
            except Exception as e:
                return img_path, {"match": "no", "part": "whole_plant", "reason": str(e)}, str(e)

        # Process images
        results = []
        if args.parallel <= 1:
            for img_path in imgs:
                results.append(_process(img_path))
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(_process, p): p for p in imgs}
                for future in as_completed(futures):
                    results.append(future.result())

        # Sort back to deterministic order
        path_order = {p: i for i, p in enumerate(imgs)}
        results.sort(key=lambda r: path_order.get(r[0], 0))

        # Copy images to output
        for img_path, tag, err in results:
            total_inspected += 1
            fname = Path(img_path).name
            if err:
                errors.append((cls, fname, err))
                continue

            part = tag["part"]
            all_tags.append((cls, fname, tag["match"], part, tag["reason"]))

            if tag["match"] == "yes":
                if quotas[cls][part] >= args.max_per_part:
                    total_skipped += 1
                    continue
                dest = output_dir / cls / part
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest / fname)
                quotas[cls][part] += 1
                total_matched += 1

                if quotas[cls][part] == args.max_per_part:
                    print(f"    {cls}/{part}: {args.max_per_part}/{args.max_per_part} reached")
            else:
                dest = output_dir / cls / "rejected"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest / fname)
                total_rejected += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Inspected: {total_inspected}")
    print(f"Matched:   {total_matched}")
    print(f"Rejected:  {total_rejected}")
    print(f"Skipped:   {total_skipped} (quota full)")
    print(f"Errors:    {len(errors)}")

    if errors:
        print(f"\nErrors:")
        for cls, fname, err in errors:
            print(f"  {cls}/{fname}: {err[:100]}")

    # Report below-quota class/parts
    print(f"\nBelow quota ({args.max_per_part}):")
    any_below = False
    for cls in classes:
        for part in sorted(quotas[cls]):
            count = quotas[cls][part]
            if count < args.max_per_part:
                print(f"  {cls}/{part}: {count}/{args.max_per_part}")
                any_below = True
    if not any_below:
        print("  None -- all class/parts met quota")

    empty = [cls for cls in classes if not quotas[cls]]
    if empty:
        print(f"\nClasses with zero matched images: {empty}")

    # Save reasoning log
    tags_path = output_dir / "tags.csv"
    with open(tags_path, "w") as f:
        f.write("class,filename,match,part,reason\n")
        for cls_name, fname, match, part, reason in all_tags:
            reason_clean = reason.replace('"', "'").replace(",", ";")
            f.write(f'{cls_name},{fname},{match},{part},"{reason_clean}"\n')
    print(f"Tags log saved: {tags_path}")

    # Save metadata
    meta = {
        "dataset": dataset,
        "seed": args.seed,
        "max_per_part": args.max_per_part,
        "max_inspect_per_class": args.max_inspect_per_class,
        "classes": len(classes),
        "total_inspected": total_inspected,
        "total_matched": total_matched,
        "total_rejected": total_rejected,
        "total_skipped": total_skipped,
        "errors": len(errors),
        "quotas": {cls: dict(quotas[cls]) for cls in classes},
    }
    meta_path = output_dir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMetadata saved: {meta_path}")


if __name__ == "__main__":
    main()
