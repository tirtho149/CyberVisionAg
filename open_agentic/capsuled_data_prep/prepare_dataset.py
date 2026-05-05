"""Filter and tag dataset images using KB descriptions (capsule version).

Self-contained variant of CyberVisionAg/open_agentic/prepare_dataset.py
intended to run on an HPC cluster where the source dataset lives locally
and the AgCrawler repo is not available. All disease_registry helpers
needed by this script are inlined below.

Usage (from this directory):
    python prepare_dataset.py \\
      --input-dir /work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/Tomato_Diseases \\
      --output-dir ./out/Tomato_Diseases \\
      --max-per-part 5 --test-per-class 5 --seed 42 --parallel 12

The script expects:
  * A KB workbook at ./kb/<Crop>/internet.xlsx (Crop derived from input
    dir name with `_Diseases` / `_Disease` stripped). All 10 supported
    crops are shipped in the capsule.
  * ANTHROPIC_API_KEY in the environment or in a .env file alongside this
    script.

Output:
  * <output-dir>/<class>/<part>/  -- reference images, max-per-part each
  * <output-dir>/<class>/rejected/ -- images that did not match the KB
  * <output-dir>_test/<class>/    -- test split (separate root)
  * <output-dir>/_tags.csv         -- audit log: class, file, match, part, split, reason
"""

import argparse
import base64
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openpyxl

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Config ────────────────────────────────────────────────────────────────────

API_MODEL = "claude-sonnet-4-6"  # keep in sync with disease_registry/config.py
API_MAX_TOKENS = 16000

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


# ── Inlined helpers (originally from disease_registry/shared.py) ──────────────

_API_KEY: str | None = None
_ANTHROPIC_CLIENT = None


def _get_api_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        if not _API_KEY:
            env_path = SCRIPT_DIR / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        _API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if not _API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found in environment or .env file"
            )
    return _API_KEY


def _get_client():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        import anthropic
        _ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=_get_api_key())
    return _ANTHROPIC_CLIENT


def _add_additional_properties_false(schema: dict) -> dict:
    """Recursively add additionalProperties: false to all object types."""
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    schema_type = result.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if is_object:
        result["additionalProperties"] = False
        if "properties" in result:
            result["properties"] = {
                k: _add_additional_properties_false(v)
                for k, v in result["properties"].items()
            }
    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _add_additional_properties_false(result["items"])
    return result


def api_query(
    prompt: str,
    system_prompt: str,
    json_schema: dict | None = None,
    content_blocks: list | None = None,
    max_tokens: int | None = None,
) -> str | None:
    client = _get_client()
    if content_blocks:
        user_content = content_blocks + [{"type": "text", "text": prompt}]
    else:
        user_content = prompt
    kwargs = {
        "model": API_MODEL,
        "max_tokens": max_tokens or API_MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }
    if json_schema:
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": _add_additional_properties_false(json_schema),
            }
        }
    response = client.messages.create(**kwargs)
    return response.content[0].text if response.content else None


def parse_json_result(raw: str | None, stage_name: str) -> dict:
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
            try:
                return json.loads("\n".join(lines))
            except json.JSONDecodeError:
                pass
        return {}


# ── KB loading (capsule path) ─────────────────────────────────────────────────

def load_kb(dataset: str) -> dict[str, str]:
    """Load KB descriptions from the capsule's local kb/<crop>/internet.xlsx."""
    crop = dataset.replace("_Diseases", "").replace("_Disease", "")
    path = SCRIPT_DIR / "kb" / crop / "internet.xlsx"
    if not path.exists():
        print(f"  WARNING: KB not found at {path}", file=sys.stderr)
        return {}
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    kb: dict[str, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        disease = str(row[0]).strip() if row[0] else ""
        parts = str(row[3]).strip() if len(row) > 3 and row[3] else ""
        visual = str(row[4]).strip() if len(row) > 4 and row[4] else ""
        if disease:
            entry = ""
            if parts:
                entry += f"Affected parts: {parts}\n"
            if visual:
                entry += f"Visual symptoms: {visual}"
            kb[disease] = entry.strip() if entry.strip() else "No description available"
    wb.close()
    return kb


# ── Image preparation ─────────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    """Load image as base64; resize if over 3.7 MB to fit Anthropic limits."""
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


def tag_image(image_path: str, class_name: str, kb_description: str) -> dict:
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
        system_prompt=(
            "You are a plant pathology expert. Analyze the image and respond "
            "with structured JSON."
        ),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and tag dataset images using KB.")
    parser.add_argument("--input-dir", required=True,
                        help="Source folder with class subfolders.")
    parser.add_argument("--output-dir", required=True,
                        help="Destination folder for filtered references.")
    parser.add_argument("--max-per-part", type=int, default=5,
                        help="Max reference images per class/part. Default: 5.")
    parser.add_argument("--test-per-class", type=int, default=0,
                        help="Max test images per class (0 = no test split). "
                             "Saved to <output-dir>_test/<class>/.")
    parser.add_argument("--max-inspect-per-class", type=int, default=None,
                        help="Cap images inspected per class (default: all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Concurrent API calls.")
    parser.add_argument("--filename-prefix", default=None,
                        help="Only process files starting with this prefix.")
    parser.add_argument("--exclude", default=None,
                        help="Comma-separated class names to skip.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"ERROR: --input-dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(2)
    dataset = input_dir.name

    user_exclude = set(args.exclude.split(",")) if args.exclude else set()
    all_exclude = EXCLUDE | user_exclude
    classes = sorted(
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name not in all_exclude
    )

    kb = load_kb(dataset)
    print(f"Dataset: {dataset}")
    print(f"Classes: {len(classes)}")
    print(f"KB entries: {len(kb)}")
    print(f"Max ref per part: {args.max_per_part}")
    print(f"Max test per class: {args.test_per_class}")
    print(f"Max inspect per class: {args.max_inspect_per_class or 'all'}")
    print(f"Seed: {args.seed}")
    print(f"Parallel: {args.parallel}")
    print()

    ref_quotas: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    test_counts: dict[str, int] = defaultdict(int)
    total_ref = 0
    total_test = 0
    total_rejected = 0
    total_skipped = 0
    total_inspected = 0
    errors: list[tuple[str, str, str]] = []
    all_tags: list[tuple[str, str, str, str, str, str]] = []

    for cls in classes:
        cls_dir = input_dir / cls
        imgs = sorted(
            str(p) for p in cls_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
            and (not args.filename_prefix or p.name.startswith(args.filename_prefix))
        )

        rng = random.Random(args.seed)
        rng.shuffle(imgs)
        if args.max_inspect_per_class and len(imgs) > args.max_inspect_per_class:
            imgs = imgs[: args.max_inspect_per_class]

        kb_desc = kb.get(cls, "No description available")
        print(f"  {cls}: inspecting {len(imgs)} images, KB={'yes' if cls in kb else 'NO'}")

        def _process(img_path: str):
            try:
                tag = tag_image(img_path, cls, kb_desc)
                return img_path, tag, None
            except Exception as e:
                return img_path, {"match": "no", "part": "whole_plant", "reason": str(e)}, str(e)

        results: list = []
        if args.parallel <= 1:
            for img_path in imgs:
                results.append(_process(img_path))
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(_process, p): p for p in imgs}
                for future in as_completed(futures):
                    results.append(future.result())

        path_order = {p: i for i, p in enumerate(imgs)}
        results.sort(key=lambda r: path_order.get(r[0], 0))

        for img_path, tag, err in results:
            total_inspected += 1
            fname = Path(img_path).name
            if err:
                errors.append((cls, fname, err))
                continue

            part = tag["part"]
            split = "skipped"

            if tag["match"] == "yes":
                if ref_quotas[cls][part] < args.max_per_part:
                    dest = output_dir / cls / part
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest / fname)
                    ref_quotas[cls][part] += 1
                    total_ref += 1
                    split = "ref"
                    if ref_quotas[cls][part] == args.max_per_part:
                        print(f"    {cls}/{part}: {args.max_per_part}/{args.max_per_part} reached")
                elif args.test_per_class > 0 and test_counts[cls] < args.test_per_class:
                    test_dir = Path(str(output_dir) + "_test")
                    dest = test_dir / cls
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest / fname)
                    test_counts[cls] += 1
                    total_test += 1
                    split = "test"
                    if test_counts[cls] == args.test_per_class:
                        print(f"    {cls}/test: {args.test_per_class}/{args.test_per_class} reached")
                else:
                    total_skipped += 1
            else:
                dest = output_dir / cls / "rejected"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest / fname)
                total_rejected += 1
                split = "rejected"

            all_tags.append((cls, fname, tag["match"], part, split, tag["reason"]))

    # Summary
    print()
    print("=" * 60)
    print(f"Inspected: {total_inspected}")
    print(f"Reference: {total_ref}")
    print(f"Test:      {total_test}")
    print(f"Rejected:  {total_rejected}")
    print(f"Skipped:   {total_skipped} (quota full)")
    print(f"Errors:    {len(errors)}")

    if errors:
        print()
        print("Errors:")
        for cls, fname, err in errors:
            print(f"  {cls}/{fname}: {err[:120]}")

    print()
    print(f"Below ref quota ({args.max_per_part}):")
    any_below = False
    for cls in classes:
        for part in sorted(ref_quotas[cls]):
            count = ref_quotas[cls][part]
            if count < args.max_per_part:
                print(f"  {cls}/{part}: {count}/{args.max_per_part}")
                any_below = True
    if not any_below:
        print("  None -- all class/parts met ref quota")

    if args.test_per_class > 0:
        print()
        print(f"Below test quota ({args.test_per_class}):")
        any_below_test = False
        for cls in classes:
            count = test_counts[cls]
            if count < args.test_per_class:
                print(f"  {cls}: {count}/{args.test_per_class}")
                any_below_test = True
        if not any_below_test:
            print("  None -- all classes met test quota")

    # Audit CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "_tags.csv"
    with audit_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "filename", "match", "part", "split", "reason"])
        w.writerows(all_tags)
    print()
    print(f"Wrote audit log: {audit_path}")


if __name__ == "__main__":
    main()
