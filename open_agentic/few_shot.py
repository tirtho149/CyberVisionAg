"""Few-shot baseline — single API call with k random labeled images.

No tools, no KB, no reasoning loop. Pure visual few-shot classification.

Usage:
    python -m CyberVisionAg.open_agentic.few_shot --num-classes 10 --k 4 --seed 42
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CYBERVISION_DIR = SCRIPT_DIR.parent
DATASET_ROOT = CYBERVISION_DIR / "Curated_Local_Dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
RESULTS_DIR = CYBERVISION_DIR / "results" / "open_agentic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are an expert plant pathologist. You will be shown labeled example images \
of plant diseases, followed by one unlabeled test image. Classify the test \
image into one of the provided disease classes.

Base your classification ONLY on visual similarity to the examples. \
Do NOT use filenames or metadata — only visual content.

Respond with JSON only (no markdown, no preamble):
{"prediction": "<class_name>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}\
"""


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(dataset, num_classes, images_per_class, seed, exclude=None):
    """Discover test classes and images (same logic as eval.py)."""
    test_dir = TEST_DIR / dataset
    if not test_dir.exists():
        sys.exit(f"ERROR: {test_dir} not found")

    all_classes = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and (not exclude or d.name not in exclude)
    ])

    rng = random.Random(seed)
    if num_classes and num_classes < len(all_classes):
        classes = sorted(rng.sample(all_classes, num_classes))
    else:
        classes = all_classes

    test_images = []
    for cls in classes:
        imgs = sorted(
            str(p) for p in (test_dir / cls).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if images_per_class and images_per_class < len(imgs):
            imgs = sorted(rng.sample(imgs, images_per_class))
        for img in imgs:
            test_images.append((img, cls))

    return classes, test_images


def collect_train_pool(dataset, classes):
    """Collect all training images for the given classes."""
    pool = []
    for cls in classes:
        cls_dir = TRAIN_DIR / dataset / cls
        if not cls_dir.exists():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                pool.append((str(img), cls))
    return pool


def _inline_image(path: str, max_bytes: int = 1_000_000) -> dict:
    """Encode image as base64 content block for Anthropic API.

    Compresses JPEGs if they exceed max_bytes to avoid 413 errors.
    """
    raw = Path(path).read_bytes()

    # Compress large images via PIL if available
    if len(raw) > max_bytes:
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(raw))
            # Resize if very large
            max_dim = 1024
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            raw = buf.getvalue()
        except ImportError:
            pass  # No PIL, send as-is

    data = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


# ── Classification ─────────────────────────────────────────────────────────────

def classify(test_image_path: str, examples: list[tuple[str, str]],
             classes: list[str], client: anthropic.Anthropic) -> dict:
    """Classify one test image with k labeled examples in a single API call.

    Args:
        examples: [(image_path, class_label), ...] — randomly sampled training images
    """
    content = []

    # Labeled examples
    for i, (img_path, label) in enumerate(examples):
        content.append(_inline_image(img_path))
        content.append({"type": "text", "text": f"Example {i+1} — Class: {label}"})

    # Test image (use neutral filename via inline encoding)
    content.append({"type": "text", "text":
        "\n══════════════════════════════════════\n"
        "TEST IMAGE — classify this one\n"
        "══════════════════════════════════════\n"
        f"Possible classes: {sorted(classes)}\n"
    })
    content.append(_inline_image(test_image_path))
    content.append({"type": "text", "text": "Classify. JSON only."})

    t0 = time.time()
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        elapsed = time.time() - t0
        text = resp.content[0].text if resp.content else ""
        pred = _parse_prediction(text, classes)
        # Sonnet pricing: $3/M input, $15/M output
        cost = (resp.usage.input_tokens * 3 + resp.usage.output_tokens * 15) / 1_000_000
        return {**pred, "cost_usd": cost, "duration_s": elapsed, "error": None}
    except Exception as e:
        return {
            "prediction": "UNKNOWN", "confidence": 0.0, "reasoning": "",
            "cost_usd": 0.0, "duration_s": time.time() - t0,
            "error": str(e)[:200],
        }


def _parse_prediction(text: str, classes: list[str]) -> dict:
    """Extract prediction from response text."""
    import re
    # Try JSON blocks
    for match in re.finditer(r'\{[^}]*"prediction"[^}]*\}', text):
        try:
            parsed = json.loads(match.group())
            if "prediction" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    # Fallback: find class name in text
    for cls in classes:
        if cls in text:
            return {"prediction": cls, "confidence": 0.0, "reasoning": "parsed from text"}
    return {"prediction": "UNKNOWN", "confidence": 0.0, "reasoning": "no prediction found"}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Few-shot baseline evaluation")
    parser.add_argument("--dataset", default="Soybean_Diseases")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--quick-test", type=int, default=None, metavar="N",
                        help="Shortcut: num-classes=N, images-per-class=1")
    parser.add_argument("--k", type=int, default=4,
                        help="Total labeled example images per API call (default: 4)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated class names to exclude")
    args = parser.parse_args()

    if args.quick_test:
        args.num_classes = args.quick_test
        args.images_per_class = 1

    exclude = set(args.exclude.split(",")) if args.exclude else None

    classes, test_images = load_dataset(
        args.dataset, args.num_classes, args.images_per_class, args.seed, exclude
    )
    train_pool = collect_train_pool(args.dataset, classes)

    print("FEW-SHOT BASELINE (single API call, no tools, no KB)")
    print(f"  Model         : {MODEL}")
    print(f"  Dataset       : {args.dataset}")
    print(f"  Classes       : {len(classes)}")
    print(f"  Test images   : {len(test_images)}")
    print(f"  k (examples)  : {args.k}")
    print(f"  Train pool    : {len(train_pool)} images")
    print(f"  Parallel      : {args.parallel}")
    print(f"  Seed          : {args.seed}")
    print()

    client = anthropic.Anthropic()
    rng = random.Random(args.seed)
    results = []
    total = len(test_images)

    def _process(idx_and_item):
        idx, (img_path, gt) = idx_and_item
        # Sample k random training images (same seed progression for reproducibility)
        examples = rng.sample(train_pool, min(args.k, len(train_pool)))
        print(f"  [{idx+1}/{total}] {gt} ...", end="", flush=True)
        result = classify(img_path, examples, classes, client)
        result["ground_truth"] = gt
        result["correct"] = result["prediction"] == gt
        status = "OK" if result["correct"] else "WRONG"
        if result["error"]:
            status = f"ERR({result['error'][:20]})"
        print(f" → {result['prediction']} ({status}, {result['duration_s']:.0f}s)", flush=True)
        return result

    if args.parallel <= 1:
        for item in enumerate(test_images):
            results.append(_process(item))
    else:
        # Pre-sample examples for each image (thread-safe)
        items_with_examples = []
        for idx, (img_path, gt) in enumerate(test_images):
            examples = rng.sample(train_pool, min(args.k, len(train_pool)))
            items_with_examples.append((idx, img_path, gt, examples))

        def _process_parallel(item):
            idx, img_path, gt, examples = item
            print(f"  [{idx+1}/{total}] {gt} ...", end="", flush=True)
            result = classify(img_path, examples, classes, client)
            result["ground_truth"] = gt
            result["correct"] = result["prediction"] == gt
            status = "OK" if result["correct"] else "WRONG"
            if result["error"]:
                status = f"ERR({result['error'][:20]})"
            print(f" → {result['prediction']} ({status}, {result['duration_s']:.0f}s)", flush=True)
            return result

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_process_parallel, item): item
                       for item in items_with_examples}
            for future in as_completed(futures):
                results.append(future.result())

    # Metrics
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r["error"])
    total_cost = sum(r["cost_usd"] for r in results)

    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        class_stats[r["ground_truth"]]["total"] += 1
        if r["correct"]:
            class_stats[r["ground_truth"]]["correct"] += 1

    print(f"\n  Accuracy       : {correct}/{n} ({correct/n*100:.1f}%)")
    print(f"  Avg cost/image : ${total_cost/n:.4f}")
    print(f"  Total cost     : ${total_cost:.4f}")
    print(f"  Errors         : {errors}")
    print(f"\n  Per-class accuracy:")
    for cls, stats in sorted(class_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"    {cls:30s} {acc:.0f}%")

    # Save summary
    log_dir = RESULTS_DIR / "few_shot" / "logs" / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "summary.json", "w") as f:
        json.dump({
            "config": {"k": args.k, "num_classes": len(classes),
                       "num_images": n, "seed": args.seed, "model": MODEL},
            "metrics": {"accuracy": correct/n*100, "correct": correct, "total": n,
                        "avg_cost_usd": total_cost/n, "total_cost_usd": total_cost},
            "per_class": {cls: s["correct"]/s["total"]*100
                          for cls, s in sorted(class_stats.items())},
        }, f, indent=2)
    print(f"\n  Summary: {log_dir}/summary.json")


if __name__ == "__main__":
    main()
