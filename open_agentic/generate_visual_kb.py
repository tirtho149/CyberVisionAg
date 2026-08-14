"""Generate a visual-symptom KB by describing training images with an LLM.

For each disease class, sends training images to Claude and asks it to
describe the visual symptoms. Output is a markdown file that can be passed to
eval.py via --kb-file.

Usage:
    python -m CyberVisionAg.open_agentic.generate_visual_kb \
        --dataset Soybean_Diseases --exclude "Class1,Class2"

This is a standalone script — it does NOT modify any existing pipeline code.
"""

import argparse
import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CYBERVISION_DIR = SCRIPT_DIR.parent
TRAIN_DIR = CYBERVISION_DIR / "Curated_Local_Dataset" / "train"
OUTPUT_DIR = SCRIPT_DIR / "visual_kbs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MODEL = "claude-sonnet-4-6"

DESCRIBE_PROMPT = """\
You are an expert plant pathologist. You are shown multiple training images of \
a single plant disease class: "{disease_name}".

Describe the VISUAL symptoms you observe across these images. Focus on:
- Leaf/stem/fruit appearance (color changes, spots, lesions, patterns)
- Shape, size, and distribution of symptoms
- Texture changes (wilting, curling, powdery coatings, etc.)
- Any distinguishing features that differentiate this from similar diseases

Be specific and concrete — describe what you SEE, not general textbook knowledge. \
If images show different stages or severities, note that.

Respond with a concise paragraph (3-6 sentences) of visual symptom description only. \
No preamble, no headers, no disease name repetition.\
"""


def _inline_image(path: str, max_bytes: int = 1_000_000) -> dict:
    """Encode image as base64 content block for Anthropic API."""
    raw = Path(path).read_bytes()
    if len(raw) > max_bytes:
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(raw))
            max_dim = 1024
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            raw = buf.getvalue()
        except ImportError:
            pass
    data = base64.standard_b64encode(raw).decode("ascii")
    ext = Path(path).suffix.lower()
    media_type = {
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def get_training_images(dataset: str, cls: str, max_images: int = 8) -> list[str]:
    """Get training image paths for a class."""
    cls_dir = TRAIN_DIR / dataset / cls
    if not cls_dir.exists():
        return []
    imgs = sorted(
        str(p) for p in cls_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if len(imgs) <= max_images:
        return imgs
    # Evenly spaced selection
    return [imgs[int(i * len(imgs) / max_images)] for i in range(max_images)]


def describe_class(client: anthropic.Anthropic, disease_name: str,
                   image_paths: list[str]) -> str:
    """Send training images to Claude and get visual symptom description."""
    content = []
    for path in image_paths:
        content.append(_inline_image(path))

    content.append({
        "type": "text",
        "text": DESCRIBE_PROMPT.format(disease_name=disease_name),
    })

    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip() if resp.content else ""


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual-symptom KB from training images"
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. Soybean_Diseases)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated class names to exclude")
    parser.add_argument("--max-images", type=int, default=8,
                        help="Max training images per class (default: 8)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Parallel API calls (default: 4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: visual_kbs/{dataset}.md)")
    args = parser.parse_args()

    exclude = set(args.exclude.split(",")) if args.exclude else set()

    # Discover classes
    dataset_dir = TRAIN_DIR / args.dataset
    if not dataset_dir.exists():
        sys.exit(f"ERROR: {dataset_dir} not found")

    classes = sorted([
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name not in exclude
    ])

    print(f"VISUAL KB GENERATION")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Classes    : {len(classes)}")
    print(f"  Max images : {args.max_images}")
    print(f"  Excluded   : {len(exclude)}")
    print()

    client = anthropic.Anthropic()
    descriptions = {}
    total = len(classes)

    def _process(cls):
        images = get_training_images(args.dataset, cls, args.max_images)
        if not images:
            print(f"  [{cls}] No images found, skipping")
            return cls, None
        t0 = time.time()
        desc = describe_class(client, cls.replace("_", " "), images)
        elapsed = time.time() - t0
        print(f"  [{cls}] {len(images)} images → {len(desc)} chars ({elapsed:.1f}s)")
        return cls, desc

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_process, cls): cls for cls in classes}
        for future in as_completed(futures):
            cls, desc = future.result()
            if desc:
                descriptions[cls] = desc

    # Build markdown KB (same format as --kb-file expects)
    lines = []
    for cls in classes:
        if cls in descriptions:
            lines.append(f"### {cls}\n{descriptions[cls]}\n")

    kb_text = "\n".join(lines)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.dataset}.md"
    out_path.write_text(kb_text)

    print(f"\n  Described  : {len(descriptions)}/{total} classes")
    print(f"  Saved      : {out_path}")


if __name__ == "__main__":
    main()
