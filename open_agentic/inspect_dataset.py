"""Generate a big grid image of all test images with KB descriptions for visual inspection.

Usage:
    python -m CyberVisionAg.open_agentic.inspect_dataset --dataset Soybean_Diseases --split test
    python -m CyberVisionAg.open_agentic.inspect_dataset --dataset Corn_Diseases --split train
"""

import argparse
import textwrap
from pathlib import Path

import openpyxl
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
CYBERVISION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CYBERVISION_DIR.parent
DATASET_ROOT = CYBERVISION_DIR / "Curated_Local_Dataset"
REGISTRY_OUTPUTS = CYBERVISION_DIR / "disease_registry" / "outputs"

EXCLUDE = {
    ".DS_Store", "Diaporthe_2015_Kanawha", "Green_stem",
    "Fusarium_healthy_vs_infected", "Stem_Canker", "Top_Dieback",
}


def load_kb(dataset: str) -> dict[str, str]:
    """Load KB visual descriptions from xlsx."""
    crop = dataset.replace("_Diseases", "").replace("_Disease", "")
    path = REGISTRY_OUTPUTS / crop / "internet.xlsx"
    kb = {}
    if not path.exists():
        return kb
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        disease = str(row[0]).strip() if row[0] else ""
        visual = str(row[4]).strip() if len(row) > 4 and row[4] else ""
        parts = str(row[3]).strip() if len(row) > 3 and row[3] else ""
        if disease:
            entry = f"Parts: {parts}\n{visual}" if parts else visual
            kb[disease] = entry if entry.strip() else "No KB entry"
    wb.close()
    return kb


def make_grid(dataset: str, split: str, output: str | None = None):
    data_dir = DATASET_ROOT / split / dataset
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return

    classes = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name not in EXCLUDE
    )

    kb = load_kb(dataset)

    # Count max images per class
    max_imgs = 0
    for cls in classes:
        imgs = [p for p in (data_dir / cls).glob("*.*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        max_imgs = max(max_imgs, len(imgs))

    # Layout
    thumb_size = 250
    text_width = 500
    label_height = 35
    row_height = thumb_size + label_height
    cols = max_imgs
    canvas_w = text_width + cols * thumb_size
    canvas_h = len(classes) * row_height

    print(f"Dataset: {dataset} ({split})")
    print(f"Classes: {len(classes)}, Max images/class: {max_imgs}")
    print(f"Canvas: {canvas_w}x{canvas_h}px")

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_kb = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font_label = ImageFont.load_default()
        font_kb = ImageFont.load_default()

    for row_idx, cls in enumerate(classes):
        y_top = row_idx * row_height

        # Class label
        label = cls.replace("_", " ")
        draw.text((5, y_top + 2), label, fill="black", font=font_label)

        # KB text
        kb_text = kb.get(cls, "No KB entry")
        wrapped = textwrap.fill(kb_text, width=55)
        lines = wrapped.split("\n")[:12]
        for i, line in enumerate(lines):
            draw.text((5, y_top + label_height + i * 16), line, fill="gray", font=font_kb)

        # Separators
        draw.line([(text_width - 5, y_top), (text_width - 5, y_top + row_height)], fill="lightgray", width=1)
        draw.line([(0, y_top + row_height - 1), (canvas_w, y_top + row_height - 1)], fill="lightgray", width=1)

        # Images
        imgs = sorted(
            p for p in (data_dir / cls).glob("*.*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )[:cols]

        for col_idx, img_path in enumerate(imgs):
            img = Image.open(img_path)
            img.thumbnail((thumb_size, thumb_size))
            x = text_width + col_idx * thumb_size
            y = y_top + label_height
            x_off = (thumb_size - img.width) // 2
            y_off = (thumb_size - img.height) // 2
            canvas.paste(img, (x + x_off, y + y_off))

    if output is None:
        output = str(SCRIPT_DIR / f"inspect_{dataset}_{split}.jpg")

    canvas.save(output, quality=90)
    size_mb = Path(output).stat().st_size / 1024 / 1024
    print(f"Saved: {output} ({size_mb:.1f} MB)")


def crop_row(grid_path: str, class_name: str, dataset: str, split: str):
    """Crop a single class row from the grid image."""
    data_dir = DATASET_ROOT / split / dataset
    classes = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name not in EXCLUDE
    )
    if class_name not in classes:
        print(f"ERROR: {class_name} not found. Available: {classes}")
        return
    row_idx = classes.index(class_name)
    row_height = 250 + 35  # thumb_size + label_height
    img = Image.open(grid_path)
    row = img.crop((0, row_idx * row_height, img.width, (row_idx + 1) * row_height))
    out = str(Path(grid_path).parent / f"inspect_{class_name}.jpg")
    row.save(out, quality=95)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset inspection grid")
    parser.add_argument("--dataset", default="Soybean_Diseases")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--crop-class", default=None, help="Crop single class row from existing grid")
    args = parser.parse_args()

    if args.crop_class:
        grid = args.output or str(SCRIPT_DIR / f"inspect_{args.dataset}_{args.split}.jpg")
        crop_row(grid, args.crop_class, args.dataset, args.split)
    else:
        make_grid(args.dataset, args.split, args.output)


if __name__ == "__main__":
    main()
