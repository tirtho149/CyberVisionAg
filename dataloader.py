"""
Local Plant Disease Data Curation Script
=========================================
Curates train (5) / test (10) splits from your local disease image directories.

SOURCE datasets used (only classes with >= 15 images are included):
  - Alfalfa Diseases/
  - Corn Diseases/
  - Soybean Diseases/
  - Wheat Diseases/
  - mango-leaf-disease-dataset/   → saved as Foliar_Disease_Stress category

EXCLUDED:
  - yellowrust19 / Disease Severity   (completely removed)
  - Healthy                           (non-disease class)
  - .venv/                            (Python env, not disease data)
  - AgVisionReason-reasoning/         (already curated output)
  - Rye Diseases/                     (all classes < 15 images)
  - dataset_analysis_output/          (not disease data)

OUTPUT STRUCTURE:
  Curated_Local_Dataset/
    train/
      Alfalfa_Diseases/<disease>/         (5 images)
      Corn_Diseases/<disease>/            (5 images)
      Soybean_Diseases/<disease>/         (5 images)
      Wheat_Diseases/<disease>/           (5 images)
      Mango_Leaf_Disease/<disease>/       (5 images)  ← mango leaf
    test/
      (same structure, 10 images each)
    metadata.json
    DATASET_SUMMARY.txt

REQUIREMENTS:
    pip install tqdm
"""

import os
import json
import shutil
import random
from tqdm import tqdm

# ============================================================================
# ★ CONFIGURE THESE PATHS ★
# ============================================================================

# Root of your local data (where Alfalfa Diseases/, Corn Diseases/ etc. live)
SOURCE_ROOT = "/home/user/Desktop/Disease Images for Arti"   # <-- SET YOUR PATH

# Where to write the curated output
OUTPUT_DIR  = "./Curated_Local_Dataset"

# Split sizes
TRAIN_PER_CLASS = 5
TEST_PER_CLASS  = 10
MIN_IMAGES      = TRAIN_PER_CLASS + TEST_PER_CLASS   # 15

RANDOM_STATE    = 42
IMAGE_EXT       = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# ============================================================================
# CLASS MAP
# Maps  output_category → { output_class_name: absolute_source_folder }
# Only classes with >= 15 images are listed.
# Disease Severity (Yellow Rust) is completely removed.
# ============================================================================

def build_class_map(root):
    R = root  # shorthand

    class_map = {

        # ------------------------------------------------------------------
        # Alfalfa Diseases  (only Verticillium wilt has >= 15 images)
        # ------------------------------------------------------------------
        "Alfalfa_Diseases": {
            "Verticillium_wilt":
                os.path.join(R, "Alfalfa Diseases", "Verticillium wilt"),
        },

        # ------------------------------------------------------------------
        # Corn Diseases  — all classes with >= 15 images
        # ------------------------------------------------------------------
        "Corn_Diseases": {
            "Anthracnose_leaf_spot_top_dieback":
                os.path.join(R, "Corn Diseases", "Anthracnose leaf spot and top dieback"),
            "Anthracnose_stalk_rot":
                os.path.join(R, "Corn Diseases", "Anthracnose stalk rot"),
            "Bacterial_stalk_rot":
                os.path.join(R, "Corn Diseases", "Bacterial stalk rot"),
            "Carbonum_leaf_spot":
                os.path.join(R, "Corn Diseases", "Carbonum leaf spot"),
            "Charcoal_stalk_rot":
                os.path.join(R, "Corn Diseases", "Charcoal stalk rot"),
            "Common_rust":
                os.path.join(R, "Corn Diseases", "Common rust"),
            "Common_smut":
                os.path.join(R, "Corn Diseases", "Common smut"),
            "Corn_Stunt_Spiroplasma":
                os.path.join(R, "Corn Diseases", "Corn Stunt Spiroplasma"),
            "Crazy_top":
                os.path.join(R, "Corn Diseases", "Crazy top"),
            "Diplodia_stalk_rot":
                os.path.join(R, "Corn Diseases", "Diplodia stalk rot"),
            "Downy_mildew":
                os.path.join(R, "Corn Diseases", "Downy mildew"),
            "Ear_rots_Aspergillus":
                os.path.join(R, "Corn Diseases", "Ear rots", "Aspergillus ear rot"),
            "Ear_rots_Diplodia":
                os.path.join(R, "Corn Diseases", "Ear rots", "Diplodia ear rot"),
            "Ear_rots_General_Mixed":
                os.path.join(R, "Corn Diseases", "Ear rots", "General and Mixed Ear Rots"),
            "Ear_rots_Gibberella":
                os.path.join(R, "Corn Diseases", "Ear rots", "Gibberella ear rot"),
            "Ear_rots_Trichoderma":
                os.path.join(R, "Corn Diseases", "Ear rots", "Trichoderma ear rot"),
            "Eyespot":
                os.path.join(R, "Corn Diseases", "Eyespot"),
            "Fusarium":
                os.path.join(R, "Corn Diseases", "Fusarium"),
            "General_Mixed_Stalk_Rots":
                os.path.join(R, "Corn Diseases", "General and Mixed Stalk Rots"),
            "Genetic_flecking_striping":
                os.path.join(R, "Corn Diseases", "Genetic flecking or striping"),
            "Goss_wilt":
                os.path.join(R, "Corn Diseases", "Goss's wilt"),
            "Gray_leaf_spot":
                os.path.join(R, "Corn Diseases", "Gray leaf spot"),
            "Head_smut":
                os.path.join(R, "Corn Diseases", "Head smut"),
            "Head_smut_South_Africa":
                os.path.join(R, "Corn Diseases", "Head smut", "Head smut - South Africa 2013 - Daren Mueller"),
            "Holcus_spot":
                os.path.join(R, "Corn Diseases", "Holcus spot"),
            "Maize_chlorotic_dwarf_virus":
                os.path.join(R, "Corn Diseases", "Maize chlorotic dwarf virus"),
            "Maize_dwarf_mosaic_virus":
                os.path.join(R, "Corn Diseases", "Maize dwarf mosaic virus"),
            "Maize_streak_virus":
                os.path.join(R, "Corn Diseases", "Maize streak virus - South Africa 2013 - Daren Mueller"),
            "Misc":
                os.path.join(R, "Corn Diseases", "Misc"),
            "Northern_corn_leaf_blight":
                os.path.join(R, "Corn Diseases", "Northern corn leaf blight"),
            "Physoderma_brown_spot":
                os.path.join(R, "Corn Diseases", "Physoderma brown spot"),
            "Physoderma_stalk_rot":
                os.path.join(R, "Corn Diseases", "Physoderma stalk rot"),
            "Southern_Corn_Leaf_Blight":
                os.path.join(R, "Corn Diseases", "Southern Corn Leaf Blight"),
            "Southern_rust":
                os.path.join(R, "Corn Diseases", "Southern rust"),
            "Stewarts_disease":
                os.path.join(R, "Corn Diseases", "Stewart's disease"),
            "Stewarts_wilt":
                os.path.join(R, "Corn Diseases", "Stewart's disease", "Stewart's wilt"),
            "Tar_spot":
                os.path.join(R, "Corn Diseases", "Tar spot"),
            "Multiple_foliar_diseases":
                os.path.join(R, "Corn Diseases", "multiple foliar diseases at once"),
        },

        # ------------------------------------------------------------------
        # Soybean Diseases  — all classes with >= 15 images
        # ------------------------------------------------------------------
        "Soybean_Diseases": {
            "Anthracnose":
                os.path.join(R, "Soybean Diseases", "Anthracnose"),
            "Bacterial_Blight":
                os.path.join(R, "Soybean Diseases", "Bacterial Blight"),
            "Bacterial_Pustule":
                os.path.join(R, "Soybean Diseases", "Bacterial Pustule"),
            "Bean_Pod_Mottle_virus":
                os.path.join(R, "Soybean Diseases", "Bean Pod Mottle virus"),
            "Brown_Stem_Rot":
                os.path.join(R, "Soybean Diseases", "Brown Stem Rot"),
            "Cercospora":
                os.path.join(R, "Soybean Diseases", "Cercospora"),
            "Charcoal_Rot":
                os.path.join(R, "Soybean Diseases", "Charcoal Rot"),
            "Diaporthe":
                os.path.join(R, "Soybean Diseases", "Diaporthe"),
            "Diaporthe_2015_Kanawha":
                os.path.join(R, "Soybean Diseases", "Diaporthe", "2015 Kanawha field"),
            "Downy_mildew":
                os.path.join(R, "Soybean Diseases", "Downy mildew"),
            "Frogeye_leaf_spot":
                os.path.join(R, "Soybean Diseases", "Frogeye leaf spot"),
            "Fusarium":
                os.path.join(R, "Soybean Diseases", "Fusarium"),
            "Fusarium_healthy_vs_infected":
                os.path.join(R, "Soybean Diseases", "Fusarium",
                             "fusarium maybe healthy vs infected a sisson"),
            "Green_stem":
                os.path.join(R, "Soybean Diseases", "Green stem"),
            "Green_stem_disorder":
                os.path.join(R, "Soybean Diseases", "Green stem", "green stem disorder A. Sisson"),
            "Phomopsis":
                os.path.join(R, "Soybean Diseases", "Phomopsis"),
            "Phyllosticta_leaf_spot":
                os.path.join(R, "Soybean Diseases", "Phyllosticta leaf spot"),
            "Phytophthora":
                os.path.join(R, "Soybean Diseases", "Phytophthora"),
            "Powdery_Mildew":
                os.path.join(R, "Soybean Diseases", "Powdery Mildew"),
            "Purple_Seed_Stain":
                os.path.join(R, "Soybean Diseases", "Purple Seed Stain"),
            "Pythium_damping_off":
                os.path.join(R, "Soybean Diseases", "Damping off",
                             "Pythium damping off and unknown insect injury to soy - Tristan Mueller",
                             "Pythium - Tristan Mueller"),
            "Rhizoctonia":
                os.path.join(R, "Soybean Diseases", "Rhizoctonia"),
            "Septoria_brown_spot":
                os.path.join(R, "Soybean Diseases", "Septoria brown spot"),
            "Soybean_Cyst_Nematode":
                os.path.join(R, "Soybean Diseases", "Soybean Cyst Nematode"),
            "Soybean_Dwarf_Mosaic_Virus":
                os.path.join(R, "Soybean Diseases", "Soybean Dwarf Mosaic Virus 2012"),
            "Soybean_Vein_necrosis_virus":
                os.path.join(R, "Soybean Diseases", "Soybean Vein necrosis virus"),
            "Soybean_rust":
                os.path.join(R, "Soybean Diseases", "Soybean rust"),
            "Stem_Canker":
                os.path.join(R, "Soybean Diseases", "Stem Canker"),
            "Sudden_death_syndrome":
                os.path.join(R, "Soybean Diseases", "Sudden death syndrome"),
            "Tobacco_Streak_Virus":
                os.path.join(R, "Soybean Diseases", "Tobacco Streak Virus"),
            "Top_Dieback":
                os.path.join(R, "Soybean Diseases", "Top Dieback"),
            "White_Mold":
                os.path.join(R, "Soybean Diseases", "White Mold"),
        },

        # ------------------------------------------------------------------
        # Wheat Diseases  (only rust qualifies)
        # ------------------------------------------------------------------
        "Wheat_Diseases": {
            "Rust":
                os.path.join(R, "Wheat Diseases", "rust"),
        },

        # ------------------------------------------------------------------
        # Mango Leaf Disease Dataset  →  Mango_Leaf_Disease
        # Healthy class is excluded.
        # ------------------------------------------------------------------
        "Mango_Leaf_Disease": {
            "Anthracnose":
                os.path.join(R, "mango-leaf-disease-dataset", "Anthracnose"),
            "Bacterial_Canker":
                os.path.join(R, "mango-leaf-disease-dataset", "Bacterial Canker"),
            "Cutting_Weevil":
                os.path.join(R, "mango-leaf-disease-dataset", "Cutting Weevil"),
            "Die_Back":
                os.path.join(R, "mango-leaf-disease-dataset", "Die Back"),
            "Gall_Midge":
                os.path.join(R, "mango-leaf-disease-dataset", "Gall Midge"),
            "Powdery_Mildew":
                os.path.join(R, "mango-leaf-disease-dataset", "Powdery Mildew"),
            "Sooty_Mould":
                os.path.join(R, "mango-leaf-disease-dataset", "Sooty Mould"),
        },

        # NOTE: Disease_Severity (Yellow Rust) has been completely removed.
        # NOTE: Mango Leaf Disease is stored under "Mango_Leaf_Disease" category.
    }

    return class_map


# ============================================================================
# UTILITIES
# ============================================================================

def get_images(folder_path):
    """Return all image paths directly inside a folder (not recursive)."""
    if not os.path.isdir(folder_path):
        return []
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXT)
        and os.path.isfile(os.path.join(folder_path, f))
    ]


def copy_images(paths, dest_dir, label, split):
    """Copy list of image paths into dest_dir with clean filenames."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for idx, src in enumerate(tqdm(paths, desc=f"    [{split}] {label[:35]}", leave=False)):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(dest_dir, f"{label}_{split}_{idx+1:03d}{ext}")
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"    ✗ Error copying {src}: {e}")
    return copied


def split_and_copy(image_paths, class_name, train_dir, test_dir, n_train, n_test):
    """
    Sample (n_train + n_test) images without overlap.
    Copy n_train → train_dir, n_test → test_dir.
    Returns (train_count, test_count, skipped:bool).
    """
    total = n_train + n_test
    if len(image_paths) < total:
        return 0, 0, True   # not enough images → skip

    random.seed(RANDOM_STATE)
    sampled = random.sample(image_paths, total)
    t_cnt = copy_images(sampled[:n_train],             train_dir, class_name, "train")
    e_cnt = copy_images(sampled[n_train:n_train+n_test], test_dir, class_name, "test")
    return t_cnt, e_cnt, False


# ============================================================================
# MAIN CURATION FUNCTION
# ============================================================================

def curate(source_root, output_dir, n_train=TRAIN_PER_CLASS, n_test=TEST_PER_CLASS):

    print(f"\n{'='*70}")
    print("LOCAL PLANT DISEASE CURATION")
    print(f"{'='*70}")
    print(f"  Source : {os.path.abspath(source_root)}")
    print(f"  Output : {os.path.abspath(output_dir)}")
    print(f"  Train  : {n_train} images/class")
    print(f"  Test   : {n_test}  images/class")
    print(f"  Min    : {n_train+n_test} images needed to include a class")
    print(f"  Note   : Disease Severity (Yellow Rust) excluded entirely")
    print(f"{'='*70}")

    class_map = build_class_map(source_root)

    metadata = {}
    included = []
    skipped  = []

    for category, classes in class_map.items():

        print(f"\n{'─'*70}")
        print(f"  CATEGORY: {category}  ({len(classes)} classes defined)")
        print(f"{'─'*70}")

        metadata[category] = []

        for class_name, src_folder in classes.items():

            images = get_images(src_folder)

            if len(images) < (n_train + n_test):
                print(f"  ⚠  SKIPPED  {class_name:<45} "
                      f"only {len(images)} images (need {n_train+n_test})")
                skipped.append(f"{category}/{class_name}  ({len(images)} imgs)")
                continue

            train_dir = os.path.join(output_dir, "train", category, class_name)
            test_dir  = os.path.join(output_dir, "test",  category, class_name)

            t_cnt, e_cnt, skip = split_and_copy(
                images, class_name, train_dir, test_dir, n_train, n_test
            )

            print(f"  ✓  {class_name:<45}  "
                  f"train: {t_cnt}  test: {e_cnt}  (from {len(images)} total)")

            included.append(f"{category}/{class_name}")
            metadata[category].append({
                "class":       class_name,
                "source":      src_folder,
                "total_avail": len(images),
                "train":       t_cnt,
                "test":        e_cnt,
            })

    # -----------------------------------------------------------------------
    # Save metadata.json
    # -----------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "source_root":     os.path.abspath(source_root),
            "output_dir":      os.path.abspath(output_dir),
            "train_per_class": n_train,
            "test_per_class":  n_test,
            "total_included":  len(included),
            "total_skipped":   len(skipped),
            "note":            "Disease_Severity (Yellow Rust) excluded entirely",
            "categories":      metadata,
        }, f, indent=2)

    # -----------------------------------------------------------------------
    # Save DATASET_SUMMARY.txt
    # -----------------------------------------------------------------------
    summary_path = os.path.join(output_dir, "DATASET_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("LOCAL PLANT DISEASE DATASET — CURATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Train per class  : {n_train}\n")
        f.write(f"Test  per class  : {n_test}\n")
        f.write(f"Classes included : {len(included)}\n")
        f.write(f"Classes skipped  : {len(skipped)}\n")
        f.write(f"Total train imgs : {len(included) * n_train}\n")
        f.write(f"Total test  imgs : {len(included) * n_test}\n")
        f.write(f"Note             : Disease_Severity (Yellow Rust) excluded entirely\n\n")

        for category, entries in metadata.items():
            if not entries:
                continue
            f.write("─"*70 + "\n")
            f.write(f"{category}  ({len(entries)} classes)\n")
            f.write("─"*70 + "\n")
            for e in entries:
                f.write(f"  {e['class']:<45}  train:{e['train']}  test:{e['test']}\n")
            f.write("\n")

        if skipped:
            f.write("─"*70 + "\n")
            f.write("SKIPPED (< 15 images)\n")
            f.write("─"*70 + "\n")
            for s in skipped:
                f.write(f"  {s}\n")

    # -----------------------------------------------------------------------
    # Final console summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CURATION COMPLETE!")
    print(f"{'='*70}")
    print(f"  ✓ Classes included : {len(included)}")
    print(f"  ⚠ Classes skipped  : {len(skipped)}  (< {n_train+n_test} images)")
    print(f"  ✓ Total train imgs : {len(included) * n_train}")
    print(f"  ✓ Total test  imgs : {len(included) * n_test}")
    print(f"  ✓ Output           : {os.path.abspath(output_dir)}")
    print(f"  ✓ metadata.json    : {meta_path}")
    print(f"  ✓ DATASET_SUMMARY  : {summary_path}")
    print(f"\n  Output structure:")
    print(f"    {output_dir}/")
    print(f"      train/")
    for cat in class_map:
        n = len(metadata.get(cat, []))
        if n:
            print(f"        {cat}/  ({n} class folders × {n_train} images)")
    print(f"      test/")
    for cat in class_map:
        n = len(metadata.get(cat, []))
        if n:
            print(f"        {cat}/  ({n} class folders × {n_test} images)")
    print(f"      metadata.json")
    print(f"      DATASET_SUMMARY.txt")

    return metadata


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize(metadata, output_dir, n_train, n_test):
    """
    Generate a professional multi-panel visualization of the curated dataset.
    Saves as:  <output_dir>/dataset_visualization.png

    Panels:
      1. Horizontal bar chart  — images per class, colored by category,
                                 stacked train vs test bars
      2. Pie chart             — class distribution across categories
      3. Summary stats table   — total classes, images, train/test counts
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        import numpy as np
    except ImportError:
        print("\n⚠  matplotlib not installed. Run:  pip install matplotlib")
        print("   Skipping visualization.")
        return

    # ── palette — one colour per category ──────────────────────────────────
    PALETTE = {
        "Alfalfa_Diseases":   "#4CAF50",
        "Corn_Diseases":      "#FF9800",
        "Soybean_Diseases":   "#2196F3",
        "Wheat_Diseases":     "#9C27B0",
        "Mango_Leaf_Disease": "#F44336",
    }
    DEFAULT_COLOR = "#607D8B"

    # ── flatten metadata into lists ─────────────────────────────────────────
    labels, trains, tests, colors, cat_labels = [], [], [], [], []
    cat_counts = {}

    for category, entries in metadata.items():
        if not entries:
            continue
        color = PALETTE.get(category, DEFAULT_COLOR)
        cat_counts[category] = len(entries)
        for e in entries:
            labels.append(e["class"].replace("_", " "))
            trains.append(e["train"])
            tests.append(e["test"])
            colors.append(color)
            cat_labels.append(category.replace("_", " "))

    n_classes  = len(labels)
    total_imgs = sum(trains) + sum(tests)

    # ── figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, max(14, n_classes * 0.38 + 6)),
                     facecolor="#0F1117")
    gs  = GridSpec(2, 2, figure=fig,
                   left=0.22, right=0.97,
                   top=0.92,  bottom=0.05,
                   hspace=0.45, wspace=0.35)

    title_kw  = dict(color="#FFFFFF", fontsize=11, fontweight="bold", pad=10)
    label_kw  = dict(color="#CCCCCC", fontsize=8.5)
    tick_kw   = dict(colors="#AAAAAA", labelsize=7.5)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1 — Stacked horizontal bar: train + test per class
    # ════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[:, 0])   # full left column
    ax1.set_facecolor("#1A1D27")

    y   = np.arange(n_classes)
    bar_h = 0.62

    # test bars (right portion, lighter shade)
    test_bars = ax1.barh(y, tests,  bar_h,
                         color=[c + "99" for c in colors],
                         label="Test")
    # train bars (left portion, full colour)
    train_bars = ax1.barh(y, trains, bar_h,
                          left=tests,
                          color=colors,
                          label="Train")

    # value labels inside bars
    for i, (tr, te) in enumerate(zip(trains, tests)):
        total = tr + te
        if total > 0:
            ax1.text(total + 0.3, i, str(total),
                     va="center", ha="left",
                     color="#DDDDDD", fontsize=6.5)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7, color="#CCCCCC")
    ax1.set_xlabel("Number of Images", **label_kw)
    ax1.set_title("Images per Disease Class  (Train + Test)", **title_kw)
    ax1.tick_params(axis="x", **tick_kw)
    ax1.tick_params(axis="y", colors="#AAAAAA", labelsize=7)
    ax1.spines[:].set_color("#333344")
    ax1.set_xlim(0, max(trains[i] + tests[i] for i in range(n_classes)) * 1.18)

    # category colour legend
    cat_patches = [
        mpatches.Patch(color=PALETTE.get(c, DEFAULT_COLOR),
                       label=c.replace("_", " "))
        for c in cat_counts
    ]
    split_patches = [
        mpatches.Patch(color="#888888",       label=f"Train  (n={n_train})"),
        mpatches.Patch(color="#88888855",     label=f"Test   (n={n_test})"),
    ]
    ax1.legend(handles=cat_patches + split_patches,
               loc="lower right", fontsize=7,
               facecolor="#1A1D27", edgecolor="#444455",
               labelcolor="#CCCCCC", framealpha=0.9)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 2 — Pie: class distribution across categories
    # ════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1A1D27")

    pie_sizes   = list(cat_counts.values())
    pie_labels  = [f"{k.replace('_',' ')}\n({v} classes)"
                   for k, v in cat_counts.items()]
    pie_colors  = [PALETTE.get(k, DEFAULT_COLOR) for k in cat_counts]

    wedges, texts, autotexts = ax2.pie(
        pie_sizes,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=1.5, edgecolor="#0F1117"),
        textprops=dict(color="#CCCCCC", fontsize=7.5),
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("#FFFFFF")
        at.set_fontweight("bold")

    ax2.set_title("Class Distribution by Category", **title_kw)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 3 — Bar: total images available per category (source data)
    # ════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1A1D27")

    cat_names  = list(cat_counts.keys())
    cat_totals = [
        sum(e["total_avail"] for e in metadata.get(c, []))
        for c in cat_names
    ]
    cat_curated = [
        sum(e["train"] + e["test"] for e in metadata.get(c, []))
        for c in cat_names
    ]
    bar_x   = np.arange(len(cat_names))
    bar_w   = 0.38
    c_list  = [PALETTE.get(c, DEFAULT_COLOR) for c in cat_names]

    b1 = ax3.bar(bar_x - bar_w/2, cat_totals,  bar_w,
                 color=[c + "66" for c in c_list],
                 label="Available in source")
    b2 = ax3.bar(bar_x + bar_w/2, cat_curated, bar_w,
                 color=c_list,
                 label="Curated (train+test)")

    ax3.set_xticks(bar_x)
    ax3.set_xticklabels([c.replace("_", "\n") for c in cat_names],
                        fontsize=6.5, color="#CCCCCC")
    ax3.set_ylabel("Image Count", **label_kw)
    ax3.set_title("Source Available vs Curated per Category", **title_kw)
    ax3.tick_params(axis="y", **tick_kw)
    ax3.spines[:].set_color("#333344")
    ax3.legend(fontsize=7, facecolor="#1A1D27",
               edgecolor="#444455", labelcolor="#CCCCCC")

    # value labels on top of bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + max(cat_totals)*0.01,
                 f"{h:,}", ha="center", va="bottom",
                 color="#BBBBBB", fontsize=6)

    # ════════════════════════════════════════════════════════════════════════
    # Super-title
    # ════════════════════════════════════════════════════════════════════════
    fig.text(0.5, 0.965,
             "Plant Disease Dataset — Toy Benchmark Curation",
             ha="center", va="top",
             color="#FFFFFF", fontsize=15, fontweight="bold",
             fontfamily="monospace")

    fig.text(0.5, 0.945,
             f"{n_classes} disease classes  ·  "
             f"{sum(trains)} train images  ·  "
             f"{sum(tests)} test images  ·  "
             f"{total_imgs} total",
             ha="center", va="top",
             color="#888899", fontsize=9)

    # ── save ────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dataset_visualization.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n  ✓ Visualization saved → {os.path.abspath(out_path)}")
    return out_path


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    metadata = curate(
        source_root=SOURCE_ROOT,
        output_dir=OUTPUT_DIR,
        n_train=TRAIN_PER_CLASS,
        n_test=TEST_PER_CLASS,
    )

    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*70}")
    visualize(metadata, OUTPUT_DIR, TRAIN_PER_CLASS, TEST_PER_CLASS)