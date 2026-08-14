import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================
SOURCE_DIR = "/home/user/Desktop/Disease Images for Arti"
OUTPUT_DIR = "dataset_analysis_output"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

TOP_K = 15  # number of diseases to visualize

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "Times New Roman"

# =====================================================
# STEP 1: COUNT IMAGES (Crop → Disease)
# =====================================================
def count_images(root_dir):
    counts = defaultdict(lambda: defaultdict(int))

    for crop in os.listdir(root_dir):
        crop_path = os.path.join(root_dir, crop)
        if not os.path.isdir(crop_path):
            continue

        for disease in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease)
            if not os.path.isdir(disease_path):
                continue

            for root, _, files in os.walk(disease_path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                        counts[crop][disease] += 1
    return counts

counts = count_images(SOURCE_DIR)

# =====================================================
# STEP 2: DATAFRAME
# =====================================================
records = []
for crop, diseases in counts.items():
    for disease, count in diseases.items():
        records.append([crop, disease, count])

df = pd.DataFrame(records, columns=["Crop", "Disease", "Image_Count"])
df = df.sort_values("Image_Count", ascending=False)

csv_path = os.path.join(OUTPUT_DIR, "crop_disease_image_counts.csv")
df.to_csv(csv_path, index=False)

# =====================================================
# STEP 3: TOP-K DISEASE SELECTION
# =====================================================
top_diseases = (
    df.groupby("Disease")["Image_Count"]
      .sum()
      .sort_values(ascending=False)
      .head(TOP_K)
      .index
)

df_top = df[df["Disease"].isin(top_diseases)]

pivot = (
    df_top.pivot(index="Crop", columns="Disease", values="Image_Count")
          .fillna(0)
)

pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

# =====================================================
# FIGURE 1: LOG-SCALED HEATMAP (TOP-K)
# =====================================================
fig, ax = plt.subplots(figsize=(16, 6))

im = ax.imshow(np.log1p(pivot.values), aspect="auto")

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))

ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=11)
ax.set_yticklabels(pivot.index, fontsize=11)

ax.set_title(
    "Top-15 Diseases Across Crops (Log-Scaled Image Counts)",
    fontsize=15,
    pad=15
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("log(1 + Image Count)", fontsize=11)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "heatmap_top15_log.png"),
    dpi=300
)
plt.close()

# =====================================================
# FIGURE 2: HORIZONTAL STACKED BAR (BEST OVERVIEW)
# =====================================================
fig, ax = plt.subplots(figsize=(14, 7))

left = np.zeros(len(pivot))

for disease in pivot.columns:
    ax.barh(
        pivot.index,
        pivot[disease],
        left=left,
        label=disease
    )
    left += pivot[disease].values

ax.set_xlabel("Number of Images", fontsize=12)
ax.set_ylabel("Crop", fontsize=12)
ax.set_title(
    "Crop-wise Distribution of Top-15 Diseases",
    fontsize=15,
    pad=15
)

ax.legend(
    title="Disease",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "stacked_bar_horizontal_top15.png"),
    dpi=300
)
plt.close()

# =====================================================
# FIGURE 3: GROUPED BAR CHART (VERY READABLE)
# =====================================================
fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(top_diseases))
bar_width = 0.8 / len(pivot.index)

for i, crop in enumerate(pivot.index):
    ax.bar(
        x + i * bar_width,
        pivot.loc[crop],
        width=bar_width,
        label=crop
    )

ax.set_xticks(x + bar_width * (len(pivot.index) - 1) / 2)
ax.set_xticklabels(top_diseases, rotation=35, ha="right")

ax.set_ylabel("Number of Images")
ax.set_title(
    "Top-15 Diseases: Image Counts Across Crops",
    fontsize=15,
    pad=15
)

ax.legend(
    title="Crop",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "grouped_bar_top15_diseases.png"),
    dpi=300
)
plt.close()

# =====================================================
# PRINT SUMMARY
# =====================================================
print("\n===== DATASET SUMMARY (TOP CLASSES) =====")
print(df_top.to_string(index=False))

print("\nSaved outputs in:", OUTPUT_DIR)
print(" - crop_disease_image_counts.csv")
print(" - heatmap_top15_log.png")
print(" - stacked_bar_horizontal_top15.png")
print(" - grouped_bar_top15_diseases.png")
