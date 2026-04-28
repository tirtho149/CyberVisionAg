# SAGE Numbers Reference

---

## Dataset Statistics

### Soybean

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 27 |
| Disease classes (evaluated, with test images) | 25 |
| Raw images inspected | 538 |
| Reference images kept | 242 |
| Test images kept | 74 |
| Images rejected | 107 |
| Images skipped | 114 |
| Errors | 1 |
| Max images per part | 5 |
| max_inspect_per_class | 20 |

**Part distribution (part → number of classes with that part in ref set):**

| Part | Classes |
|------|---------|
| leaf | 16 |
| pod | 10 |
| root | 4 |
| seed | 5 |
| stem | 11 |
| whole_plant | 17 |

---

### Corn

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 42 |
| Disease classes (evaluated, with test images) | 30 |
| Raw images inspected | 665 |
| Reference images kept | 274 |
| Test images kept | 88 |
| Images rejected | 110 |
| Images skipped | 193 |
| Errors | 0 |
| Max images per part | 5 |
| max_inspect_per_class | 20 |

**Part distribution:**

| Part | Classes |
|------|---------|
| leaf | 19 |
| pod | 7 |
| seed | 8 |
| stem | 11 |
| whole_plant | 14 |

---

### Mango

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 5 |
| Disease classes (evaluated, with test images) | 4 |
| Raw images inspected | 250 |
| Reference images kept | 35 |
| Test images kept | 40 |
| Images rejected | 104 |
| Images skipped | 71 |
| Errors | 0 |
| Max images per part | 5 |
| max_inspect_per_class | 50 |

**Part distribution:**

| Part | Classes |
|------|---------|
| leaf | 4 |
| pod | 3 |
| seed | 1 |
| stem | 2 |
| whole_plant | 1 |

---

## KB Coverage

### Soybean

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/27 | all |
| local | 26/27 | Green_stem_disorder |
| internet | 27/27 | none |

### Corn

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/31 | all |
| local | not reported (no PDF source) | — |
| internet | 29/31 | Carbonum_Leaf_Spot, Maize_Streak_Virus |

### Mango

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/4 | all |
| local | not reported (no PDF source) | — |
| internet | 4/4 | none |

---

## Main Accuracy Table

Source: `results/open_agentic/*/summary.json` (seed=42, sonnet model, collage refs, 800 px tiles).

### Soybean — 25 classes × ~3 images = 74 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 41.89% | 41.89% | 40.54% | 47.30% | 48.65% |
| Agent (no KB) | 31.08% | 40.54% | 37.84% | 45.95% | 48.65% |
| Agent + local KB | not reported | not reported | not reported | not reported | not reported |
| Agent + internet KB | not reported† | 41.89% | 48.65% | 48.65% | 51.35% |

† internet/k=0 run is a 1-image smoke test only; full run not available.

*README "Final Results" table (written before runs completed) states different values:*
*Few-shot k=1/4/8/16: 29.6% / 32.1% / 37.0% / 35.8%*
*Agent+internet k=1/4/8/16: 38.3% / 42.0% / 37.0% / 38.3%*

---

### Corn — 30 classes × ~3 images = 88 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 42.05% | 38.64% | 48.86% | 50.00% | 47.73% |
| Agent (no KB) | 42.05% | 43.18% | 46.59% | 52.27% | 54.55% |
| Agent + internet KB | 51.14% | 52.27% | 60.23% | 60.23% | 64.77% |

*README "Final Results" table states:*
*Few-shot k=1/4/8/16: 32.3% / 33.3% / 40.9% / 44.1%*
*Agent+internet k=1/4/8/16: 41.9% / 48.4% / 53.8% / 49.5%*

---

### Mango — 4 classes × 10 images = 40 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 92.50% | 95.00% | 92.50% | 92.50% | 95.00% |
| Agent (no KB) | 92.50% | 85.00% | 92.50% | 92.50% | 95.00% |
| Agent + internet KB | 87.50% | 97.50% | 92.50% | 97.50% | 95.00% |

*README "Final Results" table states (7 classes × 3 images = 21 tests):*
*Few-shot k=1/4/8/16: 52.4% / 66.7% / 90.5% / 95.2%*
*Agent+internet k=1/4/8/16: 61.9% / 76.2% / 85.7% / 85.7%*

---

## Model Ablation

At internet KB, k=8.

### Soybean

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 31.08% (23/74) | $0.0950 |
| Sonnet | 48.65% (36/74) | $0.2286 |
| Opus | 62.16% (46/74) | $0.3455 |
| gemini-flash | not reported | — |
| gemini-pro | not reported | — |

*README states: Haiku — (needs re-run), Sonnet 37.0%, Opus — (needs re-run)*

### Corn

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 42.05% (37/88) | $0.1032 |
| Sonnet | 60.23% (53/88) | $0.2505 |
| Opus | 61.36% (54/88) | $0.3709 |
| gemini-flash | not reported | — |
| gemini-pro | not reported | — |

*README states: Haiku 24.7%, Sonnet 53.8%, Opus 61.3%*

### Mango

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 82.50% (33/40) | $0.0769 |
| Sonnet | 97.50% (39/40) | $0.1340 |
| Opus | 97.50% (39/40) | $0.2184 |
| gemini-flash | 92.50% (37/40) | $0.00 (billed externally) |
| gemini-pro | 95.00% (38/40) | $0.00 (billed externally) |

*README states: Haiku 42.9%, Sonnet 85.7%, Opus 85.7%*

---

## Cost Numbers

### Cost per image per model (internet KB, k=8, sonnet sweep)

| Model | Soybean | Corn | Mango |
|-------|---------|------|-------|
| Haiku | $0.0950 | $0.1032 | $0.0769 |
| Sonnet | $0.2286 | $0.2505 | $0.1340 |
| Opus | $0.3455 | $0.3709 | $0.2184 |

### Cost per image per k value — Agent (internet KB, sonnet)

| k | Soybean | Corn | Mango |
|---|---------|------|-------|
| 0 | not reported (smoke test) | $0.0652 | $0.0415 |
| 1 | $0.0913 | $0.1045 | $0.0547 |
| 4 | $0.1471 | $0.1718 | $0.0824 |
| 8 | $0.2286 | $0.2505 | $0.1340 |
| 16 | $0.3737 | $0.3932 | $0.2306 |

### Cost per image per k value — Few-shot (sonnet)

| k | Soybean | Corn | Mango |
|---|---------|------|-------|
| 0 | $0.0061 | $0.0068 | $0.0024 |
| 1 | $0.0093 | $0.0102 | $0.0028 |
| 4 | $0.0188 | $0.0202 | $0.0038 |
| 8 | $0.0311 | $0.0337 | $0.0052 |
| 16 | $0.0558 | $0.0605 | $0.0082 |

### Total sweep cost per crop (all configs in results/open_agentic/)

| Crop | Total cost |
|------|-----------|
| Soybean | $155.64 |
| Corn | $203.13 |
| Mango | $53.68 (Gemini configs billed externally, shown as $0) |

---

## Confusion Stats

*At internet KB, sonnet model, k=8 (from summary.json per_class_accuracy).*

### Soybean

**Top confusion pairs (from README confusion analysis, agent-only results across none/sonnet/k=8 + internet/sonnet/k=8, 162 predictions):**

| Pair | Count |
|------|-------|
| Phytophthora ↔ Rhizoctonia | 5× |
| Bacterial_Blight ↔ Bacterial_Pustule | 4× |
| Brown_Stem_Rot ↔ Sudden_death_syndrome | 4× |
| Phomopsis ↔ White_Mold | 4× |
| Phyllosticta_leaf_spot ↔ Soybean_Vein_necrosis_virus | 4× |
| Septoria_brown_spot ↔ various (Cercospora, SDMV, Bacterial_Pustule) | 8× total |

**Top attractor classes (wrongly predicted most often, all configs):**

| Class | Times wrongly predicted |
|-------|------------------------|
| Sudden_death_syndrome | 25× |
| Septoria_brown_spot | 18× |
| Bacterial_Blight | 14× |
| Bacterial_Pustule | 11× |
| White_Mold | 10× |

---

### Corn

**Top confusion pairs:** not reported

---

### Mango

**Top confusion pairs:** not reported

---

## Dataset Statistics (continued)

### Sugarcane

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 17 |
| Disease classes (evaluated, with test images) | 9 |
| Raw images inspected | 255 |
| Reference images kept | 51 |
| Test images kept (prepared) | 54 |
| Test images used in eval | 26 |
| Images rejected | 101 |
| Images skipped | 36 |
| Errors | 13 |
| Max images per part | 3 |
| max_inspect_per_class | 15 |
| test_per_class (prepare) | 5 |

**Part distribution:**

| Part | Classes |
|------|---------|
| leaf | 10 |
| stem | 4 |
| whole_plant | 4 |

---

### Banana

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 16 |
| Disease classes (evaluated, with test images) | 7 |
| Raw images inspected | 113 |
| Reference images kept | 43 |
| Test images kept (prepared) | 31 |
| Test images used in eval | 19 |
| Images rejected | 18 |
| Images skipped | 21 |
| Errors | 0 |
| Max images per part | 3 |
| max_inspect_per_class | 15 |
| test_per_class (prepare) | 5 |

**Part distribution:**

| Part | Classes |
|------|---------|
| leaf | 5 |
| pod | 2 |
| seed | 2 |
| stem | 1 |
| whole_plant | 3 |

---

### Cauliflower

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 5 |
| Disease classes (evaluated, with test images) | 4 |
| Raw images inspected | 75 |
| Reference images kept | 19 |
| Test images kept (prepared) | 21 |
| Test images used in eval | 11 |
| Images rejected | 25 |
| Images skipped | 10 |
| Errors | 0 |
| Max images per part | 3 |
| max_inspect_per_class | 15 |
| test_per_class (prepare) | 5 |

**Part distribution:**

| Part | Classes |
|------|---------|
| leaf | 3 |
| whole_plant | 2 |

---

### Orange

| Stat | Value |
|------|-------|
| Disease classes (prepared) | 7 |
| Disease classes (evaluated, with test images) | 2 |
| Raw images inspected | 39 |
| Reference images kept | 14 |
| Test images kept (prepared) | 10 |
| Test images used in eval | 6 |
| Images rejected | 13 |
| Images skipped | 2 |
| Errors | 0 |
| Max images per part | 3 |
| max_inspect_per_class | 15 |
| test_per_class (prepare) | 5 |

**Part distribution:**

| Part | Classes |
|------|---------|
| leaf | 1 |
| seed | 1 |
| stem | 1 |

---

## KB Coverage (continued)

### Sugarcane

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/9 | all |
| local | not reported | — |
| internet | 9/9 | none |

### Banana

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/7 | all |
| local | not reported | — |
| internet | 10/16 raw classes matched (6 placeholders); per evaluated class: not reported |

### Cauliflower

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/4 | all |
| local | not reported | — |
| internet | 4/5 raw classes matched; 4/4 evaluated classes have KB (Bacterial_Spot_Rot excluded had no KB) |

### Orange

| Source | Classes with KB data | Classes missing KB data |
|--------|---------------------|------------------------|
| none | 0/2 | all |
| local | not reported | — |
| internet | 5/7 raw classes matched; 2/2 evaluated classes have KB |

---

## Main Accuracy Table (continued)

### Sugarcane — 9 classes × ~3 images = 26 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 30.77% | 38.46% | 57.69% | 61.54% | 76.92% |
| Agent (no KB) | 42.31% | 53.85% | 65.38% | 69.23% | 69.23% |
| Agent + internet KB | 46.15% | 53.85% | 61.54% | 57.69% | 73.08% |

### Banana — 7 classes × ~3 images = 19 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 73.68% | 78.95% | 73.68% | 78.95% | 73.68% |
| Agent (no KB) | 57.89% | 68.42% | 68.42% | 89.47% | 68.42% |
| Agent + internet KB | 68.42% | 84.21% | 84.21% | 89.47% | 78.95% |

### Cauliflower — 4 classes × ~3 images = 11 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 45.45% | 54.55% | 54.55% | 54.55% | 72.73% |
| Agent (no KB) | 54.55% | 36.36% | 54.55% | 63.64% | 72.73% |
| Agent + internet KB | 45.45% | 54.55% | 72.73% | 63.64% | 72.73% |

### Orange — 2 classes × 3 images = 6 total test images

| Method | k=0 | k=1 | k=4 | k=8 | k=16 |
|--------|-----|-----|-----|-----|------|
| Few-shot | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Agent (no KB) | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Agent + internet KB | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

---

## Model Ablation (continued)

### Sugarcane (internet KB, k=8)

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 50.00% (13/26) | $0.0893 |
| Sonnet | 57.69% (15/26) | $0.2017 |
| Opus | 84.62% (22/26) | $0.3999 |
| gemini-flash | 46.15% (12/26) | $0.00 (billed externally) |
| gemini-pro | 57.69% (15/26) | $0.00 (billed externally) |

### Banana (internet KB, k=8)

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 68.42% (13/19) | $0.0800 |
| Sonnet | 89.47% (17/19) | $0.1776 |
| Opus | 94.74% (18/19) | $0.3615 |
| gemini-flash | 42.11% (8/19) | $0.00 (billed externally) |
| gemini-pro | 84.21% (16/19) | $0.00 (billed externally) |

### Cauliflower (internet KB, k=8)

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 72.73% (8/11) | $0.0716 |
| Sonnet | 63.64% (7/11) | $0.1815 |
| Opus | 54.55% (6/11) | $0.3496 |
| gemini-flash | 54.55% (6/11) | $0.00 (billed externally) |
| gemini-pro | 45.45% (5/11) | $0.00 (billed externally) |

### Orange (internet KB, k=8)

| Model | Accuracy | Cost/image |
|-------|----------|-----------|
| Haiku | 100.00% (6/6) | $0.0690 |
| Sonnet | 100.00% (6/6) | $0.1759 |
| Opus | 100.00% (6/6) | $0.3226 |
| gemini-flash | 50.00% (3/6) | $0.00 (billed externally) |
| gemini-pro | 83.33% (5/6) | $0.00 (billed externally) |

---

## Cost Numbers (continued)

### Cost per image per k — Agent (internet KB, sonnet) — new crops

| k | Sugarcane | Banana | Cauliflower | Orange |
|---|-----------|--------|-------------|--------|
| 0 | $0.0655 | $0.0523 | $0.0536 | $0.0404 |
| 1 | $0.0869 | $0.0741 | $0.0770 | $0.0642 |
| 4 | $0.1407 | $0.1239 | $0.1154 | $0.1002 |
| 8 | $0.2017 | $0.1776 | $0.1815 | $0.1759 |
| 16 | $0.3398 | $0.3185 | $0.3218 | $0.2901 |

### Cost per image per k — Few-shot (sonnet) — new crops

| k | Sugarcane | Banana | Cauliflower | Orange |
|---|-----------|--------|-------------|--------|
| 0 | $0.0040 | $0.0035 | $0.0040 | $0.0047 |
| 1 | $0.0064 | $0.0051 | $0.0051 | $0.0072 |
| 4 | $0.0124 | $0.0106 | $0.0099 | $0.0162 |
| 8 | $0.0207 | $0.0178 | $0.0152 | $0.0286 |
| 16 | $0.0374 | $0.0323 | $0.0256 | $0.0552 |

### Total sweep cost per crop (continued)

| Crop | Total cost |
|------|-----------|
| Sugarcane | $54.96 (Gemini billed externally) |
| Banana | $37.22 (Gemini billed externally) |
| Cauliflower | $20.69 (Gemini billed externally) |
| Orange | $10.56 (Gemini billed externally) |

---

## Confusion Stats (continued)

*At internet KB, sonnet model, k=8.*

### Sugarcane

**Top confusion pairs:** not reported

---

### Banana

**Top confusion pairs:** not reported

---

### Cauliflower

**Top confusion pairs:** not reported

---

### Orange

**Top confusion pairs:** not reported

---

## Run Configuration — Exclude Lists (continued)

### Exclude list — Sugarcane

```
Brown_Rust, Eye_Spot, Leaf_Mosaic_Virus, Smut,
Banded_Chlorosis, Common_Rust, Dried_Leaves, Red_Spot
```

### Exclude list — Banana

```
Bitter_Rot_And_Anthracnose, Pestalotiopsis_Leaf_Spot, Phoma_Blight,
Phyllosticta_Maculata, Pseudocercospora_Musae, Rust,
Phaeoseptoria_Leaf_Spot, Phyllosticta_Leaf_Spot, Pseudocercospora_Leaf_Spot
```

### Exclude list — Cauliflower

```
Bacterial_Spot_Rot
```

### Exclude list — Orange

```
Penicillium_Fungi, Penicillium_Fungus, Blue_Mold, Citrus_Blight_Disease, Green_Mold
```

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Parallel workers | 12 (default); 20 used for Corn |
| Test images per class (IMAGES) | 3 (Soybean, Corn); 10 (Mango, override) |
| Max refs per part (--max-per-part) | 5 |
| Collage tile size | 800 px (IMAGES=3 runs); 400 px (IMAGES=1 runs) |
| Collage grid | 2×2 (4 training images per class per ref view) |
| Model (main sweep) | sonnet (claude-sonnet-4-6) |
| KB default | internet |

### Exclude list — Soybean

```
Diaporthe_2015_Kanawha, Green_stem, Fusarium_healthy_vs_infected, Stem_Canker, Top_Dieback,
Diaporthe, Soybean_Dwarf_Mosaic_Virus, Bacterial_Pustule_Of_Soybean_Disease, Cercospora,
Cercospora_Fungi, Diaporthe_Fungus, Fusarium_Disease, Herbicide_Injury,
Iron_Deficiency_Chlorosis, Phomopsis, Phyllosticta_Leaf_Spot, Phytophthora,
Potassium_Deficiency, Pythium, Pythium_Diseases, Rhizobium_Bacteria, Rhizoctonia,
Root_And_Stem_Rot, Sclerotinia_Timber_Rot, Septoria_Leaf_Blotch,
Soybean_Dwarf_Mosaic_Virus_2012, Wildfire_Of_Tobacco, Xylaria_Necrophora_Garcia-Aroca
```

*Note: many of these are quality-judge exclusions added after the current sweep data was generated. Effective exclusions at run time were: Diaporthe_2015_Kanawha, Green_stem, Fusarium_healthy_vs_infected, Stem_Canker, Top_Dieback (plus Diaporthe and Soybean_Dwarf_Mosaic_Virus due to 0 test images).*

### Exclude list — Corn

```
Anthracnose_Ear_Infection, Leaf_Blight, Leaf_Spot, Maize_Lethal_Necrosis,
Penicillium_On_Seedling, Pythium, Rhizoctonia, Rust, Smut, Ear_Rots,
General_And_Mixed_Ear_Rots, General_And_Mixed_Stalk_Rots, Genetic_Flecking_Or_Striping,
Genetic_Streaking, Diplodia, Chocolate_Spot, Barley_Yellow_Dwarf_Virus, Cladosporium_Ear_Rot,
Crown_Rot, Damping_Off, Downy_Mildew, Maize_White_Line_Mosaic, Nigrospora_Ear_Rot,
Penicillium_Ear_Rot, Pythium_Stalk_Rot, Rhizopus_Stolonifer, Root_Rot, Trichoderma_Stalk_Rot,
Alternaria_Black_Molds_Stem_Cankers, Bacterial_Brown_Spot_Of_Beancanker_Of_Stone_Fruit,
Bacterial_Rot_And_Blight, Brown_Spot, Cladosporium_Fungus, Dry_Rot_Of_Ears_And_Stalks_Of_Maize,
Epicoccum_Fungus, Fusarium_Disease, Fusarium_Graminearum_Schwabe, Fusarium_Wilts,
Gibberella_Disease, Goss, Green_Mold, Khuskia_Fungus, Pantoea, Penicillium_Fungi,
Pythium_Diseases, Xanthomonas_Vasicola
```

### Exclude list — Mango

```
Bacterial_Canker, Bitter_Rot_And_Anthracnose, Fungus, Pestalotiopsis_Blight,
Phaeoramularia_Fungus
```
