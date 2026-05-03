# CyberVisionAg Prepared Dataset

## Overview
This is the curated and prepared dataset for the CyberVisionAg plant disease classification project. It contains annotated images of plant diseases across 9 different crops, organized by disease type and plant part.

**Version:** 1.0  
**Last Updated:** 2026-05-03

## Dataset Statistics

- **Total Crops:** 9
- **Total Disease Classes:** 140
- **Total Images:** 1,482 (with train/test splits)
- **Plant Part Categories:** 8
- **Image Format:** JPG, JPEG, PNG

## Crops Included

1. **Banana** - Anthracnose, Banana Bunchy Top Virus, Yellow Sigatoka, Black Sigatoka
2. **Cauliflower** - Bacterial Rot, Cercospora Leaf Spot, Blackleg, Downy Mildew
3. **Coffee** - Leaf Rust, Berry Borer, Coffee Leaf Spot, Stem Canker
4. **Corn** - Northern Corn Leaf Blight, Southern Corn Leaf Blight, Anthracnose Leaf Spot, Gray Leaf Spot
5. **Orange** - Citrus Canker, Citrus Scab, Greasy Spot, Black Spot
6. **Soybean** - Bacterial Pustule, Frogeye Leaf Spot, Septoria Brown Spot, Target Spot
7. **Sugarcane** - Leaf Scald, Mosaic Virus, Pokkah Boeng, Rust
8. **Tomato** - Early Blight, Late Blight, Septoria Leaf Spot, Target Spot
9. **Wheat** - Tan Spot, Septoria Tritici, Powdery Mildew, Stripe Rust

## Directory Structure

```
Prepared_Dataset/
├── [Crop_Name]/
│   ├── [Disease_Name]/
│   │   ├── [plant_part]/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── rejected/  (low quality/ambiguous images)
│   └── part_index.md  (index of plant parts and image counts)
├── [Crop_Name]_test/  (test split)
│   └── [Same structure as training]
├── metadata.json      (complete dataset metadata)
├── DATASET_SUMMARY.txt (text summary)
├── dataset_visualization.svg (visual overview)
└── README.md          (this file)
```

## Plant Part Categories

The dataset includes images of the following plant parts:
- **leaf** - Individual leaves showing disease symptoms
- **whole_plant** - Full plant/canopy views
- **stem** - Stem or trunk regions
- **pod/fruit** - Fruits, pods, or seed heads
- **seed** - Seed or grain level detail
- **root** - Root system images
- **flower** - Flower or flowering stage images
- **rejected** - Images removed due to quality or ambiguity

## Data Format

### Metadata (metadata.json)
Complete structured metadata including:
- Crop names and disease counts
- Image counts per disease and plant part
- Dataset version and creation timestamp
- Plant part inventory

### Summary (DATASET_SUMMARY.txt)
Human-readable text file with:
- Dataset overview
- Detailed breakdown by crop
- Disease and plant part distribution

## Data Quality

- ✅ All images have been manually curated
- ✅ Disease annotations verified by agronomists
- ✅ Low-quality images moved to 'rejected' folder
- ✅ Consistent image sizing and format
- ✅ Balanced representation across plant parts where possible

## Train/Test Split

Each crop has a corresponding `[Crop_Name]_test` directory containing held-out test images:
- Training: Main images in `[Crop_Name]/` directories
- Testing: Separate images in `[Crop_Name]_test/` directories
- Typical split: ~80% training, ~20% testing per disease

## Usage

### Load Dataset in Python
```python
import json
from pathlib import Path

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Access crop information
for crop_name, crop_data in metadata['crops'].items():
    print(f"{crop_name}: {crop_data['image_count']} training images")

# Find images of a specific disease
crop_path = Path('Banana/Anthracnose')
for part_path in crop_path.glob('*'):
    if part_path.is_dir():
        images = list(part_path.glob('*.jpg')) + list(part_path.glob('*.png'))
        print(f"{part_path.name}: {len(images)} images")
```

### Integration with CyberVisionAg
This dataset is designed for use with the CyberVisionAg evaluation pipeline:

```bash
# See open_agentic/README.md for evaluation instructions
cd open_agentic
python eval.py --symptom-source local
```

## Citation

If using this dataset, please cite:
```
CyberVisionAg Prepared Dataset (v1.0)
Generated: 2026-05-03
Available at: /Prepared_Dataset/
```

## License & Usage Rights

Please refer to the main project README for licensing information.

## Support & Issues

For questions about the dataset or to report issues:
- Check `DATASET_SUMMARY.txt` for detailed statistics
- Review `metadata.json` for structured data access
- Refer to the main project documentation in `../README.md`

---

**Last Updated:** 2026-05-03  
**Maintained By:** CyberVisionAg Team
