import os
import csv
from pathlib import Path

# Define the test dataset directory
test_dataset_dir = Path('/Users/tirthoroy/Desktop/CyberVisionAg/Prepared_Dataset')

# Collect all test data
test_data = []

# Find all _test directories
for crop_dir in sorted(test_dataset_dir.glob('*_test')):
    if crop_dir.is_dir():
        crop_name = crop_dir.name.replace('_test', '')

        # Find all disease subdirectories
        for disease_dir in sorted(crop_dir.iterdir()):
            if disease_dir.is_dir():
                disease_name = disease_dir.name

                # Find all image files
                for image_file in sorted(disease_dir.iterdir()):
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        test_data.append({
                            'file_name': image_file.name,
                            'crop_name': crop_name,
                            'disease_name': disease_name
                        })

# Write to CSV
output_csv = '/Users/tirthoroy/Desktop/CyberVisionAg/test_data_registry.csv'
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['file_name', 'crop_name', 'disease_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(test_data)

print(f"CSV created: {output_csv}")
print(f"Total test images: {len(test_data)}")
print(f"\nBreakdown by crop:")
crops = {}
for row in test_data:
    crop = row['crop_name']
    crops[crop] = crops.get(crop, 0) + 1

for crop in sorted(crops.keys()):
    print(f"  {crop}: {crops[crop]} images")
