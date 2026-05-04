#!/usr/bin/env python3
"""
Generate a CSV catalog of all test datasets in Prepared_Dataset
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

def count_images(directory):
    """Count image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    count = 0
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in image_extensions:
            count += 1
    return count

def generate_test_dataset_csv(prepared_dataset_path, output_csv):
    """Generate CSV of all test datasets"""

    data = []
    prepared_path = Path(prepared_dataset_path)

    # Find all *_test directories
    test_dirs = sorted([d for d in prepared_path.iterdir()
                       if d.is_dir() and d.name.endswith('_test')])

    for test_dir in test_dirs:
        crop_name = test_dir.name.replace('_test', '')

        # Get all disease/condition subdirectories
        disease_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

        for disease_dir in disease_dirs:
            disease_name = disease_dir.name
            num_images = count_images(str(disease_dir))
            relative_path = str(disease_dir.relative_to(prepared_path))

            data.append({
                'Crop': crop_name,
                'Disease/Condition': disease_name,
                'Test_Images': num_images,
                'Path': relative_path
            })

    # Write CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Crop', 'Disease/Condition', 'Test_Images', 'Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    # Also generate summary stats
    summary_stats = defaultdict(int)
    total_images = 0

    for row in data:
        summary_stats[row['Crop']] += row['Test_Images']
        total_images += row['Test_Images']

    print(f"\n✓ Generated {output_csv}")
    print(f"\nSummary Statistics:")
    print(f"{'Crop':<20} {'Test Images':<15}")
    print("-" * 35)
    for crop in sorted(summary_stats.keys()):
        print(f"{crop:<20} {summary_stats[crop]:<15}")
    print("-" * 35)
    print(f"{'TOTAL':<20} {total_images:<15}")
    print(f"\nTotal test entries: {len(data)}")

if __name__ == '__main__':
    prepared_dataset_path = '/Users/tirthoroy/Desktop/CyberVisionAg/Prepared_Dataset'
    output_csv = '/Users/tirthoroy/Desktop/CyberVisionAg/test_dataset_catalog.csv'

    generate_test_dataset_csv(prepared_dataset_path, output_csv)
