#!/bin/bash
# Run evaluation sweeps for crop-disease classes from crop_disease_final_min5test.csv
# Hardcoded classes: 48 crop-disease pairs with ≥5 test samples

set -uo pipefail

# Install required dependencies
echo "Checking and installing required dependencies..."
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true
echo "Dependencies ready"
echo ""

# Hardcoded crop-disease classes from crop_disease_final_min5test.csv
declare -A crop_diseases

# Banana (6 classes)
crop_diseases[Banana]="Anthracnose Cigar_End_Rot Bunchy_Top Panama_Disease Cordana_Leaf_Spot Yellow_And_Black_Sigatoka"

# Cauliflower (4 classes)
crop_diseases[Cauliflower]="Black_Rot Bacterial_Soft_Rot Alternaria_Leaf_Spot Downy_Mildew"

# Coffee (5 classes)
crop_diseases[Coffee]="Berry_Blotch Brown_Eye_Spot Cerscospora Phoma Miner"

# Orange (1 class)
crop_diseases[Orange]="Whisker_Mold"

# Sugarcane (10 classes)
crop_diseases[Sugarcane]="Brown_Spot Pokkah_Boeng Sett_Rot Sugarcane_Mosaic_Virus Common_Rust Dried_Leaves Leaf_Scald Streak_Mosaic_Scsmv Yellow_Spot Banded_Chlorosis Eye_Spot Grassy_Shoot Red_Spot Yellow_Leaf"

# Tomato (8 classes)
crop_diseases[Tomato]="Tomato_Spotted_Wilt_Virus Southern_Blight Powdery_Mildew Bacterial_Speck_Of_Tomato Cucumber_Mosaic_Virus Verticillium_Wilt Leaf_Mold Septoria_Leaf_Blotch"

# Wheat (8 classes)
crop_diseases[Wheat]="Bacterial_Leaf_Streak_Black_Chaff Stem_Rust Loose_Smut Septoria_Leaf_Blotch Powdery_Mildew Resistance_Phenotype__Moderately_Resistant Resistance_Phenotype__Moderately_Susceptible Resistance_Phenotype__Resistant Resistance_Phenotype__Susceptible Stripe_Rust"

# Available crops
crops=( Banana Cauliflower Coffee Orange Sugarcane Tomato Wheat )

# Default family (can be overridden with argument)
FAMILY="${1:-claude}"

if [ "$FAMILY" != "claude" ] && [ "$FAMILY" != "gemini" ]; then
    echo "Usage: $0 [claude|gemini]"
    echo ""
    echo "Examples:"
    echo "  $0 claude       # Run all configs for available crop-disease classes (Claude models)"
    echo "  $0 gemini       # Run all configs for available crop-disease classes (Gemini models)"
    exit 1
fi

echo "=========================================="
echo "Running evaluation sweeps for crop-disease classes"
echo "Source: crop_disease_final_min5test.csv"
echo "Family: $FAMILY"
echo "=========================================="
echo ""
echo "Available crops and classes:"
total_classes=0
for crop in "${crops[@]}"; do
    num_classes=$(echo ${crop_diseases[$crop]} | wc -w)
    total_classes=$((total_classes + num_classes))
    echo "  $crop: $num_classes classes"
    for disease in ${crop_diseases[$crop]}; do
        echo "    - $disease"
    done
done
echo ""
echo "Total: ${#crops[@]} crops, $total_classes disease classes"
echo ""

read -p "Continue? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting evaluation sweeps for ${#crops[@]} crop(s)..."
echo ""

total=0
completed=0

for crop in "${crops[@]}"; do
    echo "=========================================="
    echo ">>> Running sweep for $crop..."
    echo "=========================================="

    # Get diseases for this crop
    diseases=${crop_diseases[$crop]}
    num_diseases=$(echo $diseases | wc -w)

    echo "Classes: $num_diseases"
    echo ""

    total=$((total + 1))

    bash open_agentic/run_sweeps.sh run-missing "$crop" "$FAMILY"

    if [ $? -eq 0 ]; then
        completed=$((completed + 1))
        echo "✓ $crop completed"
    else
        echo "✗ $crop failed"
    fi
    echo ""
done

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total crops: $total"
echo "Completed: $completed"
echo "Failed: $((total - completed))"
echo "=========================================="
