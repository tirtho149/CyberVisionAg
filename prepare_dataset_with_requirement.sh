#!/bin/bash

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

crops=( Tomato Sugarcane Banana Wheat )

# Function to get exclusion list for a crop
get_exclude() {
  case "$1" in
    Tomato)
      echo "Bacterial_Leaf_Spot,Leaf_Mosaic_Virus,Leaf_Yellow_Virus,Spider_Mites,Tomato_Leaf_Mould,Bitter_Rot_And_Anthracnose,Blossom_End_Rot,Brown_Spot,Clover_Proliferation_Phytoplasma,Diaporthe_Vexans_Gratz,Fusarium_Wilts,Phomopsis_Cankers_And_Twig_Blights,Phytoplasma,Remotididymella_Destructiva,Bacterial_Pith_Necrosis,Bacterial_Soft_Rot,Beet_Curly_Top_Virus,Fusarium_Damping-Off,Fusarium_Wilt,Phoma_Blight,Pythium_Diseases,Rhizoctonia_Damping-Off,Sclerotinia_Rots,Tomato_Leaf_Curl_Virus,Tomato_Ringspot_Virus,Alfalfa_Mosaic_Virus,Tomato_Necrotic_Dwarf_Virus,Phytophthora_Root_And_Crown_Rots"
      ;;
    Sugarcane)
      echo "Brown_Rust,Eye_Spot,Leaf_Mosaic_Virus,Smut,Banded_Chlorosis,Common_Rust,Dried_Leaves,Red_Spot"
      ;;
    Banana)
      echo "Bitter_Rot_And_Anthracnose,Pestalotiopsis_Leaf_Spot,Phoma_Blight,Phyllosticta_Maculata,Pseudocercospora_Musae,Rust,Phaeoseptoria_Leaf_Spot,Phyllosticta_Leaf_Spot,Pseudocercospora_Leaf_Spot"
      ;;
    Wheat)
      echo "Alternaria_Black_Molds_Stem_Cankers,Bacterial_Brown_Spot_Of_Beancanker_Of_Stone_Fruit,Blumeria_Graminis_F,Cochliobolus_Leaf_Spot,Downy_Mildew,Fusarium_Disease,Fusarium_Graminearum_Schwabe,Fusarium_Wilts,Leaf_Spot,Leaf_Streakblack_Chaff,Parastagonospora_Nodorum,Penicillium_Fungi,Phytophthora_Root_And_Crown_Rots,Resistance_Phenotype,Root_Rot,Rust,Septoria_Leaf_Spot_And_Cankers,Smut,Sooty_Mold,Zymoseptoria_Tritici"
      ;;
  esac
}

# Function to count items in comma-separated list
count_csv() {
  if [ -z "$1" ]; then
    echo 0
  else
    echo "$1" | awk -F',' '{print NF}'
  fi
}

for crop in "${crops[@]}"; do
  echo ">>> Processing $crop..."

  # Get list of raw disease directories
  raw_dir="Raw_Crops/${crop}_Diseases"
  all_diseases=$(ls -d "$raw_dir"/*/ 2>/dev/null | xargs -I {} basename {})

  if [ -z "$all_diseases" ]; then
    echo "  ⚠️  No disease directories found in $raw_dir, skipping..."
    continue
  fi

  # Count total classes
  total_classes=$(echo "$all_diseases" | wc -w)

  # Get exclusion list for this crop
  exclusions=$(get_exclude "$crop")

  # Count excluded classes
  excluded_count=$(count_csv "$exclusions")

  # Calculate kept classes
  kept_classes=$((total_classes - excluded_count))

  if [ "$kept_classes" -le 0 ]; then
    echo "  ⚠️  All classes excluded, skipping..."
    continue
  fi

  # Calculate samples per class: max(50 // kept_classes, 5)
  test_per_class=$(python3 -c "print(max(50 // $kept_classes, 5))")
  expected_total=$((kept_classes * test_per_class))

  echo "  Total classes: $total_classes, Excluded: $excluded_count, Kept: $kept_classes"
  echo "  Test samples per class: $test_per_class (expected total: $expected_total)"

  python3 -m open_agentic.prepare_dataset \
    --input-dir "$raw_dir" \
    --output-dir "Prepared_Dataset/$crop" \
    --max-per-part 5 --test-per-class "$test_per_class" --max-inspect-per-class 20 \
    --seed 42 --parallel 12 \
    ${exclusions:+--exclude "$exclusions"}

  echo "  ✓ $crop completed"
done

echo "=== ALL DONE ==="
