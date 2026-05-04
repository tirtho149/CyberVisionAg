#!/bin/bash

set -a && source .env && set +a

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

crops=( tomato sugarcane banana wheat )

# Function to get exclusion list for a crop (hardcoded from run_sweeps.sh)
get_exclude() {
  case "$1" in
    tomato)
      echo "Bacterial_Leaf_Spot,Leaf_Mosaic_Virus,Leaf_Yellow_Virus,Spider_Mites,Tomato_Leaf_Mould,Bitter_Rot_And_Anthracnose,Blossom_End_Rot,Brown_Spot,Clover_Proliferation_Phytoplasma,Diaporthe_Vexans_Gratz,Fusarium_Wilts,Phomopsis_Cankers_And_Twig_Blights,Phytoplasma,Remotididymella_Destructiva,Bacterial_Pith_Necrosis,Bacterial_Soft_Rot,Beet_Curly_Top_Virus,Fusarium_Damping-Off,Fusarium_Wilt,Phoma_Blight,Pythium_Diseases,Rhizoctonia_Damping-Off,Sclerotinia_Rots,Tomato_Leaf_Curl_Virus,Tomato_Ringspot_Virus,Alfalfa_Mosaic_Virus,Tomato_Necrotic_Dwarf_Virus,Phytophthora_Root_And_Crown_Rots"
      ;;
    sugarcane)
      echo "Brown_Rust,Eye_Spot,Leaf_Mosaic_Virus,Smut,Banded_Chlorosis,Common_Rust,Dried_Leaves,Red_Spot"
      ;;
    banana)
      echo "Bitter_Rot_And_Anthracnose,Pestalotiopsis_Leaf_Spot,Phoma_Blight,Phyllosticta_Maculata,Pseudocercospora_Musae,Rust,Phaeoseptoria_Leaf_Spot,Phyllosticta_Leaf_Spot,Pseudocercospora_Leaf_Spot"
      ;;
    wheat)
      echo "Alternaria_Black_Molds_Stem_Cankers,Bacterial_Brown_Spot_Of_Beancanker_Of_Stone_Fruit,Blumeria_Graminis_F,Cochliobolus_Leaf_Spot,Downy_Mildew,Fusarium_Disease,Fusarium_Graminearum_Schwabe,Fusarium_Wilts,Leaf_Spot,Leaf_Streakblack_Chaff,Parastagonospora_Nodorum,Penicillium_Fungi,Phytophthora_Root_And_Crown_Rots,Resistance_Phenotype,Root_Rot,Rust,Septoria_Leaf_Spot_And_Cankers,Smut,Sooty_Mold,Zymoseptoria_Tritici"
      ;;
  esac
}

echo "KB Generation with hardcoded exclusions from run_sweeps.sh"
echo ""

for crop in "${crops[@]}"; do
  echo "========================================"
  echo ">>> Generating KB for $crop..."
  echo "========================================"

  # Get exclusion list (hardcoded)
  exclusions=$(get_exclude "$crop")

  if [ -z "$exclusions" ]; then
    echo "⚠️  No exclusion list found for $crop"
    python3 -m disease_registry.pipeline --crop "$crop" --track internet --disease-dir "Raw_Crops"
  else
    # Count excluded diseases
    excluded_count=$(echo "$exclusions" | awk -F',' '{print NF}')
    echo "  Excluding $excluded_count diseases"

    # Run KB generation with exclusions
    python3 -m disease_registry.pipeline --crop "$crop" --track internet --disease-dir "Raw_Crops" --exclude "$exclusions"
  fi

  echo ""
done

echo "=== ALL KB GENERATION DONE ==="
