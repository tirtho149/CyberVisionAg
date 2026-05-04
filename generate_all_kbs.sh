#!/bin/bash
cd /Users/tirthoroy/Desktop/CyberVisionAg
set -a && source .env && set +a

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

crops=(soybean corn mango tomato sugarcane banana cauliflower coffee orange wheat)

for crop in "${crops[@]}"; do
  echo "========================================"
  echo ">>> Generating KB for $crop..."
  echo "========================================"
  python3 -m disease_registry.pipeline --crop "$crop" --track internet --disease-dir "Raw_Crops"
  echo ""
done

echo "=== ALL KB GENERATION DONE ==="
