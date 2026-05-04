#!/bin/bash
# Open Agentic Pipeline — A/B evaluation (none vs local vs internet)
# Usage: bash CyberVisionAg/open_agentic/run_eval.sh  (from AgCrawler root)
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
set -a && source .env && set +a

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

for src in none local internet; do
  PYTHONUNBUFFERED=1 python3 -m CyberVisionAg.open_agentic.eval \
    --symptom-source $src --num-classes 10 --images-per-class 3 \
    --k 4 --parallel 12 --seed 42 \
    > /tmp/open_agentic_${src}.log 2>&1 &
  echo "$src PID: $!"
done

echo "Waiting for all 3 runs..."
wait
echo "All done!"

for src in none local internet; do
  echo "=== $src ==="
  grep -E "Accuracy|Avg turns|Avg refs|Avg cost|Total cost|Errors|KB coverage|KB missing" /tmp/open_agentic_${src}.log || true
  echo
done
