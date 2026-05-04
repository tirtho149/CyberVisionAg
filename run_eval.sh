#!/bin/bash
# A/B evaluation: runs all 3 KB sources in parallel
# Usage: bash CyberVisionAg/run_eval.sh  (from AgCrawler root)
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
set -a && source .env && set +a

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

for src in default local internet; do
  PYTHONUNBUFFERED=1 python3 -m CyberVisionAg.agent \
    --symptom-source $src --num-classes 5 --images-per-class 5 \
    --k 4 --parallel 12 --seed 42 \
    > /tmp/agent_${src}.log 2>&1 &
  echo "$src PID: $!"
done

echo "Waiting for all 3 runs..."
wait
echo "All done!"

for src in default local internet; do
  echo "=== $src ==="
  grep -E "Accuracy|Avg tool|Avg ref" /tmp/agent_${src}.log
  echo "Errors: $(grep -c 'API ERROR' /tmp/agent_${src}.log)"
  echo
done
