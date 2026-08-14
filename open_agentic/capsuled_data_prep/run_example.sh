#!/usr/bin/env bash
# Example invocation. Edit CROP and DATASET_ROOT, then run:
#   bash run_example.sh
#
# Defaults assume the cluster layout you described:
#   /work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images/<Crop>_Diseases/<class>/<image>.jpg
#
# Outputs land under ./out/<Crop>_Diseases (and ./out/<Crop>_Diseases_test).

set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

CROP="${CROP:-Tomato}"            # one of: Banana Cauliflower Coffee Corn Mango_Leaf Orange Soybean Sugarcane Tomato Wheat
DATASET_ROOT="${DATASET_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"
DATASET="${CROP}_Diseases"        # use ${CROP}_Disease for Mango_Leaf if needed

INPUT_DIR="${DATASET_ROOT}/${DATASET}"
OUTPUT_DIR="./out/${DATASET}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: ${INPUT_DIR} does not exist."
    echo "Edit CROP / DATASET_ROOT in run_example.sh, or pass them inline:"
    echo "  CROP=Soybean DATASET_ROOT=/path/to/Images bash run_example.sh"
    exit 2
fi

python prepare_dataset.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-per-part 5 \
    --test-per-class 5 \
    --max-inspect-per-class 60 \
    --seed 42 \
    --parallel 12
