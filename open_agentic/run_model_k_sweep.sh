#!/usr/bin/env bash
# Model × K sweep — Internet KB, 27 classes × 2 images, seed=42
#
# Runs all combinations of model (haiku, sonnet, opus) and k (1, 2, 4, 8)
# with internet KB fixed. Results saved to sweep_results.csv.
#
# Usage:
#   cd /path/to/AgCrawler
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
#   set -a && source .env && set +a
#   bash CyberVisionAg/open_agentic/run_model_k_sweep.sh

set -uo pipefail

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

EXCLUDE="Diaporthe_2015_Kanawha,Green_stem,Fusarium_healthy_vs_infected,Stem_Canker,Top_Dieback"
MODELS=("haiku" "sonnet" "opus")
K_VALUES=(1 2 4 8)
IMAGES_PER_CLASS=2
PARALLEL=12
SEED=42
SOURCE="internet"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_CSV="${SCRIPT_DIR}/sweep_results.csv"

echo "=== Model × K Sweep ==="
echo "KB source: ${SOURCE}"
echo "Models: ${MODELS[*]}"
echo "K values: ${K_VALUES[*]}"
echo "Images/class: ${IMAGES_PER_CLASS}, parallel: ${PARALLEL}, seed: ${SEED}"
echo "Results CSV: ${RESULTS_CSV}"
echo ""

# CSV header
echo "model,k,accuracy,correct,total,avg_turns,avg_refs,avg_cost,total_cost,avg_duration_s,errors" > "${RESULTS_CSV}"

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#K_VALUES[@]} ))
RUN_NUM=0

# Helper: extract a numeric value after a label from eval output
extract_val() {
    local label="$1"
    local text="$2"
    echo "${text}" | grep "${label}" | sed 's/.*: *//' | sed 's/[^0-9.]//g' | head -1
}

for model in "${MODELS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        echo "--- Run ${RUN_NUM}/${TOTAL_RUNS}: model=${model}, k=${k} ---"

        # Run eval and capture output
        OUTPUT=$(PYTHONUNBUFFERED=1 python -m CyberVisionAg.open_agentic.eval \
            --model "${model}" \
            --symptom-source "${SOURCE}" \
            --images-per-class "${IMAGES_PER_CLASS}" \
            --k "${k}" \
            --parallel "${PARALLEL}" \
            --seed "${SEED}" \
            --exclude "${EXCLUDE}" 2>&1) || true

        # Parse accuracy line: "  Accuracy         : 23/54 (42.6%)"
        acc_line=$(echo "${OUTPUT}" | grep "Accuracy" || echo "")
        if [ -n "${acc_line}" ]; then
            correct=$(echo "${acc_line}" | sed 's/.*: *//' | sed 's|/.*||')
            total=$(echo "${acc_line}" | sed 's|.*/||' | sed 's| .*||')
            acc_pct=$(echo "${acc_line}" | sed 's/.*(//;s/%).*//')
        else
            correct=0; total=0; acc_pct=0
        fi

        avg_turns=$(extract_val "Avg turns" "${OUTPUT}")
        avg_refs=$(extract_val "Avg refs" "${OUTPUT}")
        avg_cost=$(extract_val "Avg cost" "${OUTPUT}")
        total_cost=$(extract_val "Total cost" "${OUTPUT}")
        avg_dur=$(extract_val "Avg duration" "${OUTPUT}")
        errors=$(extract_val "Errors" "${OUTPUT}")

        # Defaults
        avg_turns=${avg_turns:-0}; avg_refs=${avg_refs:-0}
        avg_cost=${avg_cost:-0}; total_cost=${total_cost:-0}
        avg_dur=${avg_dur:-0}; errors=${errors:-0}

        echo "  Result: ${correct}/${total} (${acc_pct}%), cost=\$${total_cost}"

        # Append to CSV
        echo "${model},${k},${acc_pct},${correct},${total},${avg_turns},${avg_refs},${avg_cost},${total_cost},${avg_dur},${errors}" >> "${RESULTS_CSV}"
    done
done

echo ""
echo "=== Sweep Complete ==="
echo ""
echo "Results saved to: ${RESULTS_CSV}"
echo ""
# Print summary table
printf "%-7s | %2s | %8s | %8s | %5s | %5s | %s\n" "model" "k" "accuracy" "cost/img" "refs" "turns" "errors"
printf "%-7s-|-%2s-|-%8s-|-%8s-|-%5s-|-%5s-|-%s\n" "-------" "--" "--------" "--------" "-----" "-----" "------"
while IFS=, read -r model k acc correct total turns refs cost tcost dur errs; do
    [ "${model}" = "model" ] && continue
    printf "%-7s | %2s | %6s%%  | \$%-6s | %5s | %5s | %s\n" "${model}" "${k}" "${acc}" "${cost}" "${refs}" "${turns}" "${errs}"
done < "${RESULTS_CSV}"
