#!/usr/bin/env bash
# Run all unique evaluation configs and print results.
#
# Runs 14 unique (model, kb, k) combinations. Results are stored in
# results/open_agentic/{crop}/{kb}/{model}/k{k}/ and can be read by
# the results command without re-running.
#
# Usage:
#   cd /path/to/AgCrawler
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
#   set -a && source .env && set +a
#
#   bash CyberVisionAg/open_agentic/run_sweeps.sh run          # run all 14 configs
#   bash CyberVisionAg/open_agentic/run_sweeps.sh results      # print tables from stored results
#   bash CyberVisionAg/open_agentic/run_sweeps.sh run-missing   # only run configs without results

set -uo pipefail

EXCLUDE="Diaporthe_2015_Kanawha,Green_stem,Fusarium_healthy_vs_infected,Stem_Canker,Top_Dieback"
IMAGES=1   # test images per class (1=fast directional, 3=final paper)
PARALLEL=12
SEED=42
DATASET="Soybean_Diseases"
COMMAND="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/../results/open_agentic/${DATASET}"

# All 14 unique configs: model,kb,k
CONFIGS=(
    # Main table: sonnet × 3 KB × 4 k values = 12
    "sonnet,none,1"
    "sonnet,none,4"
    "sonnet,none,8"
    "sonnet,none,16"
    "sonnet,local,1"
    "sonnet,local,4"
    "sonnet,local,8"
    "sonnet,local,16"
    "sonnet,internet,1"
    "sonnet,internet,4"
    "sonnet,internet,8"    # shared with model ablation
    "sonnet,internet,16"
    # Model ablation: haiku + opus at internet/k=8 = 2 (sonnet already above)
    "haiku,internet,8"
    "opus,internet,8"
)

if [ -z "${COMMAND}" ]; then
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  run           Run all 14 configs"
    echo "  run-missing   Run only configs without existing results"
    echo "  results       Print paper tables from stored results"
    echo "  clean         Remove ALL results (fresh start)"
    echo "  status        Show which configs have results"
    exit 1
fi

run_single() {
    local model="$1" src="$2" k="$3"
    echo -n "  ${model} | ${src} | k=${k}: "
    OUTPUT=$(PYTHONUNBUFFERED=1 python -m CyberVisionAg.open_agentic.eval \
        --model "${model}" --symptom-source "${src}" \
        --images-per-class "${IMAGES}" --k "${k}" \
        --parallel "${PARALLEL}" --seed "${SEED}" \
        --exclude "${EXCLUDE}" 2>&1) || true

    acc=$(echo "${OUTPUT}" | grep "Accuracy" | head -1 | sed 's/.*: *//')
    refs=$(echo "${OUTPUT}" | grep "Avg refs" | head -1 | sed 's/.*: *//')
    cost=$(echo "${OUTPUT}" | grep "Total cost" | head -1 | sed 's/.*: *//')
    errors=$(echo "${OUTPUT}" | grep "Errors" | head -1 | sed 's/.*: *//')
    echo "${acc}  refs=${refs}  cost=${cost}  errors=${errors}"
}

has_results() {
    local model="$1" src="$2" k="$3"
    local dir="${RESULTS_BASE}/${src}/${model}/k${k}"
    [ -f "${dir}/summary.json" ]
}

read_accuracy() {
    local model="$1" src="$2" k="$3"
    local summary="${RESULTS_BASE}/${src}/${model}/k${k}/summary.json"
    if [ -f "${summary}" ]; then
        python3 -c "
import json
d = json.load(open('${summary}'))
m = d.get('metrics', d)
acc = m.get('accuracy', 0)
n = m.get('correct', 0)
t = m.get('total', 0)
print(f'{n}/{t} ({acc:.1f}%)')
" 2>/dev/null || echo "?"
    else
        echo "—"
    fi
}

# ── Run commands ──────────────────────────────────────────────────────────────
if [ "${COMMAND}" = "run" ] || [ "${COMMAND}" = "run-missing" ]; then
    echo "=== Running evaluation configs ==="
    echo "Dataset: ${DATASET}, images/class: ${IMAGES}, seed: ${SEED}"
    echo ""

    ran=0
    skipped=0
    for config in "${CONFIGS[@]}"; do
        IFS=',' read -r model src k <<< "${config}"
        if [ "${COMMAND}" = "run-missing" ] && has_results "${model}" "${src}" "${k}"; then
            echo "  ${model} | ${src} | k=${k}: EXISTS (skipping)"
            skipped=$((skipped + 1))
            continue
        fi
        run_single "${model}" "${src}" "${k}"
        ran=$((ran + 1))
    done

    echo ""
    echo "Ran: ${ran}, Skipped: ${skipped}"
    echo ""
fi

# ── Print results tables ─────────────────────────────────────────────────────
if [ "${COMMAND}" = "results" ] || [ "${COMMAND}" = "run" ] || [ "${COMMAND}" = "run-missing" ]; then
    echo "=== Table 1: Method × k (sonnet) ==="
    echo ""
    printf "%-20s | %15s | %15s | %15s | %15s\n" "Method" "k=1" "k=4" "k=8" "k=16"
    printf "%-20s-|-%15s-|-%15s-|-%15s-|-%15s\n" "--------------------" "---------------" "---------------" "---------------" "---------------"
    for src in none local internet; do
        label="Agent"
        [ "${src}" = "none" ] && label="Agent (no KB)"
        [ "${src}" = "local" ] && label="Agent + local KB"
        [ "${src}" = "internet" ] && label="Agent + internet KB"
        r1=$(read_accuracy sonnet "${src}" 1)
        r4=$(read_accuracy sonnet "${src}" 4)
        r8=$(read_accuracy sonnet "${src}" 8)
        r16=$(read_accuracy sonnet "${src}" 16)
        printf "%-20s | %15s | %15s | %15s | %15s\n" "${label}" "${r1}" "${r4}" "${r8}" "${r16}"
    done
    echo ""

    echo "=== Table 2: Model Ablation (internet KB, k=8) ==="
    echo ""
    printf "%-10s | %15s\n" "Model" "Accuracy"
    printf "%-10s-|-%15s\n" "----------" "---------------"
    for model in haiku sonnet opus; do
        r=$(read_accuracy "${model}" internet 8)
        printf "%-10s | %15s\n" "${model}" "${r}"
    done
    echo ""
fi

# ── Clean: remove all results ─────────────────────────────────────────────────
if [ "${COMMAND}" = "clean" ]; then
    echo "=== Cleaning all results for ${DATASET} ==="
    echo "This will delete: ${RESULTS_BASE}/"
    read -p "Are you sure? (y/N) " confirm
    if [ "${confirm}" = "y" ] || [ "${confirm}" = "Y" ]; then
        rm -rf "${RESULTS_BASE}"
        echo "Cleaned."
    else
        echo "Cancelled."
    fi
fi

# ── Status: show which configs have results ───────────────────────────────────
if [ "${COMMAND}" = "status" ]; then
    echo "=== Status: ${DATASET} ==="
    echo ""
    done_count=0
    missing_count=0
    for config in "${CONFIGS[@]}"; do
        IFS=',' read -r model src k <<< "${config}"
        if has_results "${model}" "${src}" "${k}"; then
            acc=$(read_accuracy "${model}" "${src}" "${k}")
            echo "  [done]    ${model} | ${src} | k=${k} → ${acc}"
            done_count=$((done_count + 1))
        else
            echo "  [missing] ${model} | ${src} | k=${k}"
            missing_count=$((missing_count + 1))
        fi
    done
    echo ""
    echo "Done: ${done_count}/14, Missing: ${missing_count}/14"
fi

echo "=== Done ==="
