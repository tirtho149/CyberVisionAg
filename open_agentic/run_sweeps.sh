#!/usr/bin/env bash
# Run evaluation experiments for the paper.
#
# Usage:
#   cd /path/to/AgCrawler
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
#   set -a && source .env && set +a
#   bash CyberVisionAg/open_agentic/run_sweeps.sh <experiment>
#
# Experiments:
#   main-table       Method (none/local/internet KB) × k (1,4,8,16) — 12 runs
#   model-ablation   Model (haiku/sonnet/opus) at k=8, internet KB — 3 runs
#   all              All of the above — 15 runs
#
# All experiments use: collage refs, seed=42, parallel=12, 27 soybean classes

set -uo pipefail

EXCLUDE="Diaporthe_2015_Kanawha,Green_stem,Fusarium_healthy_vs_infected,Stem_Canker,Top_Dieback"
IMAGES=3
PARALLEL=12
SEED=42
EXPERIMENT="${1:-}"

if [ -z "${EXPERIMENT}" ]; then
    echo "Usage: $0 <experiment>"
    echo ""
    echo "Experiments:"
    echo "  main-table       Method × k table (12 runs)"
    echo "  model-ablation   Model comparison at k=8 (3 runs)"
    echo "  all              Everything (15 runs)"
    exit 1
fi

run_eval() {
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

# ── Main results table: Method × k ───────────────────────────────────────────
run_main_table() {
    echo "=== Main Table: Method × k (sonnet, 27 classes × ${IMAGES} images) ==="
    echo ""
    for src in none local internet; do
        echo "--- KB: ${src} ---"
        for k in 1 4 8 16; do
            run_eval sonnet "${src}" "${k}"
        done
        echo ""
    done
}

# ── Model ablation: haiku/sonnet/opus at k=8 ─────────────────────────────────
run_model_ablation() {
    echo "=== Model Ablation: k=8, internet KB, 27 classes × ${IMAGES} images ==="
    echo ""
    for model in haiku sonnet opus; do
        run_eval "${model}" internet 8
    done
    echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${EXPERIMENT}" in
    main-table)
        run_main_table
        ;;
    model-ablation)
        run_model_ablation
        ;;
    all)
        run_main_table
        run_model_ablation
        ;;
    *)
        echo "Unknown experiment: ${EXPERIMENT}"
        echo "Options: main-table, model-ablation, all"
        exit 1
        ;;
esac

echo "=== Done ==="
