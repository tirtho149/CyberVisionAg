#!/usr/bin/env bash
# Run all evaluation sweeps for the paper.
# Each sweep varies one factor while holding others at default.
#
# Defaults: sonnet, internet KB, k=8, collage, seed=42, 27 classes × 3 imgs
#
# Usage:
#   cd /path/to/AgCrawler
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
#   set -a && source .env && set +a
#   bash CyberVisionAg/open_agentic/run_sweeps.sh [sweep_number]
#
# Examples:
#   bash CyberVisionAg/open_agentic/run_sweeps.sh        # run all sweeps
#   bash CyberVisionAg/open_agentic/run_sweeps.sh 1      # k sweep only
#   bash CyberVisionAg/open_agentic/run_sweeps.sh 2      # KB sweep only
#   bash CyberVisionAg/open_agentic/run_sweeps.sh 3      # model sweep only

set -uo pipefail

EXCLUDE="Diaporthe_2015_Kanawha,Green_stem,Fusarium_healthy_vs_infected,Stem_Canker,Top_Dieback"
IMAGES_PER_CLASS=3
PARALLEL=12
SEED=42
SWEEP_FILTER="${1:-all}"

run_eval() {
    local model="$1" src="$2" k="$3"
    echo -n "  model=${model} src=${src} k=${k}: "
    OUTPUT=$(PYTHONUNBUFFERED=1 python -m CyberVisionAg.open_agentic.eval \
        --model "${model}" --symptom-source "${src}" \
        --images-per-class "${IMAGES_PER_CLASS}" --k "${k}" \
        --parallel "${PARALLEL}" --seed "${SEED}" \
        --exclude "${EXCLUDE}" 2>&1) || true

    acc=$(echo "${OUTPUT}" | grep "Accuracy" | head -1 | sed 's/.*: *//')
    refs=$(echo "${OUTPUT}" | grep "Avg refs" | head -1 | sed 's/.*: *//')
    cost=$(echo "${OUTPUT}" | grep "Total cost" | head -1 | sed 's/.*: *//')
    errors=$(echo "${OUTPUT}" | grep "Errors" | head -1 | sed 's/.*: *//')
    echo "acc=${acc}  refs=${refs}  cost=${cost}  errors=${errors}"
}

# ── Sweep 1: k sweep (sonnet, internet) ──────────────────────────────────────
if [ "${SWEEP_FILTER}" = "all" ] || [ "${SWEEP_FILTER}" = "1" ]; then
    echo "=== Sweep 1: k sweep (sonnet, internet KB) ==="
    for k in 1 2 4 8 16 27; do
        run_eval sonnet internet "${k}"
    done
    echo ""
fi

# ── Sweep 2: KB sweep (sonnet, k=8) ──────────────────────────────────────────
if [ "${SWEEP_FILTER}" = "all" ] || [ "${SWEEP_FILTER}" = "2" ]; then
    echo "=== Sweep 2: KB sweep (sonnet, k=8) ==="
    for src in none local internet; do
        run_eval sonnet "${src}" 8
    done
    echo ""
fi

# ── Sweep 3: Model sweep (internet, k=8) ─────────────────────────────────────
if [ "${SWEEP_FILTER}" = "all" ] || [ "${SWEEP_FILTER}" = "3" ]; then
    echo "=== Sweep 3: Model sweep (internet KB, k=8) ==="
    for model in haiku sonnet opus; do
        run_eval "${model}" internet 8
    done
    echo ""
fi

echo "=== All requested sweeps complete ==="
