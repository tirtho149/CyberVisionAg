#!/usr/bin/env bash
# Run evaluation sweeps for any crop dataset.
#
# Usage:
#   cd /path/to/AgCrawler
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
#   set -a && source .env && set +a
#
#   bash CyberVisionAg/open_agentic/run_sweeps.sh <command> <crop>
#
# Commands:
#   run-missing   Run only configs without existing results (resumable)
#   run           Run all configs (overwrites existing)
#   results       Print paper tables from stored results
#   status        Show which configs have results
#   clean         Remove ALL results for this crop
#
# Crops:
#   soybean       Soybean_Diseases (27 clean classes, KB: none/local/internet)
#   corn          Corn_Diseases (31 clean classes, KB: none/internet)
#
# Examples:
#   bash CyberVisionAg/open_agentic/run_sweeps.sh run-missing soybean
#   bash CyberVisionAg/open_agentic/run_sweeps.sh results corn
#   bash CyberVisionAg/open_agentic/run_sweeps.sh status soybean

set -uo pipefail

IMAGES=3   # test images per class (1=fast directional, 3=final paper)
PARALLEL=12
SEED=42
COMMAND="${1:-}"
CROP="${2:-}"

# ── Crop-specific settings ────────────────────────────────────────────────────
setup_crop() {
    # Prepared dataset paths (empty = use default Curated_Local_Dataset)
    REF_DIR=""
    TEST_DIR=""
    PART_INDEX=""

    case "${CROP}" in
        soybean)
            DATASET="Soybean_Diseases"
            EXCLUDE="Diaporthe_2015_Kanawha,Green_stem,Fusarium_healthy_vs_infected,Stem_Canker,Top_Dieback,Diaporthe,Soybean_Dwarf_Mosaic_Virus"
            KB_SOURCES=("none" "internet")
            # KB_SOURCES=("none" "local" "internet")  # local commented out: not needed for paper
            REF_DIR="CyberVisionAg/Prepared_Dataset/Soybean"
            TEST_DIR="CyberVisionAg/Prepared_Dataset/Soybean_test"
            PART_INDEX="CyberVisionAg/Prepared_Dataset/Soybean/part_index.md"
            ;;
        corn)
            DATASET="Corn_Diseases"
            EXCLUDE="Anthracnose_Ear_Infection,Leaf_Blight,Leaf_Spot,Maize_Lethal_Necrosis,Penicillium_On_Seedling,Pythium,Rhizoctonia,Rust,Smut,Ear_Rots,General_And_Mixed_Ear_Rots,General_And_Mixed_Stalk_Rots,Genetic_Flecking_Or_Striping,Genetic_Streaking,Diplodia,Chocolate_Spot,Barley_Yellow_Dwarf_Virus,Cladosporium_Ear_Rot,Crown_Rot,Damping_Off,Downy_Mildew,Maize_White_Line_Mosaic,Nigrospora_Ear_Rot,Penicillium_Ear_Rot,Pythium_Stalk_Rot,Rhizopus_Stolonifer,Root_Rot,Trichoderma_Stalk_Rot"
            KB_SOURCES=("none" "internet")
            REF_DIR="CyberVisionAg/Prepared_Dataset/Corn"
            TEST_DIR="CyberVisionAg/Prepared_Dataset/Corn_test"
            PART_INDEX="CyberVisionAg/Prepared_Dataset/Corn/part_index.md"
            ;;
        mango)
            DATASET="Mango_Leaf_Disease"
            EXCLUDE="Bacterial_Canker"
            KB_SOURCES=("none" "internet")
            REF_DIR="CyberVisionAg/Prepared_Dataset/Mango_Leaf"
            TEST_DIR="CyberVisionAg/Prepared_Dataset/Mango_Leaf_test"
            PART_INDEX="CyberVisionAg/Prepared_Dataset/Mango_Leaf/part_index.md"
            IMAGES=10  # override: use all 10 test images per class (only 4 classes)
            ;;
        tomato)
            DATASET="Tomato_Diseases"
            EXCLUDE="Bacterial_Leaf_Spot,Leaf_Bacterial_Spot,Early_Blight_Leaf,Leaf_Late_Blight,Mold_Leaf,Leaf_Mosaic_Virus,Leaf_Yellow_Virus,Spider_Mites_Two-Spotted_Spider_Mite,Two_Spotted_Spider_Mites_Leaf,Yellow_Leaf_Curl_Virus"
            KB_SOURCES=("none" "internet")
            ;;
        *)
            echo "Unknown crop: ${CROP}"
            echo "Available: soybean, corn, mango, tomato"
            exit 1
            ;;
    esac
}

if [ -z "${COMMAND}" ] || [ -z "${CROP}" ]; then
    echo "Usage: $0 <command> <crop>"
    echo ""
    echo "Commands: run, run-missing, results, status, clean"
    echo "Crops:    soybean, corn"
    echo ""
    echo "Examples:"
    echo "  $0 run-missing soybean"
    echo "  $0 results corn"
    exit 1
fi

setup_crop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/../results/open_agentic/${DATASET}"

# Build configs dynamically based on crop's KB sources
AGENTIC_CONFIGS=()
for src in "${KB_SOURCES[@]}"; do
    for k in 0 1 4 8 16; do
        AGENTIC_CONFIGS+=("sonnet,${src},${k}")
    done
done
# Model ablation: haiku + opus at internet/k=8
AGENTIC_CONFIGS+=("haiku,internet,8")
AGENTIC_CONFIGS+=("opus,internet,8")
# Gemini ablation: flash + pro at internet/k=8 (thinking budget overridden via
# CyberVisionAg/.gemini/settings.json — 0 for Flash, 128 min for Pro).
AGENTIC_CONFIGS+=("gemini-flash,internet,8")
AGENTIC_CONFIGS+=("gemini-pro,internet,8")
# Deduplicate (sonnet,internet,8 already exists from the loop)

FEWSHOT_K_VALUES=(0 1 4 8 16)

# ── Helper functions ──────────────────────────────────────────────────────────
run_single() {
    local model="$1" src="$2" k="$3"
    echo -n "  ${model} | ${src} | k=${k}: "
    local extra_flags=""
    [ -n "${REF_DIR}" ] && extra_flags+=" --ref-dir ${REF_DIR}"
    [ -n "${TEST_DIR}" ] && extra_flags+=" --test-dir ${TEST_DIR}"
    # Part index only when KB is present (it's derived from KB-guided filtering)
    [ -n "${PART_INDEX}" ] && [ "${src}" != "none" ] && extra_flags+=" --part-index ${PART_INDEX}"
    OUTPUT=$(PYTHONUNBUFFERED=1 python -m CyberVisionAg.open_agentic.eval \
        --model "${model}" --symptom-source "${src}" \
        --dataset "${DATASET}" \
        --images-per-class "${IMAGES}" --k "${k}" \
        --parallel "${PARALLEL}" --seed "${SEED}" \
        --exclude "${EXCLUDE}" \
        --no-collage --refs-per-class 3 \
        ${extra_flags} 2>&1) || true

    acc=$(echo "${OUTPUT}" | grep "Accuracy" | head -1 | sed 's/.*: *//')
    refs=$(echo "${OUTPUT}" | grep "Avg refs" | head -1 | sed 's/.*: *//')
    cost=$(echo "${OUTPUT}" | grep "Total cost" | head -1 | sed 's/.*: *//')
    errors=$(echo "${OUTPUT}" | grep "Errors" | head -1 | sed 's/.*: *//')
    echo "${acc}  refs=${refs}  cost=${cost}  errors=${errors}"
}

run_fewshot() {
    local k="$1"
    echo -n "  few-shot | k=${k}: "
    local extra_flags=""
    [ -n "${REF_DIR}" ] && extra_flags+=" --ref-dir ${REF_DIR}"
    [ -n "${TEST_DIR}" ] && extra_flags+=" --test-dir ${TEST_DIR}"
    OUTPUT=$(PYTHONUNBUFFERED=1 python -m CyberVisionAg.open_agentic.few_shot \
        --dataset "${DATASET}" \
        --images-per-class "${IMAGES}" --k "${k}" \
        --parallel "${PARALLEL}" --seed "${SEED}" \
        --exclude "${EXCLUDE}" ${extra_flags} 2>&1) || true

    acc=$(echo "${OUTPUT}" | grep "Accuracy" | head -1 | sed 's/.*: *//')
    cost=$(echo "${OUTPUT}" | grep "Total cost" | head -1 | sed 's/.*: *//')
    errors=$(echo "${OUTPUT}" | grep "Errors" | head -1 | sed 's/.*: *//')
    echo "${acc}  cost=${cost}  errors=${errors}"
}

has_results() {
    local model="$1" src="$2" k="$3"
    [ -f "${RESULTS_BASE}/${src}/${model}/k${k}/summary.json" ]
}

has_fewshot_results() {
    local k="$1"
    [ -f "${RESULTS_BASE}/few_shot/sonnet/k${k}/summary.json" ]
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

# ── Run ───────────────────────────────────────────────────────────────────────
if [ "${COMMAND}" = "run" ] || [ "${COMMAND}" = "run-missing" ]; then
    echo "=== Running: ${DATASET} (images/class: ${IMAGES}, seed: ${SEED}) ==="
    echo ""

    ran=0
    skipped=0

    echo "--- Agentic configs ---"
    seen=()
    for config in "${AGENTIC_CONFIGS[@]}"; do
        # Deduplicate
        if [[ " ${seen[*]:-} " == *" ${config} "* ]]; then continue; fi
        seen+=("${config}")

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
    echo "--- Few-shot baseline ---"
    for k in "${FEWSHOT_K_VALUES[@]}"; do
        if [ "${COMMAND}" = "run-missing" ] && has_fewshot_results "${k}"; then
            echo "  few-shot | k=${k}: EXISTS (skipping)"
            skipped=$((skipped + 1))
            continue
        fi
        run_fewshot "${k}"
        ran=$((ran + 1))
    done

    echo ""
    echo "Ran: ${ran}, Skipped: ${skipped}"
    echo ""
fi

# ── Results ───────────────────────────────────────────────────────────────────
if [ "${COMMAND}" = "results" ] || [ "${COMMAND}" = "run" ] || [ "${COMMAND}" = "run-missing" ]; then
    echo "=== Table 1: Method × k — ${DATASET} (sonnet) ==="
    echo ""
    printf "%-20s | %15s | %15s | %15s | %15s | %15s\n" "Method" "k=0" "k=1" "k=4" "k=8" "k=16"
    printf "%-20s-|-%15s-|-%15s-|-%15s-|-%15s-|-%15s\n" "--------------------" "---------------" "---------------" "---------------" "---------------" "---------------"
    # Few-shot
    fr0=$(read_accuracy sonnet "few_shot" 0)
    fr1=$(read_accuracy sonnet "few_shot" 1)
    fr4=$(read_accuracy sonnet "few_shot" 4)
    fr8=$(read_accuracy sonnet "few_shot" 8)
    fr16=$(read_accuracy sonnet "few_shot" 16)
    printf "%-20s | %15s | %15s | %15s | %15s | %15s\n" "Few-shot baseline" "${fr0}" "${fr1}" "${fr4}" "${fr8}" "${fr16}"
    # Agentic by KB source
    for src in "${KB_SOURCES[@]}"; do
        label="Agent (no KB)"
        [ "${src}" = "local" ] && label="Agent + local KB"
        [ "${src}" = "internet" ] && label="Agent + internet KB"
        r0=$(read_accuracy sonnet "${src}" 0)
        r1=$(read_accuracy sonnet "${src}" 1)
        r4=$(read_accuracy sonnet "${src}" 4)
        r8=$(read_accuracy sonnet "${src}" 8)
        r16=$(read_accuracy sonnet "${src}" 16)
        printf "%-20s | %15s | %15s | %15s | %15s | %15s\n" "${label}" "${r0}" "${r1}" "${r4}" "${r8}" "${r16}"
    done
    echo ""

    echo "=== Table 2: Model Ablation — ${DATASET} (internet KB, k=8) ==="
    echo ""
    printf "%-12s | %15s\n" "Model" "Accuracy"
    printf "%-12s-|-%15s\n" "------------" "---------------"
    for model in haiku sonnet opus gemini-flash gemini-pro; do
        r=$(read_accuracy "${model}" internet 8)
        printf "%-12s | %15s\n" "${model}" "${r}"
    done
    echo ""
fi

# ── Clean ─────────────────────────────────────────────────────────────────────
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

# ── Status ────────────────────────────────────────────────────────────────────
if [ "${COMMAND}" = "status" ]; then
    echo "=== Status: ${DATASET} ==="
    echo ""
    done_count=0
    missing_count=0

    echo "--- Agentic ---"
    seen=()
    for config in "${AGENTIC_CONFIGS[@]}"; do
        if [[ " ${seen[*]:-} " == *" ${config} "* ]]; then continue; fi
        seen+=("${config}")
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
    echo "--- Few-shot ---"
    for k in "${FEWSHOT_K_VALUES[@]}"; do
        if has_fewshot_results "${k}"; then
            acc=$(read_accuracy sonnet "few_shot" "${k}")
            echo "  [done]    few-shot | k=${k} → ${acc}"
            done_count=$((done_count + 1))
        else
            echo "  [missing] few-shot | k=${k}"
            missing_count=$((missing_count + 1))
        fi
    done
    total=$((done_count + missing_count))
    echo ""
    echo "Done: ${done_count}/${total}, Missing: ${missing_count}/${total}"
fi

echo "=== Done ==="
