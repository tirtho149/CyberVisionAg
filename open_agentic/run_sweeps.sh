#!/usr/bin/env bash
# Run evaluation sweeps for any crop dataset.
#
# Usage:
#   cd /path/to/AgCrawler
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vl-reasoning
#   set -a && source .env && set +a
#
#   bash open_agentic/run_sweeps.sh <command> <crop> [<family>]
#
# Commands:
#   run-missing   Run only configs without existing results (resumable)
#   run           Run all configs (overwrites existing)
#   results       Print paper tables from stored results
#   status        Show which configs have results
#   clean         Remove ALL results for this crop (both families)
#
# Family (3rd arg, default "claude"):
#   claude   sonnet × {KB, k} sweep + haiku/opus ablations + few-shot baseline
#   gemini   gemini-flash + gemini-pro × {KB, k} sweep (no few-shot — agent only)
#
# Storage: results live under results/open_agentic/<DATASET>/<src>/<model>/k<k>/
# so claude and gemini runs land in disjoint subdirs and never clobber each other.
#
# Crops (all 10 in SageQualityChecker/crop_disease_subset.csv):
#   Prepared datasets exist:   soybean, corn, mango, tomato, sugarcane, banana, cauliflower, orange, coffee
#   Need data preparation:     wheat
#
# Quality filter: INCORRECT + QUESTIONABLE disease labels from
# SageQualityChecker/disease_label_subset_report.json are HARDCODED into each
# crop's EXCLUDE below (previously appended at runtime via quality_exclude.py).
# To refresh after a new judge run:
#   for c in Soybean Corn Mango Tomato Sugarcane Banana Cauliflower Coffee Orange Wheat; do
#     python3 open_agentic/quality_exclude.py --crop "$c" --existing "<manual-part>"
#   done
#
# Examples:
#   bash open_agentic/run_sweeps.sh run-missing soybean claude
#   bash open_agentic/run_sweeps.sh run-missing soybean gemini
#   bash open_agentic/run_sweeps.sh results corn gemini
#   bash open_agentic/run_sweeps.sh status sugarcane

set -uo pipefail

# Install required dependencies
python3 -m pip install --quiet Pillow opencv-python numpy pandas 2>/dev/null || true

IMAGES=5  # test images per class (1=fast directional, 3=final paper)
PARALLEL=6
SEED=42
COMMAND="${1:-}"
CROP="${2:-}"
FAMILY="${3:-claude}"

# ── Crop-specific settings ────────────────────────────────────────────────────
setup_crop() {
    # Prepared dataset paths (empty = use default Curated_Local_Dataset)
    REF_DIR=""
    TEST_DIR=""
    PART_INDEX=""
    # Crop name as it appears in disease_label_subset_report.json (Title Case).
    # Leave empty to skip quality-report filtering for this crop.
    QUALITY_CROP=""

    case "${CROP}" in
        soybean)
            DATASET="Soybean_Diseases"
            EXCLUDE="Diaporthe_2015_Kanawha,Green_stem,Fusarium_healthy_vs_infected,Stem_Canker,Top_Dieback,Diaporthe,Soybean_Dwarf_Mosaic_Virus,Bacterial_Pustule_Of_Soybean_Disease,Cercospora,Cercospora_Fungi,Diaporthe_Fungus,Fusarium_Disease,Herbicide_Injury,Iron_Deficiency_Chlorosis,Phomopsis,Phyllosticta_Leaf_Spot,Phytophthora,Potassium_Deficiency,Pythium,Pythium_Diseases,Rhizobium_Bacteria,Rhizoctonia,Root_And_Stem_Rot,Sclerotinia_Timber_Rot,Septoria_Leaf_Blotch,Soybean_Dwarf_Mosaic_Virus_2012,Wildfire_Of_Tobacco,Xylaria_Necrophora_Garcia-Aroca"
            KB_SOURCES=("none" "internet")
            # KB_SOURCES=("none" "local" "internet")  # local commented out: not needed for paper
            REF_DIR="Prepared_Dataset/Soybean"
            TEST_DIR="Prepared_Dataset/Soybean_test"
            PART_INDEX="Prepared_Dataset/Soybean/part_index.md"
            QUALITY_CROP="Soybean"
            ;;
        corn)
            DATASET="Corn_Diseases"
            EXCLUDE="Anthracnose_Ear_Infection,Leaf_Blight,Leaf_Spot,Maize_Lethal_Necrosis,Penicillium_On_Seedling,Pythium,Rhizoctonia,Rust,Smut,Ear_Rots,General_And_Mixed_Ear_Rots,General_And_Mixed_Stalk_Rots,Genetic_Flecking_Or_Striping,Genetic_Streaking,Diplodia,Chocolate_Spot,Barley_Yellow_Dwarf_Virus,Cladosporium_Ear_Rot,Crown_Rot,Damping_Off,Downy_Mildew,Maize_White_Line_Mosaic,Nigrospora_Ear_Rot,Penicillium_Ear_Rot,Pythium_Stalk_Rot,Rhizopus_Stolonifer,Root_Rot,Trichoderma_Stalk_Rot,Alternaria_Black_Molds_Stem_Cankers,Bacterial_Brown_Spot_Of_Beancanker_Of_Stone_Fruit,Bacterial_Rot_And_Blight,Brown_Spot,Cladosporium_Fungus,Dry_Rot_Of_Ears_And_Stalks_Of_Maize,Epicoccum_Fungus,Fusarium_Disease,Fusarium_Graminearum_Schwabe,Fusarium_Wilts,Gibberella_Disease,Goss,Green_Mold,Khuskia_Fungus,Pantoea,Penicillium_Fungi,Pythium_Diseases,Xanthomonas_Vasicola"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Corn"
            TEST_DIR="Prepared_Dataset/Corn_test"
            PART_INDEX="Prepared_Dataset/Corn/part_index.md"
            QUALITY_CROP="Corn"
            ;;
        mango)
            DATASET="Mango_Leaf_Disease"
            EXCLUDE="Bacterial_Canker,Bitter_Rot_And_Anthracnose,Fungus,Pestalotiopsis_Blight,Phaeoramularia_Fungus"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Mango_Leaf"
            TEST_DIR="Prepared_Dataset/Mango_Leaf_test"
            PART_INDEX="Prepared_Dataset/Mango_Leaf/part_index.md"
            IMAGES=5   # override: use all 10 test images per class (only 4 classes)
            QUALITY_CROP="Mango"
            ;;
        tomato)
            # Dataset: Data/Tomato (from selected_10crops_100_fast.zip, extracted 2026-04-26).
            # KB: outputs/Tomato/internet.xlsx (built for old Curated_Local_Dataset names;
            # 15 of 21 classes have no KB entry — prepare_dataset uses generic Claude knowledge).
            # Excluded: old quality-judge flags + classes with <10 images + 0-test-image class.
            # Evaluated: 20 classes (100 test images, 5/class).
            DATASET="Tomato_Diseases"
            EXCLUDE="Bacterial_Leaf_Spot,Leaf_Mosaic_Virus,Leaf_Yellow_Virus,Spider_Mites,Tomato_Leaf_Mould,Bitter_Rot_And_Anthracnose,Blossom_End_Rot,Brown_Spot,Clover_Proliferation_Phytoplasma,Diaporthe_Vexans_Gratz,Fusarium_Wilts,Phomopsis_Cankers_And_Twig_Blights,Phytoplasma,Remotididymella_Destructiva,Bacterial_Pith_Necrosis,Bacterial_Soft_Rot,Beet_Curly_Top_Virus,Fusarium_Damping-Off,Fusarium_Wilt,Phoma_Blight,Pythium_Diseases,Rhizoctonia_Damping-Off,Sclerotinia_Rots,Tomato_Leaf_Curl_Virus,Tomato_Ringspot_Virus,Alfalfa_Mosaic_Virus,Tomato_Necrotic_Dwarf_Virus,Phytophthora_Root_And_Crown_Rots"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Tomato"
            TEST_DIR="Prepared_Dataset/Tomato_test"
            PART_INDEX="Prepared_Dataset/Tomato/part_index.md"
            IMAGES=5  # override: use all 5 test images per class (20 classes = 100 total)
            QUALITY_CROP="Tomato"
            ;;
        sugarcane)
            # Dataset: sugarcane_only/Sugarcane (17 classes × 100 imgs). Prepared
            # via prepare_dataset (15 inspect, 3 ref/part, 5 test/class). Four
            # classes produced 0 test images and are excluded so the sweep won't
            # crash: Brown_Rust, Eye_Spot, Leaf_Mosaic_Virus, Smut.
            # Quality judge also drops: Banded_Chlorosis, Common_Rust, Dried_Leaves,
            # Leaf_Mosaic_Virus, Red_Spot — appended automatically via quality_exclude.py.
            DATASET="Sugarcane_Diseases"
            EXCLUDE="Brown_Rust,Eye_Spot,Leaf_Mosaic_Virus,Smut,Banded_Chlorosis,Common_Rust,Dried_Leaves,Red_Spot"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Sugarcane"
            TEST_DIR="Prepared_Dataset/Sugarcane_test"
            PART_INDEX="Prepared_Dataset/Sugarcane/part_index.md"
            QUALITY_CROP="Sugarcane"
            ;;
        banana)
            # Dataset: Data/Banana (16 classes). KB: outputs/Banana/internet.xlsx
            # (generated 2026-04-23). Prepared via prepare_dataset (15 inspect,
            # 3 ref/part, 5 test/class). Classes with 0 test images (too few
            # source imgs) are added to EXCLUDE so the sweep doesn't crash.
            DATASET="Banana_Diseases"
            EXCLUDE="Bitter_Rot_And_Anthracnose,Pestalotiopsis_Leaf_Spot,Phoma_Blight,Phyllosticta_Maculata,Pseudocercospora_Musae,Rust,Phaeoseptoria_Leaf_Spot,Phyllosticta_Leaf_Spot,Pseudocercospora_Leaf_Spot"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Banana"
            TEST_DIR="Prepared_Dataset/Banana_test"
            PART_INDEX="Prepared_Dataset/Banana/part_index.md"
            QUALITY_CROP="Banana"
            ;;
        cauliflower)
            # Dataset: Data/Cauliflower (5 classes). KB: outputs/Cauliflower/internet.xlsx
            # (generated 2026-04-23; 4/5 matched). Prepared via prepare_dataset
            # (15 inspect, 3 ref/part, 5 test/class). Quality judge drops
            # Bacterial_Spot_Rot (INCORRECT) → 4 eval classes.
            DATASET="Cauliflower_Diseases"
            EXCLUDE="Bacterial_Spot_Rot"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Cauliflower"
            TEST_DIR="Prepared_Dataset/Cauliflower_test"
            PART_INDEX="Prepared_Dataset/Cauliflower/part_index.md"
            QUALITY_CROP="Cauliflower"
            ;;
        coffee)
            # Dataset: Data/Coffee (7 raw classes, from selected_10crops_100_fast.zip).
            # KB: outputs/Coffee/internet.xlsx (generated 2026-04-24).
            # Excluded: Berry_Blotch/Black_Rot/Cerscospora (low-confidence KB, no usable images),
            # Miner/Phoma (excluded by quality judge). Leaves 2 classes: Brown_Eye_Spot, Rust.
            # Prepared via prepare_dataset (max-per-part 3, test-per-class 5, 2026-04-25).
            DATASET="Coffee_Diseases"
            EXCLUDE="Berry_Blotch,Black_Rot,Cerscospora,Miner,Phoma,Rust"
            KB_SOURCES=("none" "internet")
            IMAGES=5   # override: use all 10 test images per class (only 2 classes)
            REF_DIR="Prepared_Dataset/Coffee"
            TEST_DIR="Prepared_Dataset/Coffee_test"
            PART_INDEX="Prepared_Dataset/Coffee/part_index.md"
            QUALITY_CROP="Coffee"
            ;;
        orange)
            # Dataset: Data/Orange (7 raw classes, from selected_10crops_100_fast.zip).
            # KB: outputs/Orange/internet.xlsx (generated 2026-04-24; Huanglongbing
            # description relaxed after the original KB's strict "asymmetrical
            # blotchy mottle" wording rejected all 100 source images).
            # Quality judge flagged Whisker_Mold as QUESTIONABLE, but it has the
            # best image coverage so we keep it in the eval; the other quality
            # flags (Penicillium_Fungi/Fungus) have 0 usable data anyway.
            # Classes with 0 test images after prepare_dataset are added to
            # EXCLUDE so the sweep won't crash.
            DATASET="Orange_Diseases"
            EXCLUDE="Penicillium_Fungi,Penicillium_Fungus,Blue_Mold,Citrus_Blight_Disease,Green_Mold"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Orange"
            TEST_DIR="Prepared_Dataset/Orange_test"
            PART_INDEX="Prepared_Dataset/Orange/part_index.md"
            QUALITY_CROP="Orange"
            ;;
        wheat)
            DATASET="Wheat_Diseases"
            EXCLUDE="Alternaria_Black_Molds_Stem_Cankers,Bacterial_Brown_Spot_Of_Beancanker_Of_Stone_Fruit,Blumeria_Graminis_F,Cochliobolus_Leaf_Spot,Downy_Mildew,Fusarium_Disease,Fusarium_Graminearum_Schwabe,Fusarium_Wilts,Leaf_Spot,Leaf_Streakblack_Chaff,Parastagonospora_Nodorum,Penicillium_Fungi,Phytophthora_Root_And_Crown_Rots,Resistance_Phenotype,Root_Rot,Rust,Septoria_Leaf_Spot_And_Cankers,Smut,Sooty_Mold,Zymoseptoria_Tritici"
            KB_SOURCES=("none" "internet")
            REF_DIR="Prepared_Dataset/Wheat"
            TEST_DIR="Prepared_Dataset/Wheat_test"
            PART_INDEX="Prepared_Dataset/Wheat/part_index.md"
            QUALITY_CROP="Wheat"
            ;;
        *)
            echo "Unknown crop: ${CROP}"
            echo "Available: soybean, corn, mango, tomato, sugarcane, banana, cauliflower, coffee, orange, wheat"
            exit 1
            ;;
    esac

    # Quality-judge excludes (INCORRECT + QUESTIONABLE) are hardcoded into
    # each crop's EXCLUDE above. QUALITY_CROP is kept for reference only.
}

if [ -z "${COMMAND}" ] || [ -z "${CROP}" ]; then
    echo "Usage: $0 <command> <crop> [<family>]"
    echo ""
    echo "Commands: run, run-missing, results, status, clean"
    echo "Crops:    soybean, corn, mango, tomato, sugarcane,"
    echo "          banana, cauliflower, coffee, orange, wheat"
    echo "Family:   claude (default) | gemini"
    echo ""
    echo "Examples:"
    echo "  $0 run-missing soybean             # claude (default)"
    echo "  $0 run-missing soybean gemini      # gemini-flash + gemini-pro sweep"
    echo "  $0 results corn gemini"
    exit 1
fi

if [ "${FAMILY}" != "claude" ] && [ "${FAMILY}" != "gemini" ]; then
    echo "Unknown family: ${FAMILY}. Use 'claude' or 'gemini'."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
setup_crop

RESULTS_BASE="${SCRIPT_DIR}/../results/open_agentic/${DATASET}"

# Build configs dynamically based on family and the crop's KB sources.
AGENTIC_CONFIGS=()
if [ "${FAMILY}" = "claude" ]; then
    # Primary sweep: sonnet, haiku, opus × all KB × all k.
    for model in "sonnet" "haiku" "opus"; do
        for src in "${KB_SOURCES[@]}"; do
            for k in 0 1 4 8 16; do
                AGENTIC_CONFIGS+=("${model},${src},${k}")
            done
        done
    done
    FEWSHOT_K_VALUES=()
else
    # Gemini family: flash + pro × all KB × all k. Thinking budget overridden
    # via .gemini/settings.json (0 for Flash, 128 min for Pro).
    for src in "${KB_SOURCES[@]}"; do
        for k in 0 1 4 8 16; do
            AGENTIC_CONFIGS+=("gemini-flash,${src},${k}")
            AGENTIC_CONFIGS+=("gemini-pro,${src},${k}")
        done
    done
    # few_shot.py is anthropic-only, so no few-shot baseline for gemini.
    FEWSHOT_K_VALUES=()
fi

# ── Helper functions ──────────────────────────────────────────────────────────
run_single() {
    local model="$1" src="$2" k="$3"
    echo -n "  ${model} | ${src} | k=${k}: "
    local extra_flags=""
    [ -n "${REF_DIR}" ] && extra_flags+=" --ref-dir ${REF_DIR}"
    [ -n "${TEST_DIR}" ] && extra_flags+=" --test-dir ${TEST_DIR}"
    # Part index only when KB is present (it's derived from KB-guided filtering)
    [ -n "${PART_INDEX}" ] && [ "${src}" != "none" ] && extra_flags+=" --part-index ${PART_INDEX}"
    OUTPUT=$(PYTHONUNBUFFERED=1 python3 -m open_agentic.eval \
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
    OUTPUT=$(PYTHONUNBUFFERED=1 python3 -m open_agentic.few_shot \
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
    echo "=== Running: ${DATASET} | family: ${FAMILY} (images/class: ${IMAGES}, seed: ${SEED}) ==="
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

    if [ "${#FEWSHOT_K_VALUES[@]}" -gt 0 ]; then
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
    fi

    echo ""
    echo "Ran: ${ran}, Skipped: ${skipped}"
    echo ""
fi

# ── Results ───────────────────────────────────────────────────────────────────
if [ "${COMMAND}" = "results" ] || [ "${COMMAND}" = "run" ] || [ "${COMMAND}" = "run-missing" ]; then
    if [ "${FAMILY}" = "claude" ]; then
        TABLE1_MODELS=("sonnet")
        TABLE2_MODELS=("haiku" "sonnet" "opus")
    else
        TABLE1_MODELS=("gemini-flash" "gemini-pro")
        TABLE2_MODELS=()  # subsumed by Table 1 for gemini
    fi

    echo "=== Table 1: Method × k — ${DATASET} (${FAMILY}) ==="
    echo ""
    printf "%-28s | %15s | %15s | %15s | %15s | %15s\n" "Method" "k=0" "k=1" "k=4" "k=8" "k=16"
    printf "%-28s-|-%15s-|-%15s-|-%15s-|-%15s-|-%15s\n" "----------------------------" "---------------" "---------------" "---------------" "---------------" "---------------"
    # Few-shot (claude family only; few_shot.py is anthropic-only)
    if [ "${FAMILY}" = "claude" ]; then
        fr0=$(read_accuracy sonnet "few_shot" 0)
        fr1=$(read_accuracy sonnet "few_shot" 1)
        fr4=$(read_accuracy sonnet "few_shot" 4)
        fr8=$(read_accuracy sonnet "few_shot" 8)
        fr16=$(read_accuracy sonnet "few_shot" 16)
        printf "%-28s | %15s | %15s | %15s | %15s | %15s\n" "Few-shot baseline" "${fr0}" "${fr1}" "${fr4}" "${fr8}" "${fr16}"
    fi
    # Agentic rows: model × KB-source
    for model in "${TABLE1_MODELS[@]}"; do
        for src in "${KB_SOURCES[@]}"; do
            kb_label="no KB"
            [ "${src}" = "local" ] && kb_label="local KB"
            [ "${src}" = "internet" ] && kb_label="internet KB"
            label="Agent ${model} (${kb_label})"
            r0=$(read_accuracy "${model}" "${src}" 0)
            r1=$(read_accuracy "${model}" "${src}" 1)
            r4=$(read_accuracy "${model}" "${src}" 4)
            r8=$(read_accuracy "${model}" "${src}" 8)
            r16=$(read_accuracy "${model}" "${src}" 16)
            printf "%-28s | %15s | %15s | %15s | %15s | %15s\n" "${label}" "${r0}" "${r1}" "${r4}" "${r8}" "${r16}"
        done
    done
    echo ""

    if [ "${#TABLE2_MODELS[@]}" -gt 0 ]; then
        echo "=== Table 2: Model Ablation — ${DATASET} (internet KB, k=8) ==="
        echo ""
        printf "%-12s | %15s\n" "Model" "Accuracy"
        printf "%-12s-|-%15s\n" "------------" "---------------"
        for model in "${TABLE2_MODELS[@]}"; do
            r=$(read_accuracy "${model}" internet 8)
            printf "%-12s | %15s\n" "${model}" "${r}"
        done
        echo ""
    fi
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
    echo "=== Status: ${DATASET} | family: ${FAMILY} ==="
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
    if [ "${#FEWSHOT_K_VALUES[@]}" -gt 0 ]; then
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
    fi
    total=$((done_count + missing_count))
    echo ""
    echo "Done: ${done_count}/${total}, Missing: ${missing_count}/${total}"
fi

echo "=== Done ==="
