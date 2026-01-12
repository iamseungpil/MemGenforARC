#!/bin/bash
# ============================================================================
# GSM8K Pipeline: Full MemGen Training & Evaluation - SmolLM3-3B
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
OUTPUT_ROOT="${PROJECT_ROOT}/results"

WITH_TRIGGER=false
for arg in "$@"; do
    case $arg in
        --with-trigger)
            WITH_TRIGGER=true
            shift
            ;;
    esac
done

echo "============================================"
echo "MemGen GSM8K Pipeline - SmolLM3-3B"
echo "Trigger training: $([ "$WITH_TRIGGER" = true ] && echo "ENABLED" || echo "DISABLED")"
echo "============================================"

echo "[Step 0] Running vanilla baseline evaluation..."
bash ${SCRIPT_DIR}/00_vanilla_eval.sh

echo "[Step 1] Running Weaver SFT training..."
bash ${SCRIPT_DIR}/01_weaver_sft.sh

echo "[Step 2] Evaluating Weaver..."
bash ${SCRIPT_DIR}/02_eval_weaver.sh

if [ "$WITH_TRIGGER" = true ]; then
    echo "[Step 3] Running Trigger GRPO training..."
    bash ${SCRIPT_DIR}/03_trigger_train.sh

    echo "[Step 4] Evaluating full MemGen..."
    bash ${SCRIPT_DIR}/04_eval_trigger.sh
fi

echo "============================================"
echo "Pipeline Completed!"
echo "Results: ${OUTPUT_ROOT}/"
echo "Logs: ${SCRIPT_DIR}/logs/"
echo "============================================"
