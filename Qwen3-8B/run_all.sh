#!/bin/bash
# ============================================================================
# GSM8K Pipeline: Full MemGen Training & Evaluation
# ============================================================================
# This script runs the complete MemGen pipeline for Qwen3-8B:
#   0. Vanilla baseline evaluation (no MemGen)
#   1. Weaver SFT training
#   2. Weaver evaluation
#   3. Trigger GRPO training (optional, use --with-trigger)
#   4. Trigger evaluation (optional)
#
# Usage:
#   ./run_all.sh              # Weaver only (default)
#   ./run_all.sh --with-trigger   # Full pipeline with Trigger
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
OUTPUT_ROOT="${PROJECT_ROOT}/results"

# Parse arguments
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
echo "MemGen GSM8K Pipeline - Qwen3-8B"
echo "============================================"
echo "Trigger training: $([ "$WITH_TRIGGER" = true ] && echo "ENABLED" || echo "DISABLED")"
echo "============================================"
echo ""

# Step 0: Vanilla baseline
echo "[Step 0] Running vanilla baseline evaluation..."
bash ${SCRIPT_DIR}/00_vanilla_eval.sh
echo ""

# Step 1: Weaver SFT
echo "[Step 1] Running Weaver SFT training..."
bash ${SCRIPT_DIR}/01_weaver_sft.sh
echo ""

# Step 2: Evaluate Weaver
echo "[Step 2] Evaluating Weaver..."
bash ${SCRIPT_DIR}/02_eval_weaver.sh
echo ""

if [ "$WITH_TRIGGER" = true ]; then
    # Step 3: Trigger training
    echo "[Step 3] Running Trigger GRPO training..."
    bash ${SCRIPT_DIR}/03_trigger_train.sh
    echo ""

    # Step 4: Evaluate Trigger
    echo "[Step 4] Evaluating full MemGen (Weaver + Trigger)..."
    bash ${SCRIPT_DIR}/04_eval_trigger.sh
    echo ""
fi

echo "============================================"
echo "Pipeline Completed!"
echo "============================================"
echo ""
echo "Results location: ${OUTPUT_ROOT}/"
echo "  - Vanilla eval: ${OUTPUT_ROOT}/evaluate/gsm8k/*/evaluate/answer.json"
echo "  - Weaver eval:  ${OUTPUT_ROOT}/evaluate/gsm8k/*/evaluate/answer.json"
if [ "$WITH_TRIGGER" = true ]; then
    echo "  - Trigger eval: ${OUTPUT_ROOT}/evaluate/gsm8k/*/evaluate/answer.json"
fi
echo ""
echo "Logs: ${SCRIPT_DIR}/logs/"
echo "============================================"
