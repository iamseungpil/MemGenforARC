#!/bin/bash
# ============================================================================
# GSM8K Pipeline: Full Automated Run (Extended with Weaver GRPO)
# ============================================================================
# This script runs the complete MemGen pipeline:
#   1. Weaver SFT pretraining (warmup)
#   2. Weaver GRPO training (RL fine-tuning) [OPTIONAL]
#   3. Weaver evaluation
#   4. Trigger GRPO training
#   5. Trigger evaluation
#   6. LTPO test-time optimization + evaluation
#   7. LTPO hyperparameter sweep [OPTIONAL]
#
# The script automatically finds checkpoint paths between stages using
# common.sh utilities.
#
# Options:
#   --skip-weaver-grpo    Skip Weaver GRPO training (use SFT only)
#   --skip-ltpo-sweep     Skip LTPO hyperparameter sweep
#   --grpo-only           Run only Weaver GRPO (requires existing SFT checkpoint)
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Parse command line arguments
SKIP_WEAVER_GRPO=false
SKIP_LTPO_SWEEP=true  # Default: skip sweep (it's time-consuming)
GRPO_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-weaver-grpo)
            SKIP_WEAVER_GRPO=true
            shift
            ;;
        --skip-ltpo-sweep)
            SKIP_LTPO_SWEEP=true
            shift
            ;;
        --with-ltpo-sweep)
            SKIP_LTPO_SWEEP=false
            shift
            ;;
        --grpo-only)
            GRPO_ONLY=true
            shift
            ;;
        *)
            ;;
    esac
done

# Wandb configuration
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"

PROJECT_ROOT=~/MemGenforARC

# Model configuration
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME_SAFE=$(get_model_name_safe "${MODEL_NAME}")
DATASET_NAME="gsm8k"

cd ${PROJECT_ROOT}
mkdir -p ${SCRIPT_DIR}/logs

# Determine total steps based on options
if [ "$GRPO_ONLY" = true ]; then
    TOTAL_STEPS=2
    echo "============================================"
    echo "GSM8K Pipeline: GRPO Only Mode"
    echo "============================================"
else
    TOTAL_STEPS=6
    if [ "$SKIP_WEAVER_GRPO" = false ]; then
        TOTAL_STEPS=$((TOTAL_STEPS + 1))
    fi
    if [ "$SKIP_LTPO_SWEEP" = false ]; then
        TOTAL_STEPS=$((TOTAL_STEPS + 1))
    fi
    echo "============================================"
    echo "GSM8K Full Pipeline Execution"
    echo "============================================"
fi

echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${OUTPUT_ROOT}"
echo "Options:"
echo "  - Weaver GRPO: $([ "$SKIP_WEAVER_GRPO" = false ] && echo "ENABLED" || echo "DISABLED")"
echo "  - LTPO Sweep: $([ "$SKIP_LTPO_SWEEP" = false ] && echo "ENABLED" || echo "DISABLED")"
echo "Total steps: ${TOTAL_STEPS}"
echo "============================================"
echo ""

CURRENT_STEP=0

# ============================================================================
# GRPO Only Mode: Just run Weaver GRPO and eval
# ============================================================================
if [ "$GRPO_ONLY" = true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Running Weaver GRPO Training..."
    bash ${SCRIPT_DIR}/02_weaver_grpo.sh 2>&1 | tee ${SCRIPT_DIR}/logs/02_weaver_grpo_full.log

    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Evaluating Weaver (GRPO)..."
    bash ${SCRIPT_DIR}/02_eval_weaver.sh 2>&1 | tee ${SCRIPT_DIR}/logs/02_eval_weaver_grpo_full.log

    echo "============================================"
    echo "GRPO Only Pipeline Completed!"
    echo "============================================"
    exit 0
fi

# ============================================================================
# Step 1: Weaver SFT Pretraining
# ============================================================================
CURRENT_STEP=$((CURRENT_STEP + 1))
echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Running Weaver SFT Pretraining..."
bash ${SCRIPT_DIR}/01_weaver_pretrain.sh 2>&1 | tee ${SCRIPT_DIR}/logs/01_weaver_pretrain_full.log

# Find the weaver checkpoint using common.sh utility
WEAVER_CHECKPOINT=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
if [ -z "$WEAVER_CHECKPOINT" ]; then
    echo "ERROR: Weaver checkpoint not found after training!"
    echo "Expected location: ${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_NAME_SAFE}/pn=*/weaver/weaver_lora"
    exit 1
fi

echo "Found Weaver (SFT) checkpoint: ${WEAVER_CHECKPOINT}"
echo ""

# ============================================================================
# Step 2: Weaver GRPO Training (Optional)
# ============================================================================
if [ "$SKIP_WEAVER_GRPO" = false ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Running Weaver GRPO Training..."
    bash ${SCRIPT_DIR}/02_weaver_grpo.sh 2>&1 | tee ${SCRIPT_DIR}/logs/02_weaver_grpo_full.log

    # Update weaver checkpoint to GRPO version
    WEAVER_CHECKPOINT=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
    if [ -z "$WEAVER_CHECKPOINT" ]; then
        echo "ERROR: Weaver GRPO checkpoint not found after training!"
        exit 1
    fi
    echo "Found Weaver (GRPO) checkpoint: ${WEAVER_CHECKPOINT}"
    echo ""
fi

# ============================================================================
# Step 3: Evaluate Weaver
# ============================================================================
CURRENT_STEP=$((CURRENT_STEP + 1))
echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Evaluating Weaver..."
bash ${SCRIPT_DIR}/02_eval_weaver.sh 2>&1 | tee ${SCRIPT_DIR}/logs/02_eval_weaver_full.log

# Backup weaver eval results (will be overwritten by trigger eval)
WEAVER_EVAL_DIR=$(find ${OUTPUT_ROOT}/evaluate/${DATASET_NAME} -type d -name "*pn=*" 2>/dev/null | sort | tail -1)
if [ -n "$WEAVER_EVAL_DIR" ] && [ -f "${WEAVER_EVAL_DIR}/evaluate/answer.json" ]; then
    cp "${WEAVER_EVAL_DIR}/evaluate/answer.json" "${WEAVER_EVAL_DIR}/evaluate/answer_weaver.json"
    echo "Backed up weaver eval results to: ${WEAVER_EVAL_DIR}/evaluate/answer_weaver.json"
fi
echo ""

# ============================================================================
# Step 4: Trigger GRPO Training
# ============================================================================
CURRENT_STEP=$((CURRENT_STEP + 1))
echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Running Trigger GRPO Training..."
bash ${SCRIPT_DIR}/03_trigger_pretrain.sh 2>&1 | tee ${SCRIPT_DIR}/logs/03_trigger_pretrain_full.log

# Find the trigger checkpoint using common.sh utility
TRIGGER_CHECKPOINT=$(find_latest_trigger_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
if [ -z "$TRIGGER_CHECKPOINT" ]; then
    echo "ERROR: Trigger checkpoint not found after training!"
    echo "Expected location: ${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_NAME_SAFE}/pn=*/trigger/trigger_lora"
    exit 1
fi

echo "Found Trigger checkpoint: ${TRIGGER_CHECKPOINT}"
echo ""

# ============================================================================
# Step 5: Evaluate Trigger
# ============================================================================
CURRENT_STEP=$((CURRENT_STEP + 1))
echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Evaluating Trigger..."
bash ${SCRIPT_DIR}/04_eval_trigger.sh 2>&1 | tee ${SCRIPT_DIR}/logs/04_eval_trigger_full.log
echo ""

# ============================================================================
# Step 6: LTPO Test-Time Optimization + Evaluation
# ============================================================================
CURRENT_STEP=$((CURRENT_STEP + 1))
echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Running LTPO Evaluation..."
bash ${SCRIPT_DIR}/05_ltpo_eval.sh 2>&1 | tee ${SCRIPT_DIR}/logs/05_ltpo_eval_full.log
echo ""

# ============================================================================
# Step 7: LTPO Hyperparameter Sweep (Optional)
# ============================================================================
if [ "$SKIP_LTPO_SWEEP" = false ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Running LTPO Hyperparameter Sweep..."
    bash ${SCRIPT_DIR}/07_ltpo_sweep.sh 2>&1 | tee ${SCRIPT_DIR}/logs/07_ltpo_sweep_full.log
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
# Re-discover checkpoints for final summary
WEAVER_CHECKPOINT=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
TRIGGER_CHECKPOINT=$(find_latest_trigger_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")

echo "============================================"
echo "Pipeline Completed Successfully!"
echo "============================================"
echo ""
echo "Checkpoints:"
echo "  - Weaver: ${WEAVER_CHECKPOINT:-NOT FOUND}"
echo "  - Trigger: ${TRIGGER_CHECKPOINT:-NOT FOUND}"
echo ""
echo "Evaluation Results:"
EVAL_BASE="${OUTPUT_ROOT}/evaluate/${DATASET_NAME}"
LTPO_BASE="${OUTPUT_ROOT}/evaluate_ltpo/${DATASET_NAME}"
echo "  - Weaver eval: ${EVAL_BASE}/*/evaluate/answer_weaver.json"
echo "  - Trigger eval: ${EVAL_BASE}/*/evaluate/answer.json"
echo "  - LTPO eval: ${LTPO_BASE}/*/evaluate/answer_ltpo.json"
if [ "$SKIP_LTPO_SWEEP" = false ]; then
    echo "  - LTPO sweep: ${SCRIPT_DIR}/logs/ltpo_sweep/*/sweep_results.csv"
fi
echo ""
echo "Logs: ${SCRIPT_DIR}/logs/"
echo "  - 01_weaver_pretrain_full.log"
if [ "$SKIP_WEAVER_GRPO" = false ]; then
    echo "  - 02_weaver_grpo_full.log"
fi
echo "  - 02_eval_weaver_full.log"
echo "  - 03_trigger_pretrain_full.log"
echo "  - 04_eval_trigger_full.log"
echo "  - 05_ltpo_eval_full.log"
if [ "$SKIP_LTPO_SWEEP" = false ]; then
    echo "  - 07_ltpo_sweep_full.log"
fi
echo "============================================"
