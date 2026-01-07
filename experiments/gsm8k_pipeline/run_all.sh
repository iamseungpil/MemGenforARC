#!/bin/bash
# ============================================================================
# GSM8K Pipeline: Full Automated Run
# ============================================================================
# This script runs the complete MemGen pipeline:
#   1. Weaver SFT pretraining
#   2. Weaver evaluation
#   3. Trigger GRPO training
#   4. Trigger evaluation
#   5. LTPO test-time optimization + evaluation
#
# The script automatically finds checkpoint paths between stages using
# common.sh utilities.
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

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

echo "============================================"
echo "GSM8K Full Pipeline Execution"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${OUTPUT_ROOT}"
echo "============================================"
echo ""

# ============================================================================
# Step 1: Weaver SFT Pretraining
# ============================================================================
echo "[Step 1/5] Running Weaver SFT Pretraining..."
bash ${SCRIPT_DIR}/01_weaver_pretrain.sh 2>&1 | tee ${SCRIPT_DIR}/logs/01_weaver_pretrain_full.log

# Find the weaver checkpoint using common.sh utility
WEAVER_CHECKPOINT=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
if [ -z "$WEAVER_CHECKPOINT" ]; then
    echo "ERROR: Weaver checkpoint not found after training!"
    echo "Expected location: ${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_NAME_SAFE}/pn=*/weaver/weaver_lora"
    exit 1
fi

echo "Found Weaver checkpoint: ${WEAVER_CHECKPOINT}"
echo ""

# ============================================================================
# Step 2: Evaluate Weaver
# ============================================================================
echo "[Step 2/5] Evaluating Weaver..."
# No need to modify script - 02_eval_weaver.sh uses auto-discovery
bash ${SCRIPT_DIR}/02_eval_weaver.sh 2>&1 | tee ${SCRIPT_DIR}/logs/02_eval_weaver_full.log

# Backup weaver eval results (will be overwritten by trigger eval)
WEAVER_EVAL_DIR=$(find ${OUTPUT_ROOT}/evaluate/${DATASET_NAME} -type d -name "*pn=*" 2>/dev/null | sort | tail -1)
if [ -n "$WEAVER_EVAL_DIR" ] && [ -f "${WEAVER_EVAL_DIR}/evaluate/answer.json" ]; then
    cp "${WEAVER_EVAL_DIR}/evaluate/answer.json" "${WEAVER_EVAL_DIR}/evaluate/answer_weaver.json"
    echo "Backed up weaver eval results to: ${WEAVER_EVAL_DIR}/evaluate/answer_weaver.json"
fi
echo ""

# ============================================================================
# Step 3: Trigger GRPO Training
# ============================================================================
echo "[Step 3/5] Running Trigger GRPO Training..."
# No need to modify script - 03_trigger_pretrain.sh uses auto-discovery
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
# Step 4: Evaluate Trigger
# ============================================================================
echo "[Step 4/5] Evaluating Trigger..."
# No need to modify script - 04_eval_trigger.sh uses auto-discovery
bash ${SCRIPT_DIR}/04_eval_trigger.sh 2>&1 | tee ${SCRIPT_DIR}/logs/04_eval_trigger_full.log
echo ""

# ============================================================================
# Step 5: LTPO Test-Time Optimization + Evaluation
# ============================================================================
echo "[Step 5/5] Running LTPO Evaluation..."
# No need to modify script - 05_ltpo_eval.sh uses auto-discovery
bash ${SCRIPT_DIR}/05_ltpo_eval.sh 2>&1 | tee ${SCRIPT_DIR}/logs/05_ltpo_eval_full.log
echo ""

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
echo ""
echo "Logs: ${SCRIPT_DIR}/logs/"
echo "  - 01_weaver_pretrain_full.log"
echo "  - 02_eval_weaver_full.log"
echo "  - 03_trigger_pretrain_full.log"
echo "  - 04_eval_trigger_full.log"
echo "  - 05_ltpo_eval_full.log"
echo "============================================"
