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
# The script automatically finds checkpoint paths between stages.
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/ubuntu/MemGenforARC"
OUTPUT_ROOT="/data/memgen"

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME_SAFE=$(echo ${MODEL_NAME} | tr '/' '_')
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

# Find the latest weaver checkpoint (sort by modification time, newest first)
WEAVER_CHECKPOINT_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_NAME_SAFE}/weaver_sft"
if [ ! -d "$WEAVER_CHECKPOINT_DIR" ]; then
    echo "ERROR: Weaver checkpoint directory not found: ${WEAVER_CHECKPOINT_DIR}"
    exit 1
fi

# Find the most recent checkpoint by sorting checkpoint directories by step number
WEAVER_CHECKPOINT=$(find ${WEAVER_CHECKPOINT_DIR} -name "model.safetensors" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$WEAVER_CHECKPOINT" ]; then
    # Fallback: try without printf (macOS compatibility)
    WEAVER_CHECKPOINT=$(find ${WEAVER_CHECKPOINT_DIR} -name "model.safetensors" -type f | head -1)
fi
if [ -z "$WEAVER_CHECKPOINT" ]; then
    echo "ERROR: No model.safetensors found in ${WEAVER_CHECKPOINT_DIR}"
    exit 1
fi

echo "Found Weaver checkpoint: ${WEAVER_CHECKPOINT}"
echo ""

# ============================================================================
# Step 2: Evaluate Weaver
# ============================================================================
echo "[Step 2/5] Evaluating Weaver..."
export LOAD_MODEL_PATH="${WEAVER_CHECKPOINT}"
# WARNING: sed -i modifies script files permanently. This is intentional to
# allow re-running individual scripts with the correct checkpoint paths.
sed -i "s|^LOAD_MODEL_PATH=.*|LOAD_MODEL_PATH=\"${WEAVER_CHECKPOINT}\"|" ${SCRIPT_DIR}/02_eval_weaver.sh
bash ${SCRIPT_DIR}/02_eval_weaver.sh 2>&1 | tee ${SCRIPT_DIR}/logs/02_eval_weaver_full.log

# Backup weaver eval results (will be overwritten by trigger eval)
WEAVER_EVAL_DIR=$(find ${OUTPUT_ROOT}/evaluate/${DATASET_NAME} -type d -name "*pn=*" | sort | tail -1)
if [ -n "$WEAVER_EVAL_DIR" ] && [ -f "${WEAVER_EVAL_DIR}/evaluate/answer.json" ]; then
    cp "${WEAVER_EVAL_DIR}/evaluate/answer.json" "${WEAVER_EVAL_DIR}/evaluate/answer_weaver.json"
    echo "Backed up weaver eval results to: ${WEAVER_EVAL_DIR}/evaluate/answer_weaver.json"
fi
echo ""

# ============================================================================
# Step 3: Trigger GRPO Training
# ============================================================================
echo "[Step 3/5] Running Trigger GRPO Training..."
sed -i "s|^LOAD_WEAVER_PATH=.*|LOAD_WEAVER_PATH=\"${WEAVER_CHECKPOINT}\"|" ${SCRIPT_DIR}/03_trigger_pretrain.sh
bash ${SCRIPT_DIR}/03_trigger_pretrain.sh 2>&1 | tee ${SCRIPT_DIR}/logs/03_trigger_pretrain_full.log

# Find the latest trigger checkpoint (sort by modification time, newest first)
TRIGGER_CHECKPOINT_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_NAME_SAFE}/trigger_grpo"
if [ ! -d "$TRIGGER_CHECKPOINT_DIR" ]; then
    echo "ERROR: Trigger checkpoint directory not found: ${TRIGGER_CHECKPOINT_DIR}"
    exit 1
fi

# Find the most recent checkpoint by sorting checkpoint directories by step number
TRIGGER_CHECKPOINT=$(find ${TRIGGER_CHECKPOINT_DIR} -name "model.safetensors" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$TRIGGER_CHECKPOINT" ]; then
    # Fallback: try without printf (macOS compatibility)
    TRIGGER_CHECKPOINT=$(find ${TRIGGER_CHECKPOINT_DIR} -name "model.safetensors" -type f | head -1)
fi
if [ -z "$TRIGGER_CHECKPOINT" ]; then
    echo "ERROR: No model.safetensors found in ${TRIGGER_CHECKPOINT_DIR}"
    exit 1
fi

echo "Found Trigger checkpoint: ${TRIGGER_CHECKPOINT}"
echo ""

# ============================================================================
# Step 4: Evaluate Trigger
# ============================================================================
echo "[Step 4/5] Evaluating Trigger..."
sed -i "s|^LOAD_MODEL_PATH=.*|LOAD_MODEL_PATH=\"${TRIGGER_CHECKPOINT}\"|" ${SCRIPT_DIR}/04_eval_trigger.sh
bash ${SCRIPT_DIR}/04_eval_trigger.sh 2>&1 | tee ${SCRIPT_DIR}/logs/04_eval_trigger_full.log
echo ""

# ============================================================================
# Step 5: LTPO Test-Time Optimization + Evaluation
# ============================================================================
echo "[Step 5/5] Running LTPO Evaluation..."
sed -i "s|^LOAD_MODEL_PATH=.*|LOAD_MODEL_PATH=\"${TRIGGER_CHECKPOINT}\"|" ${SCRIPT_DIR}/05_ltpo_eval.sh
bash ${SCRIPT_DIR}/05_ltpo_eval.sh 2>&1 | tee ${SCRIPT_DIR}/logs/05_ltpo_eval_full.log
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================"
echo "Pipeline Completed Successfully!"
echo "============================================"
echo ""
echo "Checkpoints:"
echo "  - Weaver: ${WEAVER_CHECKPOINT}"
echo "  - Trigger: ${TRIGGER_CHECKPOINT}"
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
