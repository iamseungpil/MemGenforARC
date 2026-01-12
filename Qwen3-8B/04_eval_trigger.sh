#!/bin/bash
# ============================================================================
# Step 4: Evaluate Trigger (Full MemGen)
# ============================================================================
# Evaluate the full MemGen model (Weaver + Trigger) on GSM8K test set.
#
# Usage: ./04_eval_trigger.sh [weaver_checkpoint_dir] [trigger_checkpoint_dir]
# If no checkpoint provided, will look for latest in results/train/
# ============================================================================

set -e

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_reproduce"
export DEBUG_MODE=true
export CUDA_VISIBLE_DEVICES=0

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd ${PROJECT_ROOT}

mkdir -p ${SCRIPT_DIR}/logs

# Model Configuration
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
DATASET_NAME="gsm8k"

# Wandb run name
export WANDB_RUN_NAME="eval_trigger_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

# MemGen settings (must match training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Checkpoint paths
OUTPUT_ROOT="${PROJECT_ROOT}/results"
if [ -n "$1" ]; then
    LOAD_WEAVER_PATH="$1"
else
    # Find latest weaver checkpoint from trigger training (trigger dir contains weaver too)
    BASE_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_SHORT}"
    LATEST_DIR=$(ls -td ${BASE_DIR}/pn=* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/trigger/weaver_lora" ]; then
        LOAD_WEAVER_PATH="${LATEST_DIR}/trigger"
    elif [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/weaver/weaver_lora" ]; then
        LOAD_WEAVER_PATH="${LATEST_DIR}/weaver"
    else
        LOAD_WEAVER_PATH="null"
    fi
fi

if [ -n "$2" ]; then
    LOAD_TRIGGER_PATH="$2"
else
    # Find latest trigger checkpoint
    BASE_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_SHORT}"
    LATEST_DIR=$(ls -td ${BASE_DIR}/pn=* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/trigger/trigger_lora" ]; then
        LOAD_TRIGGER_PATH="${LATEST_DIR}/trigger"
    else
        LOAD_TRIGGER_PATH="null"
    fi
fi

echo "============================================"
echo "Step 4: Evaluate Full MemGen (Weaver + Trigger)"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Weaver Checkpoint: ${LOAD_WEAVER_PATH}"
echo "Trigger Checkpoint: ${LOAD_TRIGGER_PATH}"
echo "GPU: Single GPU (CUDA_VISIBLE_DEVICES=0)"
echo "============================================"

python main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_weaver_path ${LOAD_WEAVER_PATH} \
    model.load_trigger_path ${LOAD_TRIGGER_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active True \
    run.mode evaluate \
    run.interaction.batch_size 4 \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
    2>&1 | tee ${SCRIPT_DIR}/logs/04_eval_trigger.log

echo "============================================"
echo "Full MemGen evaluation completed!"
echo "============================================"
