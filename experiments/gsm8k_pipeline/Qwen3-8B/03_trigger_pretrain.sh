#!/bin/bash
# ============================================================================
# Step 3: Trigger GRPO Training
# ============================================================================
# Train the Trigger module using GRPO to learn when to insert latent memory.
# Requires a trained Weaver model from Step 1.
#
# Expected Output: /data/memgen/train/gsm8k/<model_name>/<timestamp>/trigger/
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/03_trigger_pretrain.log"
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs (A100 40GB x2)
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"
DATASET_MODE="grpo"
TRAIN_METHOD="grpo"  # Trigger only supports GRPO

# Wandb run name
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="trigger_${TRAIN_METHOD}_${MODEL_SHORT}_$(date +%Y%m%d)"

# Augmentation configs (must match weaver training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# Checkpoint path: use argument if provided, otherwise auto-discover
# Usage: ./03_trigger_pretrain.sh [weaver_path]
# ============================================================================
MODEL_NAME_SAFE=$(get_model_name_safe "${MODEL_NAME}")
if [ -n "$1" ]; then
    LOAD_WEAVER_PATH="$1"
else
    LOAD_WEAVER_PATH=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
fi

echo "============================================"
echo "Checkpoint Discovery"
echo "============================================"
print_checkpoint_info "Weaver" "${LOAD_WEAVER_PATH}"

if [ -z "$LOAD_WEAVER_PATH" ]; then
    echo ""
    echo "ERROR: Weaver checkpoint is required for trigger training!"
    echo "Please run 01_weaver_pretrain.sh first."
    exit 1
fi
echo "============================================"

# GRPO Training hyperparameters
BATCH_SIZE=8
NUM_EPOCHS=1
NUM_GENERATIONS=8
GRADIENT_ACCUMULATION_STEPS=4

# ============================================================================
# Execute Training
# ============================================================================
echo ""
echo "============================================"
echo "Step 3: Trigger GRPO Training"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Method: ${TRAIN_METHOD}"
echo "Weaver checkpoint: ${LOAD_WEAVER_PATH}"
echo "============================================"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_weaver_path ${LOAD_WEAVER_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active True \
    dataset.mode ${DATASET_MODE} \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method ${TRAIN_METHOD} \
    run.trigger.grpo.num_train_epochs ${NUM_EPOCHS} \
    run.trigger.grpo.per_device_train_batch_size ${BATCH_SIZE} \
    run.trigger.grpo.per_device_eval_batch_size ${BATCH_SIZE} \
    run.trigger.grpo.num_generations ${NUM_GENERATIONS} \
    run.trigger.grpo.gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024

echo ""
echo "============================================"
echo "Trigger GRPO training completed!"
echo "============================================"
TRIGGER_PATH=$(find_latest_trigger_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
echo ""
echo "export WEAVER_PATH=${LOAD_WEAVER_PATH}"
echo "export TRIGGER_PATH=${TRIGGER_PATH}"
echo ""
echo "Next: ./04_eval_trigger.sh \$WEAVER_PATH \$TRIGGER_PATH"
echo "============================================"
