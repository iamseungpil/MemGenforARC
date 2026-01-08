#!/bin/bash
# ============================================================================
# Step 2: Weaver GRPO Training (after SFT warmup)
# ============================================================================
# Train the Weaver module using GRPO reinforcement learning.
# Requires a trained Weaver SFT model from Step 1.
#
# Expected Input:  SFT checkpoint from 01_weaver_sft.sh
# Expected Output: /data/memgen/train/gsm8k/<model_name>/<timestamp>/weaver/
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/02_weaver_grpo.log"
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs (A100 40GB x2)
export MAIN_PROCESS_PORT=29508
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Create log directory
mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"
DATASET_MODE="grpo"
TRAIN_METHOD="grpo"

# Wandb run name
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="weaver_${TRAIN_METHOD}_${MODEL_SHORT}_$(date +%Y%m%d)"

# Augmentation configs (must match SFT training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# Checkpoint path: use argument if provided, otherwise auto-discover
# Usage: ./02_weaver_grpo.sh [sft_weaver_path]
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
print_checkpoint_info "Weaver (SFT)" "${LOAD_WEAVER_PATH}"

if [ -z "$LOAD_WEAVER_PATH" ]; then
    echo ""
    echo "ERROR: Weaver SFT checkpoint is required for GRPO training!"
    echo "Please run 01_weaver_sft.sh first."
    exit 1
fi
echo "============================================"

# GRPO Training hyperparameters
BATCH_SIZE=8
NUM_EPOCHS=1
NUM_GENERATIONS=8
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-5

# ============================================================================
# Execute Training
# ============================================================================
echo ""
echo "============================================"
echo "Step 2: Weaver GRPO Training"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Method: ${TRAIN_METHOD}"
echo "SFT checkpoint: ${LOAD_WEAVER_PATH}"
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
    model.trigger.active False \
    dataset.mode ${DATASET_MODE} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.grpo.num_train_epochs ${NUM_EPOCHS} \
    run.weaver.grpo.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.grpo.per_device_eval_batch_size ${BATCH_SIZE} \
    run.weaver.grpo.num_generations ${NUM_GENERATIONS} \
    run.weaver.grpo.gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    run.weaver.grpo.learning_rate ${LEARNING_RATE} \
    run.weaver.grpo.bf16 True \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024

echo ""
echo "============================================"
echo "Weaver GRPO training completed!"
echo "============================================"
WEAVER_GRPO_PATH=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
echo ""
echo "export WEAVER_PATH=${WEAVER_GRPO_PATH}"
echo ""
echo "Next: ./03_eval_weaver.sh or ./04_trigger_grpo.sh"
echo "============================================"
