#!/bin/bash
# ============================================================================
# Step 1: Weaver SFT Pretraining
# ============================================================================
# Train the Weaver module to generate latent memory tokens using supervised
# fine-tuning on KodCode dataset.
#
# Expected Output: /data/memgen/train/kodcode/<model_name>/<timestamp>/weaver/
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/01_weaver_pretrain.log"
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs (A100 40GB x2)
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Create log directory
mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
# Options:
#   - Qwen/Qwen2.5-1.5B-Instruct  (quick validation, ~3GB VRAM)
#   - Qwen/Qwen2.5-7B-Instruct    (main experiment, ~14GB VRAM)
#   - Qwen/Qwen3-14B              (large scale, ~28GB VRAM)
# ============================================================================
MODEL_NAME="Qwen/Qwen3-8B"

# Dataset
DATASET_NAME="kodcode"

# Training method
TRAIN_METHOD="sft"  # SFT for warmup, GRPO for RL later

# Wandb run name (set after MODEL_NAME and TRAIN_METHOD are defined)
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="weaver_${TRAIN_METHOD}_${MODEL_SHORT}_$(date +%Y%m%d)"

# Augmentation configs (standard for reasoning tasks)
MAX_PROMPT_AUG_NUM=1       # Number of prompt-end augmentations
MAX_INFERENCE_AUG_NUM=5    # Number of mid-generation augmentations
PROMPT_LATENTS_LEN=8       # Length of prompt latent sequence
INFERENCE_LATENTS_LEN=8    # Length of inference latent sequence

# Training hyperparameters
BATCH_SIZE=4
NUM_EPOCHS=2
LEARNING_RATE=1e-5

# ============================================================================
# Execute Training
# ============================================================================
echo "============================================"
echo "Step 1: Weaver SFT Pretraining"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Method: ${TRAIN_METHOD}"
echo "============================================"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active False \
    dataset.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.sft.num_train_epochs ${NUM_EPOCHS} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
    run.weaver.sft.learning_rate ${LEARNING_RATE} \
    run.weaver.sft.bf16 True \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024

echo ""
echo "============================================"
echo "Weaver SFT training completed!"
echo "============================================"
MODEL_NAME_SAFE=$(get_model_name_safe "${MODEL_NAME}")
WEAVER_PATH=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
echo ""
echo "export WEAVER_PATH=${WEAVER_PATH}"
echo ""
echo "Next: ./02_eval_weaver.sh \$WEAVER_PATH"
echo "============================================"
