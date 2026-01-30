#!/bin/bash
# ============================================================================
# Recursive Memory Training (TRM-style compression)
# ============================================================================
# Train the recursive memory compressor to compress context into memory tokens.
# NO projection layers, NO LoRA - only Low-rank Cross-Attention + MLP.
#
# Parameters:
#   - query_latents: ~65K
#   - LowRankCrossAttention: ~2M
#   - LowRankSwiGLU: ~6M
#   - Total: ~8M trainable params
# ============================================================================

set -e  # Exit on error

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen-recursive"
export DEBUG_MODE=true
export LOG_PATH="./logs/recursive_memory_train.log"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29508

# Create log directory
mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"
TRAIN_METHOD="sft"

# Wandb run name
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="recursive_memory_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

# Augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Recursive Memory configs
RECURSIVE_ENABLED=true
RECURSIVE_TWO_LEVEL=false  # SingleLevel (simpler)
RECURSIVE_HIDDEN_SIZE=4096
RECURSIVE_NUM_HEADS=8
RECURSIVE_ATTN_RANK=64
RECURSIVE_MLP_RANK=128
RECURSIVE_MAX_CYCLES=3

# Training hyperparameters
BATCH_SIZE=4
NUM_EPOCHS=3
LEARNING_RATE=1e-4  # Higher LR for small parameter count

# ============================================================================
# Execute Training
# ============================================================================
echo "============================================"
echo "Recursive Memory Training"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Two-Level: ${RECURSIVE_TWO_LEVEL}"
echo "Max Cycles: ${RECURSIVE_MAX_CYCLES}"
echo "Trainable Params: ~8M"
echo "============================================"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
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
    model.recursive_memory.enabled ${RECURSIVE_ENABLED} \
    model.recursive_memory.two_level ${RECURSIVE_TWO_LEVEL} \
    model.recursive_memory.hidden_size ${RECURSIVE_HIDDEN_SIZE} \
    model.recursive_memory.num_heads ${RECURSIVE_NUM_HEADS} \
    model.recursive_memory.attn_rank ${RECURSIVE_ATTN_RANK} \
    model.recursive_memory.mlp_rank ${RECURSIVE_MLP_RANK} \
    model.recursive_memory.max_cycles ${RECURSIVE_MAX_CYCLES} \
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
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
    2>&1 | tee ${LOG_PATH}

echo ""
echo "============================================"
echo "Recursive Memory training completed!"
echo "============================================"
echo "Log: ${LOG_PATH}"
echo "============================================"
