#!/bin/bash
# ============================================================================
# Step 1: Weaver SFT Pretraining
# ============================================================================
# Train the Weaver module to generate latent memory tokens using supervised
# fine-tuning on GSM8K dataset.
#
# Expected Output: /data/memgen/train/gsm8k/<model_name>/weaver_sft/
# ============================================================================

set -e  # Exit on error

# Environment setup
export DEBUG_MODE=true
export LOG_PATH="./logs/01_weaver_pretrain.log"
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for 1.5B model
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
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# Dataset
DATASET_NAME="gsm8k"

# Training method
TRAIN_METHOD="sft"  # SFT for warmup, GRPO for RL later

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
echo "Check output at: /data/memgen/train/gsm8k/"
echo "Next: Run 02_eval_weaver.sh to evaluate"
echo "============================================"
