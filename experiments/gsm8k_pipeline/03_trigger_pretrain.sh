#!/bin/bash
# ============================================================================
# Step 3: Trigger GRPO Training
# ============================================================================
# Train the Trigger module using GRPO to learn when to insert latent memory.
# Requires a trained Weaver model from Step 1.
#
# Expected Output: /data/memgen/train/gsm8k/<model_name>/trigger_grpo/
# ============================================================================

set -e  # Exit on error

# Environment setup
export DEBUG_MODE=true
export LOG_PATH="./logs/03_trigger_pretrain.log"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME="gsm8k"
DATASET_MODE="grpo"
TRAIN_METHOD="grpo"  # Trigger only supports GRPO

# Augmentation configs (must match weaver training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# IMPORTANT: Set the path to your trained weaver model
# ============================================================================
# This must be the checkpoint from Step 1 (01_weaver_pretrain.sh)
# Example:
# LOAD_WEAVER_PATH="/data/memgen/train/gsm8k/Qwen2.5-1.5B-Instruct/pn=1_pl=8_in=5_il=8_20250107_120000/weaver/weaver_lora"
# ============================================================================
LOAD_WEAVER_PATH=null  # <-- UPDATE THIS with trained weaver path

# GRPO Training hyperparameters
BATCH_SIZE=8
NUM_EPOCHS=1
NUM_GENERATIONS=8
GRADIENT_ACCUMULATION_STEPS=4

# ============================================================================
# Execute Training
# ============================================================================
echo "============================================"
echo "Step 3: Trigger GRPO Training"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Method: ${TRAIN_METHOD}"
echo "Weaver checkpoint: ${LOAD_WEAVER_PATH}"
echo "============================================"

if [ "$LOAD_WEAVER_PATH" = "null" ]; then
    echo "ERROR: LOAD_WEAVER_PATH is null!"
    echo "You must first run 01_weaver_pretrain.sh and set the checkpoint path."
    exit 1
fi

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
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
echo "Check output at: /data/memgen/train/gsm8k/"
echo "Next: Run 04_eval_trigger.sh to evaluate"
echo "============================================"
