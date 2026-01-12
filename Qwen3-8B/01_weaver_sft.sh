#!/bin/bash
# ============================================================================
# Step 1: Weaver SFT Training
# ============================================================================
# Train Weaver model using Supervised Fine-Tuning (SFT).
# This trains the latent memory generation capability.
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
export WANDB_RUN_NAME="weaver_sft_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

# MemGen settings
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

echo "============================================"
echo "Step 1: Weaver SFT Training"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Training Method: SFT"
echo "GPU: Single GPU (CUDA_VISIBLE_DEVICES=0)"
echo "============================================"

python main.py \
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
    dataset.mode sft \
    run.mode train \
    run.train_weaver True \
    run.train_weaver_method sft \
    run.train_trigger False \
    run.weaver.sft.num_train_epochs 2 \
    run.weaver.sft.per_device_train_batch_size 4 \
    run.weaver.sft.per_device_eval_batch_size 4 \
    run.weaver.sft.gradient_accumulation_steps 1 \
    run.weaver.sft.learning_rate 1e-5 \
    run.weaver.sft.warmup_ratio 0.1 \
    2>&1 | tee ${SCRIPT_DIR}/logs/01_weaver_sft.log

echo "============================================"
echo "Weaver SFT training completed!"
echo "Check output at: ${PROJECT_ROOT}/results/train/${DATASET_NAME}/${MODEL_SHORT}/"
echo "Next: ./02_eval_weaver.sh"
echo "============================================"
