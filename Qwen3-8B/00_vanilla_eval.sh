#!/bin/bash
# ============================================================================
# Step 0: Vanilla Baseline Evaluation (No MemGen)
# ============================================================================
# Evaluate base model WITHOUT any MemGen augmentation.
# This provides the baseline performance for comparison.
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
DATASET_NAME="gsm8k"

# Wandb run name
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="vanilla_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "Step 0: Vanilla Baseline Evaluation"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "MemGen: DISABLED (vanilla baseline)"
echo "GPU: Single GPU (CUDA_VISIBLE_DEVICES=0)"
echo "============================================"

python main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.max_prompt_aug_num 0 \
    model.max_inference_aug_num 0 \
    model.weaver.model_name ${MODEL_NAME} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active False \
    run.mode evaluate \
    run.interaction.batch_size 4 \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 512 \
    2>&1 | tee ${SCRIPT_DIR}/logs/00_vanilla_eval.log

echo "============================================"
echo "Vanilla evaluation completed!"
echo "============================================"
