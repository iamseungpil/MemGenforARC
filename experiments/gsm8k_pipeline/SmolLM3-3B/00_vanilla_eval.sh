#!/bin/bash
# ============================================================================
# Step 0: Vanilla Baseline Evaluation
# ============================================================================
# Evaluate base model WITHOUT any training or MemGen augmentation.
# This provides the baseline performance for comparison.
#
# Expected Output: ~/data/memgen/evaluate/gsm8k/<model_name>/pn=0_.../
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/00_vanilla_eval.log"
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Create log directory
mkdir -p logs

# Model Configuration
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
DATASET_NAME="gsm8k"

MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="vanilla_eval_${MODEL_SHORT}_$(date +%Y%m%d)"

echo "============================================"
echo "Step 0: Vanilla Baseline Evaluation"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "MemGen: DISABLED (vanilla baseline)"
echo "============================================"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.max_prompt_aug_num 0 \
    model.max_inference_aug_num 0 \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len 8 \
    model.weaver.inference_latents_len 8 \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active False \
    run.mode evaluate \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 512

echo "============================================"
echo "Vanilla evaluation completed!"
echo "============================================"
