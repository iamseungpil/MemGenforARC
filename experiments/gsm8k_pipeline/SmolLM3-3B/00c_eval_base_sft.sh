#!/bin/bash
# ============================================================================
# Step 0c: Evaluate Base SFT Model
# ============================================================================
# Evaluate the base model trained with SFT (no MemGen).
#
# Usage:
#   ./00c_eval_base_sft.sh                    # Auto-discover latest checkpoint
#   ./00c_eval_base_sft.sh /path/to/checkpoint  # Use specific checkpoint
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/00c_eval_base_sft.log"
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29509
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Create log directory
mkdir -p logs

# Model Configuration
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
DATASET_NAME="gsm8k"

# Find latest base SFT checkpoint
find_latest_base_sft_checkpoint() {
    local dataset=$1
    local model_name=$2
    local base_dir="${HOME}/data/basesft/train/${dataset}"
    ls -dt ${base_dir}/${model_name}_* 2>/dev/null | head -1
}

MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')

# Get checkpoint path (from argument or auto-discover)
if [ -n "$1" ]; then
    BASE_SFT_PATH="$1"
else
    BASE_SFT_PATH=$(find_latest_base_sft_checkpoint "${DATASET_NAME}" "${MODEL_SHORT}")
fi

if [ -z "$BASE_SFT_PATH" ] || [ ! -d "$BASE_SFT_PATH" ]; then
    echo "============================================"
    echo "ERROR: No Base SFT checkpoint found"
    echo "============================================"
    echo "Expected location: ~/data/basesft/train/${DATASET_NAME}/${MODEL_SHORT}_*"
    echo ""
    echo "Please run 00b_base_sft.sh first, or provide checkpoint path as argument:"
    echo "  ./00c_eval_base_sft.sh /path/to/checkpoint"
    echo "============================================"
    exit 1
fi

export WANDB_RUN_NAME="eval_base_sft_${MODEL_SHORT}_$(date +%Y%m%d)"

echo "============================================"
echo "Step 0c: Evaluate Base SFT Model"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${BASE_SFT_PATH}"
echo "MemGen: DISABLED"
echo "============================================"

# Base SFT는 PEFT 모델이므로 base model + adapter 로드
# model.load_model_path로 PEFT adapter 로드
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
    model.load_model_path ${BASE_SFT_PATH} \
    run.mode evaluate \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 512

echo "============================================"
echo "Base SFT evaluation completed!"
echo "============================================"
