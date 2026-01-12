#!/bin/bash
# ============================================================================
# Step 0b: Base Model SFT Training (No MemGen)
# ============================================================================
# Train base model with SFT but WITHOUT MemGen augmentation.
# This measures pure SFT improvement as a control group.
#
# Expected Output: ~/data/basesft/train/gsm8k/<model_name>_<timestamp>/
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/00b_base_sft.log"
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29508
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Create log directory
mkdir -p logs

# Model Configuration
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
DATASET_NAME="gsm8k"

# Training hyperparameters (same as Weaver SFT for fair comparison)
BATCH_SIZE=4
NUM_EPOCHS=2
LEARNING_RATE=1e-5
MAX_SEQ_LENGTH=1024

MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="${HOME}/data/basesft/train/${DATASET_NAME}/${MODEL_SHORT}_${TIMESTAMP}"

export WANDB_RUN_NAME="base_sft_${MODEL_SHORT}_$(date +%Y%m%d)"

echo "============================================"
echo "Step 0b: Base Model SFT (No MemGen)"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "MemGen: DISABLED (base SFT only)"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "============================================"

# IMPORTANT: Use accelerate launch (CLAUDE.md requirement)
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    basesft_train.py \
    --model-name ${MODEL_NAME} \
    --dataset ${DATASET_NAME} \
    --output-dir ${OUTPUT_DIR} \
    --epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32

echo "============================================"
echo "Base SFT training completed!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Next: ./00c_eval_base_sft.sh"
echo "============================================"
