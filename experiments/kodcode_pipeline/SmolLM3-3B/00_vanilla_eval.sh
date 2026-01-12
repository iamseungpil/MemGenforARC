#!/bin/bash
# ============================================================================
# Step 0: Vanilla Baseline Evaluation
# ============================================================================
# Evaluate base model WITHOUT MemGen augmentation.
# This provides a baseline for comparison with MemGen-enhanced results.
#
# Key settings:
#   - max_prompt_aug_num=0 (no prompt augmentation)
#   - max_inference_aug_num=0 (no inference augmentation)
#   - No trained weaver/trigger checkpoints loaded
#
# Expected Output: ~/data/memgen/evaluate/kodcode/<model_name>/pn=0_pl=8_in=0_il=8_<timestamp>/
# ============================================================================

set -e  # Exit on error

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/00_vanilla_eval.log"
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs (A100 40GB x2)
export MAIN_PROCESS_PORT=29506
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
DATASET_NAME="kodcode"

# Wandb run name
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="vanilla_eval_${MODEL_SHORT}_$(date +%Y%m%d)"

# ============================================================================
# VANILLA: Disable all augmentation
# ============================================================================
MAX_PROMPT_AUG_NUM=0      # No prompt augmentation
MAX_INFERENCE_AUG_NUM=0   # No inference augmentation
PROMPT_LATENTS_LEN=8      # Not used when aug_num=0, but required by config
INFERENCE_LATENTS_LEN=8   # Not used when aug_num=0, but required by config

# Evaluation configs
BATCH_SIZE=4
TEMPERATURE=0.0  # Greedy decoding for evaluation
TRIGGER_ACTIVE=False

# ============================================================================
# Execute Vanilla Evaluation
# ============================================================================
echo ""
echo "============================================"
echo "Step 0: Vanilla Baseline Evaluation"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Augmentation: DISABLED (vanilla baseline)"
echo "  - max_prompt_aug_num: ${MAX_PROMPT_AUG_NUM}"
echo "  - max_inference_aug_num: ${MAX_INFERENCE_AUG_NUM}"
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
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.do_sample False \
    run.interaction.temperature ${TEMPERATURE} \
    run.interaction.max_response_length 1024

echo ""
echo "============================================"
echo "Vanilla baseline evaluation completed!"
echo "Check results at: ~/data/memgen/evaluate/kodcode/"
echo "Look for directory with pn=0_... (vanilla)"
echo "============================================"
