#!/bin/bash
# ============================================================================
# Step 4: Evaluate Trigger
# ============================================================================
# Evaluate the trained Trigger model on KodCode test set.
# This measures the quality of memory insertion timing decisions.
# Requires both Weaver and Trigger checkpoints from previous steps.
#
# Expected Output: ~/data/memgen/evaluate/kodcode/<model_name>/evaluate/answer.json
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/04_eval_trigger.log"
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs (A100 40GB x2)
export MAIN_PROCESS_PORT=29507
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
export WANDB_RUN_NAME="eval_trigger_${MODEL_SHORT}_$(date +%Y%m%d)"

# Augmentation configs (must match training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# Checkpoint paths: use arguments if provided, otherwise auto-discover
# Usage: ./04_eval_trigger.sh [weaver_path] [trigger_path]
# ============================================================================
MODEL_NAME_SAFE=$(get_model_name_safe "${MODEL_NAME}")
if [ -n "$1" ]; then
    LOAD_WEAVER_PATH="$1"
else
    LOAD_WEAVER_PATH=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
fi
if [ -n "$2" ]; then
    LOAD_TRIGGER_PATH="$2"
else
    LOAD_TRIGGER_PATH=$(find_latest_trigger_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
fi

echo "============================================"
echo "Checkpoint Discovery"
echo "============================================"
print_checkpoint_info "Weaver" "${LOAD_WEAVER_PATH}"
print_checkpoint_info "Trigger" "${LOAD_TRIGGER_PATH}"

# Validate checkpoints
MISSING_CHECKPOINTS=false
if [ -z "$LOAD_WEAVER_PATH" ]; then
    echo ""
    echo "ERROR: Weaver checkpoint not found!"
    MISSING_CHECKPOINTS=true
fi
if [ -z "$LOAD_TRIGGER_PATH" ]; then
    echo ""
    echo "ERROR: Trigger checkpoint not found!"
    MISSING_CHECKPOINTS=true
fi

if [ "$MISSING_CHECKPOINTS" = true ]; then
    echo ""
    echo "Please run the following steps first:"
    echo "  1. 01_weaver_pretrain.sh (Weaver training)"
    echo "  2. 03_trigger_pretrain.sh (Trigger training)"
    exit 1
fi
echo "============================================"

# Evaluation configs
BATCH_SIZE=4
TEMPERATURE=0.0  # Greedy decoding for evaluation
TRIGGER_ACTIVE=True  # Now using trained trigger

# ============================================================================
# Execute Evaluation
# ============================================================================
echo ""
echo "============================================"
echo "Step 4: Evaluate Trigger"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Weaver Checkpoint: ${LOAD_WEAVER_PATH}"
echo "Trigger Checkpoint: ${LOAD_TRIGGER_PATH}"
echo "Trigger Active: ${TRIGGER_ACTIVE}"
echo "============================================"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_weaver_path ${LOAD_WEAVER_PATH} \
    model.load_trigger_path ${LOAD_TRIGGER_PATH} \
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
echo "Trigger evaluation completed!"
echo "Check results at: ~/data/memgen/evaluate/kodcode/"
echo "Next: Run 05_ltpo_eval.sh for test-time optimization"
echo "============================================"
