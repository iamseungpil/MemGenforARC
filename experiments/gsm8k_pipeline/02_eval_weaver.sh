#!/bin/bash
# ============================================================================
# Step 2: Evaluate Weaver
# ============================================================================
# Evaluate the trained Weaver model on GSM8K test set.
# This measures the quality of latent memory generation.
#
# Expected Output: /data/memgen/evaluate/gsm8k/<model_name>/evaluate/answer.json
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo"
export DEBUG_MODE=true
export LOG_PATH="./logs/02_eval_weaver.log"
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
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"

# Wandb run name
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME="eval_weaver_${MODEL_SHORT}_$(date +%Y%m%d)"

# Augmentation configs (must match training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# Auto-discover weaver checkpoint from previous training
# ============================================================================
MODEL_NAME_SAFE=$(get_model_name_safe "${MODEL_NAME}")
LOAD_WEAVER_PATH=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")

echo "============================================"
echo "Checkpoint Discovery"
echo "============================================"
print_checkpoint_info "Weaver" "${LOAD_WEAVER_PATH}"

if [ -z "$LOAD_WEAVER_PATH" ]; then
    echo ""
    echo "WARNING: No weaver checkpoint found. Evaluating base model."
    echo "To evaluate trained model, run 01_weaver_pretrain.sh first."
    LOAD_WEAVER_PATH="null"
fi
echo "============================================"

# Evaluation configs
BATCH_SIZE=4
TEMPERATURE=0.0  # Greedy decoding for evaluation
TRIGGER_ACTIVE=False  # Trigger not trained yet

# ============================================================================
# Execute Evaluation
# ============================================================================
echo ""
echo "============================================"
echo "Step 2: Evaluate Weaver"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Weaver Checkpoint: ${LOAD_WEAVER_PATH}"
echo "============================================"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_weaver_path ${LOAD_WEAVER_PATH} \
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
echo "Weaver evaluation completed!"
echo "Check results at: /data/memgen/evaluate/gsm8k/"
echo "Next: Run 03_trigger_pretrain.sh to train trigger"
echo "============================================"
