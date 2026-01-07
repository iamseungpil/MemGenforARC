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

# Environment setup
export DEBUG_MODE=true
export LOG_PATH="./logs/02_eval_weaver.log"
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

# Augmentation configs (must match training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# IMPORTANT: Set the path to your trained weaver model
# ============================================================================
# After running 01_weaver_pretrain.sh, find the checkpoint path:
# /data/memgen/train/gsm8k/<model_name>/<timestamp>/weaver/weaver_lora/
#
# Example:
# LOAD_MODEL_PATH="/data/memgen/train/gsm8k/Qwen2.5-1.5B-Instruct/pn=1_pl=8_in=5_il=8_20250107_120000/weaver/weaver_lora"
# ============================================================================
LOAD_MODEL_PATH=null  # <-- UPDATE THIS after training

# Evaluation configs
BATCH_SIZE=4
TEMPERATURE=0.0  # Greedy decoding for evaluation
TRIGGER_ACTIVE=False  # Trigger not trained yet

# ============================================================================
# Execute Evaluation
# ============================================================================
echo "============================================"
echo "Step 2: Evaluate Weaver"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${LOAD_MODEL_PATH}"
echo "============================================"

if [ "$LOAD_MODEL_PATH" = "null" ]; then
    echo "WARNING: LOAD_MODEL_PATH is null. Evaluating base model without weaver training."
    echo "To evaluate trained model, update LOAD_MODEL_PATH in this script."
    echo ""
fi

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_model_path ${LOAD_MODEL_PATH} \
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
