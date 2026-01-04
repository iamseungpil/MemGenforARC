#!/bin/bash
# ============================================================================
# Step 4: Evaluate Trigger
# ============================================================================
# Evaluate the trained Trigger model on GSM8K test set.
# This measures the quality of memory insertion timing decisions.
#
# Expected Output: /data/memgen/evaluate/gsm8k/<model_name>/evaluate/answer.json
# ============================================================================

set -e  # Exit on error

# Environment setup
export DEBUG_MODE=true
export LOG_PATH="./logs/04_eval_trigger.log"
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
# IMPORTANT: Set the path to your trained model (after trigger training)
# ============================================================================
# After running 03_trigger_pretrain.sh, find the checkpoint path:
# /data/memgen/train/gsm8k/<model_name>/trigger_grpo/checkpoint-<step>/model.safetensors
#
# Example:
# LOAD_MODEL_PATH="/data/memgen/train/gsm8k/Qwen_Qwen2.5-1.5B-Instruct/trigger_grpo/checkpoint-500/model.safetensors"
# ============================================================================
LOAD_MODEL_PATH=null  # <-- UPDATE THIS after trigger training

# Evaluation configs
BATCH_SIZE=4
TEMPERATURE=0.0  # Greedy decoding for evaluation
TRIGGER_ACTIVE=True  # Now using trained trigger

# ============================================================================
# Execute Evaluation
# ============================================================================
echo "============================================"
echo "Step 4: Evaluate Trigger"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${LOAD_MODEL_PATH}"
echo "Trigger Active: ${TRIGGER_ACTIVE}"
echo "============================================"

if [ "$LOAD_MODEL_PATH" = "null" ]; then
    echo "WARNING: LOAD_MODEL_PATH is null. Evaluating without trained trigger."
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
echo "Trigger evaluation completed!"
echo "Check results at: /data/memgen/evaluate/gsm8k/"
echo "Next: Run 05_ltpo_eval.sh for test-time optimization"
echo "============================================"
