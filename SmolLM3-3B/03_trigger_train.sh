#!/bin/bash
# ============================================================================
# Step 3: Trigger GRPO Training
# ============================================================================

set -e

export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_reproduce"
export DEBUG_MODE=true
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd ${PROJECT_ROOT}

mkdir -p ${SCRIPT_DIR}/logs

MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
DATASET_NAME="gsm8k"
export WANDB_RUN_NAME="trigger_grpo_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

OUTPUT_ROOT="${PROJECT_ROOT}/results"
if [ -n "$1" ]; then
    LOAD_WEAVER_PATH="$1"
else
    BASE_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_SHORT}"
    LATEST_DIR=$(ls -td ${BASE_DIR}/pn=* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/weaver/weaver_lora" ]; then
        LOAD_WEAVER_PATH="${LATEST_DIR}/weaver"
    else
        echo "ERROR: No weaver checkpoint found."
        exit 1
    fi
fi

echo "============================================"
echo "Step 3: Trigger GRPO Training"
echo "Model: ${MODEL_NAME}"
echo "Weaver Checkpoint: ${LOAD_WEAVER_PATH}"
echo "GPU: Single GPU"
echo "============================================"

python main.py \
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
    model.trigger.active True \
    dataset.mode grpo \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method grpo \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.per_device_train_batch_size 4 \
    run.trigger.grpo.per_device_eval_batch_size 4 \
    run.trigger.grpo.num_generations 4 \
    run.trigger.grpo.gradient_accumulation_steps 1 \
    run.trigger.grpo.learning_rate 1e-5 \
    2>&1 | tee ${SCRIPT_DIR}/logs/03_trigger_train.log

echo "Trigger GRPO training completed!"
