#!/bin/bash
# ============================================================================
# Step 4: Evaluate Trigger (Full MemGen)
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
export WANDB_RUN_NAME="eval_trigger_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

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
    if [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/trigger/weaver_lora" ]; then
        LOAD_WEAVER_PATH="${LATEST_DIR}/trigger"
    elif [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/weaver/weaver_lora" ]; then
        LOAD_WEAVER_PATH="${LATEST_DIR}/weaver"
    else
        LOAD_WEAVER_PATH="null"
    fi
fi

if [ -n "$2" ]; then
    LOAD_TRIGGER_PATH="$2"
else
    BASE_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_SHORT}"
    LATEST_DIR=$(ls -td ${BASE_DIR}/pn=* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/trigger/trigger_lora" ]; then
        LOAD_TRIGGER_PATH="${LATEST_DIR}/trigger"
    else
        LOAD_TRIGGER_PATH="null"
    fi
fi

echo "============================================"
echo "Step 4: Evaluate Full MemGen"
echo "Model: ${MODEL_NAME}"
echo "Weaver Checkpoint: ${LOAD_WEAVER_PATH}"
echo "Trigger Checkpoint: ${LOAD_TRIGGER_PATH}"
echo "GPU: Single GPU"
echo "============================================"

python main.py \
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
    model.trigger.active True \
    run.mode evaluate \
    run.interaction.batch_size 4 \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
    2>&1 | tee ${SCRIPT_DIR}/logs/04_eval_trigger.log

echo "Full MemGen evaluation completed!"
