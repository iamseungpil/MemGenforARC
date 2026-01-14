#!/bin/bash
# ============================================================================
# Step 5: LTPO Evaluation (Test-Time Optimization)
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
export WANDB_RUN_NAME="ltpo_eval_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# LTPO 설정
LTPO_LR=0.03
LTPO_SIGMA=0.1
LTPO_SIGMA_DECAY=0.99
LTPO_MAX_STEPS=10
LTPO_TOP_K=10
LTPO_USE_AUTO_GRAD=true
LTPO_VERBOSE=1

OUTPUT_ROOT="${PROJECT_ROOT}/results"
if [ -n "$1" ]; then
    LOAD_WEAVER_PATH="$1"
else
    BASE_DIR="${OUTPUT_ROOT}/train/${DATASET_NAME}/${MODEL_SHORT}"
    LATEST_DIR=$(ls -td ${BASE_DIR}/pn=* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -d "${LATEST_DIR}/weaver/weaver_lora" ]; then
        LOAD_WEAVER_PATH="${LATEST_DIR}/weaver"
    else
        LOAD_WEAVER_PATH="null"
    fi
fi

echo "============================================"
echo "Step 5: LTPO Evaluation"
echo "Model: ${MODEL_NAME}"
echo "Weaver Checkpoint: ${LOAD_WEAVER_PATH}"
echo "LTPO Settings:"
echo "  - lr: ${LTPO_LR}"
echo "  - sigma: ${LTPO_SIGMA}"
echo "  - max_steps: ${LTPO_MAX_STEPS}"
echo "  - use_auto_grad: ${LTPO_USE_AUTO_GRAD}"
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
    model.trigger.active False \
    run.mode evaluate_ltpo \
    run.interaction.batch_size 1 \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
    run.ltpo.enabled true \
    run.ltpo.lr ${LTPO_LR} \
    run.ltpo.sigma ${LTPO_SIGMA} \
    run.ltpo.sigma_decay ${LTPO_SIGMA_DECAY} \
    run.ltpo.max_steps ${LTPO_MAX_STEPS} \
    run.ltpo.top_k ${LTPO_TOP_K} \
    run.ltpo.use_auto_grad ${LTPO_USE_AUTO_GRAD} \
    run.ltpo.verbose ${LTPO_VERBOSE} \
    2>&1 | tee ${SCRIPT_DIR}/logs/05_ltpo_eval.log

echo "LTPO evaluation completed!"
