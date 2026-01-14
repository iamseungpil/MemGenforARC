#!/bin/bash
# ============================================================================
# Step 1: Weaver SFT Training
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
export WANDB_RUN_NAME="weaver_sft_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Resume from checkpoint (comment out or set empty to disable)
RESUME_CHECKPOINT="/home/jovyan/MemGenforARC/MemGen_reproduce/results/train/gsm8k/SmolLM3-3B/pn=1_pl=8_in=5_il=8_20260113-094143/weaver/checkpoint-1682"
# RESUME_CHECKPOINT=""  # Uncomment this line to disable resume

echo "============================================"
echo "Step 1: Weaver SFT Training"
echo "Model: ${MODEL_NAME}"
echo "GPU: Single GPU"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "Resume from: ${RESUME_CHECKPOINT}"
fi
echo "============================================"

# Build resume option if checkpoint is set
RESUME_OPT=""
if [ -n "${RESUME_CHECKPOINT}" ]; then
    RESUME_OPT="run.weaver.sft.resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

python main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active False \
    dataset.mode sft \
    run.mode train \
    run.train_weaver True \
    run.train_weaver_method sft \
    run.train_trigger False \
    run.weaver.sft.num_train_epochs 2 \
    run.weaver.sft.per_device_train_batch_size 4 \
    run.weaver.sft.per_device_eval_batch_size 4 \
    run.weaver.sft.gradient_accumulation_steps 1 \
    run.weaver.sft.learning_rate 1e-5 \
    run.weaver.sft.warmup_ratio 0.1 \
    ${RESUME_OPT} \
    2>&1 | tee ${SCRIPT_DIR}/logs/01_weaver_sft.log

echo "Weaver SFT training completed!"
echo "Next: ./02_eval_weaver.sh"
