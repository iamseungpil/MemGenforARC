#!/bin/bash
#
# Evaluation script: Vanilla (No Augmentation) - Qwen3-8B
#
# This script evaluates the model with:
# - max_prompt_aug_num: 0 (prompt augmentation DISABLED)
# - max_inference_aug_num: 0 (inference augmentation DISABLED)
#
# Purpose: Baseline without any MemGen augmentation

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export WANDB_ENTITY=gistdslab
export WANDB_PROJECT=memgen_reproduce
export WANDB_RUN_NAME="eval_gsm8k_qwen3-8b_vanilla"

# Model settings
MODEL_NAME="Qwen/Qwen3-8B"
# No weaver checkpoint needed for vanilla evaluation
LOAD_WEAVER_PATH="null"

# Dataset
DATASET_NAME="gsm8k"

# Augmentation settings - ALL DISABLED
MAX_PROMPT_AUG_NUM=0      # Prompt augmentation: DISABLED
MAX_INFERENCE_AUG_NUM=0   # Inference augmentation: DISABLED

# Latent settings (not used, but required for config)
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Output log
LOG_FILE="./logs/eval_vanilla_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./logs

echo "========================================"
echo "Evaluation: Vanilla (Qwen3-8B)"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Weaver checkpoint: ${LOAD_WEAVER_PATH} (not used)"
echo "max_prompt_aug_num: ${MAX_PROMPT_AUG_NUM} (DISABLED)"
echo "max_inference_aug_num: ${MAX_INFERENCE_AUG_NUM} (DISABLED)"
echo "Log file: ${LOG_FILE}"
echo "========================================"

cd /home/jovyan/MemGenforARC

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
    run.mode evaluate \
    run.interaction.batch_size 4 \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
    run.ltpo.enabled False \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "Evaluation complete! Log saved to: ${LOG_FILE}"
