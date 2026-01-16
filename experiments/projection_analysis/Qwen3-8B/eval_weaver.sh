#!/bin/bash
#
# Evaluation script: Standard Weaver Evaluation (Qwen3-8B)
#
# This script evaluates the trained Weaver model with standard settings.

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export WANDB_ENTITY=gistdslab
export WANDB_PROJECT=memgen_reproduce
export WANDB_RUN_NAME="eval_gsm8k_qwen3-8b_weaver"

# Model settings
MODEL_NAME="Qwen/Qwen3-8B"
LOAD_WEAVER_PATH="/home/jovyan/data/memgen/train/gsm8k/Qwen3-8B/pn=1_pl=8_in=5_il=8_20260113-221706/weaver"

# Dataset
DATASET_NAME="gsm8k"

# Augmentation settings (standard)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5

# Latent settings (must match training config)
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Skip-LoRA mode
SKIP_LORA=True

# Output log
LOG_FILE="./logs/eval_weaver_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./logs

echo "========================================"
echo "Evaluation: Standard Weaver (Qwen3-8B)"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Weaver checkpoint: ${LOAD_WEAVER_PATH}"
echo "max_prompt_aug_num: ${MAX_PROMPT_AUG_NUM}"
echo "max_inference_aug_num: ${MAX_INFERENCE_AUG_NUM}"
echo "skip_lora: ${SKIP_LORA}"
echo "Log file: ${LOG_FILE}"
echo "========================================"

cd /home/jovyan/MemGenforARC

python main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_weaver_path ${LOAD_WEAVER_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.skip_lora ${SKIP_LORA} \
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
