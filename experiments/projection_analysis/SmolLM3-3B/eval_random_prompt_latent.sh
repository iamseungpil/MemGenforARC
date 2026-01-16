#!/bin/bash
#
# Evaluation script: Random Prompt Latent Only - SmolLM3-3B
#
# This script evaluates the model with:
# - load_prompt_query_latents: false (prompt latents RANDOM)
# - load_inference_query_latents: true (inference latents LOADED)
#
# Purpose: Ablation study - measure contribution of prompt latents

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=online
export WANDB_ENTITY=gistdslab
export WANDB_PROJECT=memgen_reproduce
export WANDB_RUN_NAME="eval_gsm8k_smollm3-3b_random-prompt-latent"

# Model settings
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
LOAD_WEAVER_PATH="/home/jovyan/data/memgen/train/gsm8k/SmolLM3-3B/pn=1_pl=8_in=5_il=8_20260110-195742/weaver"

# Dataset
DATASET_NAME="gsm8k"

# Augmentation settings
MAX_PROMPT_AUG_NUM=1      # Prompt augmentation: ENABLED
MAX_INFERENCE_AUG_NUM=5   # Inference augmentation: ENABLED

# Latent settings (must match training config)
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Skip-LoRA mode (same as training)
SKIP_LORA=True

# Latent loading settings (NEW)
LOAD_PROMPT_QUERY_LATENTS=False   # Keep prompt latents RANDOM
LOAD_INFERENCE_QUERY_LATENTS=True # Load trained inference latents

# Output log
LOG_FILE="./logs/eval_random_prompt_latent_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./logs

echo "========================================"
echo "Evaluation: Random Prompt Latent (SmolLM3-3B)"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Weaver checkpoint: ${LOAD_WEAVER_PATH}"
echo "max_prompt_aug_num: ${MAX_PROMPT_AUG_NUM}"
echo "max_inference_aug_num: ${MAX_INFERENCE_AUG_NUM}"
echo "skip_lora: ${SKIP_LORA}"
echo "load_prompt_query_latents: ${LOAD_PROMPT_QUERY_LATENTS}"
echo "load_inference_query_latents: ${LOAD_INFERENCE_QUERY_LATENTS}"
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
    model.load_prompt_query_latents ${LOAD_PROMPT_QUERY_LATENTS} \
    model.load_inference_query_latents ${LOAD_INFERENCE_QUERY_LATENTS} \
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
