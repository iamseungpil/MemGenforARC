#!/bin/bash
# ============================================================================
# Evaluate Full Mode Checkpoint
# ============================================================================

set -e

# Activate conda environment
source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate memgen

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_check"
export WANDB_RUN_NAME="eval_full_mode_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES=0

# Project root
cd /home/jovyan/MemGenWorkspace/ltpo_sub

# Model Configuration
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"

# Checkpoint path (just trained)
CHECKPOINT_PATH="/home/jovyan/data/memgen/train/gsm8k/Qwen3-8B/pn=1_pl=8_in=5_il=8_20260117-200714/weaver/weaver_lora"

# MemGen settings (same as training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

BATCH_SIZE=8

echo "============================================"
echo "Evaluating Full Mode Checkpoint"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "============================================"

mkdir -p logs

python main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_weaver_path ${CHECKPOINT_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active False \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.do_sample False \
    run.ltpo.enabled False \
    2>&1 | tee logs/eval_full_mode_$(date +%Y%m%d_%H%M%S).log

echo "============================================"
echo "Evaluation completed!"
echo "============================================"
