#!/bin/bash
# ============================================================================
# Evaluate Skip-Projection Mode (Query Latents + LLM, NO Projections)
# ============================================================================
# Tests whether projections are necessary for MemGen performance
# ============================================================================

set -e

# Activate conda environment
source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate memgen

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_check"
export WANDB_RUN_NAME="eval_skip_projection_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES=0

# Project root
cd /home/jovyan/MemGenWorkspace/ltpo_sub

# Model Configuration
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"

# Skip-projection trained checkpoint
CHECKPOINT_PATH="/home/jovyan/data/memgen/train/gsm8k/Qwen3-8B/pn=1_pl=8_in=5_il=8_20260118-220333/weaver"

# MemGen settings (same as 1/8 checkpoint)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

BATCH_SIZE=8

echo "============================================"
echo "Evaluating Skip-Projection Mode"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Mode: skip_projection=True + skip_lora=True (Query Latents only, NO LoRA, NO Projections)"
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
    model.skip_projection True \
    model.skip_lora True \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.do_sample False \
    run.ltpo.enabled False \
    2>&1 | tee logs/eval_skip_projection_$(date +%Y%m%d_%H%M%S).log

echo "============================================"
echo "Skip-Projection Evaluation completed!"
echo "============================================"
