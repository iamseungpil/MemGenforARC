#!/bin/bash
# ============================================================================
# Evaluate Output-Projection-Only Mode
# ============================================================================
# Tests: Query Latents + LLM (no LoRA) + weaver_to_reasoner only
# Skip: reasoner_to_weaver (input projection)
# ============================================================================

set -e

# Activate conda environment
source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate memgen

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_check"
export WANDB_RUN_NAME="eval_output_proj_only_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES=0

# Project root
cd /home/jovyan/MemGenWorkspace/ltpo_sub

# Model Configuration
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"

# 1/8 Checkpoint path (for query_latents and weaver_to_reasoner)
CHECKPOINT_PATH="/home/jovyan/MemGenWorkspace/_shared/MemGen_reproduce/results/train/gsm8k/Qwen3-8B/pn=1_pl=8_in=5_il=8_20260110-014247/weaver/weaver_lora"

# MemGen settings
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

BATCH_SIZE=8

echo "============================================"
echo "Evaluating Output-Projection-Only Mode"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Mode: output_projection_only=True"
echo "  - Input: direct (NO reasoner_to_weaver)"
echo "  - LLM: base model (NO LoRA)"
echo "  - Output: weaver_to_reasoner (YES)"
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
    model.output_projection_only True \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.do_sample False \
    run.ltpo.enabled False \
    2>&1 | tee logs/eval_output_projection_only_$(date +%Y%m%d_%H%M%S).log

echo "============================================"
echo "Output-Projection-Only Evaluation completed!"
echo "============================================"
