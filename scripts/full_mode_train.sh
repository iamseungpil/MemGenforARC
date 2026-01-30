#!/bin/bash
# ============================================================================
# Full Mode Training (LoRA + Query Latents + Projections)
# ============================================================================
# Reproduce 1/8 checkpoint with same settings
# ============================================================================

set -e

# Activate conda environment
source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate memgen

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_check"
export WANDB_RUN_NAME="full_mode_qwen3_8b_$(date +%Y%m%d_%H%M%S)"
export DEBUG_MODE=true
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29600
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Project root
cd /home/jovyan/MemGenWorkspace/ltpo_sub

# Model Configuration (same as 1/8 checkpoint)
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"
TRAIN_METHOD="sft"

# MemGen Paper settings
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Training hyperparameters (same as 1/8 checkpoint)
BATCH_SIZE=4  # May need to reduce if OOM
NUM_EPOCHS=2
LEARNING_RATE=1e-5

echo "============================================"
echo "Full Mode Training (LoRA + Query Latents)"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Method: ${TRAIN_METHOD}"
echo "Augmentation: prompt=${MAX_PROMPT_AUG_NUM}, inference=${MAX_INFERENCE_AUG_NUM}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "============================================"

mkdir -p logs

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py \
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
    dataset.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.sft.num_train_epochs ${NUM_EPOCHS} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
    run.weaver.sft.learning_rate ${LEARNING_RATE} \
    run.weaver.sft.bf16 True \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024 \
    2>&1 | tee logs/full_mode_train_$(date +%Y%m%d_%H%M%S).log

echo "============================================"
echo "Full Mode training completed!"
echo "============================================"
