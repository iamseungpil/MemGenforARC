#!/bin/bash
# ============================================================================
# Step 1: Recursive Memory Training (Qwen3-8B, GSM8K)
# ============================================================================
# Train WeaverStyleCompressor (~7.9M params, skip_projection mode).
# Base config: gsm8k_recursive_skip_proj.yaml
#
# Usage:
#   bash 01_recursive_memory_train.sh
#
# Output:
#   ~/data/memgen/train/gsm8k/Qwen3-8B/<EXPERIMENT_DIR>/<EXPERIMENT_SUBDIR>/weaver/
#
# For sbatch: set EXPERIMENT_NAME, SLURM_JOB_ID before calling this script.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================================
# Model / Dataset
# ============================================================================
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's|.*/||')
DATASET_NAME="gsm8k"

# Augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Recursive Memory configs (Qwen3-8B: hidden=4096, heads=8)
RECURSIVE_HIDDEN_SIZE=4096
RECURSIVE_NUM_HEADS=8

# Training hyperparameters
BATCH_SIZE=4
NUM_EPOCHS=2
LEARNING_RATE=1e-4

# ============================================================================
# EXPERIMENT_DIR pattern
# ============================================================================
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recursive_memory_gsm8k_qwen3}
TIMESTAMP=$(date +%y%m%d_%H%M%S)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  TIMESTAMP="${TIMESTAMP}_job${SLURM_JOB_ID}"
fi

export EXPERIMENT_DIR="${EXPERIMENT_NAME}"
export EXPERIMENT_SUBDIR="pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_${TIMESTAMP}"

# Deterministic checkpoint path
CHECKPOINT_DIR="${HOME}/data/memgen/train/${DATASET_NAME}/${MODEL_SHORT}/${EXPERIMENT_DIR}/${EXPERIMENT_SUBDIR}"
WEAVER_PATH="${CHECKPOINT_DIR}/weaver"

# ============================================================================
# Environment
# ============================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29541}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-gistdslab}
export WANDB_PROJECT=${WANDB_PROJECT:-RecursiveMem}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"train_recursive_memory_${MODEL_SHORT}_${TIMESTAMP}"}

echo "============================================"
echo "Recursive Memory Training (GSM8K, Qwen3-8B)"
echo "============================================"
echo "Model:          ${MODEL_NAME}"
echo "Dataset:        ${DATASET_NAME}"
echo "EXPERIMENT_DIR: ${EXPERIMENT_DIR}"
echo "EXPERIMENT_SUB: ${EXPERIMENT_SUBDIR}"
echo "Checkpoint:     ${WEAVER_PATH}"
echo "Start:          $(date)"
echo "============================================"

python -m accelerate.commands.launch \
  --config_file=configs/zero2.yaml \
  --num_processes=1 \
  main.py \
  --cfg-path configs/latent_memory/gsm8k_recursive_skip_proj.yaml \
  --options \
  model.model_name ${MODEL_NAME} \
  model.weaver.model_name ${MODEL_NAME} \
  model.recursive_memory.hidden_size ${RECURSIVE_HIDDEN_SIZE} \
  model.recursive_memory.num_heads ${RECURSIVE_NUM_HEADS} \
  run.weaver.sft.num_train_epochs ${NUM_EPOCHS} \
  run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
  run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
  run.weaver.sft.learning_rate ${LEARNING_RATE}

# Verify
if [ ! -f "${WEAVER_PATH}/recursive_memory.pt" ]; then
  echo "ERROR: Checkpoint not found at ${WEAVER_PATH}/recursive_memory.pt"
  exit 1
fi

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo "Checkpoint: ${WEAVER_PATH}"
echo ""
echo "Next: bash 02_eval_recursive_memory.sh ${WEAVER_PATH}"
echo "============================================"
