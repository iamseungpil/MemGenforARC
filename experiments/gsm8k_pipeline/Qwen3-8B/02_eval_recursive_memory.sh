#!/bin/bash
# ============================================================================
# Recursive Memory Evaluation (Qwen3-8B, GSM8K)
# ============================================================================
# Evaluate trained recursive_memory checkpoint on GSM8K test set.
# Base config: gsm8k_recursive_eval.yaml (Qwen3-8B)
#
# Usage:
#   bash 02_eval_recursive_memory.sh <weaver_path>
#   bash 02_eval_recursive_memory.sh /path/to/weaver
#
# For sbatch: set LOAD_WEAVER_PATH env var before calling this script.
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

# Recursive Memory configs (Qwen3-8B: hidden=4096, heads=8)
RECURSIVE_HIDDEN_SIZE=4096
RECURSIVE_NUM_HEADS=8

# ============================================================================
# Checkpoint path: argument > env var > error
# ============================================================================
if [ -n "$1" ]; then
  LOAD_WEAVER_PATH="$1"
elif [ -z "${LOAD_WEAVER_PATH:-}" ]; then
  echo "Usage: bash 02_eval_recursive_memory.sh <weaver_path>"
  echo "  or set LOAD_WEAVER_PATH env var"
  exit 1
fi

if [ ! -f "${LOAD_WEAVER_PATH}/recursive_memory.pt" ]; then
  echo "ERROR: ${LOAD_WEAVER_PATH}/recursive_memory.pt not found"
  exit 1
fi

# ============================================================================
# EXPERIMENT_DIR pattern (for eval output directory)
# ============================================================================
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recursive_memory_gsm8k_qwen3}
TIMESTAMP=$(date +%y%m%d_%H%M%S)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  TIMESTAMP="${TIMESTAMP}_job${SLURM_JOB_ID}"
fi

export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${EXPERIMENT_NAME}}"
export EXPERIMENT_SUBDIR="${EXPERIMENT_SUBDIR:-eval_${TIMESTAMP}}"

# ============================================================================
# Environment
# ============================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29533}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-gistdslab}
export WANDB_PROJECT=${WANDB_PROJECT:-RecursiveMem}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"eval_recursive_memory_${MODEL_SHORT}_${TIMESTAMP}"}

echo "============================================"
echo "Recursive Memory Evaluation (GSM8K, Qwen3-8B)"
echo "============================================"
echo "Model:          ${MODEL_NAME}"
echo "Dataset:        ${DATASET_NAME}"
echo "Checkpoint:     ${LOAD_WEAVER_PATH}"
echo "EXPERIMENT_DIR: ${EXPERIMENT_DIR}"
echo "Start:          $(date)"
echo "============================================"

python -m accelerate.commands.launch \
  --config_file=configs/zero2.yaml \
  --num_processes=1 \
  main.py \
  --cfg-path configs/latent_memory/gsm8k_recursive_eval.yaml \
  --options \
  model.model_name ${MODEL_NAME} \
  model.weaver.model_name ${MODEL_NAME} \
  model.load_weaver_path "${LOAD_WEAVER_PATH}" \
  model.recursive_memory.hidden_size ${RECURSIVE_HIDDEN_SIZE} \
  model.recursive_memory.num_heads ${RECURSIVE_NUM_HEADS} \
  model.recursive_memory.skip_projection true

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "============================================"
