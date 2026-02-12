#!/bin/bash
# ============================================================================
# 1-Cycle Recursive Memory: Train + Eval (SmolLM3-3B, GSM8K)
# ============================================================================
# Trains 1-cycle recursive memory, then evaluates the resulting checkpoint.
#
# Usage:
#   bash runs/gsm8k/SmolLM3-3B/train_eval_1cycle.sh
#
# wandb:
#   Project: ReMem_confidence
#   Train Run: smollm3_1cycle_<timestamp>
#   Eval Run: eval_smollm3_1cycle_<timestamp>
# ============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================================
# Model / Dataset
# ============================================================================
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's|.*/||')
DATASET_NAME="gsm8k"

# Augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Recursive Memory configs (SmolLM3: hidden=2048, heads=16)
RECURSIVE_HIDDEN_SIZE=2048
RECURSIVE_NUM_HEADS=16
MAX_CYCLES=1  # 1-cycle training

# Training hyperparameters
BATCH_SIZE=4
NUM_EPOCHS=2
LEARNING_RATE=1e-4

# ============================================================================
# EXPERIMENT_DIR pattern
# ============================================================================
EXPERIMENT_NAME="smollm3_1cycle_confidence"
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
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29531}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-gistdslab}
export WANDB_PROJECT="ReMem_confidence"

# ============================================================================
# [1/2] Training
# ============================================================================
export WANDB_RUN_NAME="smollm3_1cycle_${TIMESTAMP}"

echo "============================================"
echo "[1/2] 1-Cycle Recursive Memory Training"
echo "============================================"
echo "Model:          ${MODEL_NAME}"
echo "Dataset:        ${DATASET_NAME}"
echo "Max Cycles:     ${MAX_CYCLES}"
echo "EXPERIMENT_DIR: ${EXPERIMENT_DIR}"
echo "EXPERIMENT_SUB: ${EXPERIMENT_SUBDIR}"
echo "Checkpoint:     ${WEAVER_PATH}"
echo "wandb Project:  ${WANDB_PROJECT}"
echo "wandb Run:      ${WANDB_RUN_NAME}"
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
  model.recursive_memory.max_cycles ${MAX_CYCLES} \
  model.recursive_memory.two_level false \
  run.weaver.sft.num_train_epochs ${NUM_EPOCHS} \
  run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
  run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
  run.weaver.sft.learning_rate ${LEARNING_RATE}

# Verify training checkpoint
if [ ! -f "${WEAVER_PATH}/recursive_memory.pt" ]; then
  echo "ERROR: Checkpoint not found at ${WEAVER_PATH}/recursive_memory.pt"
  exit 1
fi

echo ""
echo "============================================"
echo "Training completed! Starting evaluation..."
echo "Checkpoint: ${WEAVER_PATH}"
echo "============================================"

# ============================================================================
# [2/2] Evaluation
# ============================================================================
export WANDB_RUN_NAME="eval_smollm3_1cycle_${TIMESTAMP}"

echo ""
echo "============================================"
echo "[2/2] 1-Cycle Recursive Memory Evaluation"
echo "============================================"
echo "wandb Run:      ${WANDB_RUN_NAME}"
echo "============================================"

python main.py \
  --cfg-path configs/latent_memory/gsm8k_recursive_skip_proj.yaml \
  --options \
  model.model_name ${MODEL_NAME} \
  model.weaver.model_name ${MODEL_NAME} \
  model.load_weaver_path ${WEAVER_PATH} \
  model.recursive_memory.hidden_size ${RECURSIVE_HIDDEN_SIZE} \
  model.recursive_memory.num_heads ${RECURSIVE_NUM_HEADS} \
  model.recursive_memory.max_cycles ${MAX_CYCLES} \
  model.recursive_memory.two_level false \
  run.mode evaluate \
  run.interaction.do_sample false \
  run.interaction.temperature 0.0 \
  run.interaction.max_response_length 512

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
echo "Checkpoint: ${WEAVER_PATH}"
echo "wandb Project: ${WANDB_PROJECT}"
echo "============================================"
