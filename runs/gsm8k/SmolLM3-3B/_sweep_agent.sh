#!/bin/bash
# ============================================================================
# wandb sweep agent: Recursive Memory (SmolLM3-3B, GSM8K)
# ============================================================================
# wandb agent가 호출하는 스크립트. --key=value 형태로 sweep 파라미터 수신.
# 직접 실행도 가능: bash sweep_recursive_memory.sh --max_cycles=5 --learning_rate=5e-5
# ============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================================
# Fixed: Model / Dataset (sweep 대상 아님)
# ============================================================================
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's|.*/||')
DATASET_NAME="gsm8k"
BASE_CONFIG="configs/latent_memory/gsm8k_recursive_skip_proj.yaml"

# SmolLM3-3B architecture (model-dependent, fixed)
RECURSIVE_HIDDEN_SIZE=2048
RECURSIVE_NUM_HEADS=16

# ============================================================================
# Defaults (sweep 파라미터로 override 가능)
# ============================================================================
# Cycle / Training
MAX_CYCLES=10
LEARNING_RATE=1e-4
NUM_EPOCHS=2
BATCH_SIZE=4

# Architecture toggles
BIDIRECTIONAL=false
CONTEXT_UPDATE=false
FULL_RANK_MLP=false

# Architecture numeric
ATTN_RANK=64
MLP_RANK=128
NUM_LATENTS=8

# Stepwise
STEPWISE_TRAINING=false
STEPWISE_LOSS_WEIGHT=0.5

# Two-level
TWO_LEVEL=false
L_CYCLES=6
MAX_H_CYCLES=5

# Projection / Augmentation
SKIP_PROJECTION=true
MAX_INFERENCE_AUG_NUM=5

# ============================================================================
# Parse sweep arguments (--key=value)
# ============================================================================
normalize_bool() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    true|1) echo "true" ;;
    false|0) echo "false" ;;
    *) echo "$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    # Cycle / Training
    --max_cycles=*)         MAX_CYCLES="${1#*=}" ;;
    --learning_rate=*)      LEARNING_RATE="${1#*=}" ;;
    --num_train_epochs=*)   NUM_EPOCHS="${1#*=}" ;;
    --batch_size=*)         BATCH_SIZE="${1#*=}" ;;
    # Architecture toggles
    --bidirectional=*)      BIDIRECTIONAL=$(normalize_bool "${1#*=}") ;;
    --context_update=*)     CONTEXT_UPDATE=$(normalize_bool "${1#*=}") ;;
    --full_rank_mlp=*)      FULL_RANK_MLP=$(normalize_bool "${1#*=}") ;;
    # Architecture numeric
    --attn_rank=*)          ATTN_RANK="${1#*=}" ;;
    --mlp_rank=*)           MLP_RANK="${1#*=}" ;;
    --num_latents=*)        NUM_LATENTS="${1#*=}" ;;
    # Stepwise
    --stepwise_training=*)  STEPWISE_TRAINING=$(normalize_bool "${1#*=}") ;;
    --stepwise_loss_weight=*) STEPWISE_LOSS_WEIGHT="${1#*=}" ;;
    # Two-level
    --two_level=*)          TWO_LEVEL=$(normalize_bool "${1#*=}") ;;
    --l_cycles=*)           L_CYCLES="${1#*=}" ;;
    --max_h_cycles=*)       MAX_H_CYCLES="${1#*=}" ;;
    # Projection / Augmentation
    --skip_projection=*)    SKIP_PROJECTION=$(normalize_bool "${1#*=}") ;;
    --max_inference_aug_num=*) MAX_INFERENCE_AUG_NUM="${1#*=}" ;;
    *) echo "WARN: Unknown sweep arg: $1" ;;
  esac
  shift
done

# ============================================================================
# EXPERIMENT_DIR
# ============================================================================
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-sweep_recursive_memory_gsm8k_smollm3}
TIMESTAMP=$(date +%y%m%d_%H%M%S)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  TIMESTAMP="${TIMESTAMP}_job${SLURM_JOB_ID}"
fi
export EXPERIMENT_DIR="${EXPERIMENT_NAME}"
export EXPERIMENT_SUBDIR="pn=1_pl=${NUM_LATENTS}_in=${MAX_INFERENCE_AUG_NUM}_il=${NUM_LATENTS}_${TIMESTAMP}"

# ============================================================================
# Environment
# ============================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29540}

# wandb: sweep agent가 WANDB_RUN_ID를 설정하므로 여기서는 건드리지 않음
export WANDB_ENTITY=${WANDB_ENTITY:-gistdslab}
export WANDB_PROJECT=${WANDB_PROJECT:-RecursiveMem}
export WANDB_RUN_NAME="mc${MAX_CYCLES}_lr${LEARNING_RATE}_ep${NUM_EPOCHS}_bi${BIDIRECTIONAL}_cu${CONTEXT_UPDATE}_nl${NUM_LATENTS}_tl${TWO_LEVEL}_${TIMESTAMP}"

# ============================================================================
# Print config
# ============================================================================
echo "============================================"
echo "Sweep Agent: Recursive Memory (GSM8K, SmolLM3-3B)"
echo "============================================"
echo "max_cycles=${MAX_CYCLES}  lr=${LEARNING_RATE}  epochs=${NUM_EPOCHS}  bs=${BATCH_SIZE}"
echo "bidirectional=${BIDIRECTIONAL}  context_update=${CONTEXT_UPDATE}  two_level=${TWO_LEVEL}"
echo "attn_rank=${ATTN_RANK}  mlp_rank=${MLP_RANK}  num_latents=${NUM_LATENTS}"
echo "stepwise=${STEPWISE_TRAINING}  two_level=${TWO_LEVEL}  skip_proj=${SKIP_PROJECTION}"
echo "max_inference_aug_num=${MAX_INFERENCE_AUG_NUM}"
echo "EXPERIMENT: ${EXPERIMENT_DIR}/${EXPERIMENT_SUBDIR}"
echo "============================================"

# ============================================================================
# Shared --options (train과 eval 모두 동일한 architecture 설정 필요)
# ============================================================================
SHARED_OPTIONS="\
  model.model_name ${MODEL_NAME} \
  model.weaver.model_name ${MODEL_NAME} \
  model.recursive_memory.hidden_size ${RECURSIVE_HIDDEN_SIZE} \
  model.recursive_memory.num_heads ${RECURSIVE_NUM_HEADS} \
  model.recursive_memory.max_cycles ${MAX_CYCLES} \
  model.recursive_memory.bidirectional ${BIDIRECTIONAL} \
  model.recursive_memory.context_update ${CONTEXT_UPDATE} \
  model.recursive_memory.full_rank_mlp ${FULL_RANK_MLP} \
  model.recursive_memory.attn_rank ${ATTN_RANK} \
  model.recursive_memory.mlp_rank ${MLP_RANK} \
  model.recursive_memory.skip_projection ${SKIP_PROJECTION} \
  model.recursive_memory.stepwise_training ${STEPWISE_TRAINING} \
  model.recursive_memory.stepwise_loss_weight ${STEPWISE_LOSS_WEIGHT} \
  model.recursive_memory.two_level ${TWO_LEVEL} \
  model.recursive_memory.l_cycles ${L_CYCLES} \
  model.recursive_memory.max_h_cycles ${MAX_H_CYCLES} \
  model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
  model.weaver.prompt_latents_len ${NUM_LATENTS} \
  model.weaver.inference_latents_len ${NUM_LATENTS}"

# Deterministic checkpoint path
WEAVER_PATH="${HOME}/data/memgen/train/${DATASET_NAME}/${MODEL_SHORT}/${EXPERIMENT_DIR}/${EXPERIMENT_SUBDIR}/weaver"

# ============================================================================
# Train
# ============================================================================
echo "[1/2] Training..."

python -m accelerate.commands.launch \
  --config_file=configs/zero2.yaml \
  --num_processes=1 \
  main.py \
  --cfg-path ${BASE_CONFIG} \
  --options \
  ${SHARED_OPTIONS} \
  run.weaver.sft.num_train_epochs ${NUM_EPOCHS} \
  run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
  run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
  run.weaver.sft.learning_rate ${LEARNING_RATE} \
  run.weaver.sft.run_name ${WANDB_RUN_NAME}

# ============================================================================
# Verify checkpoint
# ============================================================================
if [ ! -f "${WEAVER_PATH}/recursive_memory.pt" ]; then
  echo "ERROR: Checkpoint not found at ${WEAVER_PATH}/recursive_memory.pt"
  exit 1
fi

# ============================================================================
# Eval
# ============================================================================
echo ""
echo "[2/2] Evaluating: ${WEAVER_PATH}"

EVAL_CONFIG="configs/latent_memory/gsm8k_recursive_eval.yaml"

python -m accelerate.commands.launch \
  --config_file=configs/zero2.yaml \
  --num_processes=1 \
  main.py \
  --cfg-path ${EVAL_CONFIG} \
  --options \
  ${SHARED_OPTIONS} \
  model.load_weaver_path "${WEAVER_PATH}"

echo "============================================"
echo "Sweep trial done: $(date)"
echo "Checkpoint: ${WEAVER_PATH}"
echo "============================================"
