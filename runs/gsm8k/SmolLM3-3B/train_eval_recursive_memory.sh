#!/bin/bash
# ============================================================================
# Wrapper: Recursive Memory Train + Eval (SmolLM3-3B, GSM8K)
# ============================================================================
# Trains recursive memory, then evaluates the resulting checkpoint.
#
# Usage:
#   bash runs/gsm8k/SmolLM3-3B/train_eval_recursive_memory.sh
#
# Customizable env vars (all optional):
#   EXPERIMENT_NAME        - experiment group name
#   CUDA_VISIBLE_DEVICES   - GPU selection (default: 0)
#   WANDB_MODE             - online/offline/disabled (default: online)
# ============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-recursive_memory_gsm8k_smollm3}

# --- Train ---
TRAIN_LOG=$(mktemp /tmp/train_recursive_memory_XXXXXX.log)

bash "${PROJECT_ROOT}/experiments/gsm8k_pipeline/SmolLM3-3B/01_recursive_memory_train.sh" 2>&1 | tee "${TRAIN_LOG}"

# Parse checkpoint path from training output
WEAVER_PATH=$(grep "^Checkpoint:" "${TRAIN_LOG}" | tail -1 | sed 's/^Checkpoint:[[:space:]]*//')

rm -f "${TRAIN_LOG}"

if [ -z "${WEAVER_PATH}" ] || [ ! -d "${WEAVER_PATH}" ]; then
  echo "ERROR: Could not find checkpoint directory from training output"
  echo "Parsed: ${WEAVER_PATH}"
  exit 1
fi

echo ""
echo "============================================"
echo "Training done. Starting evaluation..."
echo "Checkpoint: ${WEAVER_PATH}"
echo "============================================"

# --- Eval ---
bash "${PROJECT_ROOT}/experiments/gsm8k_pipeline/SmolLM3-3B/02_eval_recursive_memory.sh" "${WEAVER_PATH}"
