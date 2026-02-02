#!/bin/bash
# ============================================================================
# Wrapper: Recursive Memory Training (SmolLM3-3B, GSM8K)
# ============================================================================
# Sets experiment naming and calls the training script.
#
# Usage:
#   bash runs/gsm8k/SmolLM3-3B/train_recursive_memory.sh
#
# Customizable env vars (all optional):
#   EXPERIMENT_NAME        - experiment group name
#   CUDA_VISIBLE_DEVICES   - GPU selection (default: 0)
#   WANDB_MODE             - online/offline/disabled (default: online)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-recursive_memory_gsm8k_smollm3}

bash "${PROJECT_ROOT}/experiments/gsm8k_pipeline/SmolLM3-3B/01_recursive_memory_train.sh"
