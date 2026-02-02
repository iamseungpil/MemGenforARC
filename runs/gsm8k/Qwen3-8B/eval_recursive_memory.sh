#!/bin/bash
# ============================================================================
# Wrapper: Recursive Memory Evaluation (Qwen3-8B, GSM8K)
# ============================================================================
# Sets experiment naming and calls the evaluation script.
#
# Usage:
#   bash runs/gsm8k/Qwen3-8B/eval_recursive_memory.sh <weaver_path>
#
# Customizable env vars (all optional):
#   EXPERIMENT_NAME        - experiment group name
#   CUDA_VISIBLE_DEVICES   - GPU selection (default: 0)
#   WANDB_MODE             - online/offline/disabled (default: online)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-recursive_memory_gsm8k_qwen3}

bash "${PROJECT_ROOT}/experiments/gsm8k_pipeline/Qwen3-8B/02_eval_recursive_memory.sh" "$@"
