#!/bin/bash
# ============================================================================
# Sweep Launcher: Recursive Memory (SmolLM3-3B, GSM8K)
# ============================================================================
# sweep 생성 + agent 실행을 한 번에 수행.
# 사용법:
#   bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh
#   bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --count 10
#   CUDA_VISIBLE_DEVICES=0 bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

SWEEP_YAML="runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.yaml"
SWEEP_COUNT=${1:-30}

# --count=N 형태도 지원
if [[ "${SWEEP_COUNT}" == --count=* ]]; then
  SWEEP_COUNT="${SWEEP_COUNT#*=}"
elif [[ "${SWEEP_COUNT}" == "--count" ]]; then
  SWEEP_COUNT="${2:-30}"
fi

echo "============================================"
echo "Sweep Launcher: Recursive Memory (GSM8K, SmolLM3-3B)"
echo "============================================"
echo "YAML:   ${SWEEP_YAML}"
echo "COUNT:  ${SWEEP_COUNT}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================"

# sweep 생성
SWEEP_OUTPUT=$(wandb sweep "${SWEEP_YAML}" 2>&1)
echo "${SWEEP_OUTPUT}"

SWEEP_ID=$(echo "${SWEEP_OUTPUT}" | grep -oP 'wandb agent \K\S+')
if [ -z "${SWEEP_ID}" ]; then
  echo "ERROR: sweep ID를 파싱할 수 없습니다."
  exit 1
fi

echo ""
echo "Sweep ID: ${SWEEP_ID}"
echo "Starting agent (count=${SWEEP_COUNT})..."
echo "============================================"

wandb agent --count "${SWEEP_COUNT}" "${SWEEP_ID}"

echo "============================================"
echo "Sweep done: $(date)"
echo "============================================"
