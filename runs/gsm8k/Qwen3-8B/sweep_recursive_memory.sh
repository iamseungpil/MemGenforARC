#!/bin/bash
# ============================================================================
# Sweep Launcher: Recursive Memory (Qwen3-8B, GSM8K)
# ============================================================================
# sweep 생성 + agent 실행을 한 번에 수행.
# 사용법:
#   bash runs/gsm8k/Qwen3-8B/sweep_recursive_memory.sh
#   bash runs/gsm8k/Qwen3-8B/sweep_recursive_memory.sh --count 10
#   bash runs/gsm8k/Qwen3-8B/sweep_recursive_memory.sh --count 30 --gpus 0,1
#   bash runs/gsm8k/Qwen3-8B/sweep_recursive_memory.sh --count 30 --gpus 0,1 --agents-per-gpu 2
#
# 옵션:
#   --count N            총 trial 수 (default: 30)
#   --gpus 0,1,2         사용할 GPU (default: CUDA_VISIBLE_DEVICES 또는 0)
#   --agents-per-gpu N   GPU당 agent 수 (default: 1)
#   --sweep-id ID        기존 sweep에 agent 붙이기 (새 sweep 생성 안 함)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

SWEEP_YAML="runs/gsm8k/Qwen3-8B/sweep_recursive_memory.yaml"
SWEEP_COUNT=30
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
AGENTS_PER_GPU=1
SWEEP_ID=""

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --count=*)          SWEEP_COUNT="${1#*=}" ;;
    --count)            SWEEP_COUNT="${2}"; shift ;;
    --gpus=*)           GPUS="${1#*=}" ;;
    --gpus)             GPUS="${2}"; shift ;;
    --agents-per-gpu=*) AGENTS_PER_GPU="${1#*=}" ;;
    --agents-per-gpu)   AGENTS_PER_GPU="${2}"; shift ;;
    --sweep-id=*)       SWEEP_ID="${1#*=}" ;;
    --sweep-id)         SWEEP_ID="${2}"; shift ;;
    *) echo "WARN: Unknown arg: $1" ;;
  esac
  shift
done

# GPU 목록 → 배열
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}
TOTAL_AGENTS=$((NUM_GPUS * AGENTS_PER_GPU))

echo "============================================"
echo "Sweep Launcher: Recursive Memory (GSM8K, Qwen3-8B)"
echo "============================================"
echo "YAML:            ${SWEEP_YAML}"
echo "COUNT:           ${SWEEP_COUNT}"
echo "GPUs:            ${GPUS} (${NUM_GPUS}개)"
echo "Agents per GPU:  ${AGENTS_PER_GPU}"
echo "Total agents:    ${TOTAL_AGENTS}"
echo "============================================"

# ============================================================================
# Sweep 생성 또는 기존 sweep 사용
# ============================================================================
if [ -z "${SWEEP_ID}" ]; then
  SWEEP_OUTPUT=$(wandb sweep "${SWEEP_YAML}" 2>&1)
  echo "${SWEEP_OUTPUT}"

  SWEEP_ID=$(echo "${SWEEP_OUTPUT}" | grep -oP 'wandb agent \K\S+')
  if [ -z "${SWEEP_ID}" ]; then
    echo "ERROR: sweep ID를 파싱할 수 없습니다."
    exit 1
  fi
else
  echo "기존 sweep 사용: ${SWEEP_ID}"
fi

echo ""
echo "Sweep ID: ${SWEEP_ID}"
echo "Starting ${TOTAL_AGENTS} agent(s)..."
echo "============================================"

# ============================================================================
# Agent 실행
# ============================================================================
if [ "${TOTAL_AGENTS}" -eq 1 ]; then
  # 단일 agent: foreground 실행
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" wandb agent --count "${SWEEP_COUNT}" "${SWEEP_ID}"
else
  # 복수 agent: background 실행
  PIDS=()
  for gpu in "${GPU_LIST[@]}"; do
    for ((a=0; a<AGENTS_PER_GPU; a++)); do
      echo "  → GPU ${gpu}, agent $((a+1))/${AGENTS_PER_GPU} 시작"
      CUDA_VISIBLE_DEVICES="${gpu}" wandb agent --count "${SWEEP_COUNT}" "${SWEEP_ID}" &
      PIDS+=($!)
    done
  done

  echo ""
  echo "${#PIDS[@]} agents running (PIDs: ${PIDS[*]})"
  echo "============================================"

  # 모든 agent 종료 대기
  FAILED=0
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || ((FAILED++))
  done

  if [ "${FAILED}" -gt 0 ]; then
    echo "WARN: ${FAILED}/${#PIDS[@]} agent(s) failed"
  fi
fi

echo "============================================"
echo "Sweep done: $(date)"
echo "Sweep ID: ${SWEEP_ID}"
echo "============================================"
