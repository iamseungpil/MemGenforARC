#!/bin/bash
# =============================================================================
# ARC Two-Stage Training Script
# =============================================================================
#
# This script runs the ARC two-stage training pipeline:
# - Stage 1: Generate instructions (with memory capture)
# - Stage 2: Generate grids using instructions + memory
#
# Usage:
#   ./scripts/train_arc_twostage.sh [GPU_ID] [CONFIG_OVERRIDE...]
#
# Examples:
#   # Train with default config on GPU 0
#   ./scripts/train_arc_twostage.sh
#
#   # Train on specific GPU
#   ./scripts/train_arc_twostage.sh 0
#
#   # Train with Qwen instead of GPT-OSS
#   ./scripts/train_arc_twostage.sh 0 model.model_name Qwen/Qwen2.5-7B-Instruct
#
#   # Train with fewer candidates for faster iteration
#   ./scripts/train_arc_twostage.sh 0 arc.instruction_candidates 3 arc.refinement_turns 5
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# GPU selection (default: 2,3 for multi-GPU training)
GPU_IDS="${1:-2,3}"
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
shift 2>/dev/null || true  # Shift past GPU_IDS if provided

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${PROJECT_DIR}/configs/arc_twostage.yaml"
# Use 2-GPU config for multi-GPU, single GPU config for single GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    ACCELERATE_CONFIG="${PROJECT_DIR}/configs/zero2_2gpu.yaml"
else
    ACCELERATE_CONFIG="${PROJECT_DIR}/configs/zero2.yaml"
fi

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/outputs/arc_twostage_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Environment Setup
# =============================================================================

echo "========================================"
echo "ARC Two-Stage Training (GPT-OSS)"
echo "========================================"
echo "Project: ${PROJECT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Accelerate: ${ACCELERATE_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPU_IDS} (${NUM_GPUS} GPUs)"
echo "========================================"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Change to project directory
cd "${PROJECT_DIR}"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "memgen_arc"; then
        conda activate memgen_arc
        echo "Activated conda environment: memgen_arc"
    elif conda env list | grep -q "gptoss"; then
        conda activate gptoss
        echo "Activated conda environment: gptoss"
    fi
fi

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo "Starting training..."
echo ""

# Build options string from remaining arguments
OPTIONS=""
if [ $# -gt 0 ]; then
    OPTIONS="--options run.output_dir ${OUTPUT_DIR} $*"
else
    OPTIONS="--options run.output_dir ${OUTPUT_DIR}"
fi

# Run with accelerate for distributed training support
python -m accelerate.commands.launch \
    --config_file="${ACCELERATE_CONFIG}" \
    --num_processes=${NUM_GPUS} \
    --main_process_port=29502 \
    -m arc.runner \
    --cfg-path "${CONFIG_PATH}" \
    ${OPTIONS}

# =============================================================================
# Post-Training
# =============================================================================

echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Show training results if available
if [ -f "${OUTPUT_DIR}/weaver/trainer_state.json" ]; then
    echo "Final training state:"
    python -c "
import json
with open('${OUTPUT_DIR}/weaver/trainer_state.json') as f:
    state = json.load(f)
    print(f'  Epoch: {state.get(\"epoch\", \"N/A\")}')
    print(f'  Global step: {state.get(\"global_step\", \"N/A\")}')
    print(f'  Best metric: {state.get(\"best_metric\", \"N/A\")}')
"
fi

echo ""
echo "To evaluate the trained model:"
echo "  python -m arc.runner --cfg-path ${CONFIG_PATH} --options run.mode eval run.output_dir ${OUTPUT_DIR} model.load_model_path ${OUTPUT_DIR}/weaver"
echo ""
