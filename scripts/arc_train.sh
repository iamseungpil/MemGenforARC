#!/bin/bash
# ARC Training Script for MemGen Weaver
# Trains Weaver with GRPO using grid_similarity reward

set -e

echo "=========================================="
echo "MemGen ARC Training"
echo "=========================================="

# Configuration
CONFIG_FILE="configs/latent_memory/arc.yaml"
OUTPUT_DIR="outputs/arc_weaver_grpo"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Set environment
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with accelerate
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=$NUM_GPUS \
    main.py \
    --cfg-path $CONFIG_FILE \
    --options \
    run.mode train \
    run.train_weaver true \
    run.train_weaver_method grpo \
    2>&1 | tee $OUTPUT_DIR/train.log

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
