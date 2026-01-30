#!/bin/bash
# RecursiveMemory GRPO Training Script
# Uses Qwen3-8B with 10 compression cycles
# GRPO is compatible with RecursiveMemory - valid_mask filtering handles shape matching

set -e

# Configuration
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-memgen_check}

# Model settings
MODEL_NAME="Qwen/Qwen3-8B"
MAX_CYCLES=10

# GRPO settings (must satisfy: batch_size divisible by num_generations)
BATCH_SIZE=2
NUM_GENERATIONS=2
MAX_COMPLETION_LENGTH=1024

# Optional: load from SFT checkpoint
LOAD_WEAVER_PATH="${1:-}"

echo "======================================"
echo "RecursiveMemory GRPO Training"
echo "======================================"
echo "Model: $MODEL_NAME"
echo "Max Cycles: $MAX_CYCLES"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE"
echo "Num Generations: $NUM_GENERATIONS"
if [ -n "$LOAD_WEAVER_PATH" ]; then
    echo "Loading from: $LOAD_WEAVER_PATH"
fi
echo "======================================"

# Build options
OPTIONS=(
    model.model_name "$MODEL_NAME"
    model.weaver.model_name "$MODEL_NAME"
    model.recursive_memory.enabled true
    model.recursive_memory.max_cycles "$MAX_CYCLES"
    run.train_weaver_method grpo
    run.mode train
    run.weaver.grpo.per_device_train_batch_size "$BATCH_SIZE"
    run.weaver.grpo.num_generations "$NUM_GENERATIONS"
    run.weaver.grpo.max_completion_length "$MAX_COMPLETION_LENGTH"
)

# Add checkpoint path if provided
if [ -n "$LOAD_WEAVER_PATH" ]; then
    OPTIONS+=(model.load_weaver_path "$LOAD_WEAVER_PATH")
fi

# Run training
/home/jovyan/miniconda3/envs/memgen/bin/python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py --cfg-path configs/latent_memory/gsm8k.yaml \
    --options "${OPTIONS[@]}"
