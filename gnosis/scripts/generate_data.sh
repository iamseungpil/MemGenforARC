#!/bin/bash
# Gnosis Data Generation Script
# Generates completions using vLLM for Gnosis training

set -e

# Default values
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
DATASET_ID="${DATASET_ID:-open-r1/DAPO-Math-17k-Processed}"
DATASET_CONFIG="${DATASET_CONFIG:-en}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATA_MODE="${DATA_MODE:-hf}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_QUESTIONS="${MAX_QUESTIONS:-40000}"
SHARD_SIZE="${SHARD_SIZE:-4000}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Please reason step by step, and put your final answer within \\boxed{}.}"
SAVE_DIR="${SAVE_DIR:-data/gnosis/${MODEL_ID##*/}_$(echo $DATASET_ID | tr '/' '_')}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_id) MODEL_ID="$2"; shift 2 ;;
        --dataset_id) DATASET_ID="$2"; shift 2 ;;
        --dataset_config) DATASET_CONFIG="$2"; shift 2 ;;
        --dataset_split) DATASET_SPLIT="$2"; shift 2 ;;
        --data_mode) DATA_MODE="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --num_generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --max_questions) MAX_QUESTIONS="$2"; shift 2 ;;
        --system_prompt) SYSTEM_PROMPT="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --test) MAX_QUESTIONS=10; SHARD_SIZE=10; echo "Test mode: 10 questions" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Gnosis Data Generation"
echo "=========================================="
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_ID"
echo "Mode: $DATA_MODE"
echo "Generations per question: $NUM_GENERATIONS"
echo "Max questions: $MAX_QUESTIONS"
echo "Save directory: $SAVE_DIR"
echo "=========================================="

# Create save directory
mkdir -p "$SAVE_DIR"

# Run generator
python -c "
import sys
sys.path.insert(0, '.')

from gnosis.data import CompletionGenerator

generator = CompletionGenerator(
    model_id='$MODEL_ID',
    system_prompt='$SYSTEM_PROMPT',
)

if '$DATA_MODE' == 'hf':
    from datasets import load_dataset
    dataset = load_dataset('$DATASET_ID', '$DATASET_CONFIG', split='$DATASET_SPLIT')
    generator.generate_from_dataset(
        dataset=dataset,
        question_col='prompt',
        answer_col='solution',
        num_generations=$NUM_GENERATIONS,
        max_questions=$MAX_QUESTIONS,
        shard_size=$SHARD_SIZE,
        save_dir='$SAVE_DIR',
    )
else:
    import pandas as pd
    if '$DATA_MODE' == 'csv':
        df = pd.read_csv('$DATA_PATH')
    else:
        df = pd.read_parquet('$DATA_PATH')
    generator.generate_from_dataset(
        dataset=df,
        question_col='question',
        answer_col='answer',
        num_generations=$NUM_GENERATIONS,
        max_questions=$MAX_QUESTIONS,
        shard_size=$SHARD_SIZE,
        save_dir='$SAVE_DIR',
    )

print('Generation complete!')
"

echo "=========================================="
echo "Generation complete. Output: $SAVE_DIR"
echo "=========================================="
