#!/bin/bash
# Gnosis Training Script
# Trains Gnosis self-awareness head on labeled data

set -e

# Default values
CONFIG="${CONFIG:-gnosis/configs/gnosis.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./gnosis_output}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-data/gnosis/Final/merged_balanced.parquet}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:--1}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model_id) MODEL_ID="$2"; shift 2 ;;
        --train_data) TRAIN_DATA="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Gnosis Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Model: $MODEL_ID"
echo "Train data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python -c "
import sys
sys.path.insert(0, '.')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnosis import GnosisModule, GnosisConfig
from gnosis.trainer import GnosisTrainer
from gnosis.configs import load_config

# Load config
config = load_config('$CONFIG') if os.path.exists('$CONFIG') else {}

# Override with CLI args
config.setdefault('model', {})
config['model']['name'] = '$MODEL_ID'
config.setdefault('training', {})
config['training']['output_dir'] = '$OUTPUT_DIR'
config['training']['num_train_epochs'] = $NUM_EPOCHS
config['training']['per_device_train_batch_size'] = $BATCH_SIZE
config['training']['learning_rate'] = $LEARNING_RATE
if $MAX_STEPS > 0:
    config['training']['max_steps'] = $MAX_STEPS

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    config['model']['name'],
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

print('Initializing Gnosis...')
gnosis_config = GnosisConfig(
    hidden_size=model.config.hidden_size,
    num_layers=model.config.num_hidden_layers,
    num_heads=model.config.num_attention_heads,
)
gnosis = GnosisModule(gnosis_config)

print('Loading training data...')
train_df = pd.read_parquet('$TRAIN_DATA')
print(f'Loaded {len(train_df)} training samples')

# Split into train/eval
from sklearn.model_selection import train_test_split
train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=42)

print('Initializing trainer...')
trainer = GnosisTrainer(
    model=model,
    gnosis_module=gnosis,
    train_dataset=train_df,
    eval_dataset=eval_df,
    training_args=config['training'],
)

print('Starting training...')
metrics = trainer.train()
print(f'Training complete. Metrics: {metrics}')

print('Saving checkpoint...')
trainer.save('$OUTPUT_DIR')
print(f'Saved to $OUTPUT_DIR')
"

echo "=========================================="
echo "Training complete. Output: $OUTPUT_DIR"
echo "=========================================="
