#!/bin/bash
# Gnosis Data Rebalancing Script
# Merges and rebalances labeled data for training

set -e

# Default values
INPUT_DIRS="${INPUT_DIRS:-data/gnosis/Qwen3_8B_DAPO_Math/verified}"
OUTPUT_DIR="${OUTPUT_DIR:-data/gnosis/Final}"
STRATEGY="${STRATEGY:-downsample}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dirs) INPUT_DIRS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Gnosis Data Rebalancing"
echo "=========================================="
echo "Input directories: $INPUT_DIRS"
echo "Output directory: $OUTPUT_DIR"
echo "Strategy: $STRATEGY"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run rebalancer
python -c "
import sys
sys.path.insert(0, '.')

from pathlib import Path
from gnosis.data import Rebalancer

# Parse input directories (comma-separated)
input_dirs = [Path(d.strip()) for d in '$INPUT_DIRS'.split(',')]

rebalancer = Rebalancer(strategy='$STRATEGY')
rebalancer.rebalance(
    input_dirs=input_dirs,
    output_dir=Path('$OUTPUT_DIR'),
)

print('Rebalancing complete!')
"

echo "=========================================="
echo "Rebalancing complete. Output: $OUTPUT_DIR"
echo "=========================================="

# Print statistics
python -c "
import pandas as pd
from pathlib import Path

output_file = Path('$OUTPUT_DIR') / 'merged_balanced.parquet'
if output_file.exists():
    df = pd.read_parquet(output_file)
    print('Final Dataset Statistics:')
    print(f'Total samples: {len(df)}')
    print()
    print('Label Distribution:')
    print(df['correctness_label'].value_counts())
    print()
    print(f'Balance ratio: {df[\"correctness_label\"].mean():.2%} correct')
    if 'task' in df.columns:
        print()
        print('Task Distribution:')
        print(df['task'].value_counts())
"
