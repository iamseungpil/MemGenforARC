#!/bin/bash
# Gnosis Data Labeling Script
# Labels completions with correctness (0/1) based on task evaluator

set -e

# Default values
INPUT_DIR="${INPUT_DIR:-data/gnosis/Qwen3_8B_DAPO_Math}"
TASK="${TASK:-math}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="${INPUT_DIR}/verified"

echo "=========================================="
echo "Gnosis Data Labeling"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Task: $TASK"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run labeler
python -c "
import sys
sys.path.insert(0, '.')

from pathlib import Path
from gnosis.data import Labeler

labeler = Labeler(task='$TASK')
labeler.label_directory(
    input_dir=Path('$INPUT_DIR'),
    output_dir=Path('$OUTPUT_DIR'),
)

print('Labeling complete!')
"

echo "=========================================="
echo "Labeling complete. Output: $OUTPUT_DIR"
echo "=========================================="

# Print statistics
python -c "
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
parquet_files = list(output_dir.glob('*.verified.parquet'))

if parquet_files:
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    print('Label Distribution:')
    print(df['correctness_label'].value_counts())
    print()
    print(f'Total samples: {len(df)}')
    print(f'Correct rate: {df[\"correctness_label\"].mean():.2%}')
    print(f'Parse rate: {df[\"pred_parsed\"].mean():.2%}')
"
