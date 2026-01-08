#!/bin/bash
# ============================================================================
# Experiment Runner Wrapper
# ============================================================================
# Convenience wrapper for run_pipeline.py with environment setup.
#
# Usage:
#   ./run_experiment.sh --config configs/full_pipeline.yaml
#   ./run_experiment.sh --config configs/weaver_grpo_only.yaml --dry-run
#   ./run_experiment.sh --config configs/ltpo_sweep.yaml --stage ltpo_sweep
#   ./run_experiment.sh --config configs/full_pipeline.yaml --list-stages
# ============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate memgen 2>/dev/null || true
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Run the pipeline
python "${SCRIPT_DIR}/run_pipeline.py" "$@"
