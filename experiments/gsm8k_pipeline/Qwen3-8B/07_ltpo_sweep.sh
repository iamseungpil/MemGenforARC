#!/bin/bash
# ============================================================================
# Step 7: LTPO Hyperparameter Sweep
# ============================================================================
# Run LTPO evaluation with multiple hyperparameter configurations.
# Performs grid search over lr, sigma, and max_steps.
#
# Results are saved to separate directories for comparison.
# ============================================================================

set -e  # Exit on error

# Source common utilities for checkpoint discovery
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Environment setup
export WANDB_ENTITY="gistdslab"
export WANDB_PROJECT="memgen_ltpo_sweep"
export DEBUG_MODE=true
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29510
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Create log directory
mkdir -p logs/ltpo_sweep

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="gsm8k"

# Wandb run name prefix
MODEL_SHORT=$(echo ${MODEL_NAME} | sed 's|.*/||')
export WANDB_RUN_NAME_PREFIX="ltpo_sweep_${MODEL_SHORT}"

# Augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# Checkpoint paths (use arguments if provided, otherwise auto-discover)
# Usage: ./07_ltpo_sweep.sh [weaver_path] [trigger_path]
# ============================================================================
MODEL_NAME_SAFE=$(get_model_name_safe "${MODEL_NAME}")
if [ -n "$1" ]; then
    LOAD_WEAVER_PATH="$1"
else
    LOAD_WEAVER_PATH=$(find_latest_weaver_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
fi

if [ -n "$2" ]; then
    LOAD_TRIGGER_PATH="$2"
else
    LOAD_TRIGGER_PATH=$(find_latest_trigger_checkpoint "${DATASET_NAME}" "${MODEL_NAME_SAFE}")
fi

echo "============================================"
echo "Checkpoint Discovery"
echo "============================================"
print_checkpoint_info "Weaver" "${LOAD_WEAVER_PATH}"
print_checkpoint_info "Trigger" "${LOAD_TRIGGER_PATH}"
echo "============================================"

# ============================================================================
# LTPO Hyperparameter Grid
# ============================================================================
# Define sweep parameters
LR_VALUES=(0.01 0.03 0.1)
SIGMA_VALUES=(0.05 0.1 0.2)
MAX_STEPS_VALUES=(5 10 20)

# Fixed parameters
SIGMA_DECAY=0.99
TOP_K=10
USE_AUTO_GRAD=true

# Results tracking
RESULTS_DIR="${SCRIPT_DIR}/logs/ltpo_sweep/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RESULTS_DIR}"
RESULTS_FILE="${RESULTS_DIR}/sweep_results.csv"

# Initialize results file
echo "lr,sigma,max_steps,reward_mean,reward_std,time_sec" > "${RESULTS_FILE}"

echo ""
echo "============================================"
echo "LTPO Hyperparameter Sweep"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Results: ${RESULTS_DIR}"
echo ""
echo "Grid Search Parameters:"
echo "  lr: ${LR_VALUES[*]}"
echo "  sigma: ${SIGMA_VALUES[*]}"
echo "  max_steps: ${MAX_STEPS_VALUES[*]}"
echo ""
echo "Total configurations: $((${#LR_VALUES[@]} * ${#SIGMA_VALUES[@]} * ${#MAX_STEPS_VALUES[@]}))"
echo "============================================"

# ============================================================================
# Run Grid Search
# ============================================================================
config_idx=0
total_configs=$((${#LR_VALUES[@]} * ${#SIGMA_VALUES[@]} * ${#MAX_STEPS_VALUES[@]}))

for lr in "${LR_VALUES[@]}"; do
    for sigma in "${SIGMA_VALUES[@]}"; do
        for max_steps in "${MAX_STEPS_VALUES[@]}"; do
            config_idx=$((config_idx + 1))
            config_name="lr${lr}_sigma${sigma}_steps${max_steps}"

            echo ""
            echo "============================================"
            echo "[${config_idx}/${total_configs}] Running: ${config_name}"
            echo "============================================"

            export WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${config_name}"

            start_time=$(date +%s)

            # Build command with optional checkpoint paths
            CMD="python -m accelerate.commands.launch \
                --config_file=configs/zero2.yaml \
                --num_processes=2 \
                main.py \
                --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
                --options \
                model.model_name ${MODEL_NAME} \
                model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
                model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
                model.weaver.model_name ${MODEL_NAME} \
                model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
                model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
                model.trigger.model_name ${MODEL_NAME}"

            # Add checkpoint paths if available
            if [ -n "${LOAD_WEAVER_PATH}" ]; then
                CMD="${CMD} model.load_weaver_path ${LOAD_WEAVER_PATH}"
            fi
            if [ -n "${LOAD_TRIGGER_PATH}" ]; then
                CMD="${CMD} model.load_trigger_path ${LOAD_TRIGGER_PATH} model.trigger.active True"
            else
                CMD="${CMD} model.trigger.active False"
            fi

            # Add LTPO configuration
            CMD="${CMD} \
                run.mode evaluate_ltpo \
                run.ltpo.enabled true \
                run.ltpo.lr ${lr} \
                run.ltpo.sigma ${sigma} \
                run.ltpo.sigma_decay ${SIGMA_DECAY} \
                run.ltpo.max_steps ${max_steps} \
                run.ltpo.top_k ${TOP_K} \
                run.ltpo.use_auto_grad ${USE_AUTO_GRAD} \
                run.ltpo.verbose false \
                run.interaction.do_sample False \
                run.interaction.temperature 0.0"

            # Run the command and capture output
            LOG_FILE="${RESULTS_DIR}/${config_name}.log"
            eval ${CMD} 2>&1 | tee "${LOG_FILE}"

            end_time=$(date +%s)
            elapsed=$((end_time - start_time))

            # Extract reward from log (assuming it's printed in a specific format)
            # This may need adjustment based on actual output format
            reward_mean=$(grep -oP 'reward_mean[=:]\s*\K[\d.]+' "${LOG_FILE}" 2>/dev/null | tail -1 || echo "N/A")
            reward_std=$(grep -oP 'reward_std[=:]\s*\K[\d.]+' "${LOG_FILE}" 2>/dev/null | tail -1 || echo "N/A")

            # Record results
            echo "${lr},${sigma},${max_steps},${reward_mean},${reward_std},${elapsed}" >> "${RESULTS_FILE}"

            echo ""
            echo "Completed: ${config_name} in ${elapsed}s"
            echo "Reward: mean=${reward_mean}, std=${reward_std}"
        done
    done
done

echo ""
echo "============================================"
echo "LTPO Sweep Completed!"
echo "============================================"
echo ""
echo "Results Summary:"
echo "  Results file: ${RESULTS_FILE}"
echo "  Individual logs: ${RESULTS_DIR}/"
echo ""
echo "Best configurations (by reward_mean):"
if command -v sort &> /dev/null; then
    # Sort by reward_mean (4th column) and show top 5
    tail -n +2 "${RESULTS_FILE}" | sort -t',' -k4 -rn | head -5
fi
echo "============================================"
