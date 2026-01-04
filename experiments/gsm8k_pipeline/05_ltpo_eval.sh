#!/bin/bash
# ============================================================================
# Step 5: LTPO Test-Time Optimization + Evaluation
# ============================================================================
# Use Latent Thought Policy Optimization (LTPO) to optimize latent tokens
# at test time without updating model weights.
#
# LTPO performs gradient ascent on the latent embeddings to maximize
# confidence reward, adapting the model to each test problem individually.
#
# Expected Output: /data/memgen/evaluate_ltpo/gsm8k/<model_name>/evaluate/answer_ltpo.json
# ============================================================================

set -e  # Exit on error

# Environment setup
export DEBUG_MODE=true
export LOG_PATH="./logs/05_ltpo_eval.log"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

mkdir -p logs

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME="gsm8k"

# Augmentation configs (must match training)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# ============================================================================
# IMPORTANT: Set the path to your trained model
# ============================================================================
# Use the checkpoint from Step 3 (after trigger training) or Step 1 (weaver only)
# Example:
# LOAD_MODEL_PATH="/data/memgen/train/gsm8k/Qwen_Qwen2.5-1.5B-Instruct/trigger_grpo/checkpoint-500/model.safetensors"
# ============================================================================
LOAD_MODEL_PATH=null  # <-- UPDATE THIS with your best checkpoint

# Trigger setting (use True if trigger was trained)
TRIGGER_ACTIVE=True

# ============================================================================
# LTPO Configuration
# ============================================================================
# lr: Learning rate for latent optimization (higher = faster but less stable)
# sigma: Initial noise std for exploration (higher = more exploration)
# sigma_decay: Noise decay per step (0.99 = slow decay, 0.9 = fast decay)
# max_steps: Maximum optimization steps per sample
# reward_threshold: Early stopping threshold (-1 = disabled)
# top_k: Number of top tokens for confidence calculation
# use_auto_grad: Use PyTorch autograd (True) vs REINFORCE (False)
# verbose: Print detailed optimization logs
# ============================================================================
LTPO_ENABLED=true
LTPO_LR=0.03
LTPO_SIGMA=0.1
LTPO_SIGMA_DECAY=0.99
LTPO_MAX_STEPS=10
LTPO_REWARD_THRESHOLD=-1.0  # -1 = no early stopping
LTPO_TOP_K=10
LTPO_USE_AUTO_GRAD=true
LTPO_VERBOSE=false  # Set to true for debugging

# Evaluation configs
BATCH_SIZE=4
TEMPERATURE=0.0

# ============================================================================
# Execute LTPO Evaluation
# ============================================================================
echo "============================================"
echo "Step 5: LTPO Test-Time Optimization"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Checkpoint: ${LOAD_MODEL_PATH}"
echo "============================================"
echo "LTPO Configuration:"
echo "  - Learning rate: ${LTPO_LR}"
echo "  - Sigma: ${LTPO_SIGMA}"
echo "  - Max steps: ${LTPO_MAX_STEPS}"
echo "  - Top-k: ${LTPO_TOP_K}"
echo "============================================"

if [ "$LOAD_MODEL_PATH" = "null" ]; then
    echo "WARNING: LOAD_MODEL_PATH is null. Running LTPO on base model."
    echo "For best results, use a trained checkpoint."
    echo ""
else
    # Auto-detect if checkpoint is weaver-only (no trigger training)
    # If path contains "weaver_sft" but not "trigger_grpo", disable trigger
    if [[ "$LOAD_MODEL_PATH" == *"weaver_sft"* ]] && [[ "$LOAD_MODEL_PATH" != *"trigger_grpo"* ]]; then
        echo "INFO: Detected weaver-only checkpoint. Setting TRIGGER_ACTIVE=False"
        TRIGGER_ACTIVE=False
    fi
fi

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${MODEL_NAME} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${MODEL_NAME} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${MODEL_NAME} \
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate_ltpo \
    run.ltpo.enabled ${LTPO_ENABLED} \
    run.ltpo.lr ${LTPO_LR} \
    run.ltpo.sigma ${LTPO_SIGMA} \
    run.ltpo.sigma_decay ${LTPO_SIGMA_DECAY} \
    run.ltpo.max_steps ${LTPO_MAX_STEPS} \
    run.ltpo.reward_threshold ${LTPO_REWARD_THRESHOLD} \
    run.ltpo.top_k ${LTPO_TOP_K} \
    run.ltpo.use_auto_grad ${LTPO_USE_AUTO_GRAD} \
    run.ltpo.verbose ${LTPO_VERBOSE} \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.do_sample False \
    run.interaction.temperature ${TEMPERATURE} \
    run.interaction.max_response_length 1024

echo ""
echo "============================================"
echo "LTPO evaluation completed!"
echo "Check results at: /data/memgen/evaluate_ltpo/gsm8k/"
echo "============================================"
echo ""
echo "Pipeline Complete! Compare results:"
echo "  - Baseline (no training): evaluate without checkpoints"
echo "  - Weaver only: 02_eval_weaver.sh results"
echo "  - Weaver + Trigger: 04_eval_trigger.sh results"
echo "  - Weaver + Trigger + LTPO: This script's results"
echo "============================================"
