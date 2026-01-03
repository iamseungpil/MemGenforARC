#!/bin/bash
# LTPO-Enhanced Evaluation Script for MemGen
# Uses Latent Thought Policy Optimization at test time to optimize weaver-generated latents

export DEBUG_MODE=true
export LOG_PATH="./debug_log_ltpo.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# Model configs
REASONER_MODEL="Qwen/Qwen3-14B"
WEAVER_MODEL="Qwen/Qwen3-14B"
TRIGGER_MODEL="Qwen/Qwen3-14B"
TRIGGER_ACTIVE=False

# Dataset configs
DATASET_NAME="arc"

# MemGen augmentation configs
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# LTPO configs (override yaml defaults if needed)
LTPO_ENABLED=true
LTPO_LR=0.03
LTPO_SIGMA=0.1
LTPO_SIGMA_DECAY=0.99
LTPO_MAX_STEPS=10
LTPO_REWARD_THRESHOLD=-1.0
LTPO_TOP_K=10
LTPO_USE_AUTO_GRAD=true
LTPO_VERBOSE=false

# Trained model path:
# - Must point to a checkpoint file ending with .safetensors (e.g. <output_dir>/model.safetensors)
# - Required when evaluating the model
LOAD_MODEL_PATH=null

# evaluate with LTPO
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
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
    run.interaction.batch_size 4 \
    run.interaction.do_sample False \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 512 \
