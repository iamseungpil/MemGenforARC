#!/bin/bash
# ============================================================================
# Common Utilities for GSM8K Pipeline
# ============================================================================
# This script provides shared functions for checkpoint discovery across
# all pipeline scripts. Source this file to use the functions.
#
# Usage:
#   source "$(dirname "$0")/common.sh"
#   MODEL_NAME_SAFE=$(get_model_name_safe "Qwen/Qwen2.5-1.5B-Instruct")  # Returns "Qwen2.5-1.5B-Instruct"
#   WEAVER_PATH=$(find_latest_weaver_checkpoint "gsm8k" "$MODEL_NAME_SAFE")
# ============================================================================

# Default output root (can be overridden by environment variable)
# NOTE: Must match main.py build_working_dir() which uses ~/data/memgen
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/data/memgen}"

# ============================================================================
# find_latest_timestamp_dir
# ============================================================================
# Find the most recent timestamp directory for a given dataset and model.
#
# Args:
#   $1: dataset name (e.g., "gsm8k", "arc")
#   $2: model name (last part only, e.g., "Qwen2.5-1.5B-Instruct")
#
# Returns:
#   Path to the latest timestamp directory, or empty string if not found
# ============================================================================
find_latest_timestamp_dir() {
    local dataset=$1
    local model_name=$2
    local base_dir="${OUTPUT_ROOT}/train/${dataset}/${model_name}"

    # Find latest timestamp directory (pn=*_pl=*_in=*_il=*_YYYYMMDD-HHMMSS)
    local latest_dir=$(ls -td ${base_dir}/pn=* 2>/dev/null | head -1)

    if [ -z "$latest_dir" ] || [ ! -d "$latest_dir" ]; then
        echo ""
        return 1
    fi

    echo "$latest_dir"
    return 0
}

# ============================================================================
# find_latest_weaver_checkpoint
# ============================================================================
# Find the most recent weaver checkpoint for a given dataset and model.
#
# Args:
#   $1: dataset name (e.g., "gsm8k", "arc")
#   $2: model name (last part only, e.g., "Qwen2.5-1.5B-Instruct")
#
# Returns:
#   Path to weaver_lora directory, or empty string if not found
# ============================================================================
find_latest_weaver_checkpoint() {
    local dataset=$1
    local model_name=$2

    local latest_dir=$(find_latest_timestamp_dir "$dataset" "$model_name")
    if [ -z "$latest_dir" ]; then
        echo ""
        return 1
    fi

    # Check for weaver_lora directory
    local weaver_lora="${latest_dir}/weaver/weaver_lora"
    if [ -d "$weaver_lora" ]; then
        echo "$weaver_lora"
        return 0
    fi

    echo ""
    return 1
}

# ============================================================================
# find_latest_trigger_checkpoint
# ============================================================================
# Find the most recent trigger checkpoint for a given dataset and model.
#
# Args:
#   $1: dataset name (e.g., "gsm8k", "arc")
#   $2: model name with slashes replaced by underscores (e.g., "Qwen_Qwen2.5-1.5B-Instruct")
#
# Returns:
#   Path to trigger_lora directory, or empty string if not found
# ============================================================================
find_latest_trigger_checkpoint() {
    local dataset=$1
    local model_name=$2

    local latest_dir=$(find_latest_timestamp_dir "$dataset" "$model_name")
    if [ -z "$latest_dir" ]; then
        echo ""
        return 1
    fi

    # Check for trigger_lora directory
    local trigger_lora="${latest_dir}/trigger/trigger_lora"
    if [ -d "$trigger_lora" ]; then
        echo "$trigger_lora"
        return 0
    fi

    echo ""
    return 1
}

# ============================================================================
# get_model_name_safe
# ============================================================================
# Extract model name for filesystem path (last part after /)
# NOTE: Must match main.py build_working_dir() which uses split("/")[-1]
#
# Args:
#   $1: model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
#
# Returns:
#   Model name without org prefix (e.g., "Qwen2.5-1.5B-Instruct")
# ============================================================================
get_model_name_safe() {
    echo "$1" | awk -F'/' '{print $NF}'
}

# ============================================================================
# print_checkpoint_info
# ============================================================================
# Print formatted checkpoint information for logging
#
# Args:
#   $1: checkpoint type ("weaver" or "trigger")
#   $2: checkpoint path (can be empty)
# ============================================================================
print_checkpoint_info() {
    local type=$1
    local path=$2

    if [ -n "$path" ]; then
        echo "  ${type}: ${path}"
    else
        echo "  ${type}: NOT FOUND"
    fi
}

# ============================================================================
# find_timestamp_dir_by_index
# ============================================================================
# Find a timestamp directory by index (0 = latest, 1 = second latest, etc.)
#
# Args:
#   $1: dataset name (e.g., "gsm8k", "arc")
#   $2: model name (last part only, e.g., "Qwen2.5-1.5B-Instruct")
#   $3: index (0 = latest, 1 = second latest, etc.)
#
# Returns:
#   Path to the timestamp directory, or empty string if not found
# ============================================================================
find_timestamp_dir_by_index() {
    local dataset=$1
    local model_name=$2
    local index=${3:-0}
    local base_dir="${OUTPUT_ROOT}/train/${dataset}/${model_name}"

    # Find timestamp directories sorted by time (newest first)
    local dirs=$(ls -td ${base_dir}/pn=* 2>/dev/null)

    if [ -z "$dirs" ]; then
        echo ""
        return 1
    fi

    # Get the directory at the specified index (1-indexed for sed)
    local selected_dir=$(echo "$dirs" | sed -n "$((index + 1))p")

    if [ -z "$selected_dir" ] || [ ! -d "$selected_dir" ]; then
        echo ""
        return 1
    fi

    echo "$selected_dir"
    return 0
}

# ============================================================================
# find_weaver_checkpoint_by_index
# ============================================================================
# Find a weaver checkpoint by index (0 = latest, 1 = second latest, etc.)
# Useful for loading specific checkpoints (e.g., SFT before GRPO)
#
# Args:
#   $1: dataset name (e.g., "gsm8k", "arc")
#   $2: model name (last part only, e.g., "Qwen2.5-1.5B-Instruct")
#   $3: index (0 = latest, 1 = second latest, etc.)
#
# Returns:
#   Path to weaver_lora directory, or empty string if not found
# ============================================================================
find_weaver_checkpoint_by_index() {
    local dataset=$1
    local model_name=$2
    local index=${3:-0}

    local target_dir=$(find_timestamp_dir_by_index "$dataset" "$model_name" "$index")
    if [ -z "$target_dir" ]; then
        echo ""
        return 1
    fi

    local weaver_lora="${target_dir}/weaver/weaver_lora"
    if [ -d "$weaver_lora" ]; then
        echo "$weaver_lora"
        return 0
    fi

    echo ""
    return 1
}

# ============================================================================
# list_all_checkpoints
# ============================================================================
# List all available checkpoints for a given dataset and model
#
# Args:
#   $1: dataset name (e.g., "gsm8k", "arc")
#   $2: model name (last part only, e.g., "Qwen2.5-1.5B-Instruct")
#
# Prints:
#   Indexed list of all checkpoints with their paths
# ============================================================================
list_all_checkpoints() {
    local dataset=$1
    local model_name=$2
    local base_dir="${OUTPUT_ROOT}/train/${dataset}/${model_name}"

    echo "Available checkpoints for ${dataset}/${model_name}:"
    echo "============================================"

    local index=0
    for dir in $(ls -td ${base_dir}/pn=* 2>/dev/null); do
        local dirname=$(basename "$dir")
        local has_weaver="N"
        local has_trigger="N"

        [ -d "${dir}/weaver/weaver_lora" ] && has_weaver="Y"
        [ -d "${dir}/trigger/trigger_lora" ] && has_trigger="Y"

        printf "[%d] %s (weaver:%s, trigger:%s)\n" "$index" "$dirname" "$has_weaver" "$has_trigger"
        index=$((index + 1))
    done

    if [ $index -eq 0 ]; then
        echo "  No checkpoints found"
    fi
    echo "============================================"
}

# ============================================================================
# get_checkpoint_info
# ============================================================================
# Get information about a specific checkpoint
#
# Args:
#   $1: checkpoint path (timestamp directory)
#
# Prints:
#   Checkpoint metadata (creation time, components, etc.)
# ============================================================================
get_checkpoint_info() {
    local checkpoint_dir=$1

    if [ ! -d "$checkpoint_dir" ]; then
        echo "Checkpoint not found: $checkpoint_dir"
        return 1
    fi

    local dirname=$(basename "$checkpoint_dir")
    echo "Checkpoint: $dirname"
    echo "  Path: $checkpoint_dir"
    echo "  Components:"

    if [ -d "${checkpoint_dir}/weaver/weaver_lora" ]; then
        local weaver_size=$(du -sh "${checkpoint_dir}/weaver" 2>/dev/null | cut -f1)
        echo "    - Weaver: ${weaver_size}"
    fi

    if [ -d "${checkpoint_dir}/trigger/trigger_lora" ]; then
        local trigger_size=$(du -sh "${checkpoint_dir}/trigger" 2>/dev/null | cut -f1)
        echo "    - Trigger: ${trigger_size}"
    fi
}
