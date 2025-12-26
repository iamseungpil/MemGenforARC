# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemGen (Memory Generator) is a framework for self-evolving AI agents that generates latent memory tokens within the model's reasoning stream. It consists of two core modules:
- **Memory Weaver**: Synthesizes past experiences into compact latent sequences for reasoning augmentation
- **Memory Trigger**: Decides when to recall and insert memory during generation

## Common Commands

### Environment Setup
```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
```

### Training

**Train Weaver model (SFT or GRPO):**
```bash
bash scripts/weaver_train.sh
```

**Train Trigger model (GRPO only):**
```bash
bash scripts/trigger_train.sh
```

### Evaluation
```bash
# Update LOAD_MODEL_PATH in eval.sh first
bash scripts/eval.sh
```

### Running with Custom Config
```bash
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/<dataset>.yaml \
    --options <key> <value> ...
```

## Architecture

### Core Components

**MemGenModel** (`memgen/model/modeling_memgen.py`):
- Main model class inheriting from `PreTrainedModel`
- Contains three sub-models: `reasoner` (base LLM), `weaver`, and `trigger`
- Uses LoRA adapters for weaver and trigger to avoid full fine-tuning
- Projection layers (`reasoner_to_weaver`, `weaver_to_reasoner`) map embeddings between components

**MemGenWeaver** (`memgen/model/weaver.py`):
- Generates latent memory tokens via learnable query latents
- Two modes: `augment_prompt()` for prompt-end augmentation, `augment_inference()` for mid-generation augmentation
- Uses `prompt_query_latents` and `inference_query_latents` as trainable parameters

**MemGenTrigger** (`memgen/model/trigger.py`):
- Binary classifier deciding whether to insert memory at each position
- Output layer maps hidden states to 2-class logits (insert/skip)
- When `active=False`, always returns logits favoring insertion

**MemGenRunner** (`memgen/runner.py`):
- Orchestrates training and evaluation
- Two-stage training: weaver first, then trigger
- Supports SFT and GRPO training methods for weaver, GRPO only for trigger

### Data Pipeline

**BaseBuilder** (`data/base_builder.py`):
- Abstract class for dataset construction
- Returns `DatasetDict` with train/valid/test splits
- Provides environment class via `get_env_cls()`

**BaseEnv** (`data/base_env.py`):
- Two environment types: `StaticEnv` (single-turn) and `DynamicEnv` (multi-turn)
- `compute_reward()` method for RL training
- Dynamic envs implement `step()`, `set_env()`, and `feedback()` for interaction loops

**Supported Datasets** (in `data/`):
- `gsm8k`: Math word problems (Static)
- `gpqa`: Graduate-level QA (Static)
- `kodcode`: Code generation (Static)
- `triviaqa`: Retrieval-augmented QA (Dynamic, multi-turn)

### Interaction System

**InteractionManager** (`interactions/base_interaction.py`):
- Manages model generation during training/evaluation
- `SingleTurnInteractionManager`: For static environments
- `MultiTurnInteractionManager`: For dynamic environments with tool use

### Configuration

YAML configs in `configs/latent_memory/` define:
- `model`: Base LLM, weaver/trigger settings, LoRA configs, augmentation parameters
- `dataset`: Dataset name, mode (sft/grpo), validation ratio
- `run`: Training mode, trainer configs (SFT/GRPO hyperparameters), interaction settings

Key augmentation parameters:
- `max_prompt_aug_num`: Number of prompt-end augmentations (1 for reasoning tasks, 6+ for retrieval)
- `max_inference_aug_num`: Number of mid-generation augmentations (5 for reasoning, 0 for retrieval)
- `prompt_latents_len`, `inference_latents_len`: Length of latent sequences

## Key Implementation Details

- Models use `bfloat16` precision and Flash Attention 2
- Training uses Accelerate with DeepSpeed ZeRO-2 (`configs/zero2.yaml`)
- Weaver training fixes trigger params and vice versa via `fix_component()`/`open_component()`
- Multi-turn forward processes conversation turns sequentially, with latents not visible across turns
- Generation loop interleaves trigger decisions with weaver augmentation at delimiter positions
