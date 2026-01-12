# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemGen (Memory Generator) is a framework for self-evolving AI agents that generates latent memory tokens within the model's reasoning stream. It consists of two core modules:
- **Memory Weaver**: Synthesizes past experiences into compact latent sequences for reasoning augmentation
- **Memory Trigger**: Decides when to recall and insert memory during generation

## 🚨 개발 원칙 (2025-01-08)

### Master Branch 보호 원칙
1. **항상 master branch와 비교하며 작업**
   - `git diff origin/master --stat`로 변경 범위 확인
   - 불필요한 변경 최소화

2. **master 코드 변경 최소화**
   - 반드시 필요하거나 명시적 요청이 있는 경우에만 수정
   - 기능 추가 시 기존 코드 수정보다 새 파일/함수 추가 선호

3. **변경 전 확인 사항**
   - 해당 변경이 정말 필요한가?
   - 기존 기능에 영향을 주지 않는가?
   - 더 작은 범위로 해결 가능한가?

## 🔧 원본 MemGen 복원 작업 (2025-01-08)

ltpo 브랜치에서 원본 master 대비 변경되었던 부분을 복원한 내역:

### 1. `_grpo_forward` 메서드 삭제 (`modeling_memgen.py`)
- **문제**: ltpo 브랜치에서 `_grpo_forward` 메서드가 새로 추가됨
- **원인**: prompt augmentation만 수행하고, inference augmentation(`_select_augment_points_after_delimiter`)을 수행하지 않음
- **결과**: GRPO 학습 시 latent memory가 prompt 끝에만 삽입되고, 생성 중간에 삽입되지 않음
- **수정**: `_grpo_forward` 메서드 삭제, 원본처럼 `_forward` 사용

### 2. `is_grpo` 플래그 삭제 (`modeling_memgen.py`, `weaver_grpo_trainer.py`)
- **문제**: `forward()`에서 `is_grpo=True`면 `_grpo_forward` 호출하는 분기 추가됨
- **원인**: trainer에서 `"is_grpo": True`를 전달하여 위의 불완전한 `_grpo_forward` 사용
- **수정**: `is_grpo` 체크 로직 삭제, trainer에서 `"is_grpo": True` 전달 삭제

### 3. `compute_loss` 메서드 주석처리 (`weaver_grpo_trainer.py`)
- **문제**: `compute_loss` 메서드가 새로 오버라이드됨
- **원인**: loss 계산 공식이 GRPO가 아닌 BNPO 방식 사용
  - GRPO: `((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()` (샘플별 정규화 후 평균)
  - BNPO (잘못됨): `(per_token_loss * mask).sum() / mask.sum()` (전체 정규화)
- **수정**: `compute_loss` 메서드 전체 주석처리, 원본 `_compute_loss` 사용

### 4. projection layer dtype 제거 (`modeling_memgen.py`)
- **문제**: `reasoner_to_weaver`, `weaver_to_reasoner` Linear 레이어에 `dtype=torch.bfloat16` 추가됨
- **원인**: 학습 가능 파라미터가 bfloat16으로 초기화되어 정밀도 저하
- **원본**: dtype 미지정 (기본 float32)
- **수정**: `dtype=torch.bfloat16` 제거

### 5. query_latents dtype 제거 (`weaver.py`)
- **문제**: `prompt_query_latents`, `inference_query_latents`에 `dtype=torch.bfloat16` 추가됨
- **원인**: 학습 가능 파라미터가 bfloat16으로 초기화되어 정밀도 저하
- **원본**: dtype 미지정 (기본 float32)
- **수정**: `dtype=torch.bfloat16` 제거

### 6. chat_template 오버라이드 복원 (`modeling_memgen.py`)
- **문제**: `self.tokenizer.chat_template = CONVERSATION_TEMPLATE` 라인이 주석으로 대체됨
- **원인**: multi-turn 대화 시 `_is_conversation()`, `_postprocess_assistant_labels()`가 `<|im_start|>` 토큰에 의존
- **수정**: `self.tokenizer.chat_template = CONVERSATION_TEMPLATE` 복원
- **주의**: GPT-OSS 등 다른 chat template 사용하는 모델은 `CONVERSATION_TEMPLATE` 수정 필요

### 7. mixed_precision 복원 (`configs/zero2.yaml`)
- **문제**: `mixed_precision: bf16`으로 변경됨
- **원본**: `mixed_precision: 'no'` (full precision)
- **수정**: `mixed_precision: 'no'`로 복원

### 8. temperature 버그 수정 (`modeling_memgen.py`) (2026-01-09)
- **문제**: `generate()` 루프에서 temperature 처리 시 falsy 값 버그
- **원인**: Python에서 `0.0`은 falsy 값이므로 `if temperature else 1.0`에서 `temperature=0.0` → `1.0`으로 변환됨
- **증상**: 평가 시 `temperature=0.0` (greedy decoding) 설정에도 `temperature=1.0` (sampling) 사용됨
- **영향 범위**:
  - Vanilla 평가: 영향 없음 (별도 경로 사용)
  - Weaver/Trigger/LTPO 평가: **영향** (sampling 모드로 실행됨)
  - 학습 (SFT/GRPO): 영향 없음 (temperature=1.0 사용)
- **원본 (Master)**: `do_sample=False, temperature=0.0` (Line 689-690 하드코딩)
- **수정**: Master 방식으로 복원 - `do_sample=False, temperature=0.0` 하드코딩

```python
# 수정 전 (버그)
do_sample=generation_config.do_sample,
temperature=generation_config.temperature if generation_config.temperature else 1.0

# 수정 후 (Master 방식)
do_sample=False,
temperature=0.0
```

### 9. SmolLM3 generation_config 오버라이드 문제 (2026-01-10)
- **문제**: SmolLM3 모델이 `generation_config` 기본값을 강제로 덮어씀
- **증상**: 로그에 다음 경고 발생:
  ```
  `generation_config` default values have been modified to match model-specific defaults:
  {'do_sample': True, 'temperature': 0.6, 'top_p': 0.95}
  ```
- **원인**: SmolLM3의 `model.generation_config`에 기본값이 설정되어 있음

**해결**: 8번의 Master 방식 하드코딩으로 해결됨
- `_append_one_step()` 호출 시 `do_sample=False, temperature=0.0` 하드코딩
- `generation_config` 값을 사용하지 않으므로 SmolLM3 오버라이드 영향 없음
- 추가 수정 불필요

### 10. LTPO confidence 계산 범위 수정 (`memgen_ltpo.py`) (2026-01-12)
- **문제**: confidence 계산 시 `range(latent_start_idx, latent_end_idx + 1)` 사용
- **원인**: 원본 LTPO는 special tokens 뒤에 `gen_prompt` 토큰이 있어서 +1 위치까지 포함
  - 원본 구조: `[prompt] [special tokens] [gen_prompt]` → `thought_idx[1]` 위치에 실제 토큰 존재
  - MemGen 구조: `[prompt] [latent tokens]` → latent가 sequence 끝, 다음 토큰 없음
- **증상**:
  - `latent_end_idx + 1 = seq_len + 1`로 sequence 범위 초과
  - boundary check로 마지막 iteration 스킵되지만, `num_tokens`는 `n+1`로 계산
  - 실제 합산은 `n`개인데 `n+1`로 나눔 → **confidence 과소평가**
- **수정**: latent positions만 계산하도록 변경

```python
# 수정 전 (버그)
for idx in range(latent_start_idx, latent_end_idx + 1):
    if idx < probs.shape[0]:
        topk = torch.topk(probs[idx], k=self.top_k, largest=True)[0]
        confidence -= torch.sum(torch.log(topk + 1e-10)) / self.top_k
num_tokens = latent_end_idx - latent_start_idx + 1

# 수정 후 (올바름)
for idx in range(latent_start_idx, latent_end_idx):
    topk = torch.topk(probs[idx], k=self.top_k, largest=True)[0]
    confidence -= torch.sum(torch.log(topk + 1e-10)) / self.top_k
num_tokens = latent_end_idx - latent_start_idx
```

- **핵심 차이**: 원본 LTPO는 `gen_prompt` 첫 토큰 예측까지 포함, MemGen은 latent만

---

## 🚨 학습 시 반드시 accelerate launch 사용 (2025-01-08)

### 문제 상황
직접 `python main.py`로 학습 실행 시 체크포인트 저장에서 크래시 발생:
```
RuntimeError: The weights trying to be saved contained shared tensors
[{'trigger.model.base_model.model.model.embed_tokens.weight',
  'reasoner.model.embed_tokens.weight',
  'weaver.model.base_model.model.model.embed_tokens.weight'}, ...]
```

### 원인
- MemGen은 `reasoner`, `weaver`, `trigger`가 동일한 base model 가중치를 **공유**
- HuggingFace Trainer의 기본 `save_pretrained`는 shared tensors를 처리하지 못함
- DeepSpeed ZeRO-2와 함께 `accelerate launch`를 사용해야 올바르게 저장됨

### 올바른 실행 방법

**✅ 올바름 (accelerate launch 사용):**
```bash
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py \
    --cfg-path configs/latent_memory/<dataset>.yaml \
    --options ...
```

**❌ 잘못됨 (직접 python 실행):**
```bash
python main.py --cfg-path configs/latent_memory/<dataset>.yaml --options ...
```

### 단일 GPU 사용 시에도 accelerate 필수
```bash
# GPU 0만 사용하더라도 accelerate launch 필요
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py ...
```

### 평가(evaluate)는 직접 실행 가능
- 평가 모드에서는 체크포인트 저장이 없으므로 직접 `python main.py` 사용 가능
- 하지만 일관성을 위해 `accelerate launch` 권장

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

---

## ARC Experiment (2025-01 Update)

### Overview

ARC (Abstract Reasoning Corpus) 실험을 위한 코드 생성 기반 접근법. BARC-style로 모델이 Python 코드를 생성하고, training examples에서 코드를 실행하여 정확도로 reward를 계산합니다.

### Three Main Execution Modes

#### 1. Pretrain (Weaver SFT/GRPO)
Weaver 모델을 학습하여 latent memory 생성 능력을 훈련합니다.

```bash
# SFT Warmup (권장: GRPO 전 사전학습)
bash scripts/arc_train.sh

# 또는 직접 실행
GPU_IDS=0,1 python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=2 \
    main.py \
    --cfg-path configs/latent_memory/arc.yaml \
    --options \
    run.mode train \
    run.train_weaver true \
    run.train_weaver_method sft  # or grpo
```

**학습 과정:**
1. `main.py` → `MemGenRunner.train()` → `_train_weaver()`
2. Weaver의 LoRA 파라미터만 학습 (trigger 고정)
3. SFT: supervised learning으로 latent 생성 학습
4. GRPO: code execution accuracy를 reward로 강화학습

**출력 위치:** `/data/memgen/train/arc/<model_name>/`

#### 2. Eval (Standard Evaluation)
학습된 모델 또는 base 모델의 ARC 문제 해결 능력을 평가합니다.

```bash
bash scripts/eval.sh

# 또는 직접 실행
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/arc.yaml \
    --options \
    run.mode evaluate \
    model.load_model_path <checkpoint_path>
```

**평가 과정:**
1. `main.py` → `MemGenRunner.evaluate()` → `_static_evaluate()`
2. Weaver가 latent tokens 생성
3. Reasoner가 latent + prompt로 Python 코드 생성
4. 코드 실행하여 training examples 정확도 계산

**출력 위치:** `/data/memgen/evaluate/arc/<model_name>/evaluate/answer.json`

#### 3. Test-Time Train with LTPO
LTPO (Latent Thought Policy Optimization)를 사용하여 inference 시 latent를 최적화합니다.

```bash
bash scripts/eval_ltpo.sh

# 또는 직접 실행
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/arc.yaml \
    --options \
    run.mode evaluate_ltpo \
    run.ltpo.enabled true \
    run.ltpo.lr 0.03 \
    run.ltpo.max_steps 10
```

**LTPO 최적화 과정:**
1. `main.py` → `MemGenRunner.evaluate_with_ltpo()` → `_static_evaluate_with_ltpo()`
2. Weaver가 초기 latent hidden states 생성
3. `MemGenLTPOOptimizer.optimize()`:
   - 초기 latent에 noise 추가 (exploration)
   - confidence reward 계산 (top-k token probability)
   - gradient ascent로 latent 업데이트
   - max_steps만큼 반복
4. 최적화된 latent로 코드 생성

**출력 위치:** `/data/memgen/evaluate_ltpo/arc/<model_name>/evaluate/answer_ltpo.json`

### LTPO Module (`ltpo/`)

| 파일 | 역할 |
|------|------|
| `ltpo.py` | 원본 LTPO 구현 (standalone) |
| `memgen_ltpo.py` | MemGen 통합 LTPO 최적화기 |
| `reward.py` | Reward model 인터페이스 |

**핵심 파라미터 (`configs/latent_memory/arc.yaml`):**
```yaml
run:
  ltpo:
    enabled: true        # LTPO 활성화
    lr: 0.03             # 최적화 learning rate
    sigma: 0.1           # exploration noise std
    sigma_decay: 0.99    # noise decay per step
    max_steps: 10        # 최대 최적화 스텝
    reward_threshold: -1 # early stopping threshold (-1=disabled)
    top_k: 10            # confidence 계산용 top-k tokens
    use_auto_grad: true  # PyTorch autograd 사용 (vs REINFORCE)
```

### ARC Environment (`data/arc/env.py`)

| 클래스 | 타입 | 용도 |
|--------|------|------|
| `ARCEnv` | Static | Single-turn 코드 생성 + 평가 |
| `ARCDynamicEnv` | Dynamic | Multi-turn 코드 refinement |

**Reward 계산:**
- Binary reward: ALL training examples 통과 → 1.0, otherwise → 0.0
- `validate_code_on_examples()`: 코드 파싱 → 실행 → 정확도 계산

### 설정 파일

**`configs/latent_memory/arc.yaml`:**
```yaml
model:
  model_name: Qwen/Qwen3-14B
  max_prompt_aug_num: 1      # prompt 끝 latent 개수
  max_inference_aug_num: 5   # 생성 중 latent 삽입 횟수
  weaver:
    prompt_latents_len: 8    # prompt latent 길이
    inference_latents_len: 8 # inference latent 길이

dataset:
  name: arc
  data_path: /home/ubuntu/arc-lang-public/data/arc-prize-2024

run:
  mode: train/evaluate/evaluate_ltpo
```

### 실행 워크플로우 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Flow                            │
├─────────────────────────────────────────────────────────────────┤
│  1. arc_train.sh → main.py (mode=train)                         │
│  2. MemGenRunner.train() → _train_weaver()                      │
│  3. WeaverGRPOTrainer: prompt → weaver latents → code generation│
│  4. ARCEnv.compute_reward(): execute code → accuracy → reward   │
│  5. GRPO loss: optimize weaver LoRA parameters                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Flow                           │
├─────────────────────────────────────────────────────────────────┤
│  1. eval.sh → main.py (mode=evaluate)                           │
│  2. MemGenRunner.evaluate() → _static_evaluate()                │
│  3. Weaver generates latents → Reasoner generates code          │
│  4. StaticEvalRecorder: compute_reward() → log results          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LTPO Test-Time Flow                         │
├─────────────────────────────────────────────────────────────────┤
│  1. eval_ltpo.sh → main.py (mode=evaluate_ltpo)                 │
│  2. MemGenRunner.evaluate_with_ltpo()                           │
│  3. Create MemGenLTPOOptimizer                                  │
│  4. For each sample:                                            │
│     a. Weaver → initial latents                                 │
│     b. LTPO loop: noise → confidence reward → gradient update   │
│     c. Optimized latents → code generation                      │
│  5. Log results to answer_ltpo.json                             │
└─────────────────────────────────────────────────────────────────┘
```

### 주요 코드 파일

| 파일 | 역할 |
|------|------|
| `main.py` | 진입점, mode에 따라 train/evaluate/evaluate_ltpo 분기 |
| `memgen/runner.py` | Training/Evaluation orchestration |
| `memgen/model/modeling_memgen.py` | MemGenModel (reasoner + weaver + trigger) |
| `memgen/model/weaver.py` | Latent memory 생성 (augment_prompt/augment_inference) |
| `ltpo/memgen_ltpo.py` | Test-time latent optimization |
| `data/arc/env.py` | ARC environment + reward computation |
| `data/arc/builder.py` | ARC dataset builder |
| `arc/utils.py` | 코드 파싱/실행/검증 유틸리티 |

### 디버깅 팁

1. **LTPO 최적화 확인:** `run.ltpo.verbose: true`로 설정하면 각 step의 reward 출력
2. **코드 실행 에러:** `arc/utils.py`의 `validate_code_on_examples()` 로그 확인
3. **메모리 부족:** `max_prompt_aug_num`, `max_inference_aug_num` 줄이기
4. **Reward가 0:** training examples JSON 파싱 확인, 코드 블록 형식 확인

---

## ⚠️ 중요 개념 정리 (2025-01-04)

### Test-Time Optimization vs Test-Time Training

```
┌────────────────────────────────────────────────────────────────────────┐
│                    ⚠️ LTPO는 Test-Time OPTIMIZATION이다!                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Test-Time Training (TTT)         │  Test-Time Optimization (LTPO)   │
│  ─────────────────────────────    │  ────────────────────────────    │
│  • 모델 가중치 업데이트 O          │  • 모델 가중치 업데이트 X         │
│  • 영구적 변경                    │  • inference 시에만 임시 최적화    │
│  • 별도 구현 필요                 │  • eval_ltpo.sh로 실행            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**LTPO가 최적화하는 것:**
- Weaver가 생성한 `latent_hidden_states` (embedding 벡터)
- 모델 파라미터가 아닌 **중간 표현(latent embeddings)**만 최적화
- 각 샘플마다 독립적으로 최적화, 샘플 간 정보 공유 없음

**LTPO Reward:**
- confidence-based reward (top-k token probability)
- ARC binary reward (code execution accuracy)와 **별개**

### Binary Reward 구현 (ARC 전용)

```python
# data/arc/env.py - ARCEnv.compute_reward()
if accuracy == 1.0:   # 모든 training examples 통과
    reward = 1.0
else:                 # 하나라도 실패
    reward = 0.0
```

**이유:** ARC에서 부분 정답(2/3 맞음)은 완전 오답과 동일 - 규칙이 완전히 맞거나 완전히 틀리거나

### 세 가지 파이프라인 핵심 정리

| 파이프라인 | 스크립트 | 모델 업데이트 | 사용 목적 |
|-----------|---------|-------------|----------|
| **Training** | `weaver_train.sh` | ✅ Yes (LoRA) | Weaver/Trigger 학습 |
| **Evaluation** | `eval.sh` | ❌ No | 성능 측정 |
| **LTPO Eval** | `eval_ltpo.sh` | ❌ No | Latent 최적화 후 평가 |

---

## 🔧 최근 수정 사항 (2025-01-04)

### Critical Fixes Applied

| # | 파일 | 이슈 | 수정 |
|---|------|------|------|
| 1 | `memgen/runner.py:110-123` | `_filter_dataset()` evaluate 모드 crash | `interaction_config` fallback 추가 |
| 2 | `data/base_env.py:29` | `preprocess_action(self,...)` | `self` → `cls` (classmethod) |
| 3 | `memgen/trainer/trigger_grpo_trainer.py` | Missing imports | `SamplingParams`, `gather`, `is_conversational` 추가 |
| 4 | `memgen/trainer/trigger_grpo_trainer.py:126-181` | Missing method | `_calculate_rewards()` 메서드 추가 |

### 삭제된 코드 (의도적)

| 파일/클래스 | 이유 |
|------------|------|
| `ARCCodeEnv` | `ARCEnv`와 중복 (동일 기능) |
| `configs/arc_twostage.yaml` | 2-stage training 미사용 |
| `configs/arc_instruction_sft.yaml` | instruction → code 방식 전환 |
| `interactions/arc_multiturn_interaction.py` | 현재 single-turn만 사용 |

### 유지해야 할 코드 (삭제 금지!)

| 파일 | 이유 |
|------|------|
| `data/triviaqa/` | 다른 실험용 dynamic env |
| `interactions/multiturn_interaction.py` | TriviaQA 등 multi-turn 지원 |
| `ARCDynamicEnv` | 향후 multi-turn ARC 확장용 |
| `ltpo/` 전체 | Test-time optimization 핵심 |

---

## 📁 ARC Single-Turn Code Generation 접근법

### 왜 Code Generation인가?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BARC-Style Approach                               │
├─────────────────────────────────────────────────────────────────────┤
│  기존 방식 (Instruction)          │  현재 방식 (Code Generation)     │
│  ───────────────────────────      │  ──────────────────────────────  │
│  "상단 2줄을 하단으로 복사"         │  def main(input_grid):          │
│  → 모호한 자연어 지시              │      return input_grid[:2]       │
│  → 실행 불가                      │  → 명확한 코드                    │
│  → 평가 어려움                    │  → 실행 가능                      │
│                                   │  → 정확도로 평가                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
ARC Task JSON
    │
    ▼
┌─────────────────┐
│  ARCBuilder     │ → training examples를 prompt로 변환
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Prompt Example:                                                 │
│  ───────────────                                                 │
│  Example 1:                                                      │
│  Input (3x3):                                                    │
│  0 0 1                                                           │
│  0 1 0                                                           │
│  1 0 0                                                           │
│                                                                  │
│  Output (3x3):                                                   │
│  1 0 0                                                           │
│  0 1 0                                                           │
│  0 0 1                                                           │
│                                                                  │
│  Write a Python function `main(input_grid)` that implements...   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  MemGen Model   │ → Weaver latents + Reasoner generation
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Generated Code:                                                 │
│  ────────────────                                                │
│  ```python                                                       │
│  def main(input_grid):                                           │
│      import numpy as np                                          │
│      grid = np.array(input_grid)                                 │
│      return np.flip(grid, axis=1).tolist()                       │
│  ```                                                             │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  ARCEnv Reward  │ → 코드 실행 → training examples 정확도
└────────┬────────┘
         │
         ▼
    Binary Reward: 1.0 (all pass) or 0.0 (any fail)
```

### 코드 실행 흐름 (`arc/utils.py`)

```python
# 1. 코드 파싱
code = parse_code_from_text(completion)  # ```python ... ``` 추출

# 2. 각 training example에서 실행
for example in train_examples:
    result = execute_code_on_input(code, example["input"])
    if result == example["output"]:
        passed += 1

# 3. 정확도 계산
accuracy = passed / total

# 4. Binary reward
reward = 1.0 if accuracy == 1.0 else 0.0
```

---

## 🛡️ 수정 시 주의사항

### 절대 건드리지 말 것
1. `ltpo/memgen_ltpo.py` - LTPO 핵심 로직
2. `data/arc/env.py`의 binary reward 로직
3. `memgen/runner.py`의 `evaluate_with_ltpo()` 메서드
4. `main.py`의 mode 분기 로직

### 수정 전 확인할 것
1. **Import chain**: 순환 참조 확인 (`arc/__init__.py` 주의)
2. **Type signatures**: `@classmethod`는 `cls` 사용
3. **Trainer methods**: `_calculate_rewards()` 존재 확인
4. **Config keys**: YAML 키와 코드 파라미터명 일치 확인

### 테스트 방법
```bash
# 모든 import 검증
python -c "from memgen.runner import MemGenRunner; from data.arc.env import ARCEnv, ARCDynamicEnv; from ltpo import MemGenLTPOOptimizer; print('OK')"

# LTPO 메서드 존재 확인
python -c "from memgen.runner import MemGenRunner; assert hasattr(MemGenRunner, 'evaluate_with_ltpo'); print('OK')"
```

---

## 📚 관련 논문 핵심 요약 (2025-01-08)

### LTPO (arXiv:2510.04182)
**제목**: "Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization"

- **목적**: Test-time에 latent thought 벡터를 최적화하여 추론 성능 향상
- **핵심**:
  - **Parameter-free**: 모델 가중치 업데이트 없음
  - **Confidence-based intrinsic reward**: frozen LLM 출력 분포에서 계산
  - **Noise (sigma)는 exploration용으로 test-time에만 추가**
  - 외부 supervision이나 text generation 없이 최적화

### MemGen (arXiv:2509.24704)
**제목**: "MemGen: Weaving Generative Latent Memory for Self-Evolving Agents"

- **목적**: Self-evolving agent를 위한 generative latent memory 프레임워크
- **핵심 모듈**:
  - **Memory Weaver**: 현재 상태 → latent token sequence 생성
  - **Memory Trigger**: memory 호출 시점 결정
- **차별점**: parametric/retrieval memory의 한계 극복, human-like cognitive ability

---

## ⚠️ Training과 LTPO의 명확한 구분 (2025-01-08)

### Noise 적용 규칙

| 모드 | Noise 적용 | 적용 위치 |
|------|-----------|----------|
| **SFT Training** | ❌ 없음 | - |
| **GRPO Training** | ❌ 없음 | - |
| **LTPO Eval** | ✅ 적용 | `ltpo/memgen_ltpo.py:157-163` |

**핵심**: SFT/GRPO 학습에서는 noise 없음. LTPO test-time에서만 exploration을 위해 noise 추가.

### Reward 사용 규칙

| 모드 | Reward 타입 | 용도 |
|------|-----------|------|
| **SFT Training** | 없음 (supervised labels) | Cross-entropy loss |
| **GRPO Training** | Binary (task accuracy) | Policy gradient |
| **LTPO Eval** | Confidence (top-k prob) | Latent optimization |

### 코드 흐름 확인

```
Training (SFT/GRPO):
├── Noise: ❌ 없음
├── Reward: Binary (1.0 or 0.0)
└── 학습 대상: Weaver/Trigger LoRA parameters

Test-Time (LTPO):
├── Noise: ✅ sigma로 exploration
├── Reward: Confidence-based (top-k token probability)
└── 최적화 대상: Latent embeddings (모델 파라미터 X)
```

---

## 🔬 GSM8K Pipeline 실험 가이드 (`experiments/gsm8k_pipeline/`)

### 환경 설정
```bash
# 반드시 memgen conda 환경 사용
conda activate memgen
```

### 체크포인트 자동 검색
각 스크립트는 `common.sh`를 통해 최신 체크포인트를 자동으로 찾습니다:
- `find_latest_weaver_checkpoint()`: 최신 weaver_lora 경로 반환
- `find_latest_trigger_checkpoint()`: 최신 trigger_lora 경로 반환

### 개별 실험 실행 순서
```bash
# 1. Weaver 학습 (SFT)
bash experiments/gsm8k_pipeline/01_weaver_pretrain.sh

# 2. Weaver 평가 (자동으로 최신 weaver 체크포인트 사용)
bash experiments/gsm8k_pipeline/02_eval_weaver.sh

# 3. Trigger 학습 (자동으로 최신 weaver 체크포인트 사용)
bash experiments/gsm8k_pipeline/03_trigger_pretrain.sh

# 4. Trigger 평가 (자동으로 최신 weaver + trigger 체크포인트 사용)
bash experiments/gsm8k_pipeline/04_eval_trigger.sh

# 5. LTPO 평가 (자동으로 최신 체크포인트 사용)
bash experiments/gsm8k_pipeline/05_ltpo_eval.sh

# 전체 파이프라인 자동 실행
bash experiments/gsm8k_pipeline/run_all.sh
```

### 수동 경로 지정 (필요시)
```bash
# 방법 1: 커맨드라인 인자로 전달
bash experiments/gsm8k_pipeline/02_eval_weaver.sh /path/to/weaver_lora
bash experiments/gsm8k_pipeline/03_trigger_pretrain.sh /path/to/weaver_lora
bash experiments/gsm8k_pipeline/04_eval_trigger.sh /path/to/weaver_lora /path/to/trigger_lora
bash experiments/gsm8k_pipeline/05_ltpo_eval.sh /path/to/weaver_lora /path/to/trigger_lora

# 방법 2: 스크립트 내 변수 직접 수정
LOAD_WEAVER_PATH="/path/to/weaver_lora"
LOAD_TRIGGER_PATH="/path/to/trigger_lora"
```

### 체크포인트 저장 위치
- **학습**: `~/data/memgen/train/<dataset>/<model_name>/pn=*_pl=*_in=*_il=*_<timestamp>/`
- **평가**: `~/data/memgen/evaluate/<dataset>/<model_name>/.../evaluate/answer.json`
- **LTPO**: `~/data/memgen/evaluate_ltpo/<dataset>/<model_name>/.../evaluate/answer_ltpo.json`

### 핵심 코드 경로 참조
| 기능 | 파일 위치 |
|------|----------|
| LTPO optimizer | `ltpo/memgen_ltpo.py:110-213` |
| Noise 적용 | `ltpo/memgen_ltpo.py:157-163` |
| GRPO reward | `memgen/trainer/weaver_grpo_trainer.py:186-241` |
| Binary reward | `data/arc/env.py:107-116` |

---

## ✅ 의도된 설계 결정사항 (오류 아님) (2025-01-08)

코드 리뷰 시 오류로 잡지 않아야 하는 master branch 설계 결정사항:

### 1. GSM8KEnv.compute_reward() 시그니처
**파일:** `data/gsm8k/env.py:10`

```python
@classmethod
def compute_reward(cls, completions: list[str], solution: list[str], **kwargs) -> list[float]:
```

- **의도**: `prompts` 파라미터는 `**kwargs`로 전달되어 무시됨
- **이유**: GSM8K는 completion과 solution만으로 reward 계산 가능
- **상태**: master branch와 동일 (변경 불필요)
- **참고**: GRPO trainer에서 `prompts`를 전달해도 kwargs에서 무시되므로 정상 작동

### 2. LTPO logits 인덱싱 (batch_size=1 가정)
**파일:** `ltpo/memgen_ltpo.py:96`

```python
logits = outputs.logits[0]  # 첫 번째 샘플만 사용
```

- **의도**: LTPO 최적화는 샘플당 개별 실행 (batch_size=1)
- **이유**: 각 샘플마다 latent를 독립적으로 최적화하므로 batch 처리 불필요
- **상태**: ltpo_sub 브랜치에서 새로 작성된 파일 (master에 없음)
- **주의**: batch_size > 1 사용 시 수정 필요 (현재는 해당 없음)

### 3. KodCode 데이터셋 크기
- **문제 수**: 10,000개
- **파일**: `data/kodcode/builder.py`

### 코드 리뷰 체크리스트

| 항목 | 상태 | 설명 |
|------|------|------|
| GSM8KEnv prompts kwargs | ✅ 정상 | 의도된 설계 |
| LTPO batch_size=1 | ✅ 정상 | 의도된 설계 |
| float32 dtype | ✅ 복원됨 | CLAUDE.md 1-7번 항목 |
| is_grpo 플래그 | ✅ 삭제됨 | CLAUDE.md 2번 항목 |
| _grpo_forward | ✅ 삭제됨 | CLAUDE.md 1번 항목 |
