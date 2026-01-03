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
