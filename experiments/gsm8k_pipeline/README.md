# GSM8K Pipeline Experiments

이 폴더는 GSM8K 데이터셋을 사용한 Recursive Memory 파이프라인 실험 스크립트를 포함합니다.

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GSM8K Recursive Memory Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  00_vanilla_eval.sh         (베이스라인: 메모리 없이 평가)                   │
│                                                                             │
│  01_recursive_memory_train.sh  →  02_eval_recursive_memory.sh              │
│  (WeaverStyleCompressor 학습)      (Recursive Memory 평가)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 실행 방법

### 방법 1: 단계별 실행 (권장)

```bash
cd /home/hjkim/projects/RecursiveMem/RecursiveMLP

# Step 0: Vanilla 베이스라인 평가 (메모리 없이)
bash experiments/gsm8k_pipeline/SmolLM3-3B/00_vanilla_eval.sh

# Step 1: Recursive Memory 학습
bash experiments/gsm8k_pipeline/SmolLM3-3B/01_recursive_memory_train.sh

# Step 2: Recursive Memory 평가
bash experiments/gsm8k_pipeline/SmolLM3-3B/02_eval_recursive_memory.sh <checkpoint_path>
```

### 방법 2: runs/ wrapper 사용

```bash
cd /home/hjkim/projects/RecursiveMem/RecursiveMLP

# Train + Eval
bash runs/gsm8k/SmolLM3-3B/train_eval_recursive_memory.sh

# 또는 sweep
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh
```

## 모델별 디렉토리

| 모델 | 디렉토리 | hidden_size | num_heads |
|------|----------|-------------|-----------|
| SmolLM3-3B | `SmolLM3-3B/` | 2048 | 16 |
| Qwen3-8B | `Qwen3-8B/` | 4096 | 8 |

## 스크립트 설명

| 스크립트 | 설명 |
|----------|------|
| `00_vanilla_eval.sh` | 베이스라인 평가 (메모리 비활성화) |
| `00b_base_sft.sh` | Base SFT 학습 (비교용) |
| `00c_eval_base_sft.sh` | Base SFT 모델 평가 |
| `01_recursive_memory_train.sh` | Recursive Memory (WeaverStyleCompressor) 학습 |
| `02_eval_recursive_memory.sh` | Recursive Memory 평가 |

## 출력 경로

| 단계 | 출력 경로 |
|------|----------|
| Recursive Memory 학습 | `~/data/memgen/train/gsm8k/<model>/<experiment>/weaver/` |
| 평가 결과 | `~/data/memgen/evaluate/gsm8k/<model>/...` |

## 핵심 하이퍼파라미터

### Augmentation 설정 (GSM8K 권장)
- `MAX_PROMPT_AUG_NUM=1`: 프롬프트 끝 latent 삽입 횟수
- `MAX_INFERENCE_AUG_NUM=5`: 생성 중 latent 삽입 횟수
- `PROMPT_LATENTS_LEN=8`: 프롬프트 latent 시퀀스 길이
- `INFERENCE_LATENTS_LEN=8`: 추론 latent 시퀀스 길이

### Recursive Memory 설정
- `max_cycles=10`: compression cycle 반복 횟수
- `attn_rank=64`: self-attention low-rank 차원
- `mlp_rank=128`: SwiGLU MLP low-rank 차원
- `skip_projection=true`: projection 없이 compressor만 학습 (~7.9M params)

## 주의사항

1. **accelerate launch 필수**: 학습 시 반드시 `accelerate launch` 사용
2. **체크포인트 확인**: `recursive_memory.pt` 파일 존재 확인
3. **GPU 메모리**: SmolLM3-3B는 단일 GPU로 충분, Qwen3-8B는 multi-GPU 권장
