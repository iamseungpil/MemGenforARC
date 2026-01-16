# CLAUDE.md

## Project Overview

MemGen은 자가 진화 AI 에이전트를 위한 latent memory 프레임워크입니다.
- **Memory Weaver**: 경험을 compact latent sequences로 합성
- **Memory Trigger**: 메모리 호출 시점 결정

---

## 🚨 버그 수정 내역 (재발 방지)

| # | 파일 | 핵심 내용 |
|---|------|----------|
| 1 | `modeling_memgen.py` | `_grpo_forward` 삭제 - inference augmentation 누락으로 삭제됨 |
| 2 | `modeling_memgen.py` | `is_grpo` 플래그 삭제 - 위 문제로 인해 삭제됨 |
| 3 | `weaver_grpo_trainer.py` | `compute_loss` 주석처리 - BNPO가 아닌 GRPO 방식 사용 |
| 4 | `modeling_memgen.py` | projection dtype - `bfloat16` 제거 → float32 |
| 5 | `weaver.py` | query_latents dtype - `bfloat16` 제거 → float32 |
| 6 | `modeling_memgen.py` | chat_template 복원 - multi-turn `<\|im_start\|>` 토큰 의존 |
| 7 | `configs/zero2.yaml` | `mixed_precision: 'no'`로 복원 |
| 8 | `modeling_memgen.py` | temperature falsy 버그 - `0.0→1.0` 변환 문제, `do_sample=False, temperature=0.0` 하드코딩 |
| 9 | - | SmolLM3 config - 8번으로 해결됨 |
| 10 | `memgen_ltpo.py` | confidence 범위 - `+1` 범위 오류, `range(start, end)` (end 제외) |
| 10-1 | `memgen_ltpo.py` | `last_position_only=True` 기본값 - 마지막 latent만 vocab token 예측 |
| 11 | `modeling_memgen.py` | LoRA 키 변환 - `lora_A.weight` → `lora_A.weaver.weight` |
| 12 | `runner.py` | projections 저장 - `_save_weaver_projections()` 추가, `projections.pt` 저장 |
| 13 | 여러 파일 | Projection-Only 모드 추가 - `model.projection_only: true` |
| 14 | `modeling_utils.py` | `max_prompt_aug_num=0` 무시 버그 - `_should_augment()`에서 `generate()` 시 prompt aug 항상 실행됨 → 체크 추가 |

---

## ⚠️ 삭제 금지 (DO NOT DELETE)

### Skip-LoRA 관련 코드
- `weaver.py`: `_augment_skip_lora()`, `augment_prompt_skip_lora()`, `augment_inference_skip_lora()`
- `modeling_memgen.py`: `skip_lora` 분기 코드
- `modeling_utils.py`: `open_component()` skip_lora 파라미터
- `runner.py`: `_save_skip_lora_checkpoint()`
- `configuration_memgen.py`: `skip_lora` config

### LTPO 관련 코드
- `ltpo/memgen_ltpo.py` 전체
- `memgen/runner.py`의 `evaluate_with_ltpo()` 메서드

### 기타 유지 코드
- `data/triviaqa/` - multi-turn 지원
- `ARCDynamicEnv` - 향후 확장용
- `data/arc/env.py`의 binary reward 로직
- `main.py`의 mode 분기 로직

---

## 🔧 세 가지 학습/추론 모드

| 모드 | LoRA | Query Latents | Projections | Config | 파라미터 |
|------|------|---------------|-------------|--------|----------|
| **Full** | ✅ | ✅ | ✅ | (기본값) | ~42.6M |
| **Skip-LoRA** | ❌ disabled | ✅ | ✅ | `skip_lora: true` | ~33.6M |
| **Projection-Only** | ❌ | ❌ | ✅ | `projection_only: true` | ~33.5M |

### 체크포인트 파일
| 파일 | 내용 | 모드 |
|------|------|------|
| `projections.pt` | Projections + Query Latents | Full, Skip-LoRA |
| `skip_lora.pt` | Projections + Query Latents | Skip-LoRA 전용 |
| `projections_only.pt` | Projections Only | Projection-Only |
| `weaver_lora/` | LoRA Adapter | Full |

---

## 📊 핵심 평가 결과 (2026-01-15)

| 모델 | 방식 | 정확도 | 비고 |
|------|------|--------|------|
| Qwen3-8B | **Skip-LoRA** | **88.93%** | Query Latents + Projections만 |
| Qwen3-8B | random LoRA | 81.44% | LoRA 로드 안됨 (버그) |

**핵심 발견**: Skip-LoRA가 +7.49% 높음 - LoRA가 필수가 아님

---

## 🚨 학습 시 accelerate launch 필수

```bash
# ✅ 올바름
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py --cfg-path configs/latent_memory/<dataset>.yaml

# ❌ 잘못됨 (shared tensors 크래시)
python main.py --cfg-path configs/latent_memory/<dataset>.yaml
```

**원인**: MemGen은 reasoner/weaver/trigger가 가중치 공유 → HF Trainer가 처리 못함

**평가는 직접 실행 가능** (체크포인트 저장 없음)

---

## 개발 원칙

1. **master branch와 비교하며 작업**: `git diff origin/master --stat`
2. **master 코드 변경 최소화**: 필요시에만 수정, 새 파일 추가 선호
3. **변경 전 확인**: 꼭 필요한가? 기존 기능에 영향 없는가?

---

## Architecture 요약

### 핵심 컴포넌트
- **MemGenModel** (`modeling_memgen.py`): reasoner + weaver + trigger
- **MemGenWeaver** (`weaver.py`): `augment_prompt()`, `augment_inference()`
- **MemGenTrigger** (`trigger.py`): 메모리 삽입 결정 (binary classifier)
- **MemGenRunner** (`runner.py`): 학습/평가 orchestration

### 데이터셋
| 데이터셋 | 타입 | 용도 |
|----------|------|------|
| gsm8k | Static | Math 문제 |
| gpqa | Static | Graduate-level QA |
| kodcode | Static | 코드 생성 |
| triviaqa | Dynamic | Multi-turn QA |
| arc | Static | ARC 코드 생성 |

### 실행 모드
| 모드 | 스크립트 | 모델 업데이트 |
|------|----------|---------------|
| Training | `weaver_train.sh` | ✅ LoRA |
| Evaluation | `eval.sh` | ❌ |
| LTPO | `eval_ltpo.sh` | ❌ (latent만 최적화) |

---

## LTPO (Test-Time Optimization)

**핵심**: 모델 파라미터 업데이트 없이 latent embeddings만 최적화

| 모드 | Noise | Reward |
|------|-------|--------|
| SFT/GRPO Training | ❌ | Binary (task accuracy) |
| LTPO Eval | ✅ sigma | Confidence (top-k prob) |

```yaml
# LTPO 핵심 파라미터
run.ltpo:
  enabled: true
  lr: 0.03
  sigma: 0.1
  max_steps: 10
  top_k: 10
```

---

## ✅ 의도된 설계 (오류 아님)

| 항목 | 파일 | 설명 |
|------|------|------|
| GSM8KEnv `**kwargs` | `data/gsm8k/env.py` | `prompts` 파라미터 무시됨 (의도적) |
| LTPO `batch_size=1` | `ltpo/memgen_ltpo.py` | 샘플별 개별 최적화 |

---

## Common Commands

```bash
# 환경 설정
conda create -n memgen python=3.10 && pip install -r requirements.txt

# Weaver 학습
bash scripts/weaver_train.sh

# 평가
bash scripts/eval.sh

# LTPO 평가
bash scripts/eval_ltpo.sh
```

---

## 수정 전 테스트

```bash
# Import 검증
python -c "from memgen.runner import MemGenRunner; from data.arc.env import ARCEnv; from ltpo import MemGenLTPOOptimizer; print('OK')"
```

---

## 주요 파일 경로

| 기능 | 파일 |
|------|------|
| LTPO optimizer | `ltpo/memgen_ltpo.py` |
| GRPO reward | `memgen/trainer/weaver_grpo_trainer.py` |
| Binary reward | `data/arc/env.py` |
| LoRA 키 변환 | `modeling_memgen.py:_load_pretrained_weaver` |
