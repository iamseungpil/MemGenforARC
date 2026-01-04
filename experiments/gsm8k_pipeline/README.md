# GSM8K Pipeline Experiments

이 폴더는 GSM8K 데이터셋을 사용한 MemGen 파이프라인 실험 스크립트를 포함합니다.

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GSM8K Training Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  01_weaver_pretrain.sh    →    02_eval_weaver.sh                            │
│  (Weaver SFT 학습)              (Weaver 평가)                                │
│        │                                                                     │
│        ▼                                                                     │
│  03_trigger_pretrain.sh   →    04_eval_trigger.sh                           │
│  (Trigger GRPO 학습)            (Trigger 평가)                               │
│        │                                                                     │
│        ▼                                                                     │
│  05_ltpo_eval.sh                                                            │
│  (LTPO 테스트 시점 최적화 + 평가)                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 실행 방법

### 방법 1: 단계별 실행 (권장)

```bash
cd /home/ubuntu/MemGenforARC

# Step 1: Weaver SFT 학습
bash experiments/gsm8k_pipeline/01_weaver_pretrain.sh

# Step 2: Weaver 평가 (체크포인트 경로 수정 필요)
# LOAD_MODEL_PATH 수정 후 실행
bash experiments/gsm8k_pipeline/02_eval_weaver.sh

# Step 3: Trigger GRPO 학습 (weaver 체크포인트 필요)
# LOAD_WEAVER_PATH 수정 후 실행
bash experiments/gsm8k_pipeline/03_trigger_pretrain.sh

# Step 4: Trigger 평가 (체크포인트 경로 수정 필요)
# LOAD_MODEL_PATH 수정 후 실행
bash experiments/gsm8k_pipeline/04_eval_trigger.sh

# Step 5: LTPO 테스트 시점 최적화
# LOAD_MODEL_PATH 수정 후 실행
bash experiments/gsm8k_pipeline/05_ltpo_eval.sh
```

### 방법 2: 자동 실행

```bash
cd /home/ubuntu/MemGenforARC
bash experiments/gsm8k_pipeline/run_all.sh
```

## 모델 크기 선택

| 모델 | VRAM 요구량 | 용도 |
|------|------------|------|
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3GB | 파이프라인 검증, 빠른 테스트 |
| `Qwen/Qwen2.5-7B-Instruct` | ~14GB | 메인 실험 |
| `Qwen/Qwen3-14B` | ~28GB | 대규모 실험 |

모델을 변경하려면 각 스크립트의 `MODEL_NAME` 변수를 수정하세요.

## 출력 경로

| 단계 | 출력 경로 |
|------|----------|
| Weaver 학습 | `/data/memgen/train/gsm8k/<model>/weaver_sft/` |
| Trigger 학습 | `/data/memgen/train/gsm8k/<model>/trigger_grpo/` |
| 평가 결과 | `/data/memgen/evaluate/gsm8k/<model>/evaluate/answer.json` |
| LTPO 결과 | `/data/memgen/evaluate_ltpo/gsm8k/<model>/evaluate/answer_ltpo.json` |

## 핵심 하이퍼파라미터

### Augmentation 설정 (GSM8K 권장)
- `MAX_PROMPT_AUG_NUM=1`: 프롬프트 끝 latent 삽입 횟수
- `MAX_INFERENCE_AUG_NUM=5`: 생성 중 latent 삽입 횟수
- `PROMPT_LATENTS_LEN=8`: 프롬프트 latent 시퀀스 길이
- `INFERENCE_LATENTS_LEN=8`: 추론 latent 시퀀스 길이

### LTPO 설정
- `LTPO_LR=0.03`: Latent 최적화 학습률
- `LTPO_SIGMA=0.1`: 탐색 노이즈 표준편차
- `LTPO_MAX_STEPS=10`: 최대 최적화 스텝
- `LTPO_TOP_K=10`: 신뢰도 계산용 상위 토큰 수

## 기대 결과 비교

| 설정 | 예상 성능 변화 |
|------|---------------|
| Base (학습 없음) | 기준선 |
| Weaver only | +5~10% |
| Weaver + Trigger | +10~15% |
| Weaver + Trigger + LTPO | +15~20% |

## 주의사항

1. **체크포인트 경로**: 각 단계에서 이전 단계의 체크포인트 경로를 수동으로 업데이트해야 합니다.

2. **GPU 메모리**: 1.5B 모델은 단일 GPU로 충분하지만, 7B 이상은 multi-GPU가 필요할 수 있습니다.

3. **학습 시간 예상**:
   - 1.5B Weaver SFT: ~1-2시간
   - 1.5B Trigger GRPO: ~2-3시간
   - LTPO 평가: ~30분-1시간

4. **로그 확인**: `logs/` 폴더에서 각 단계의 로그를 확인할 수 있습니다.
