# MemGen + LTPO 구현 요약

## 1. 개요

MemGen의 Weaver가 생성한 latent memory를 test-time에 LTPO 방식으로 최적화하는 기능 구현.

```
기존 MemGen:
  query_latents → Weaver → memory → Reasoner → 생성

MemGen + LTPO:
  query_latents → Weaver → memory → projection → [LTPO 최적화] → Reasoner → 생성
```

---

## 2. 파일 구조

### 새로 생성된 파일

```
ltpo/
├── __init__.py          # exports (LTPOConfig, MemGenLTPOOptimizer)
├── config.py            # LTPOConfig 설정 클래스
└── memgen_ltpo.py       # MemGen용 LTPO 최적화기 (핵심)
```

### 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `memgen/model/modeling_memgen.py` | `generate_with_ltpo()` 메서드 추가 |
| `memgen/runner.py` | `evaluate_with_ltpo()`, `_static_evaluate_with_ltpo()` 메서드 추가 |
| `configs/latent_memory/gsm8k.yaml` | `run.ltpo` 설정 섹션 추가 |

---

## 3. 핵심 구현

### 3.1 `ltpo/config.py` - LTPOConfig

LTPO 원본 파라미터를 그대로 유지:

```python
@dataclass
class LTPOConfig:
    enabled: bool = False
    lr: float = 0.03              # learning rate
    sigma: float = 0.1            # noise std
    sigma_decay: float = 0.99     # noise decay per step
    max_steps: int = 10           # 최대 최적화 스텝
    reward_threshold: float = -1  # early stopping (-1 = disabled)
    top_k: int = 10               # confidence 계산용 top-k
    use_auto_grad: bool = True    # PyTorch autograd 사용
    disable_best_reward: bool = False
    verbose: int = 1
```

### 3.2 `ltpo/memgen_ltpo.py` - MemGenLTPOOptimizer

LTPO 원본(LTPO_backup/ltpo.py)을 MemGen에 맞게 수정:

| 메서드 | LTPO 원본 위치 | 설명 |
|--------|---------------|------|
| `get_confidence()` | ltpo.py:102-117 | top-k confidence reward 계산 |
| `optimize()` | ltpo.py:157-224 | 최적화 루프 |

**MemGen 수정사항:**
- `range(start, end+1)` → `range(start, end)` (gen_prompt 없음)
- `build_inputs()` 불필요 (Weaver가 이미 memory 생성)

### 3.3 `modeling_memgen.py` - generate_with_ltpo()

기존 `generate()` 메서드를 기반으로 LTPO 최적화 추가:

```python
def generate_with_ltpo(self, input_ids, attention_mask, generation_config, ltpo_config):
    # 1. Weaver로 memory 생성
    weaver_hidden_states = weaver.augment_prompt(...)

    # 2. Reasoner 공간으로 projection
    latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

    # 3. concat
    current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)

    # 4. LTPO 최적화
    current_inputs_embeds = ltpo_optimizer.optimize(
        model=reasoner,
        inputs_embeds=current_inputs_embeds,
        attention_mask=current_attention_mask,
        latent_start_idx=latent_start_idx,
        latent_end_idx=latent_end_idx,
    )

    # 5. 생성 계속...
```

### 3.4 `runner.py` - evaluate_with_ltpo()

LTPO를 적용한 평가 메서드:

```python
def evaluate_with_ltpo(self, ltpo_config: LTPOConfig = None):
    # Static 환경만 지원
    # batch_size=1 강제
    return self._static_evaluate_with_ltpo(ltpo_config)
```

---

## 4. 결정사항 반영

| # | 검토 사항 | 결정 | 구현 위치 |
|---|---------|------|----------|
| 1 | Confidence 계산 범위 | +1 불필요 (latent만) | `memgen_ltpo.py:47` |
| 2 | optimize_what="query" | Phase 1 보류, memory만 | `memgen_ltpo.py` |
| 3 | optimize_what="both" | Phase 1 보류 | - |
| 4 | Batch 처리 | batch_size=1 | `modeling_memgen.py:587` |
| 5 | Projection 시점 | Option B (Reasoner 공간) | `modeling_memgen.py:621-622` |
| 6 | KV Cache | 최적화 중 cache 없음 | `modeling_memgen.py:641` |
| 7 | Inference Latent 여러 개 | 삽입 시점마다 최적화 | `modeling_memgen.py:628-638` |
| 8 | Trigger 상호작용 | Phase 1: trigger 무시 | `modeling_memgen.py:598-602` |
| 9 | dtype | bfloat16 그대로 | dtype 변환 없음 |
| 10 | Memory/autograd | use_auto_grad=True | `config.py:14` |
| 11 | Confidence 모델 | Reasoner 사용 | `modeling_memgen.py:633` |

---

## 5. 최대 Augment 설정

Config에서 두 가지 설정:

```yaml
model:
  max_prompt_aug_num: 1      # prompt 끝에 붙이는 memory 수
  max_inference_aug_num: 5   # 생성 중 붙이는 memory 수
```

**예시:**
```
[prompt] [latent_0] [생성...] [latent_1] [생성...] ... [latent_5] [생성...]
          ↑ prompt용        ↑ inference용 (최대 5개)
```

---

## 6. 최적화 흐름

```
Weaver 공간                          Reasoner 공간

weaver_hidden_states
       │
       ▼
 weaver_to_reasoner()  ────────────► latent_inputs_embeds
                                            │
                                            ▼
                                     concat with prompt
                                            │
                                            ▼
                                     [LTPO 최적화]
                                            │
                                            ▼
                                     optimized_inputs_embeds
                                            │
                                            ▼
                                     Reasoner 생성
```

---

## 7. Config 설정 (gsm8k.yaml)

```yaml
run:
  ltpo:
    enabled: false              # LTPO 활성화 여부
    lr: 0.03                    # learning rate
    sigma: 0.1                  # noise std
    sigma_decay: 0.99           # noise decay per step
    max_steps: 10               # 최대 최적화 스텝
    reward_threshold: -1        # early stopping (-1 = disabled)
    top_k: 10                   # confidence 계산용 top-k
    use_auto_grad: true         # PyTorch autograd 사용 (OOM 시 false로)
    disable_best_reward: false  # best step 결과 사용 여부
    verbose: 1                  # 로깅 레벨
```

---

## 8. 사용 방법

### Python에서 직접 호출

```python
from ltpo import LTPOConfig
from memgen.runner import MemGenRunner

# LTPOConfig 생성
ltpo_config = LTPOConfig(
    enabled=True,
    lr=0.03,
    max_steps=10,
)

# Runner로 평가
runner = MemGenRunner(...)
runner.evaluate_with_ltpo(ltpo_config)
```

### 모델에서 직접 생성

```python
from ltpo import LTPOConfig

ltpo_config = LTPOConfig(enabled=True)

output_ids = model.generate_with_ltpo(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=gen_config,
    ltpo_config=ltpo_config,
)
```

---

## 9. 관련 문서

| 문서 | 내용 |
|------|------|
| `LTPO_INTEGRATION_PLAN.md` | 통합 계획 및 결정사항 |
| `LTPO_FUTURE_EXPERIMENTS.md` | 향후 실험 계획 (query 최적화, OOM 대응 등) |
| `LTPO_backup/` | LTPO 원본 코드 백업 |

---

## 10. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-13 | 초기 구현 완료 |
