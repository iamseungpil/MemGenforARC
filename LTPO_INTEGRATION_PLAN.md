# MemGen + LTPO Integration Plan

## 1. 개요

### 목표
MemGen의 Weaver가 생성한 latent memory를 test-time에 LTPO 방식으로 최적화

### 핵심 아이디어
```
기존 MemGen:
  query_latents → Weaver → memory → Reasoner → 생성

MemGen + LTPO:
  query_latents → Weaver → memory → [LTPO 최적화] → optimized_memory → Reasoner → 생성
```

---

## 2. 설정 옵션

### 2.1 optimize_target: 어떤 latent를 최적화할 것인가?

| 값 | 설명 |
|---|------|
| `prompt` | Prompt latent만 최적화 (i=0) |
| `inference` | Inference latent만 최적화 (i>0) |
| `both` | 둘 다 최적화 **(default)** |

### 2.2 optimize_what: 무엇을 최적화할 것인가?

| 값 | 설명 |
|---|------|
| `memory` | Weaver 출력 (weaver_hidden_states) 직접 최적화 **(default)** |
| `query` | query_latents 최적화 → Weaver 재실행 → 새 memory |
| `both` | 둘 다 최적화 |

---

## 3. LTPO 원본 파라미터 (그대로 유지)

```python
# LTPO/main.py 에서 가져온 기본값
num_thought_tokens: int = 10      # MemGen에서는 prompt_latents_len / inference_latents_len 사용
lr: float = 0.03                  # learning rate
sigma: float = 0.1                # noise std
sigma_decay: float = 0.99         # noise decay per step
max_rl_steps: int = 10            # 최대 최적화 스텝
reward_threshold: float = -1      # early stopping (-1 = disabled)
top_k: int = 10                   # confidence 계산용 top-k
use_auto_grad: bool = True        # PyTorch autograd 사용 여부
disable_best_reward: bool = False # best step 결과 사용 여부
verbose: int = 1                  # 로깅 레벨
```

---

## 4. Confidence 계산 방식

### LTPO 원본 (`LTPO/ltpo.py:102-117`)
```python
def get_confidence(model, inputs, thought_idx, thought_hidden_states, k=10):
    inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    logits = model(**inputs, return_dict=True)['logits'][0]
    probs = torch.softmax(logits, dim=-1)

    confidence = 0.0
    for idx in range(thought_idx[0], thought_idx[1] + 1):
        topk = torch.topk(probs[idx], k=k, largest=True)[0]
        confidence -= torch.sum(torch.log(topk + 1e-10)) / k

    num_tokens = thought_idx[1] - thought_idx[0] + 1
    return confidence / num_tokens
```

### MemGen 적용 시
- `thought_idx` → latent가 삽입된 위치 (sequence 끝)
- `model` → `self.reasoner`
- 나머지 로직은 동일하게 유지

---

## 5. 최적화 루프

### LTPO 원본 (`LTPO/ltpo.py:168-218`)
```python
best_reward = 0.0
best_thought_hidden_states = thought_hidden_states.clone()

for i in range(max_rl_steps):
    # 1. Noise 추가
    epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape)
    thought_hidden_states_cand = thought_hidden_states + epsilon

    # 2. Confidence reward 계산
    if use_auto_grad:
        reward = get_confidence(model, inputs, thought_idx, thought_hidden_states_cand, k=top_k)
        reward.backward(retain_graph=True)
        optimizer.step()
    else:
        # REINFORCE style update
        grad_ascent = lr * reward * epsilon / sigma**2
        thought_hidden_states += grad_ascent

    # 3. Noise decay
    sigma *= sigma_decay

    # 4. Best 저장
    if reward > best_reward:
        best_reward = reward
        best_thought_hidden_states = thought_hidden_states.clone()

    # 5. Early stopping
    if reward_threshold > 0 and reward >= reward_threshold:
        break

# 최종 결과
if disable_best_reward:
    return thought_hidden_states
else:
    return best_thought_hidden_states
```

---

## 6. 파일 구조

### 새로 생성
```
memgen/
└── ltpo_optimizer.py    # LTPO 최적화 클래스
```

### 수정
```
memgen/model/modeling_memgen.py  # generate_with_ltpo() 메서드 추가
memgen/runner.py                 # evaluate_with_ltpo() 메서드 추가
configs/latent_memory/gsm8k.yaml # ltpo 설정 섹션 추가
```

---

## 7. 구현 상세

### 7.1 `memgen/ltpo_optimizer.py`

```python
# TODO: 구현 예정
class LTPOConfig:
    """LTPO 설정 - 원본 파라미터 그대로"""
    pass

class MemGenLTPOOptimizer:
    """MemGen용 LTPO 최적화기"""

    def get_confidence(self, ...):
        """LTPO 원본과 동일"""
        pass

    def optimize(self, weaver_hidden_states, ...):
        """LTPO 원본 최적화 루프"""
        pass
```

### 7.2 `modeling_memgen.py` 수정

```python
def generate_with_ltpo(self, input_ids, attention_mask, generation_config, ltpo_config):
    """
    기존 generate()와 동일하되:
    - weaver.augment_prompt() 후 LTPO 최적화 추가
    - weaver.augment_inference() 후 LTPO 최적화 추가 (옵션)
    """
    pass
```

### 7.3 Config 추가

```yaml
run:
  ltpo:
    enabled: false
    lr: 0.03
    sigma: 0.1
    sigma_decay: 0.99
    max_rl_steps: 10
    reward_threshold: -1
    top_k: 10
    use_auto_grad: true
    disable_best_reward: false
    verbose: 1
    optimize_target: both      # prompt | inference | both
    optimize_what: memory      # memory | query | both
```

---

## 8. 검토 필요 사항

### 8.1 [검토 1] Confidence 계산 범위

LTPO 원본:
```python
for idx in range(thought_idx[0], thought_idx[1] + 1):  # +1 포함
```

질문: MemGen에서도 `latent_end_idx + 1`까지 포함해야 하는가?

- LTPO 원본은 latent 다음의 `gen_prompt` 토큰까지 포함
- MemGen은 latent가 sequence 끝이므로 상황이 다름

**결정**: [x] **+1 불필요** - latent positions만 계산
- LTPO 원본의 +1은 gen_prompt (`<|im_start|>assistant\n`) 첫 토큰 예측을 포함하기 위함
- MemGen은 gen_prompt 없이 latent가 sequence 끝에 위치
- latent의 마지막 위치가 이미 "다음 토큰 예측"을 수행
- 따라서 `range(latent_start_idx, latent_end_idx)` 사용 (Python convention)

---

### 8.2 [검토 2] optimize_what = "query" 구현 방식

query_latents를 최적화할 경우:
1. `self.weaver.prompt_query_latents`를 clone하여 최적화
2. 매 step마다 `weaver._augment()` 재실행 필요
3. 원본 query_latents는 변경하지 않음 (test-time only)

**결정**: [x] **Phase 1에서는 memory만 구현**
- query 최적화는 매 step마다 Weaver forward 필요 → 계산 비용 증가
- memory 최적화가 LTPO 원본과 더 유사 (embedding 직접 최적화)
- query 구현은 `LTPO_FUTURE_EXPERIMENTS.md`에 상세 계획 기록

---

### 8.3 [검토 3] optimize_what = "both" 구현 방식

두 가지 옵션:
- Option A: query 먼저 최적화 → 결과로 memory 최적화 (순차)
- Option B: query와 memory를 동시에 최적화 (joint)

**결정**: [x] **Phase 1에서는 구현 보류**
- query 최적화 자체가 Phase 2이므로, both도 Phase 2로 연기
- 상세 계획은 `LTPO_FUTURE_EXPERIMENTS.md` 참조

---

### 8.4 [검토 4] Batch 처리

현재 MemGen generate()는 batch 지원. LTPO 최적화도 batch로?

- LTPO 원본: batch_size=1
- MemGen: batch 지원

**결정**: [x] **batch_size=1** (LTPO 원본과 동일)
- 각 샘플마다 개별 최적화
- 구현 단순, 디버깅 쉬움
- 나중에 속도 향상 필요 시 batch 지원 추가

---

### 8.5 [검토 5] weaver_to_reasoner projection

최적화 후 projection 적용 시점:
```
Option A: weaver_hidden_states 최적화 → weaver_to_reasoner → reasoner
Option B: latent_inputs_embeds (projection 후) 최적화 → reasoner
```

LTPO 원본은 embedding 공간에서 직접 최적화하므로 Option B가 더 유사.
하지만 Option A가 weaver 공간에서 최적화하므로 더 자연스러울 수 있음.

**결정**: [x] **Option B** - Reasoner embedding 공간에서 최적화 (LTPO 원본과 유사)
- Option A는 나중에 실험 가능 → `LTPO_FUTURE_EXPERIMENTS.md` 참조

---

### 8.6 [검토 6] KV Cache Invalidation

MemGen generate()는 KV cache 사용:
```python
current_cache = outputs.past_key_values
```

문제:
- LTPO 최적화 중 latent embedding이 바뀜
- 이전에 계산한 KV cache가 무효화됨
- 매 optimization step마다 cache 없이 full forward 필요

```
Step 1: latent_v1 → forward (no cache) → confidence
Step 2: latent_v2 → forward (no cache) → confidence  ← cache 재사용 불가
...
```

**결정**: [x] **최적화 중 cache 없이 구현**
- LTPO 최적화는 생성이 아닌 confidence 계산만 수행
- 각 step은 독립적인 single forward pass → cache 불필요
- 최적화 완료 후 생성 시에만 cache 사용
- 구현 단순

---

### 8.7 [검토 7] Inference Latent 여러 개일 때

`max_inference_aug_num > 1`이면 생성 중 latent가 여러 번 삽입됨:
```
[prompt] [latent_0] [생성...] [latent_1] [생성...] [latent_2] [생성...]
```

질문: 각 latent를 어떻게 최적화?

| Option | 설명 |
|--------|------|
| A | 각 latent 독립적으로 최적화 (삽입 시점마다) |
| B | 첫 번째 latent만 최적화, 나머지는 그대로 |
| C | 모든 latent를 한번에 joint 최적화 (복잡) |

**결정**: [x] **Option A** - 삽입 시점마다 독립적으로 최적화
- 각 latent 삽입 시점에서 LTPO 최적화 실행
- 이전 생성 결과를 컨텍스트로 활용하여 다음 latent 최적화
- LTPO 원본 철학과 유사

---

### 8.8 [검토 8] Trigger와의 상호작용

Trigger가 `active=True`일 때:
- Trigger가 latent 삽입 여부를 결정
- LTPO로 최적화해도 Trigger가 "삽입 안 함"을 선택할 수 있음

질문: LTPO 사용 시 Trigger를 어떻게 처리?

| Option | 설명 |
|--------|------|
| A | Trigger 무시, 항상 latent 삽입 (LTPO 우선) |
| B | Trigger 결정 존중, 삽입할 때만 LTPO 적용 |
| C | Trigger도 LTPO와 함께 최적화 (복잡) |

**결정**: [x] 단계별 접근

**Phase 1 (현재 구현)**: `trigger.active=False`
- Trigger 비활성화, 항상 memory 삽입
- LTPO 구현이 단순해짐
- 모든 latent에 LTPO 적용

**Phase 2 (나중에)**: `trigger.active=True`
- Trigger 결정 존중
- Trigger가 "삽입"을 결정한 latent에 대해서만 LTPO 적용
- 구현 흐름:
  ```
  1. Trigger가 삽입 결정 → Yes
  2. Weaver가 memory 생성
  3. LTPO가 해당 memory 최적화
  4. 최적화된 memory를 Reasoner에 삽입
  ```
- Trigger가 "삽입 안 함" 결정 시 → LTPO 스킵

---

### 8.9 [검토 9] dtype 및 precision

MemGen은 `bfloat16` 사용:
```python
base_model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
```

LTPO 최적화 시:
- Gradient 계산이 `bfloat16`에서 불안정할 수 있음
- `float32`로 변환 후 최적화 → 다시 `bfloat16`?

**결정**: [x] **bfloat16 그대로 사용**
- LTPO 원본도 bfloat16 사용
- LTPO 업데이트 크기가 큼 (lr=0.03, sigma=0.1) → 정밀도 문제 적음
- 메모리 절약, 속도 빠름
- 문제 발생 시 float32로 전환 가능 (3줄 추가로 쉽게 변경)

---

### 8.10 [검토 10] Memory 사용량

LTPO 최적화 중:
- `use_auto_grad=True`: gradient 저장 필요
- 큰 모델에서 OOM 위험

완화 방법:
- `use_auto_grad=False`: REINFORCE style (gradient 저장 불필요)
- Gradient checkpointing
- 더 작은 `max_rl_steps`

**결정**: [x] **`use_auto_grad=True`로 시작**
- 정확한 gradient로 더 좋은 최적화 성능
- OOM 발생 시 `False`로 전환 → `LTPO_FUTURE_EXPERIMENTS.md` 참조

---

### 8.11 [검토 11] Confidence 계산 시 어떤 모델 사용?

MemGen 구조:
```
Reasoner (base LLM)
Weaver (base LLM + LoRA)
Trigger (base LLM + LoRA)
```

Confidence 계산 시:
| Option | 모델 | 설명 |
|--------|------|------|
| A | Reasoner | LTPO 원본과 유사 (생성하는 모델로 confidence) |
| B | Weaver | Memory 생성 모델로 confidence |

LTPO 원본은 생성 모델 = confidence 모델 (같은 모델)
MemGen은 Weaver ≠ Reasoner (다른 역할)

**결정**: [x] **Option A** - Reasoner 사용
- Memory가 들어간 Reasoner로 confidence 계산
- 실제 생성하는 모델의 관점에서 평가

---

## 9. 구현 순서

1. [ ] 검토 사항 결정
2. [ ] `ltpo_optimizer.py` 작성
3. [ ] `modeling_memgen.py`에 `generate_with_ltpo()` 추가
4. [ ] `runner.py`에 `evaluate_with_ltpo()` 추가
5. [ ] Config 추가
6. [ ] 테스트

---

## 10. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-01-13 | 초안 작성 |
