# LTPO Future Experiments

나중에 변경/실험해볼 수 있는 사항들

---

## 1. Projection 시점 변경

### 현재 선택: Option B
```
weaver_hidden_states → weaver_to_reasoner → latent_embeds → [LTPO 최적화] → reasoner
```
- LTPO 원본과 유사하게 reasoner embedding 공간에서 최적화

### 나중에 시도: Option A
```
weaver_hidden_states → [LTPO 최적화] → weaver_to_reasoner → reasoner
```
- Weaver 공간에서 최적화
- 장점: Weaver가 학습한 representation 공간 활용
- 단점: projection이 최적화된 값을 왜곡할 수 있음

### 실험 방법
```python
# Config에 옵션 추가
ltpo:
  optimize_space: "reasoner"  # "reasoner" (현재) | "weaver" (Option A)
```

---

## 2. Query Latents 최적화 (`optimize_what = "query"`)

### 현재 선택: memory만 최적화
```
query_latents (고정) → Weaver forward → memory → [LTPO 최적화] → reasoner
```

### 나중에 시도: query_latents 최적화
```
query_latents → [LTPO 최적화] → Weaver forward → memory → reasoner
```

### 핵심 차이점

| 항목 | memory 최적화 | query 최적화 |
|------|--------------|-------------|
| 최적화 대상 | Weaver 출력 (hidden_states) | Weaver 입력 (query_latents) |
| Weaver forward | 1회 (초기에만) | 매 step마다 |
| 계산 비용 | 낮음 | ~2배 |
| 의미 | "가져온 정보" 수정 | "어떤 정보를 가져올지" 수정 |

### 구현 계획

#### 2.1 필요한 수정 사항

**`memgen/ltpo_optimizer.py`에 추가:**
```python
class MemGenLTPOOptimizer:
    def optimize_query(
        self,
        query_latents: torch.Tensor,      # [1, query_len, hidden_dim]
        weaver: MemGenWeaver,
        reasoner_inputs: dict,
        ...
    ) -> torch.Tensor:
        """
        Query latents를 최적화하여 더 좋은 memory 생성 유도

        Returns:
            optimized_query_latents: 최적화된 query latents
        """
        # 1. query_latents clone (원본 보존)
        query = query_latents.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([query], lr=self.lr)

        best_reward = float('-inf')
        best_query = query.clone()

        for step in range(self.max_steps):
            # 2. Noise 추가 (exploration)
            epsilon = torch.normal(mean=0.0, std=self.sigma, size=query.shape)
            query_cand = query + epsilon

            # 3. Weaver forward로 memory 생성
            #    - weaver._augment() 호출 필요
            #    - query_cand를 사용하여 새 memory 계산
            memory = self._weaver_forward_with_query(weaver, query_cand, reasoner_inputs)

            # 4. Confidence reward 계산
            reward = self.get_confidence(memory, reasoner_inputs)

            # 5. Gradient update
            if self.use_auto_grad:
                reward.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            else:
                # REINFORCE style
                grad = self.lr * reward * epsilon / (self.sigma ** 2)
                query = query + grad

            # 6. Best 저장
            if reward > best_reward:
                best_reward = reward
                best_query = query.clone()

            # 7. Sigma decay
            self.sigma *= self.sigma_decay

        return best_query if not self.disable_best_reward else query

    def _weaver_forward_with_query(
        self,
        weaver: MemGenWeaver,
        query_latents: torch.Tensor,
        reasoner_inputs: dict,
    ) -> torch.Tensor:
        """
        주어진 query_latents로 Weaver forward 실행

        Note: weaver._augment()는 self.prompt_query_latents를 사용하므로
              임시로 교체하거나, 직접 forward 로직 구현 필요
        """
        # Option 1: query_latents 임시 교체
        original_query = weaver.prompt_query_latents.data.clone()
        weaver.prompt_query_latents.data = query_latents

        # Weaver forward
        memory = weaver._augment(
            base_inputs_embeds=reasoner_inputs['inputs_embeds'],
            base_attention_mask=reasoner_inputs['attention_mask'],
            query_latents=query_latents,
        )

        # 원본 복원
        weaver.prompt_query_latents.data = original_query

        return memory

        # Option 2: weaver._augment() 내부 로직 직접 구현
        # - query_latents를 embedding에 추가
        # - Weaver LLM forward
        # - hidden_states 추출
```

#### 2.2 Gradient 흐름

```
query_latents (requires_grad=True)
    │
    ▼
weaver._augment()
    │
    ├── query_latents를 input에 concat
    ├── Weaver LLM forward (LoRA 포함)
    └── hidden_states 추출
    │
    ▼
memory (weaver_hidden_states)
    │
    ▼
weaver_to_reasoner projection
    │
    ▼
latent_embeds
    │
    ▼
reasoner forward → logits
    │
    ▼
get_confidence() → reward
    │
    ▼
backward() → gradient to query_latents
```

#### 2.3 주의사항

1. **Gradient 차단 방지**: Weaver forward 시 `torch.no_grad()` 사용하면 안 됨
2. **메모리 사용량**: 매 step마다 full forward → GPU 메모리 증가
3. **LoRA 가중치**: Weaver의 LoRA는 고정 (학습 안 함), gradient만 query로 전파

#### 2.4 Config 추가
```yaml
ltpo:
  optimize_what: "query"  # memory (현재) | query | both
```

### 실험 가설

**Query 최적화가 유리한 경우:**
- Memory가 이미 좋은 정보를 담고 있지만, "어떤 정보를 가져올지"가 잘못된 경우
- Query latents가 Weaver의 attention을 특정 위치로 유도하는 역할을 할 때

**Memory 최적화가 유리한 경우:**
- Weaver가 이미 좋은 정보를 추출했지만, representation이 최적이 아닌 경우
- 계산 비용이 중요한 경우

---

## 3. Query + Memory 동시 최적화 (`optimize_what = "both"`)

### 두 가지 접근법

#### Option A: 순차 최적화
```
1. Query 최적화 (N steps) → optimized_query
2. optimized_query로 memory 생성
3. Memory 최적화 (M steps) → optimized_memory
```

#### Option B: Joint 최적화
```
각 step에서:
1. Query + Memory를 하나의 parameter group으로 묶음
2. 동시에 gradient update
3. Query 변경 시 Weaver forward 재실행
```

### 구현 복잡도
- Option A: 비교적 단순, 두 최적화를 순차 호출
- Option B: 복잡, gradient 관리 및 Weaver forward 타이밍 고려 필요

### 나중에 결정할 사항
- 어떤 Option을 기본값으로 할지
- Step 수 분배 (Query:Memory 비율)

---

## 4. OOM 발생 시 `use_auto_grad=False` 전환

### 현재 선택: `use_auto_grad=True`
- 정확한 gradient 계산
- 더 좋은 최적화 성능

### OOM 발생 시 전환 방법

**Config 변경:**
```yaml
ltpo:
  use_auto_grad: false  # True → False로 변경
```

**또는 코드에서:**
```python
ltpo_config.use_auto_grad = False
```

### REINFORCE Style 동작 방식
```python
# use_auto_grad=False일 때
epsilon = torch.normal(0, sigma, size=latent.shape)
latent_noisy = latent + epsilon
reward = get_confidence(latent_noisy)

# Gradient 추정 (역전파 없음)
estimated_grad = reward * epsilon / (sigma ** 2)
latent = latent + lr * estimated_grad
```

### 성능 차이 완화 방법
- `max_steps` 증가 (10 → 20)
- `lr` 조정
- Step 수 늘리면 True와 비슷한 결과 도달 가능

---

## 5. 하이퍼파라미터 변경 위치

### 방법 1: 스크립트에서 직접 변경
```
SmolLM3-3B/05_ltpo_eval.sh
```
- `LTPO_LR`, `LTPO_SIGMA`, `LTPO_MAX_STEPS` 등 변수 수정

### 방법 2: Config yaml 변경
```
configs/latent_memory/gsm8k.yaml → run.ltpo 섹션
```

### 주요 파라미터

| 파라미터 | 스크립트 변수 | yaml 키 | 기본값 |
|---------|-------------|---------|-------|
| Learning rate | `LTPO_LR` | `run.ltpo.lr` | 0.03 |
| Noise std | `LTPO_SIGMA` | `run.ltpo.sigma` | 0.1 |
| Noise decay | `LTPO_SIGMA_DECAY` | `run.ltpo.sigma_decay` | 0.99 |
| 최적화 스텝 | `LTPO_MAX_STEPS` | `run.ltpo.max_steps` | 10 |
| Top-k | `LTPO_TOP_K` | `run.ltpo.top_k` | 10 |
| Autograd | `LTPO_USE_AUTO_GRAD` | `run.ltpo.use_auto_grad` | true |

