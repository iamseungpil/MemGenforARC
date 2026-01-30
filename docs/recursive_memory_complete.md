# Recursive Memory 완전 가이드

## 1. 구조 명확화

### 1.1 현재 구현: LowRank Self-Attention + LowRank SwiGLU

```
WeaverStyleCompressor (~7.9M params)
├── query_latents
│   ├── prompt_query_latents: (8, 4096)      # 32K params
│   └── inference_query_latents: (8, 4096)   # 32K params
├── self_attn: LowRankCausalSelfAttention    # ~2.1M params
│   ├── q_proj: LowRankLinear (4096→4096, rank=64)
│   ├── k_proj: LowRankLinear (4096→4096, rank=64)
│   ├── v_proj: LowRankLinear (4096→4096, rank=64)
│   └── o_proj: LowRankLinear (4096→4096, rank=64)
└── mlp: LowRankSwiGLU                       # ~5.7M params
    ├── gate: LowRankLinear (4096→10936, rank=128)
    ├── up: LowRankLinear (4096→10936, rank=128)
    └── down: LowRankLinear (10936→4096, rank=128)
```

### 1.2 LoRA vs LowRank 차이

| 구분 | LoRA | LowRank (현재 구현) |
|------|------|---------------------|
| 수식 | `W_orig + B @ A` | `W_down @ W_up` (원본 없음) |
| 용도 | Pre-trained weight 적응 | Scratch 학습 |
| 파라미터 | Delta만 학습 | 전체 학습 |

**결론**: 현재 구현은 LoRA가 아니라 **Low-rank Factorized Self-Attention + SwiGLU**입니다.

---

## 2. 구현 상태 체크리스트

### 2.1 구현 완료

| 기능 | 파일 | 상태 |
|------|------|------|
| WeaverStyleCompressor | `recursive_memory.py` | ✅ |
| 10 cycle 고정 학습 | `recursive_memory.py` | ✅ |
| Confidence early stop (추론용) | `recursive_memory.py:251-260` | ✅ |
| skip_projection 옵션 | `configuration_memgen.py` | ✅ |
| verbose_cycles 로깅 | `configuration_memgen.py` | ✅ |

### 2.2 구현 완료 (2차)

| 기능 | 파일 | 상태 |
|------|------|------|
| **H-cycle/L-cycle 분리** | `recursive_memory.py:_forward_two_level()` | ✅ |
| **Two-level config 옵션** | `configuration_memgen.py` | ✅ |
| **Stepwise Training** | `modeling_memgen.py:_forward()` | ✅ |
| **Stepwise config 옵션** | `configuration_memgen.py` | ✅ |

---

## 3. Confidence Score의 현재 동작

### 3.1 현재 구현 (추론 시 early stop)

```python
# recursive_memory.py Line 251-260
if reasoner is not None and self.confidence_threshold > 0:
    conf = self._compute_confidence(reasoner, context, z, attention_mask)
    if conf <= self.confidence_threshold:
        return z, cycle + 1  # Early stop
```

- `confidence_threshold > 0`일 때만 활성화
- 추론 시에만 동작 (학습 시에도 호출되지만 gradient flow에 영향 없음)
- `_compute_confidence()`는 **no_grad**로 실행

### 3.2 Confidence 계산 방식

```python
# Line 182-192
with torch.no_grad():
    outputs = reasoner(inputs_embeds=full_embeds, ...)
    logits = outputs.logits[:, -1]

probs = F.softmax(logits, dim=-1)
topk_probs = torch.topk(probs, k=self.top_k).values
confidence = -torch.log(topk_probs + 1e-10).mean().item()
```

- **낮을수록 confident** (top-k 확률이 높으면 entropy가 낮음)
- 마지막 위치의 next token 예측 확률 사용

---

## 4. 현재 학습 방식 분석

### 4.1 Forward Flow (학습)

```python
# modeling_memgen.py forward()

for aug_point in augmentation_points:
    # 1. Context 준비
    if skip_projection:
        context = current_inputs_embeds
    else:
        context = reasoner_to_weaver(current_inputs_embeds)

    # 2. Recursive compression (10 cycles 고정)
    memory, cycles = recursive_compressor(context, ...)

    # 3. Projection (optional)
    if not skip_projection:
        memory = weaver_to_reasoner(memory)

    # 4. Memory injection
    current_inputs_embeds = concat([current_inputs_embeds, memory])

# 5. Final forward through reasoner
logits = reasoner(inputs_embeds=current_inputs_embeds)

# 6. Loss computation (memory 위치 제외)
loss = cross_entropy(logits[valid_mask], labels)
```

### 4.2 Gradient Flow

```
Loss
  ↓
logits (reasoner, frozen)
  ↓
current_inputs_embeds (includes memory)
  ↓
memory = weaver_to_reasoner(z)  ← gradient (if not skip_projection)
  ↓
z = recursive_compressor(...)   ← gradient (모든 cycle)
  ↓
context = reasoner_to_weaver(embeds)  ← gradient (if not skip_projection)
  ↓
embeds (from frozen reasoner)   ← STOP
```

**문제점**: 현재는 모든 cycle에 gradient가 흐름 (TRM은 마지막만)

---

## 5. Two-Level Cycle 구조 (H-cycle / L-cycle)

### 5.1 개념

TRM 스타일의 2단계 반복 구조로, 효율적인 compression과 early stopping을 가능하게 합니다.

```
┌─────────────────────────────────────────────────────────┐
│                    H-cycle (외부 루프)                    │
│  max_h_cycles = 5 (confidence 체크 포함)                 │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              L-cycle (내부 루프)                  │   │
│  │  l_cycles = 6 (고정 반복)                        │   │
│  │                                                 │   │
│  │  for l in range(6):                            │   │
│  │      combined = [context, z]                   │   │
│  │      combined = SelfAttn(combined)             │   │
│  │      z = combined[:, -num_latents:]            │   │
│  │      z = MLP(z)                                │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│              Confidence 체크 (H-cycle 끝)               │
│                         ↓                               │
│          conf <= threshold → Early Stop                │
│          conf > threshold → 다음 H-cycle               │
└─────────────────────────────────────────────────────────┘
```

### 5.2 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `two_level` | `false` | Two-level 모드 활성화 |
| `l_cycles` | `6` | 내부 루프 반복 횟수 (고정) |
| `max_h_cycles` | `5` | 외부 루프 최대 반복 횟수 |
| **총 최대 연산** | **30** | 6 × 5 = 30 ops |

### 5.3 구현 코드

```python
# recursive_memory.py: _forward_two_level()

def _forward_two_level(self, z, context, attention_mask, reasoner, verbose, logger):
    for h in range(self.max_h_cycles):  # H-cycle: max 5회
        # L-cycle (inner loop): 고정 6회
        for l in range(self.l_cycles):
            combined = torch.cat([context, z], dim=1)
            combined = rms_norm(self.self_attn(combined))
            z = combined[:, -num_latents:]
            z = rms_norm(self.mlp(z))

        # H-cycle 끝에 confidence 체크
        if reasoner is not None and self.confidence_threshold > 0:
            conf = self._compute_confidence(reasoner, context, z, attention_mask)
            if conf <= self.confidence_threshold:
                return z, (h + 1, self.l_cycles)  # Early stop

    return z, (self.max_h_cycles, self.l_cycles)
```

### 5.4 반환값

- **Single-level**: `cycles` = `int` (총 cycle 수)
- **Two-level**: `cycles` = `Tuple[int, int]` (완료된 H-cycles, L-cycles per H)

예시:
- H=2, L=6 → 총 12 ops 후 early stop
- H=5, L=6 → 총 30 ops (최대)

### 5.5 학습 vs 추론

| 모드 | Confidence 체크 | Early Stop | 설명 |
|------|-----------------|------------|------|
| **학습** | ❌ (선택) | ❌ | 모든 cycle 실행, gradient flow |
| **추론** | ✅ | ✅ | 충분히 confident하면 조기 종료 |

학습 시에도 `confidence_threshold > 0`이면 confidence를 계산하지만,
gradient flow에는 영향 없음 (`no_grad` 내에서 계산).

---

## 6. Stepwise Training (분할 학습)

### 6.1 개념

기존 학습에서는 모든 augmentation point를 처리한 후 **한 번만** reasoner forward → loss 계산을 합니다.
Stepwise training은 **각 augmentation point마다** intermediate loss를 계산하여
compressor에 즉각적인 gradient 신호를 제공합니다.

```
기존 방식:
  [aug1] → [aug2] → [aug3] → reasoner → loss

Stepwise 방식:
  [aug1] → reasoner → loss₁
  [aug1, aug2] → reasoner → loss₂
  [aug1, aug2, aug3] → reasoner → loss₃ (final)
  total_loss = final_loss + weight × mean(loss₁, loss₂)
```

### 6.2 구현 방식

각 augmentation point에서:
1. 현재까지 축적된 `current_inputs_embeds` (원본 토큰 + latent 토큰)에
2. **다음 segment**의 원본 토큰을 lookahead로 추가 (loss 계산 대상 확보)
3. 부분 시퀀스로 reasoner forward 실행
4. Latent 위치 필터링 후 cross-entropy loss 계산
5. Intermediate losses를 평균하여 최종 loss에 가중합

```python
# modeling_memgen.py _forward() 내부
if stepwise_enabled:
    # 다음 augmentation point까지의 토큰을 lookahead
    lookahead_embeds = inputs_embeds[:, aug_point_idx:next_aug_or_end]

    # 부분 시퀀스 구성
    temp_embeds = cat([current_inputs_embeds, lookahead_embeds])

    # Reasoner forward (frozen, grad flows through inputs_embeds)
    partial_logits = reasoner(inputs_embeds=temp_embeds).logits

    # Latent 위치 제거 후 loss 계산
    step_loss = CrossEntropyLoss(valid_partial_logits, labels[:, :next_end])
    intermediate_losses.append(step_loss)

# forward() 에서:
# total_loss = final_loss + stepwise_weight * mean(intermediate_losses)
```

### 6.3 Loss 결합 공식

```
total_loss = L_final + w × mean(L_step₁, L_step₂, ..., L_stepN)
```

- `L_final`: 전체 시퀀스 기반 최종 loss (weight = 1.0)
- `L_step_i`: i번째 augmentation point에서의 intermediate loss
- `w`: `recursive_stepwise_loss_weight` (기본값 0.5)
- Intermediate losses는 **평균** (aug point 수에 비례하지 않음)

### 6.4 Config 옵션

```yaml
recursive_memory:
  stepwise_training: true     # Stepwise 학습 활성화
  stepwise_loss_weight: 0.5   # Intermediate loss 가중치 (final = 1.0)
```

### 6.5 비용 고려사항

- 각 augmentation point마다 reasoner forward가 추가 (N개 aug point → N+1회 forward)
- Reasoner는 frozen이므로 파라미터 gradient 저장 없음
- 초기 augmentation point의 부분 시퀀스가 짧아 상대적으로 빠름
- **메모리 사용량 증가**: 각 intermediate forward의 computation graph가 유지됨

### 6.6 Gradient Flow

```
L_step₁ ← partial_reasoner(accumulated₁ + lookahead₁)
  ← latent₁ (recursive_compressor output at aug point 1)
  ← context₁ (projection output)

L_step₂ ← partial_reasoner(accumulated₂ + lookahead₂)
  ← latent₂ (recursive_compressor output at aug point 2)
  ← latent₁도 accumulated₂에 포함 → latent₁에도 gradient!

L_final ← full_reasoner(full_sequence)
  ← 모든 latent에 gradient
```

**핵심**: 초기 augmentation point의 compressor가 **여러 intermediate loss에서 gradient를 받음**.
이는 초기 memory 품질 향상에 유리합니다.

---

## 7. 실험 설계

### 7.1 기본 실험 (현재 가능)

| 실험 | Projection | Cycles | Stepwise | Config |
|------|------------|--------|----------|--------|
| A | ✅ With | 10 고정 | ❌ | `skip_projection: false` |
| B | ❌ Without | 10 고정 | ❌ | `skip_projection: true` |
| C | ❌ Without | Two-level (L=6, H=5) | ❌ | `two_level: true, skip_projection: true` |
| D | ✅ With | Two-level (L=6, H=5) | ❌ | `two_level: true, skip_projection: false` |
| E | ❌ Without | 10 고정 | ✅ | `skip_projection: true, stepwise_training: true` |
| F | ❌ Without | Two-level | ✅ | `two_level: true, stepwise_training: true` |

---

## 8. 발견된 문제점

### 8.1 Residual Connection 없음

현재 구현:
```python
combined = rms_norm(self.self_attn(combined))  # No residual
z = combined[:, -num_latents:]
z = rms_norm(self.mlp(z))  # No residual
```

표준 Transformer:
```python
x = x + self.self_attn(self.norm1(x))  # With residual
x = x + self.mlp(self.norm2(x))        # With residual
```

**영향**: Gradient vanishing 가능성, 학습 불안정

### 8.2 Pre-norm vs Post-norm

현재: Post-norm (output에 RMSNorm)
```python
combined = rms_norm(self.self_attn(combined))
```

권장: Pre-norm (input에 RMSNorm, modern Transformers 표준)
```python
combined = combined + self.self_attn(rms_norm(combined))
```

### 8.3 Query Latents 초기화

현재:
```python
z = query_latents.unsqueeze(0).expand(B, -1, -1).clone()
```

매 cycle마다 **같은 query_latents에서 시작** → context 정보가 z에 축적되는 방식

---

## 9. Config 옵션 정리

### 9.1 기본 옵션

```yaml
recursive_memory:
  enabled: true
  skip_projection: false    # true: projection 없이 학습
  weaver_style: true        # WeaverStyle compressor 사용
  hidden_size: 4096
  num_heads: 8
  attn_rank: 64
  mlp_rank: 128
  max_cycles: 10
  confidence_threshold: -1.0  # >0: early stop 활성화
  top_k: 10
  verbose_cycles: false       # cycle별 confidence 로깅
```

### 9.2 Two-Level Cycle 옵션

```yaml
recursive_memory:
  two_level: true         # Two-level 모드 활성화
  l_cycles: 6             # L-cycle 반복 횟수 (내부 루프)
  max_h_cycles: 5         # H-cycle 최대 반복 횟수 (외부 루프)
  # 총 최대 연산: l_cycles × max_h_cycles = 30
```

### 9.3 Stepwise Training 옵션

```yaml
recursive_memory:
  stepwise_training: true     # Delimiter 단위 intermediate loss 활성화
  stepwise_loss_weight: 0.5   # Intermediate loss 가중치 (final = 1.0)
```

---

## 10. 파라미터 수 비교

| 모드 | recursive_compressor | Projections | Total |
|------|---------------------|-------------|-------|
| With Projection | ~7.9M | ~33.6M | ~41.5M |
| Without Projection | ~7.9M | 0 | ~7.9M |

**Without Projection이 5배 적은 파라미터**

---

## 11. 권장 실험 순서

1. **실험 B**: Without Projection, 10 cycles 고정 (가장 단순, 현재 진행 중)
2. **실험 A**: With Projection, 10 cycles 고정 (비교)
3. **실험 C**: Without Projection, Two-level (L=6, H=5)
4. **실험 E**: Without Projection, 10 cycles + Stepwise Training
5. **실험 F**: Without Projection, Two-level + Stepwise Training
6. **Residual 추가** 후 재실험 (필요 시)

---

## 12. 다음 단계

### 즉시 실행 가능
- ✅ 실험 B: `skip_projection: true`, 10 cycles (진행 중)
- 실험 C: `skip_projection: true`, `two_level: true`
- 실험 E: `skip_projection: true`, `stepwise_training: true`
- 실험 F: `skip_projection: true`, `two_level: true`, `stepwise_training: true`

### 선택적 개선
1. Residual connection 추가
2. Pre-norm으로 변경
