# MemGen Ablation Study - 중간 결과 보고

## 1. 실험 개요

MemGen의 각 구성요소(LoRA, Weaver LLM, Projections, Query Latents)가 성능에 미치는 영향을 분석하기 위한 ablation study.

**기본 설정**: GSM8K, Qwen3-8B, prompt_aug=1, inference_aug=5

---

## 2. 실험 결과 요약

### 2.1 주요 모드별 정확도 (동일 체크포인트 0108 기준)

| 모드 | 정확도 | LoRA | LLM | Projection | Query Latents | MLP |
|------|--------|------|-----|------------|---------------|-----|
| No-LoRA | **89.16%** | - | O | O | O | - |
| **Skip-LoRA** | **88.93%** | - | O | O | O | - |
| No Inference Aug | 87.64% | - | O | O | O (prompt only) | - |
| **Full (LoRA ON)** | **81.73%** | O | O | O | O | - |
| LatentProcessor* | 79.91% | - | O | O | O | O |
| Projection-Only* | 54.89% | - | - | O | - | - |
| Query-Projection | 진행중 | - | - | O | O | - |

*다른 체크포인트 사용 (아래 참조)

### 2.2 LTPO (Test-Time Optimization)

| 실험 | 정확도 | 체크포인트 | 설명 |
|------|--------|-----------|------|
| LTPO Aggressive | 88.63% | 0108 | latent 최적화 (lr=0.03, sigma=0.1) |
| LTPO 기본 | 81.43% | 0108 | 기본 LTPO 설정 |

### 2.3 Ablation 결과 (통제 실험)

| 실험 | 정확도 | 변경 내용 |
|------|--------|----------|
| Inference Aug만 사용 | 64.90% | Prompt aug 없이 inference만 |
| Random Prompt Latent | 62.85% | 학습된 latent 대신 random |
| Random Inference Latent | 62.85% | 학습된 latent 대신 random |

### 2.4 사용된 체크포인트

| 체크포인트 | 학습 날짜 | 내용 |
|-----------|----------|------|
| `pn=1_pl=8_in=5_il=8_20260108-053458` | 01/08 | Full mode 학습 (projections.pt + weaver_lora/) |
| `pn=1_pl=8_in=5_il=8_20260115-123022` | 01/15 | Skip-LoRA 학습 (skip_lora.pt) |
| `pn=1_pl=8_in=5_il=8_20260116-213205` | 01/16 | LatentProcessor 학습 (latent_processor.pt) |

---

## 3. 핵심 발견

### 3.1 LoRA는 학습 시 필요하지만, 추론 시 불필요

```
학습: Query Latents ←(gradient)← LLM(+LoRA) ←(gradient)← Loss
추론: Query Latents → LLM(base) → Memory Tokens
```

- Full mode로 학습 후 LoRA를 빼고 평가 (Skip-LoRA) → **88.93%**
- Full mode로 학습 후 LoRA 포함 평가 → **~87%**
- LoRA가 오히려 추론 시 **방해**가 될 수 있음

### 3.2 메모리는 Query Latents + Projections에 저장됨

| 구성요소 | 제거 시 영향 | 역할 추정 |
|----------|-------------|----------|
| Query Latents | 62.85% (random) | 학습된 "질문 템플릿" 역할 |
| Projections | ~54.89% (proj-only) | 공간 변환 매핑 |
| Weaver LLM | 진행중 (query-proj) | 문맥 인코딩 |
| LoRA | +1.93% (유해) | 학습 시 gradient steering |

### 3.3 LatentProcessor (MLP)는 LoRA를 대체하지 못함

- Skip-LoRA (88.93%) vs LatentProcessor (79.91%)
- MLP가 LLM의 attention-based 문맥 인코딩을 대체하기엔 부족
- 단순 `Linear + SiLU` 구조의 한계

---

## 4. 아키텍처 플로우

### 4.1 Original MemGen (Full Mode)

```
┌─────────────────────────────────────────────────────┐
│                    학습 시                            │
│                                                     │
│  [Input Tokens]                                     │
│       │                                             │
│       ▼                                             │
│  Reasoner Embedding Layer                           │
│       │                                             │
│       ▼                                             │
│  ┌─────────────┐     ┌──────────────────────┐      │
│  │ Input Embeds │────▶│ reasoner_to_weaver   │      │
│  │ (4096-dim)   │     │ (Linear 4096→4096)   │      │
│  └─────────────┘     └──────────┬───────────┘      │
│                                  │                   │
│                                  ▼                   │
│                    ┌─────────────────────────┐      │
│                    │  Weaver LLM + LoRA      │      │
│                    │  Input: [embeds] + [Q]  │      │
│                    │  Q = query_latents (8)   │      │
│                    └────────────┬────────────┘      │
│                                 │                    │
│                                 ▼                    │
│                    ┌─────────────────────────┐      │
│                    │ Hidden States[-8:]       │      │
│                    │ (마지막 8개 = Q 위치)     │      │
│                    └────────────┬────────────┘      │
│                                 │                    │
│                                 ▼                    │
│                    ┌─────────────────────────┐      │
│                    │ weaver_to_reasoner       │      │
│                    │ (Linear 4096→4096)       │      │
│                    └────────────┬────────────┘      │
│                                 │                    │
│                                 ▼                    │
│                    ┌─────────────────────────┐      │
│                    │ Memory Tokens (8개)      │      │
│                    │ (4096-dim soft vectors)  │      │
│                    └────────────┬────────────┘      │
│                                 │                    │
│                                 ▼                    │
│  ┌──────────────────────────────────────────┐       │
│  │ Reasoner LLM                              │       │
│  │ Input: [원본 embeds] + [memory tokens]    │       │
│  │        (inputs_embeds로 직접 입력)         │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### 4.2 Skip-LoRA Mode (평가 시 최적)

```
┌─────────────────────────────────────────────────────┐
│  [Input Embeds]                                     │
│       │                                             │
│       ▼                                             │
│  reasoner_to_weaver (Linear)                        │
│       │                                             │
│       ▼                                             │
│  ┌─────────────────────────────────────────┐        │
│  │  Weaver LLM (LoRA 비활성화!)             │        │
│  │                                         │        │
│  │  [weaver_embeds] + [query_latents]      │        │
│  │       ↓                                 │        │
│  │  Base LLM Forward (LoRA OFF)            │        │
│  │       ↓                                 │        │
│  │  Hidden States[-8:]                     │        │
│  └──────────────────┬──────────────────────┘        │
│                     │                                │
│                     ▼                                │
│  weaver_to_reasoner (Linear)                        │
│                     │                                │
│                     ▼                                │
│          [Memory Tokens] → Reasoner                 │
└─────────────────────────────────────────────────────┘

핵심: LoRA 없이도 query_latents가 attention으로
     전체 문맥을 보고 memory를 생성
```

### 4.3 LatentProcessor Mode

```
┌─────────────────────────────────────────────────────┐
│  [Input Embeds]                                     │
│       │                                             │
│       ▼                                             │
│  reasoner_to_weaver (Linear)                        │
│       │                                             │
│       ▼                                             │
│  Weaver LLM (LoRA OFF) + query_latents             │
│       │                                             │
│       ▼ Hidden States[-8:]                          │
│       │                                             │
│       ▼                                             │
│  ┌─────────────────────────────────────────┐        │
│  │  LatentProcessor (MLP)                   │        │
│  │  x + Sequential(Linear→SiLU→Linear→SiLU)│        │
│  │  (residual connection)                   │        │
│  └──────────────────┬──────────────────────┘        │
│                     │                                │
│                     ▼                                │
│  weaver_to_reasoner (Linear)                        │
│                     │                                │
│                     ▼                                │
│          [Memory Tokens] → Reasoner                 │
└─────────────────────────────────────────────────────┘

결과: 79.91% (Skip-LoRA 대비 -9.02%)
→ MLP가 attention-based 인코딩을 대체하기엔 부족
```

### 4.4 Projection-Only Mode

```
┌─────────────────────────────────────────────────────┐
│  [Input Embeds] (원본 토큰 일부)                     │
│       │                                             │
│       ▼                                             │
│  reasoner_to_weaver (Linear)                        │
│       │                                             │
│       │  ※ Weaver LLM 건너뜀!                      │
│       │  ※ Query Latents 사용 안함!                 │
│       │                                             │
│       ▼                                             │
│  weaver_to_reasoner (Linear)                        │
│       │                                             │
│       ▼                                             │
│  [Memory Tokens] → Reasoner                        │
└─────────────────────────────────────────────────────┘

결과: 54.89%
→ 단순 Linear 변환만으로는 의미있는 memory 생성 불가
→ Input tokens의 정보가 projection만으로는 충분히 변환되지 않음
```

### 4.5 Query-Projection Only Mode (진행중)

```
┌─────────────────────────────────────────────────────┐
│  [Query Latents] (학습된 8×4096 파라미터)            │
│       │                                             │
│       │  ※ Input Embeds 사용 안함!                  │
│       │  ※ Weaver LLM 사용 안함!                   │
│       │                                             │
│       ▼                                             │
│  reasoner_to_weaver (Linear)                        │
│       │                                             │
│       ▼                                             │
│  weaver_to_reasoner (Linear)                        │
│       │                                             │
│       ▼                                             │
│  [Memory Tokens] → Reasoner                        │
└─────────────────────────────────────────────────────┘

예상: Projection-Only(54.89%) < Query-Projection < Skip-LoRA(88.93%)
→ Query Latents 자체에 얼마나 정보가 인코딩됐는지 측정
→ 문맥 독립적인 "기본 메모리" 효과 측정
```

### 4.6 Recursive MemGen (WeaverStyleCompressor, 구현 완료)

```
┌──────────────────────────────────────────────────────────┐
│  WeaverStyleCompressor (~7.9M params)                    │
│                                                          │
│  [Context Embeds] (Reasoner 공간, 4096-dim)              │
│  [Query Latents z] (8×4096, 학습됨)                      │
│                                                          │
│  ┌─── Cycle (매 반복) ──────────────────────────────┐    │
│  │                                                   │    │
│  │  combined = concat([context, z])                  │    │
│  │       │                                           │    │
│  │       ▼                                           │    │
│  │  Causal Self-Attention (Low-Rank)                 │    │
│  │  [context, z] → z가 모든 context를 attend         │    │
│  │       │                                           │    │
│  │       ▼                                           │    │
│  │  z = RMSNorm(attn_output)[:, -8:]                 │    │
│  │       │                                           │    │
│  │       ▼                                           │    │
│  │  SwiGLU MLP (Low-Rank)                            │    │
│  │       │                                           │    │
│  │       ▼                                           │    │
│  │  z = RMSNorm(mlp_output)                          │    │
│  │                                                   │    │
│  └───────────────────────────────────────────────────┘    │
│                                                          │
│  Cycle 구조 옵션:                                         │
│                                                          │
│  ┌─ Single-level (기본) ─────────────────────────────┐   │
│  │  10 cycles 고정 반복                               │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ Two-level (옵션) ────────────────────────────────┐   │
│  │  H-cycle (외부, max 5회, confidence early stop)    │   │
│  │    └─ L-cycle (내부, 6회 고정)                     │   │
│  │  총 최대: 6 × 5 = 30 ops                          │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  Stepwise Training (옵션):                               │
│  각 augmentation point마다 intermediate loss 계산         │
│  total = final_loss + weight × mean(step_losses)         │
│                                                          │
│  Confidence Check (추론 시):                              │
│  [context + z] → Reasoner → -log(top_k_prob) ≤ threshold │
│  → early stop                                            │
│                                                          │
│  [Memory Tokens = z] → Reasoner                          │
│                                                          │
│  ※ Projection 선택 (skip_projection 옵션)                │
│  ※ LoRA 불필요                                           │
│  ※ Weaver LLM 불필요                                    │
│  ※ 파라미터: ~7.9M (Low-Rank)                           │
│  ※ Projection 포함 시: ~41.5M                           │
│                                                          │
│  MLP 옵션:                                               │
│  ┌─ LowRankSwiGLU (기본) ──────────────────────────┐    │
│  │  gate: LowRank(4096→10936, r=128)               │    │
│  │  up:   LowRank(4096→10936, r=128)               │    │
│  │  down: LowRank(10936→4096, r=128)               │    │
│  │  output = down(silu(gate(x)) * up(x))           │    │
│  │  파라미터: ~5.7M                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─ Full-Rank Linear (full_rank_mlp=true) ─────────┐    │
│  │  nn.Linear(4096, 4096)                           │    │
│  │  W2R projection과 동일 구조                       │    │
│  │  파라미터: ~16.8M                                │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## 5. 추정 및 해석

### 5.1 메모리 저장 위치에 대한 가설

```
                    정보 저장 위치
                    ┌─────────┐
                    │         │
    ┌───────────────┴───┐ ┌───┴───────────────┐
    │  Query Latents    │ │   Projections     │
    │  (static memory)  │ │ (space mapping)   │
    └───────┬───────────┘ └───────┬───────────┘
            │                     │
            ▼                     ▼
    학습된 "질문 패턴"      차원 변환 규칙
    문맥 독립적 정보        reasoner↔weaver 매핑
```

1. **Query Latents = 정적 메모리**: 학습 과정에서 task-specific한 정보 패턴이 인코딩됨
2. **Projections = 공간 변환**: reasoner↔weaver 사이의 최적 매핑 학습
3. **Weaver LLM = 동적 인코더**: 실시간 문맥을 query latents에 반영 (attention)
4. **LoRA = gradient steering**: 학습 시에만 필요한 gradient 전달 도구

### 5.2 왜 Skip-LoRA > Full? (+7.2%)

```
동일 체크포인트(0108)에서:
  Full Mode (LoRA ON):  81.73%
  Skip-LoRA (LoRA OFF): 88.93%  (+7.20%!)

Full Mode:     LLM(base + LoRA) → 학습 분포에 과적합된 출력
Skip-LoRA:     LLM(base)        → 더 일반적인 출력

가설: LoRA가 학습 데이터에 과적합된 attention 패턴을 강제하여
      평가 시 일반화 성능을 저하시킴

증거: LoRA는 Q, V projection에만 적용됨
      → attention 패턴을 직접 변경
      → 학습 데이터 패턴에 맞춘 attention이 평가 시 방해
```

### 5.3 Recursive MemGen의 기대 효과

| 기존 MemGen | Recursive MemGen |
|-------------|-----------------|
| Weaver LLM 전체 forward | Causal Self-Attention + SwiGLU (경량) |
| 단일 pass | 반복 refinement (10 cycles 또는 L×H) |
| 고정 압축 수준 | Confidence 기반 적응적 압축 (two-level) |
| ~8B 파라미터 활용 | ~7.9M 파라미터만 (Low-Rank) |
| LoRA 의존 | LoRA 불필요 |
| 최종 loss만 | Stepwise: 각 aug point별 intermediate loss |

---

## 6. 추가 실험 계획

### 6.1 단기 (현재 진행중)

- [ ] Query-Projection Only 평가 완료 → Query Latents 단독 효과 측정
- [ ] Skip-LoRA 재현성 확인 (memgen_check) → **88.93% 확인 완료**

### 6.2 Recursive Memory 실험 (구현 완료)

| 실험 | Projection | Cycles | Stepwise | Config |
|------|------------|--------|----------|--------|
| A | ✅ With | 10 고정 | ❌ | `skip_projection: false` |
| B | ❌ Without | 10 고정 | ❌ | `skip_projection: true` |
| C | ❌ Without | Two-level (L=6, H=5) | ❌ | `two_level: true` |
| D | ✅ With | Two-level (L=6, H=5) | ❌ | `two_level: true, skip_projection: false` |
| E | ❌ Without | 10 고정 | ✅ | `stepwise_training: true` |
| F | ❌ Without | Two-level | ✅ | `two_level: true, stepwise_training: true` |
| G | ❌ Without | 10 고정 | ❌ | `full_rank_mlp: true` |
| H | ❌ Without | 10 고정 | ✅ | `full_rank_mlp: true, stepwise_training: true` |

### 6.3 중기

| 실험 | 목적 | 예상 |
|------|------|------|
| SwiGLU MLP LatentProcessor | MLP 표현력 향상 효과 | 79.91% → ? |
| 다른 체크포인트로 Skip-LoRA | 학습 시점에 따른 차이 | 80.29% (0115) vs 88.93% (0108) |

### 6.4 장기

| 실험 | 목적 |
|------|------|
| Multi-task 학습 후 Skip-LoRA | 다양한 task에서의 일반화 |
| Recursive + Confidence threshold 조절 | 최적 압축 수준 탐색 |
| 다른 모델 크기 (3B, 70B) | 스케일링 효과 확인 |

---

## 7. 핵심 질문 (미해결)

1. **Query Latents에 인코딩되는 정보의 본질은?**
   - Static한 task pattern? Domain knowledge?
   - Query-Projection 결과로 일부 답 가능

2. **왜 0108 체크포인트가 0115보다 좋은가?**
   - 0108: 88.93% vs 0115: 80.29%
   - 학습 설정 차이? 과적합?

3. **Recursive Memory가 기존 MemGen을 대체할 수 있는가?**
   - 10M 파라미터로 8B LLM 효과를 낼 수 있는가?
   - Cross-Attention이 Full LLM forward를 대체 가능한가?

---

## 8. 실험 환경

| 항목 | 설정 |
|------|------|
| GPU | A100 40GB x 2 |
| Model | Qwen3-8B (Reasoner & Weaver) |
| Dataset | GSM8K (1319 samples, batch=8) |
| Prompt Latents | 8 tokens |
| Inference Latents | 8 tokens |
| Prompt Augmentation | 1회 |
| Inference Augmentation | 5회 |
| WandB | memgen_check project |
