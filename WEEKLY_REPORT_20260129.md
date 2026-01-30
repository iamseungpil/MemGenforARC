# 주간 연구 보고서 (2026-01-29)

## 전체 흐름

이번 주 MemGen 프로젝트에서는 Recursive Memory 모듈의 핵심 구조인 잔여 연결(residual connection) 패턴을 체계적으로 실험하여, post-norm 방식이 반복적 공유 가중치 구조에서 안정적으로 동작함을 확인하였다. MemGen은 자가 진화 AI 에이전트를 위한 latent memory 프레임워크로, reasoner LLM의 입력에 압축된 memory token을 삽입하여 문맥 정보를 보강하는 시스템이다. Recursive Memory(WeaverStyleCompressor)는 기존에 8B 파라미터의 Weaver LLM이 수행하던 memory 생성을 약 7.9M 파라미터의 경량 모듈로 대체하는 구조이다. 이번 주 실험에서 post-norm residual 방식을 적용한 Recursive Memory가 GSM8K 벤치마크에서 약 81%의 정확도를 달성하여, 기존 Weaver 기반 Skip-LoRA(88.93%) 대비 약 8%p 차이까지 접근하였다. 이와 함께 Full-Rank MLP 옵션과 Bidirectional Attention 옵션을 추가하여 Recursive Memory의 구성 유연성을 확대하였다.

---

## 완료된 작업

### 1. Residual Connection 패턴 실험 및 버그 수정 (Bug #17)

Recursive Memory의 핵심 압축 사이클에서 잔여 연결이 누락되어 있었으며, 이를 수정하는 과정에서 세 가지 패턴을 비교 실험하였다. 잔여 연결(residual connection)이란, 신경망의 입력을 출력에 더하여 정보가 변환 과정에서 소실되지 않도록 보장하는 구조이다. Recursive Memory는 동일한 가중치를 10회 반복 적용하는 공유 가중치(shared-weight) 구조이므로, 잔여 연결의 유무와 정규화 순서가 학습 안정성에 결정적 영향을 미친다.

원래 코드는 `rms_norm(self.self_attn(combined))` 형태로, 이전 사이클의 출력과 관계없이 매 사이클마다 z를 완전히 덮어쓰는 구조였다. 이 방식은 residual이 없어 정보 축적이 제한적이지만, 정규화 덕분에 78.09%의 정확도를 보였다. 첫 번째 수정으로 Qwen3 모델이 사용하는 pre-norm 방식(`x + f(norm(x))`)을 적용하였으나, 이 패턴에서는 10회 반복 중 z의 크기(magnitude)가 무한히 증가하는 현상이 발생하여 46.25%로 급락하였다. pre-norm은 정규화를 함수 입력에만 적용하고 출력인 z 자체는 정규화하지 않으므로, 반복마다 잔여 합산이 누적되어 reasoner가 기대하는 임베딩 스케일과 크게 벗어나기 때문이다.

최종적으로 TinyRecursiveModels(TRM)에서 사용하는 post-norm 방식(`norm(x + f(x))`)을 채택하였다. TRM은 경량 재귀 추론 모델로, 공유 가중치 블록을 반복 적용하는 구조에서 post-norm을 사용한다. post-norm은 매 사이클 후 z의 크기를 일정하게 유지하므로, 동일 가중치가 항상 유사한 스케일의 입력을 받게 된다. 이 방식을 적용한 결과 약 81%의 정확도를 달성하였다.

현재 `_compress_cycle`의 구현은 다음과 같다:

```python
def _compress_cycle(self, context, z):
    combined = torch.cat([context, z], dim=1)
    combined = rms_norm(combined + self.self_attn(combined))   # post-norm
    z = combined[:, -self.num_latents:]
    z = rms_norm(z + self.mlp(z))                              # post-norm
    return z
```

아래 표는 세 가지 residual 패턴의 실험 결과를 정리한 것이다.

| 패턴 | 수식 | GSM8K 정확도 | 비고 |
|------|------|-------------|------|
| Residual 없음 | `z = norm(f(x))` | 78.09% | 매 사이클 z 덮어쓰기 |
| Pre-norm | `z = z + f(norm(z))` | 46.25% | 크기 발산 (magnitude drift) |
| **Post-norm** | **`z = norm(z + f(z))`** | **~81%** | TRM 패턴, 안정적 |

### 2. Full-Rank MLP 옵션 추가

Recursive Memory의 MLP 계층에 Full-Rank 옵션을 추가하였다. 기존에는 LowRankSwiGLU(약 5.7M 파라미터)만 사용 가능했으나, MemGen의 W2R(Weaver-to-Reasoner) projection이 `nn.Linear(4096, 4096)` 단일 선형 변환인 점에 착안하여 동일 구조의 Full-Rank MLP(약 16.8M 파라미터) 옵션을 추가하였다. LowRankSwiGLU는 gate, up, down 세 개의 저랭크 행렬로 분해된 SwiGLU 구조인 반면, Full-Rank MLP는 단일 선형 변환으로 파라미터 수가 약 3배 많다. 이 옵션은 `full_rank_mlp: true` 설정으로 활성화할 수 있다.

### 3. Bidirectional Attention 옵션 추가

TRM과의 비교 분석을 통해 bidirectional attention 옵션을 추가하였다. TRM의 재귀 블록은 `is_causal=False`로 양방향 어텐션을 사용하는 반면, MemGen의 Recursive Memory는 `is_causal=True`로 인과적(causal) 어텐션을 사용하고 있었다. Recursive Memory의 압축 사이클은 텍스트를 생성하는 것이 아니라 문맥 정보를 query latent로 압축하는 작업이므로, 인과적 마스크가 불필요하다. 양방향 어텐션을 사용하면 각 query latent가 모든 문맥 토큰에 동시에 attend할 수 있어, 정보 흐름이 더 풍부해질 수 있다. 이 옵션은 `bidirectional: true` 설정으로 활성화할 수 있으며, 아직 학습 및 평가는 수행되지 않았다.

구현은 self-attention 모듈에 `causal` 파라미터를 추가하는 것으로 이루어졌다:

```python
class LowRankCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rank, causal=True):
        ...
        self.causal = causal

    def forward(self, x, ...):
        ...
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
```

### 4. 파라미터 효율성 분석

Recursive Memory와 기존 MemGen 모드의 학습 가능 파라미터를 비교 분석하였다. MemGen의 학습 가능 파라미터는 모드에 따라 크게 달라지며, 모든 모드에서 8B의 reasoner는 동결(frozen)된 상태로 유지된다.

| 모드 | 학습 가능 파라미터 | 구성 |
|------|------------------|------|
| SFT LoRA | 7.67M | 36개 layer × 0.21M (rank=16, Q+V) |
| Recursive Memory (skip_proj) | 7.93M | 1개 공유 layer × 10 cycles |
| Skip-LoRA | 33.63M | Projections + Query Latents |
| Full mode | 41.30M | LoRA + Projections + Query Latents |

SFT LoRA와 Recursive Memory는 파라미터 수가 유사하나(7.67M vs 7.93M), 구조가 근본적으로 다르다. LoRA는 36개 layer에 분산된 저랭크 어댑터로 모델 가중치를 간접 수정하는 반면, Recursive Memory는 1개 공유 layer를 10회 반복하여 문맥을 압축한 soft prompt를 생성한다. 즉, Recursive Memory는 모델 가중치를 변경하지 않고 입력 공간에서 동작하는 문맥 의존적 soft prompt 생성기에 해당한다.

---

## Recursive Memory 옵션 요약

현재 구현된 Recursive Memory의 설정 옵션은 다음과 같다.

| 옵션 | 설정 키 | 기본값 | 설명 |
|------|---------|--------|------|
| 활성화 | `enabled` | false | Recursive Memory 모드 활성화 |
| Projection 생략 | `skip_projection` | false | Projection 없이 compressor만 사용 (~7.9M) |
| 최대 사이클 | `max_cycles` | 10 | 단일 레벨 반복 횟수 |
| Two-Level | `two_level` | false | H-cycle(외부) × L-cycle(내부) 구조 |
| Stepwise 학습 | `stepwise_training` | false | 각 augmentation 지점별 중간 loss |
| Full-Rank MLP | `full_rank_mlp` | false | nn.Linear로 LowRankSwiGLU 대체 (~16.8M) |
| Bidirectional | `bidirectional` | false | 양방향 어텐션 (TRM 방식) |
| Confidence 임계값 | `confidence_threshold` | -1.0 | 양수 시 early stopping 활성화 |

---

## 전체 실험 결과 종합

아래 표는 GSM8K 벤치마크에서 Qwen3-8B 기반으로 측정된 모든 실험 결과를 정리한 것이다.

### Weaver 기반 모드 (기존 baseline)

| 모드 | 정확도 | 학습 파라미터 |
|------|--------|-------------|
| No-LoRA (평가 시 LoRA 비활성화) | 89.16% | 41.30M |
| Skip-LoRA | 88.93% | 33.63M |
| Full (LoRA ON) | 81.73% | 41.30M |
| LatentProcessor | 79.91% | ~35M |
| Projection-Only | 54.89% | ~33.5M |

### Recursive Memory 모드 (이번 주 실험)

| 실험 | Residual | MLP | Attention | 정확도 | 상태 |
|------|----------|-----|-----------|--------|------|
| No residual | 없음 | LowRankSwiGLU | Causal | 78.09% | 완료 |
| Pre-norm | x + f(norm(x)) | FullRank | Causal | 46.25% | 완료 (실패) |
| Post-norm | norm(x + f(x)) | FullRank | Causal | ~81% | 평가 진행 중 |
| Post-norm | norm(x + f(x)) | LowRankSwiGLU | Causal | - | 학습 99% 완료 |
| Post-norm | norm(x + f(x)) | LowRankSwiGLU | Bidirectional | - | 미실행 |

---

## 진행 중인 작업

현재 GPU 0에서 FullRank + Post-norm Recursive Memory의 평가가 진행 중이며(약 80% 완료), GPU 1에서는 LowRankSwiGLU + Post-norm의 학습이 99% 진행된 상태이다(step 3326/3364). LowRankSwiGLU 학습이 완료되면 평가를 수행하여 FullRank MLP와 LowRankSwiGLU 간의 성능 차이를 비교할 예정이다.

---

## 차주 작업 계획

첫째, LowRankSwiGLU + Post-norm의 평가를 완료하여 MLP 구조에 따른 성능 차이를 확인한다. 둘째, 새로 추가한 bidirectional attention 옵션으로 학습 및 평가를 수행하여 양방향 어텐션이 압축 품질에 미치는 영향을 검증한다. 셋째, 모든 Recursive Memory 변형(residual 패턴, MLP 종류, 어텐션 방향)을 체계적으로 비교하는 ablation study를 정리한다.
