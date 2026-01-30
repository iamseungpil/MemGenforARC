# Recursive Memory 변경 사항 (2026-01-30)

## 개요

Weaver LLM (8B params)을 경량 모듈 (~7.9M params)로 대체하는 **Recursive Memory (WeaverStyleCompressor)** 구현.
동일한 가중치를 10회 반복 적용하여 문맥을 query latent로 압축하는 구조.

---

## 새로 추가된 파일

### 핵심 모듈

| 파일 | 설명 |
|------|------|
| `memgen/model/recursive_memory.py` | WeaverStyleCompressor 전체 구현 (265 lines) |

주요 구성 요소:
- `LowRankLinear`: 저랭크 분해 선형 변환 (`down → up`)
- `LowRankCausalSelfAttention`: Low-Rank Q/K/V/O self-attention (causal 또는 bidirectional)
- `LowRankSwiGLU`: 저랭크 SwiGLU MLP
- `WeaverStyleCompressor`: 전체 압축 모듈
  - `_compress_cycle()`: post-norm residual (`norm(x + f(x))`)
  - `compress()`: 10-cycle 반복 압축
  - Two-level cycle (H-cycle × L-cycle) 지원
  - Confidence-based early stopping 지원

### Config 파일

| 파일 | 용도 |
|------|------|
| `configs/latent_memory/gsm8k_recursive_skip_proj.yaml` | Recursive Memory 학습 (lr=1e-4, skip projection) |
| `configs/latent_memory/gsm8k_recursive_eval.yaml` | Recursive Memory 평가 |
| `configs/latent_memory/gsm8k_weaver_style.yaml` | Weaver-style 학습 |
| `configs/latent_memory/arc_recursive.yaml` | ARC 데이터셋용 Recursive Memory |

### 스크립트

| 파일 | 용도 |
|------|------|
| `scripts/train_skip_projection.sh` | Skip-projection 학습 |
| `scripts/eval_skip_projection.sh` | Skip-projection 평가 |
| `scripts/eval_full_mode.sh` | Full mode 평가 |
| `scripts/eval_output_projection_only.sh` | Output-projection-only 평가 |
| `scripts/full_mode_train.sh` | Full mode 학습 |
| `scripts/recursive_memory_grpo_train.sh` | GRPO 학습 |
| `scripts/draw_recursive_memory_arch.py` | 아키텍처 다이어그램 생성 |
| `experiments/gsm8k_pipeline/Qwen3-8B/01_recursive_memory_train.sh` | GSM8K 파이프라인 |

### 문서

| 파일 | 내용 |
|------|------|
| `docs/recursive_memory_complete.md` | Recursive Memory 전체 문서 |
| `docs/recursive_memory_implementation.md` | 구현 상세 문서 |
| `WEEKLY_REPORT_20260129.md` | 주간 연구 보고서 (실험 결과 포함) |
| `memgen_ablation_study.md` | Ablation study 정리 |

---

## 수정된 파일

### `memgen/model/configuration_memgen.py` (+46 lines)

Recursive Memory 관련 config 필드 추가:

```python
# 새로 추가된 필드
recursive_memory: bool = False
recursive_weaver_style: bool = True
recursive_hidden_size: int = 4096
recursive_num_heads: int = 8
recursive_attn_rank: int = 64
recursive_mlp_rank: int = 128
recursive_max_cycles: int = 10
recursive_confidence_threshold: float = 0.5
recursive_skip_projection: bool = False
recursive_two_level: bool = False
recursive_stepwise_training: bool = False
recursive_full_rank_mlp: bool = False     # nn.Linear 대체 (~16.8M)
recursive_bidirectional: bool = False     # TRM-style 양방향 attention
```

기타 mode 플래그: `query_projection_only`, `skip_projection`, `output_projection_only`

### `memgen/model/modeling_memgen.py` (+407, -322 lines)

주요 변경:
1. **Recursive Memory 초기화** — `WeaverStyleCompressor` 생성 및 config 매핑
2. **Forward 분기** — `recursive_memory_enabled` 시 Weaver LLM 대신 compressor 사용
3. **Generate 분기** — 추론 시 recursive memory 경로
4. **Stepwise training** — 각 augmentation point마다 intermediate loss 계산
5. **체크포인트 로딩** — `recursive_memory.pt` 로드 로직
6. **valid_mask 필터링** — memory token 위치 제거하여 loss 계산 정합성 보장

### `memgen/model/modeling_utils.py` (+60, -16 lines)

`open_component()` 함수에 recursive memory 파라미터 추가:
- `recursive_memory`, `latent_processor`, `skip_projection` 등 mode별 학습 가능 파라미터 분리
- 각 mode에서 열어야 하는 파라미터만 선택적으로 `requires_grad=True` 설정

### `memgen/model/weaver.py` (+66 lines)

1. **LatentProcessor 클래스** — LoRA 대신 MLP로 latent 후처리 (`x + MLP(x)` residual)
2. **augment 관련 메서드** — skip_projection, output_projection_only 분기 추가

### `memgen/runner.py` (+108 lines)

1. **`_save_recursive_memory_checkpoint()`** — `recursive_memory.pt` 저장
2. **`_save_latent_processor_checkpoint()`** — `latent_processor.pt` 저장
3. **recursive memory 학습 파라미터 로깅** — 학습 가능 파라미터 이름/shape 출력
4. **auto-save 비활성화** — shared tensor 크래시 방지 (`save_strategy="no"`)

### `CLAUDE.md` (+116 lines)

- 버그 수정 내역 #15~#17 추가 (zero init, stepwise loss, post-norm residual)
- LatentProcessor, Recursive Memory 삭제 금지 목록 추가
- Recursive Memory config 전체 문서화
- 학습/추론 모드 표 업데이트

### `configs/latent_memory/gsm8k.yaml` (+4, -4 lines)

LTPO 파라미터 조정 (aggressive settings)

---

## 핵심 구현: `_compress_cycle` (Post-norm Residual)

```python
def _compress_cycle(self, context, z):
    combined = torch.cat([context, z], dim=1)
    combined = rms_norm(combined + self.self_attn(combined))  # post-norm after attention
    z = combined[:, -self.num_latents:]
    z = rms_norm(z + self.mlp(z))                             # post-norm after MLP
    return z
```

- TinyRecursiveModels (TRM)와 동일 패턴
- 공유 가중치 10회 반복에서 magnitude drift 방지
- Context는 매 cycle 고정 (Perceiver-style), z만 evolve

---

## 설정 옵션 요약

```yaml
model:
  recursive_memory:
    enabled: true
    skip_projection: true        # projection 없이 (~7.9M params)
    max_cycles: 10
    full_rank_mlp: false         # true: nn.Linear (~16.8M), false: LowRankSwiGLU (~5.7M)
    bidirectional: false         # true: TRM-style 양방향 attention
    two_level: false             # H-cycle × L-cycle 구조
    stepwise_training: false     # 각 aug point별 intermediate loss
    confidence_threshold: -1.0   # >0: early stopping 활성화
```

---

## 실험 결과 (GSM8K, Qwen3-8B)

| Residual | MLP | Attention | 정확도 | 파라미터 |
|----------|-----|-----------|--------|---------|
| 없음 | LowRankSwiGLU | Causal | 78.09% | ~7.9M |
| Pre-norm | FullRank | Causal | 46.25% | ~16.8M |
| **Post-norm** | **FullRank** | **Causal** | **81.35%** | ~16.8M |
| Post-norm | LowRankSwiGLU | Causal | 79.23% | ~7.9M |
| Post-norm | FullRank | Bidirectional | 평가 중 | ~16.8M |
| Post-norm | LowRankSwiGLU | Bidirectional | 학습 중 | ~7.9M |

Baseline: Skip-LoRA = 88.93% (33.63M params)
