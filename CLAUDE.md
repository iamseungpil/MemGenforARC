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
| 15 | `recursive_memory.py` | **LowRankLinear zero init → grad_norm=0** - `nn.init.zeros_(self.up.weight)`는 LoRA 전용 초기화. Standalone low-rank에서는 `output = up(down(x))` → up이 0이면 output=0, gradient=0. **해결**: `nn.init.xavier_uniform_(self.up.weight)`로 변경. LoRA는 `base + alpha*lora`이므로 zero init OK, standalone은 불가. |
| 16 | `modeling_memgen.py` | **Stepwise training loss 범위 수정** - `step_labels = labels[:, :lookahead_end]`가 위치 0부터 loss 계산 → 이전 segment가 중복 포함됨. **해결**: `.clone()` 후 `step_labels[:, :aug_point_idx] = -100`으로 다음 segment만 loss 계산. 마지막 aug point는 스킵하지 않음 (final loss가 전체를 커버하므로 모든 stepwise가 동일하게 보조 신호). Per-delimiter backward는 `retain_graph=True` 필요 + HF Trainer 충돌로 불가, 누적 후 단일 backward가 수학적으로 동일. |
| 17 | `recursive_memory.py` | **_compress_cycle residual + post-norm** - 원래 `rms_norm(self.self_attn(combined))`로 residual 없이 z를 매 cycle 덮어씀. 1차 수정: pre-norm residual (`x + f(norm(x))`). 2차 수정: post-norm residual (`norm(x + f(x))`) — 동일 weight를 10+ cycle 반복하는 recursive 구조에서 magnitude drift 방지. TinyRecursiveModel과 동일 패턴. |

---

## ⚠️ 삭제 금지 (DO NOT DELETE)

### Skip-LoRA 관련 코드
- `weaver.py`: `_augment_skip_lora()`, `augment_prompt_skip_lora()`, `augment_inference_skip_lora()`
- `modeling_memgen.py`: `skip_lora` 분기 코드
- `modeling_utils.py`: `open_component()` skip_lora 파라미터
- `runner.py`: `_save_skip_lora_checkpoint()`
- `configuration_memgen.py`: `skip_lora` config

### LatentProcessor 관련 코드
- `weaver.py`: `LatentProcessor` 클래스 정의
- `modeling_memgen.py`: `latent_processor_enabled` 초기화, forward/generate 분기
- `modeling_utils.py`: `open_component()` latent_processor 파라미터
- `runner.py`: `_save_latent_processor_checkpoint()`
- `configuration_memgen.py`: `latent_processor`, `latent_processor_depth` config

### Recursive Memory 관련 코드
- `memgen/model/recursive_memory.py`: `WeaverStyleCompressor` 전체 (LowRank layers, two-level cycle)
- `modeling_memgen.py`: `recursive_memory` 분기 (forward, generate), stepwise training 로직
- `modeling_utils.py`: `open_component()` recursive_memory 파라미터
- `runner.py`: `_save_recursive_memory_checkpoint()`, recursive memory 로깅
- `configuration_memgen.py`: `recursive_*` config 전체 (memory, two_level, stepwise)

### LTPO 관련 코드
- `ltpo/memgen_ltpo.py` 전체
- `memgen/runner.py`의 `evaluate_with_ltpo()` 메서드

### 기타 유지 코드
- `data/triviaqa/` - multi-turn 지원
- `ARCDynamicEnv` - 향후 확장용
- `data/arc/env.py`의 binary reward 로직
- `main.py`의 mode 분기 로직

---

## 🔧 학습/추론 모드

### Weaver-based 모드 (기존)

| 모드 | LoRA | Query Latents | Projections | LatentProcessor | Config | 파라미터 |
|------|------|---------------|-------------|-----------------|--------|----------|
| **Full** | ✅ | ✅ | ✅ | ❌ | (기본값) | ~42.6M |
| **Skip-LoRA** | ❌ disabled | ✅ | ✅ | ❌ | `skip_lora: true` | ~33.6M |
| **LatentProcessor** | ❌ | ✅ | ✅ | ✅ MLP | `latent_processor: true` | ~33.6M + MLP |
| **Projection-Only** | ❌ | ❌ | ✅ | ❌ | `projection_only: true` | ~33.5M |

### Recursive Memory 모드 (WeaverStyleCompressor)

Weaver LLM을 **완전히 대체**. Low-Rank Causal Self-Attention + SwiGLU로 memory 생성.

**⚠️ 핵심: Weaver LLM은 recursive_memory 모드에서 전혀 호출되지 않음**
- `weaver.augment_prompt()`, `weaver.augment_inference()` 호출 안 됨
- Weaver의 query_latents도 미사용 — WeaverStyleCompressor가 자체 query_latents 보유
- Weaver LLM + LoRA는 초기화만 되고 forward에 참여하지 않음 (frozen)

#### 훈련 시 코드 흐름

1. **`runner.py`**: `open_component("weaver", recursive_memory=True)` 호출
2. **`modeling_utils.py:72-83`**: recursive_compressor만 `requires_grad=True`, weaver 전체 frozen
3. **`modeling_memgen.py:216-238`**: forward에서 `if self.recursive_memory:` 분기 → weaver 건너뛰고 `self.recursive_compressor()` 직접 호출

```
# Forward 흐름 (modeling_memgen.py:216-238)
각 augmentation point에서:
  context → (선택) reasoner_to_weaver 프로젝션 →
  recursive_compressor(context, is_prompt) → latent_embeds →
  (선택) weaver_to_reasoner 역프로젝션 →
  [기존 시퀀스 + latent_embeds] 연결 → reasoner → logits → loss
```

#### 학습 가능 파라미터

| 컴포넌트 | 학습 | 비고 |
|----------|:---:|------|
| `recursive_compressor` (Self-Attn + MLP + query_latents) | ✅ | 핵심 학습 대상 |
| `reasoner_to_weaver` / `weaver_to_reasoner` | ✅/❌ | `skip_projection=False`일 때만 |
| Weaver LLM + LoRA + Weaver query_latents | ❌ | 전부 frozen, 미호출 |
| Reasoner | ❌ | frozen |

#### WeaverStyleCompressor 내부 (`recursive_memory.py`)

```python
# 자체 query_latents (weaver와 별개)
self.prompt_query_latents = nn.Parameter(...)     # prompt용
self.inference_query_latents = nn.Parameter(...)  # inference용

def _compress_cycle(context, z):
    combined = cat([context, z])                   # [컨텍스트, 잠재토큰]
    combined = rms_norm(combined + self_attn(combined))  # post-norm residual
    z = combined[:, -num_latents:]                 # 잠재 토큰 추출
    z = rms_norm(z + mlp(z))                       # post-norm MLP
    return z
```

#### 옵션

| 옵션 | 설명 | Config |
|------|------|--------|
| **기본** | 10 cycles 고정, projection 포함 | `recursive_memory.enabled: true` |
| **skip_projection** | projection 없이 compressor만 | `recursive_memory.skip_projection: true` |
| **two_level** | H-cycle(max 5) × L-cycle(6) 구조 | `recursive_memory.two_level: true` |
| **stepwise_training** | 각 aug point마다 intermediate loss | `recursive_memory.stepwise_training: true` |
| **context_update** | TRM-style context update (z_H=context, z_L=z) | `recursive_memory.context_update: true` |

```yaml
# Recursive Memory 전체 config
model:
  recursive_memory:
    enabled: true
    skip_projection: true        # projection 없이 (~7.9M params)
    weaver_style: true
    hidden_size: 4096
    num_heads: 8
    attn_rank: 64
    mlp_rank: 128
    num_latents: 8               # memory token 수 (query latents)
    max_cycles: 10               # single-level 시 cycle 수
    confidence_threshold: -1.0   # >0: early stop 활성화
    top_k: 10
    verbose_cycles: false
    # Two-level cycle
    two_level: false             # H-cycle/L-cycle 구조
    l_cycles: 6                  # 내부 루프 반복 (고정)
    max_h_cycles: 5              # 외부 루프 최대 반복
    # Stepwise training
    stepwise_training: false     # 각 aug point별 intermediate loss
    stepwise_loss_weight: 0.5    # intermediate loss 가중치
    # MLP / Attention options
    full_rank_mlp: false         # nn.Linear 대체 (~16.8M)
    bidirectional: false         # TRM-style bidirectional attention
    context_update: false        # TRM-style context update (z_H=context, z_L=z)
```

- **파라미터**: skip_projection 시 ~7.9M, projection 포함 시 ~41.5M
- **Weaver LLM 미사용**: forward/generate 모두 weaver 호출 없음, reasoner 공간에서 직접 compression
- **체크포인트**: `recursive_memory.pt` (+ `projections.pt` if not skip_projection)
- **auto-save 비활성**: shared tensor 충돌 방지로 `save_strategy="no"`, `_save_recursive_memory_checkpoint()`로 수동 저장

### LatentProcessor 모드
- **목적**: LoRA 대신 MLP로 latent 후처리 (gradient modulation 효과 대체)
- **구조**: `x + MLP(x)` residual 연결, depth 조절 가능
- **설정**: `latent_processor: true`, `latent_processor_depth: 2`
- **위치**: `weaver.py:LatentProcessor` 클래스

### 체크포인트 파일
| 파일 | 내용 | 모드 |
|------|------|------|
| `projections.pt` | Projections + Query Latents | Full, Skip-LoRA |
| `skip_lora.pt` | Projections + Query Latents | Skip-LoRA 전용 |
| `latent_processor.pt` | Projections + Query Latents + MLP | LatentProcessor 전용 |
| `projections_only.pt` | Projections Only | Projection-Only |
| `skip_projection.pt` | Query Latents Only (projection 없음) | Skip-Projection |
| `recursive_memory.pt` | WeaverStyleCompressor weights (query_latents 포함) | Recursive Memory |
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
- **WeaverStyleCompressor** (`recursive_memory.py`): Low-Rank Causal Self-Attn + SwiGLU 기반 memory 압축 (Weaver LLM 대체)
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
| Recursive Memory compressor | `memgen/model/recursive_memory.py` |
| LTPO optimizer | `ltpo/memgen_ltpo.py` |
| GRPO reward | `memgen/trainer/weaver_grpo_trainer.py` |
| Binary reward | `data/arc/env.py` |
| LoRA 키 변환 | `modeling_memgen.py:_load_pretrained_weaver` |
| Recursive Memory 문서 | `docs/recursive_memory_complete.md` |

---

## 폴더 구조 (2026-02-02 업데이트)

```
/home/hjkim/projects/RecursiveMem/
├── MemGenforARC/               # 메인 작업 디렉토리 (git repo)
├── MemGenforARC_crossattn/     # Cross-attention 변형 브랜치
├── MemGenforARC_old/           # 이전 버전 스냅샷
├── MemGenforARC_self_attn/     # Self-attention 변형 브랜치
├── Sbatch/                     # SLURM 배치 스크립트 및 로그
└── recursive_memory_visual.md  # Recursive Memory 시각화 문서
```

**작업 디렉토리**: `/home/hjkim/projects/RecursiveMem/MemGenforARC`

### 각 폴더별 용도
| 폴더 | 용도 |
|------|------|
| `MemGenforARC` | 메인 개발 작업 (이 폴더) |
| `MemGenforARC_crossattn` | Cross-attention 실험 변형 |
| `MemGenforARC_old` | 이전 버전 참조용 |
| `MemGenforARC_self_attn` | Self-attention 실험 변형 |
| `Sbatch` | SLURM 배치 작업 스크립트/로그 |

### 실험 스크립트 구조

```
MemGenforARC/
├── experiments/gsm8k_pipeline/       # 순수 실행 로직 (모델, config, accelerate launch)
│   ├── common.sh
│   ├── Qwen3-8B/
│   │   ├── 01_weaver_pretrain.sh
│   │   ├── 01_recursive_memory_train.sh
│   │   ├── 02_eval_weaver.sh
│   │   ├── 02_weaver_grpo.sh
│   │   ├── 03~05_...
│   │   └── 07_ltpo_sweep.sh
│   └── SmolLM3-3B/
│       ├── 00_vanilla_eval.sh, 00b_base_sft.sh, 00c_eval_base_sft.sh
│       ├── 01_weaver_pretrain.sh
│       ├── 01_recursive_memory_train.sh
│       ├── 02_eval_weaver.sh
│       ├── 02_eval_recursive_memory.sh
│       ├── 03~05_...
│       └── run_all.sh
│
└── runs/                             # Wrapper: EXPERIMENT_DIR 설정 → experiments/ 호출
    └── gsm8k/SmolLM3-3B/
        ├── train_recursive_memory.sh       → 01_recursive_memory_train.sh 호출
        ├── eval_recursive_memory.sh        → 02_eval_recursive_memory.sh 호출
        └── train_eval_recursive_memory.sh  → 01_ 실행 → stdout에서 checkpoint 파싱 → 02_ 호출
```

- **experiments/**: 하이퍼파라미터, 모델, config만 담당. 단독 실행 가능 (fallback 경로 사용)
- **runs/**: EXPERIMENT_NAME 환경변수 설정 후 experiments/ 호출. sbatch 변환 시 이 레벨에서 `#SBATCH` 헤더 추가
- **번호 규칙**: 같은 번호에 여러 스크립트 가능 (01_weaver_pretrain + 01_recursive_memory_train)

### wandb sweep 구조

```
runs/gsm8k/
├── SmolLM3-3B/
│   ├── sweep_recursive_memory.yaml   # sweep 설정 (파라미터, method, metric)
│   └── sweep_recursive_memory.sh     # sweep agent 스크립트 (train + eval)
└── Qwen3-8B/
    ├── sweep_recursive_memory.yaml
    └── sweep_recursive_memory.sh

Sbatch/
├── sweep_recursive_memory_gsm8k_smollm3.sbatch  # SLURM 제출용 (laal_rtx6000)
└── sweep_recursive_memory_gsm8k_qwen3.sbatch    # SLURM 제출용 (laal_a6000)
```

#### sweep 파라미터 (활성/비활성)

| 파라미터 | 값 | 활성 | 설명 |
|----------|-----|:---:|------|
| `max_cycles` | [3, 5, 10] | ✅ | compression cycle 반복 횟수 |
| `learning_rate` | [1e-4, 5e-5, 1e-5] | ✅ | 학습률 |
| `num_train_epochs` | [2, 3] | ✅ | 학습 epoch 수 |
| `bidirectional` | [true, false] | ✅ | 양방향 vs causal attention |
| `context_update` | [true, false] | ✅ | TRM-style context update |
| `full_rank_mlp` | [true, false] | ✅ | nn.Linear full-rank vs LowRankSwiGLU |
| `attn_rank` | [32, 64, 128] | ❌ | self-attention low-rank 차원 |
| `mlp_rank` | [64, 128, 256] | ❌ | SwiGLU MLP low-rank 차원 |
| `num_latents` | [4, 8, 16] | ❌ | memory token 수 |
| `stepwise_training` | [true, false] | ❌ | intermediate loss |
| `stepwise_loss_weight` | [0.3, 0.5, 0.7] | ❌ | intermediate loss 가중치 |
| `two_level` | [true, false] | ❌ | H-cycle × L-cycle 구조 |
| `l_cycles` | [4, 6, 8] | ❌ | 내부 루프 반복 |
| `max_h_cycles` | [3, 5] | ❌ | 외부 루프 최대 반복 |
| `skip_projection` | [true, false] | ❌ | compressor만 학습 |
| `max_inference_aug_num` | [3, 5, 10] | ❌ | inference augmentation 수 |
| `batch_size` | [2, 4, 8] | ❌ | per-device batch size |

비활성 파라미터는 YAML에서 주석 해제로 활성화.

#### wandb run 이름 규칙

wandb agent는 run을 미리 생성하여 자동 이름을 부여함 (예: `worthy-sweep-1`).
커스텀 이름은 **HF TrainingArguments의 `run_name`** 으로 설정:

```bash
# sweep_recursive_memory.sh 내부
export WANDB_RUN_NAME="mc${MAX_CYCLES}_lr${LEARNING_RATE}_ep${NUM_EPOCHS}_bi${BIDIRECTIONAL}_cu${CONTEXT_UPDATE}_frm${FULL_RANK_MLP}_${TIMESTAMP}"

# training 옵션에 전달 → wandb.init(name=...) 에 반영됨
python -m accelerate.commands.launch ... main.py \
  --options ... \
  run.weaver.sft.run_name ${WANDB_RUN_NAME}
```

결과: `mc5_lr1e-05_ep2_bifalse_cufalse_frmfalse_260202_101500_job37191`

**주의**: `WANDB_RUN_NAME` 환경변수만으로는 안 됨. wandb agent가 스크립트 실행 전에 run을 미리 생성하므로, 반드시 `run.weaver.sft.run_name`으로 HF Trainer에 전달해야 함.

#### sweep agent 흐름

각 trial마다 `sweep_recursive_memory.sh`가 호출됨:
1. wandb agent가 `--key=value` 형태로 파라미터 전달
2. 파라미터 파싱 → EXPERIMENT_DIR/EXPERIMENT_SUBDIR 설정
3. **[1/2] Training**: accelerate launch + SHARED_OPTIONS + training hyperparams
4. **Checkpoint 검증**: `${WEAVER_PATH}/recursive_memory.pt` 존재 확인
5. **[2/2] Eval**: 동일 SHARED_OPTIONS + `model.load_weaver_path`로 평가

모델별 고정값:
| | SmolLM3-3B | Qwen3-8B |
|---|---|---|
| hidden_size | 2048 | 4096 |
| num_heads | 16 | 8 |
| conda env | memgen_grpo | memgen |
| port | 29540 | 29541 |

#### sweep 파일 구조

```
runs/gsm8k/<Model>/
├── sweep_recursive_memory.yaml    # sweep 설정 (파라미터, method, metric)
├── sweep_recursive_memory.sh      # 실행 진입점 (sweep 생성 + agent 자동 실행)
└── _sweep_agent.sh                # agent가 매 trial마다 호출하는 스크립트 (직접 실행 X)
```

#### sweep 실행

```bash
# 기본 (30 trials)
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh

# trial 수 지정
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --count 10

# GPU 지정
CUDA_VISIBLE_DEVICES=1 bash runs/gsm8k/Qwen3-8B/sweep_recursive_memory.sh
```

`sweep_recursive_memory.sh`가 sweep 생성 + agent 실행을 한 번에 수행. sweep ID 수동 관리 불필요.

#### sweep 파라미터 변경

`sweep_recursive_memory.yaml`만 수정하면 됨. 다음 실행 시 새 sweep이 자동 생성됨.

```yaml
# 파라미터 추가: 주석 해제
num_latents:
  values: [4, 8, 16]

# 파라미터 값 변경
max_cycles:
  values: [5, 10, 20]   # 기존 [3, 5, 10]에서 변경

# 파라미터 비활성화: 주석 처리
# bidirectional:
#   values: [true, false]
```

`_sweep_agent.sh`는 모든 파라미터를 이미 파싱하므로 YAML만 수정하면 됨.

### 주의사항
- `configs/latent_memory/arc.yaml`의 `data_path`는 환경에 맞게 수동 업데이트 필요

---

## 실험 디렉토리 형식 (EXPERIMENT_DIR 패턴)

### 새 형식 (2단계 구조)

```
~/data/memgen/train/<dataset>/<model>/<experiment_name>/pn=<N>_pl=<N>_in=<N>_il=<N>_<YYMMDD_HHMMSS>[_job<SLURM_ID>]/weaver/
```

- 1단계: `<experiment_name>` — 실험 이름으로 그룹핑 (시간 없음)
- 2단계: `pn=<N>_pl=<N>_in=<N>_il=<N>_<timestamp>` — config + 실행 시간으로 구분

### 예시
```
~/data/memgen/train/gsm8k/SmolLM3-3B/recursive_memory_gsm8k_smollm3/pn=1_pl=8_in=5_il=8_260202_153000_job37001/weaver/
~/data/memgen/train/gsm8k/SmolLM3-3B/recursive_memory_gsm8k_smollm3_1cycle/pn=1_pl=8_in=5_il=8_260202_160000_job37002/weaver/
~/data/memgen/train/kodcode/SmolLM3-3B/recursive_memory_kodcode_smollm3/pn=1_pl=8_in=5_il=8_260202_170000_job37003/weaver/
```

### 기존 형식 (fallback, 환경변수 미설정 시)
```
~/data/memgen/train/<dataset>/<model>/pn=<N>_pl=<N>_in=<N>_il=<N>_<YYYYMMDD-HHMMSS>/weaver/
```

### 사용 방법

#### sbatch 스크립트에서
```bash
# 파라미터 정의
MODEL_SHORT="SmolLM3-3B"
DATASET_NAME="gsm8k"
MAX_PROMPT_AUG_NUM=1
PROMPT_LATENTS_LEN=8
MAX_INFERENCE_AUG_NUM=5
INFERENCE_LATENTS_LEN=8

# EXPERIMENT_DIR 패턴 설정
TIMESTAMP=$(date +%y%m%d_%H%M%S)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  TIMESTAMP="${TIMESTAMP}_job${SLURM_JOB_ID}"
fi
export EXPERIMENT_DIR="${BASE_EXPERIMENT_NAME}"
export EXPERIMENT_SUBDIR="pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_${TIMESTAMP}"

# 결정적 체크포인트 경로 조립
CHECKPOINT_DIR="${HOME}/data/memgen/train/${DATASET_NAME}/${MODEL_SHORT}/${EXPERIMENT_DIR}/${EXPERIMENT_SUBDIR}"
LOAD_WEAVER_PATH="${CHECKPOINT_DIR}/weaver"
```

#### main.py
- `EXPERIMENT_DIR` + `EXPERIMENT_SUBDIR` 환경변수가 둘 다 설정되면 2단계 형식 사용
- 없으면 기존 형식(pn=...) fallback

### Sbatch 로그 네이밍 규칙
```
~/projects/RecursiveMem/Sbatch/logs/<job_name>/<job_name>_YYMMDD_<job_id>.out
```

### sbatch 실행 방법
```bash
# 디렉토리 생성 후 sbatch 제출 (반드시!)
mkdir -p ~/projects/RecursiveMem/Sbatch/logs/<job_name> && sbatch script.sbatch
```

**주의**: SLURM은 `--output` 디렉토리가 미리 존재해야 함. 없으면 job 실패.
