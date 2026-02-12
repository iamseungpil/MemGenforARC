# CLAUDE.md

## Project Overview

**RecursiveMLP**는 LLM을 위한 경량 Recursive Memory 프레임워크입니다.

- **WeaverStyleCompressor**: Low-Rank Self-Attention + SwiGLU MLP로 context를 latent memory로 압축
- **Delimiter 기반 Augmentation**: 문장 구분자(`,`, `.`, `\n`)에서 자동으로 memory 삽입

> **Note**: 기존 MemGen의 Weaver LLM, Trigger, LTPO 모듈은 제거되었습니다. (2026-02-12 리팩토링)

---

## 🔄 리팩토링 내역 (2026-02-12)

### 삭제된 모듈

| 모듈 | 파일 | 이유 |
|------|------|------|
| **Weaver LLM** | `weaver.py` | WeaverStyleCompressor로 대체 |
| **Trigger** | `trigger.py` | Delimiter 기반 augmentation으로 대체 |
| **Trigger Trainer** | `trigger_grpo_trainer.py` | Trigger 제거로 불필요 |
| **LTPO** | `ltpo/` 폴더 전체 | Test-time 최적화 불필요 |
| **ARC** | `arc/`, `data/arc/` | ARC 작업 불필요 |

### 단순화된 코드

| 파일 | 변경 내용 |
|------|----------|
| `modeling_memgen.py` | Weaver/Trigger import 제거, recursive_compressor만 사용 |
| `modeling_utils.py` | LoRA 스위칭 mixin 제거, delimiter 기반 augmentation만 유지 |
| `runner.py` | Trigger 학습 코드 제거, 단일 train/evaluate 흐름 |
| `configuration_memgen.py` | `recursive_memory=True`, `recursive_skip_projection=True` 기본값 |

### 코드 감소량
- **~2,500줄 이상 삭제** (weaver.py, trigger.py, trigger_grpo_trainer.py, ltpo/, 브랜치 코드 등)

---

## 🚨 버그 수정 내역 (재발 방지)

| # | 파일 | 핵심 내용 |
|---|------|----------|
| 1 | `recursive_memory.py` | **LowRankLinear zero init → grad_norm=0** - `nn.init.zeros_(self.up.weight)`는 LoRA 전용 초기화. Standalone low-rank에서는 `output = up(down(x))` → up이 0이면 output=0, gradient=0. **해결**: `nn.init.xavier_uniform_(self.up.weight)`로 변경. |
| 2 | `recursive_memory.py` | **_compress_cycle residual + post-norm** - 원래 residual 없이 z를 매 cycle 덮어씀. **해결**: post-norm residual (`norm(x + f(x))`) — 동일 weight를 10+ cycle 반복하는 recursive 구조에서 magnitude drift 방지. |
| 3 | `modeling_memgen.py` | **Stepwise training loss 범위 수정** - `step_labels[:, :aug_point_idx] = -100`으로 다음 segment만 loss 계산. |
| 4 | `modeling_memgen.py` | chat_template 복원 - multi-turn `<\|im_start\|>` 토큰 의존 |
| 5 | `configs/zero2.yaml` | `mixed_precision: 'no'`로 복원 |
| 6 | `modeling_memgen.py` | temperature falsy 버그 - `do_sample=False, temperature=0.0` 하드코딩 |
| 7 | `runner.py` | **Static eval chat template 미적용** - 기존: `self.processing_class(text=prompts, ...)` 단순 토크나이징. **해결**: `apply_chat_template(messages_list, add_generation_prompt=True, ...)` 사용하여 `<\|im_start\|>user...<\|im_end\|><\|im_start\|>assistant` 형식 적용. |

---

## Architecture

### 핵심 컴포넌트

```
MemGenModel (modeling_memgen.py)
├── reasoner (base LLM, frozen)
└── recursive_compressor (WeaverStyleCompressor, trainable)
    ├── prompt_query_latents
    ├── inference_query_latents
    ├── low_rank_self_attn
    └── low_rank_mlp (SwiGLU)
```

### WeaverStyleCompressor 내부 (`recursive_memory.py`)

```python
# 자체 query_latents
self.prompt_query_latents = nn.Parameter(...)     # prompt용
self.inference_query_latents = nn.Parameter(...)  # inference용

def _compress_cycle(context, z):
    combined = cat([context, z])                   # [컨텍스트, 잠재토큰]
    combined = rms_norm(combined + self_attn(combined))  # post-norm residual
    z = combined[:, -num_latents:]                 # 잠재 토큰 추출
    z = rms_norm(z + mlp(z))                       # post-norm MLP
    return z
```

### Forward 흐름

```
각 augmentation point (delimiter 위치)에서:
  context →
  recursive_compressor(context, is_prompt) → latent_embeds →
  [기존 시퀀스 + latent_embeds] 연결 → reasoner → logits → loss
```

### 학습 가능 파라미터

| 컴포넌트 | 학습 | 파라미터 |
|----------|:---:|----------|
| `recursive_compressor` | ✅ | ~7.9M (skip_projection) |
| Reasoner | ❌ | frozen |

---

## Config 옵션

```yaml
model:
  recursive_memory:
    enabled: true
    skip_projection: true        # projection 없이 (~7.9M params)
    hidden_size: 4096
    num_heads: 8
    attn_rank: 64
    mlp_rank: 128
    max_cycles: 10               # compression cycle 수
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
    # Options
    full_rank_mlp: false         # nn.Linear 대체 (~16.8M)
    bidirectional: false         # 양방향 attention
    context_update: false        # TRM-style context update

  # Latent 설정 (weaver 키는 하위호환용)
  max_prompt_aug_num: 1
  max_inference_aug_num: 5
  weaver:
    prompt_latents_len: 8
    inference_latents_len: 8
```

### 주요 옵션 설명

| 옵션 | 설명 |
|------|------|
| `max_cycles` | compression cycle 반복 횟수 (기본 10) |
| `confidence_threshold` | >0이면 early stop 활성화. 낮은 confidence(확실한 예측)에서 종료 |
| `two_level` | H-cycle × L-cycle 2단계 구조 |
| `bidirectional` | causal 대신 양방향 attention |
| `context_update` | TRM-style context update (z_H=context, z_L=z) |

---

## 체크포인트

| 파일 | 내용 |
|------|------|
| `recursive_memory.pt` | WeaverStyleCompressor weights (query_latents 포함) |
| `projections.pt` | r2w/w2r projections (skip_projection=False일 때만) |

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

**평가는 직접 실행 가능** (체크포인트 저장 없음)

---

## 데이터셋

| 데이터셋 | 타입 | 용도 |
|----------|------|------|
| gsm8k | Static | Math 문제 |
| gpqa | Static | Graduate-level QA |
| kodcode | Static | 코드 생성 |
| triviaqa | Dynamic | Multi-turn QA |

---

## 수정 전 테스트

```bash
# Import 검증
python -c "from memgen.model import MemGenModel; from memgen.runner import MemGenRunner; print('OK')"
```

---

## 주요 파일 경로

| 기능 | 파일 |
|------|------|
| Model | `memgen/model/modeling_memgen.py` |
| Recursive Memory | `memgen/model/recursive_memory.py` |
| Config | `memgen/model/configuration_memgen.py` |
| Runner | `memgen/runner.py` |
| GRPO Trainer | `memgen/trainer/weaver_grpo_trainer.py` |

---

## 폴더 구조

```
RecursiveMLP/
├── memgen/
│   ├── model/
│   │   ├── configuration_memgen.py
│   │   ├── modeling_memgen.py
│   │   ├── modeling_utils.py
│   │   └── recursive_memory.py      # WeaverStyleCompressor
│   ├── trainer/
│   │   ├── utils.py
│   │   └── weaver_grpo_trainer.py
│   ├── runner.py
│   └── utils.py
├── data/
│   ├── gsm8k/
│   ├── gpqa/
│   ├── kodcode/
│   └── triviaqa/
├── gnosis/                          # Self-awareness 모듈 (향후 확장용)
├── interactions/
├── common/
├── configs/
│   └── latent_memory/
└── main.py
```

---

## Gnosis (향후 확장용)

LLM이 자신의 답변이 맞는지 예측하는 Self-awareness 모듈.

```
gnosis/
├── model/gnosis.py      # MemGenGnosis 클래스
├── trainer/             # 학습 로직
└── data/                # 데이터 생성/레이블링
```

현재 메인 파이프라인에 통합되지 않음. 향후 Recursive Memory의 confidence 계산이나 memory augmentation 결정에 활용 가능.

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
```

### 기존 형식 (fallback, 환경변수 미설정 시)
```
~/data/memgen/train/<dataset>/<model>/pn=<N>_pl=<N>_in=<N>_il=<N>_<YYYYMMDD-HHMMSS>/weaver/
```

### 체크포인트 경로 조립 (sbatch 스크립트)

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

---

## wandb sweep 구조

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

### sweep 파라미터 (활성/비활성)

| 파라미터 | 값 | 활성 | 설명 |
|----------|-----|:---:|------|
| `max_cycles` | min:3, max:10 | ✅ | compression cycle 반복 횟수 (Bayesian) |
| `learning_rate` | [1e-4, 5e-5] | ✅ | 학습률 |
| `num_train_epochs` | min:2, max:4 | ✅ | 학습 epoch 수 (Bayesian) |
| `bidirectional` | [true, false] | ✅ | 양방향 vs causal attention |
| `context_update` | [true, false] | ✅ | TRM-style context update |
| `num_latents` | [4, 8, 16] | ✅ | memory token 수 |
| `two_level` | [true, false] | ✅ | H-cycle × L-cycle 구조 |
| `full_rank_mlp` | [true, false] | ❌ | nn.Linear full-rank vs LowRankSwiGLU |
| `attn_rank` | [32, 64, 128] | ❌ | self-attention low-rank 차원 |
| `mlp_rank` | [64, 128, 256] | ❌ | SwiGLU MLP low-rank 차원 |
| `stepwise_training` | [true, false] | ❌ | intermediate loss |
| `stepwise_loss_weight` | [0.3, 0.5, 0.7] | ❌ | intermediate loss 가중치 |

비활성 파라미터는 YAML에서 주석 해제로 활성화.

### wandb run 이름 규칙

wandb agent는 run을 미리 생성하여 자동 이름을 부여함 (예: `worthy-sweep-1`).
커스텀 이름은 **HF TrainingArguments의 `run_name`** 으로 설정:

```bash
# sweep_recursive_memory.sh 내부
export WANDB_RUN_NAME="mc${MAX_CYCLES}_lr${LEARNING_RATE}_ep${NUM_EPOCHS}_bi${BIDIRECTIONAL}_cu${CONTEXT_UPDATE}_nl${NUM_LATENTS}_tl${TWO_LEVEL}_${TIMESTAMP}"

# training 옵션에 전달 → wandb.init(name=...) 에 반영됨
python -m accelerate.commands.launch ... main.py \
  --options ... \
  run.weaver.sft.run_name ${WANDB_RUN_NAME}
```

결과: `mc5_lr1e-04_ep2_bifalse_cufalse_nl8_tlfalse_260202_101500_job37191`

**주의**: `WANDB_RUN_NAME` 환경변수만으로는 안 됨. wandb agent가 스크립트 실행 전에 run을 미리 생성하므로, 반드시 `run.weaver.sft.run_name`으로 HF Trainer에 전달해야 함.

### sweep agent 흐름

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

### sweep 파일 구조

```
runs/gsm8k/<Model>/
├── sweep_recursive_memory.yaml    # sweep 설정 (파라미터, method, metric)
├── sweep_recursive_memory.sh      # 실행 진입점 (sweep 생성 + agent 자동 실행)
└── _sweep_agent.sh                # agent가 매 trial마다 호출하는 스크립트 (직접 실행 X)
```

### sweep 실행

```bash
# 기본: GPU 1개, agent 1개, 30 trials
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh

# trial 수 지정
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --count 10

# GPU 1개에 agent 2개 (GPU 여유 있을 때, 병렬 탐색)
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --agents-per-gpu 2

# GPU 2개, 각 1개 agent
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --gpus 0,1

# GPU 2개 × agent 2개 = 4개 병렬
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --gpus 0,1 --agents-per-gpu 2

# 기존 sweep에 agent 추가 (새 sweep 생성 안 함)
bash runs/gsm8k/SmolLM3-3B/sweep_recursive_memory.sh --sweep-id gistdslab/RecursiveMem/<id>
```

`sweep_recursive_memory.sh`가 sweep 생성 + agent 실행을 한 번에 수행. sweep ID 수동 관리 불필요.
복수 agent는 같은 sweep을 공유하며 wandb 서버가 파라미터를 중복 없이 분배.

### sweep 파라미터 변경

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

---

## Common Commands

```bash
# 환경 설정
conda create -n memgen python=3.10 && pip install -r requirements.txt

# 학습
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=1 \
    main.py --cfg-path configs/latent_memory/gsm8k_recursive_skip_proj.yaml

# 평가
python main.py --cfg-path configs/latent_memory/gsm8k_recursive_eval.yaml \
    --options run.mode evaluate model.load_weaver_path <checkpoint_path>
```
