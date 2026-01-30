# Recursive Memory Implementation

현재 구현된 Recursive Memory 시스템에 대한 정확한 문서입니다.

---

## 1. Recursive Memory 구조

### 1.1 WeaverStyleCompressor

파일: `memgen/model/recursive_memory.py`

```
WeaverStyleCompressor
├── prompt_query_latents: nn.Parameter (num_latents, hidden_size)
├── inference_query_latents: nn.Parameter (num_latents, hidden_size)
├── self_attn: LowRankCausalSelfAttention
│   ├── q_proj: LowRankLinear (hidden_size → hidden_size, rank=attn_rank)
│   ├── k_proj: LowRankLinear (hidden_size → hidden_size, rank=attn_rank)
│   ├── v_proj: LowRankLinear (hidden_size → hidden_size, rank=attn_rank)
│   └── o_proj: LowRankLinear (hidden_size → hidden_size, rank=attn_rank)
└── mlp: LowRankSwiGLU
    ├── gate: LowRankLinear (hidden_size → inter_size, rank=mlp_rank)
    ├── up: LowRankLinear (hidden_size → inter_size, rank=mlp_rank)
    └── down: LowRankLinear (inter_size → hidden_size, rank=mlp_rank)
```

### 1.2 LowRankLinear

```python
class LowRankLinear(nn.Module):
    # W = W_down @ W_up
    # 파라미터: (in_features × rank) + (rank × out_features)
    def __init__(self, in_features, out_features, rank=64):
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.up(self.down(x))
```

### 1.3 Forward Flow (Cycle 내부)

```python
def forward(self, context, attention_mask, is_prompt, reasoner=None, verbose=False):
    # query_latents 선택 (prompt/inference)
    query_latents = self.prompt_query_latents if is_prompt else self.inference_query_latents
    z = query_latents.unsqueeze(0).expand(B, -1, -1).clone()

    for cycle in range(self.max_cycles):
        # 1. Concatenate [context, z]
        combined = torch.cat([context, z], dim=1)

        # 2. Causal Self-Attention + RMSNorm
        combined = rms_norm(self.self_attn(combined))

        # 3. Extract query positions
        z = combined[:, -num_latents:]

        # 4. SwiGLU MLP + RMSNorm
        z = rms_norm(self.mlp(z))

        # 5. Confidence check (optional)
        if reasoner is not None and self.confidence_threshold > 0:
            conf = self._compute_confidence(reasoner, context, z, attention_mask)
            if conf <= self.confidence_threshold:
                return z, cycle + 1  # Early stop

    return z, self.max_cycles
```

---

## 2. 전체 시스템 구조 (현재 구현)

파일: `memgen/model/modeling_memgen.py`

### 2.1 컴포넌트 구성

```
MemGenModel
├── reasoner: AutoModelForCausalLM (frozen)
├── weaver: MemGenWeaver (unused in recursive_memory mode)
├── trigger: MemGenTrigger (optional)
├── reasoner_to_weaver: nn.Linear (reasoner_hidden → weaver_hidden)
├── weaver_to_reasoner: nn.Linear (weaver_hidden → reasoner_hidden)
└── recursive_compressor: WeaverStyleCompressor
```

### 2.2 Data Flow (forward / generate)

```
[Forward 시]
1. context = reasoner.get_input_embeddings()(input_ids)
2. projected_context = reasoner_to_weaver(context)           # Linear projection
3. memory, cycles = recursive_compressor(projected_context)  # Recursive cycles
4. memory = weaver_to_reasoner(memory)                       # Linear projection
5. full_embeds = concat([context, memory])
6. logits = reasoner(inputs_embeds=full_embeds)
7. loss = cross_entropy(logits, labels)  # memory 위치 제외
```

### 2.3 Projection Layers

| Layer | Input Size | Output Size | 비고 |
|-------|------------|-------------|------|
| reasoner_to_weaver | reasoner_hidden (4096) | weaver_hidden (4096) | 동일 크기 |
| weaver_to_reasoner | weaver_hidden (4096) | reasoner_hidden (4096) | 동일 크기 |

**참고**: 현재 구현에서는 reasoner와 weaver의 hidden_size가 동일(4096)하므로, projection은 identity mapping에 가까운 역할을 합니다.

---

## 3. Cycle 구조

### 3.1 현재 구현: L-Cycle Only

현재 구현에는 **L-cycle만 존재**합니다. TRM 논문의 H-cycle/L-cycle 구분이 없습니다.

```
현재 구현:
for cycle in range(max_cycles):   # max_cycles = 10 (default)
    combined = [context, z]
    z = RMSNorm(SelfAttn(combined))[-num_latents:]
    z = RMSNorm(SwiGLU(z))
    # confidence check → early stop (optional)
```

### 3.2 Confidence-based Early Stopping

`confidence_threshold > 0`일 때 활성화됩니다.

```python
def _compute_confidence(self, reasoner, context, memory, attention_mask):
    # 1. Memory를 context에 붙여서 reasoner에 입력
    full_embeds = torch.cat([context, memory], dim=1)

    # 2. Reasoner의 next token prediction
    with torch.no_grad():
        outputs = reasoner(inputs_embeds=full_embeds)
        logits = outputs.logits[:, -1]  # 마지막 위치

    # 3. Top-k probability의 negative log mean
    probs = F.softmax(logits, dim=-1)
    topk_probs = torch.topk(probs, k=self.top_k).values
    confidence = -torch.log(topk_probs + 1e-10).mean().item()

    return confidence  # 낮을수록 confident
```

**Early Stop 조건**: `confidence <= threshold`

---

## 4. 학습 방법

### 4.1 학습 가능한 파라미터

파일: `memgen/model/modeling_utils.py` (Line 72-82)

```python
if recursive_memory and name == "weaver":
    # 1. recursive_compressor 학습 (query_latents 포함)
    open_model_parameters(self.recursive_compressor)

    # 2. projections 학습
    open_model_parameters(self.weaver_to_reasoner)
    open_model_parameters(self.reasoner_to_weaver)

    # 3. weaver (LoRA, base model) frozen
    fix_model_parameters(component)
```

| 컴포넌트 | 상태 | 파라미터 수 (approx) |
|----------|------|---------------------|
| reasoner | Frozen | 0 |
| weaver (LoRA) | Frozen | 0 |
| recursive_compressor | **Trainable** | ~7.9M |
| reasoner_to_weaver | **Trainable** | 16.8M |
| weaver_to_reasoner | **Trainable** | 16.8M |
| **Total** | | ~41.5M |

### 4.2 학습 방식: SFT

파일: `memgen/runner.py`

```python
weaver_trainer = SFTTrainer(
    model=self.model,
    args=self.weaver_sft_training_args,
    train_dataset=self.weaver_train_dataset,
    processing_class=self.processing_class,
)
weaver_trainer.train()
```

- **Loss**: Cross-entropy (memory token 위치 제외)
- **Gradient Flow**: 모든 cycle을 통해 흐름 (truncated BPTT 없음)
- **Checkpoint**: `recursive_memory.pt` + `projections.pt`

### 4.3 Gradient Flow

```
Loss
  ↓
logits (reasoner output)
  ↓
full_embeds = [context, memory]
  ↓
memory = weaver_to_reasoner(z)        ← gradient flows
  ↓
z = recursive_compressor(projected)   ← gradient flows (all cycles)
  ↓
projected = reasoner_to_weaver(context) ← gradient flows
  ↓
context (from frozen reasoner)        ← gradient STOPS (frozen)
```

**주의**: 현재 구현은 TRM과 달리 **모든 cycle에 gradient가 흐릅니다**.

---

## 5. 학습 vs 추론 차이

### 5.1 Forward (학습)

파일: `modeling_memgen.py` Line 249-267

```python
if self.recursive_memory:
    projected_context = self.reasoner_to_weaver(current_inputs_embeds)

    latent_inputs_embeds, cycles = self.recursive_compressor(
        context=projected_context,
        attention_mask=current_attention_mask,
        is_prompt=is_prompt_end_aug,
        reasoner=self.reasoner if self.config.recursive_confidence_threshold > 0 else None,
        verbose=self.config.recursive_verbose_cycles,
    )

    latent_inputs_embeds = self.weaver_to_reasoner(latent_inputs_embeds)
```

### 5.2 Generate (추론)

파일: `modeling_memgen.py` Line 711-729

```python
if self.recursive_memory:
    projected_context = self.reasoner_to_weaver(candidate_inputs_embeds)

    latent_inputs_embeds, cycles = self.recursive_compressor(
        context=projected_context,
        attention_mask=candidate_attention_mask,
        is_prompt=(i == 0),  # i=0이면 prompt, 아니면 inference
        reasoner=self.reasoner if self.config.recursive_confidence_threshold > 0 else None,
        verbose=self.config.recursive_verbose_cycles,
    )

    latent_inputs_embeds = self.weaver_to_reasoner(latent_inputs_embeds)
```

### 5.3 차이점 요약

| 항목 | Forward (학습) | Generate (추론) |
|------|---------------|-----------------|
| is_prompt 판단 | labels 기반 (`labels[:, aug_point_idx] != -100`) | loop index 기반 (`i == 0`) |
| Gradient | ✅ 있음 | ❌ 없음 (no_grad) |
| Confidence check | threshold > 0이면 활성화 | 동일 |
| Memory injection | augmentation points에서 | 동일 |

---

## 6. Config 옵션

파일: `memgen/model/configuration_memgen.py`

```yaml
recursive_memory:
  enabled: true                    # Recursive memory 활성화
  weaver_style: true               # WeaverStyle 사용 (현재 유일한 옵션)
  hidden_size: 4096                # Compressor hidden size
  num_heads: 8                     # Self-attention heads
  attn_rank: 64                    # Attention low-rank dimension
  mlp_rank: 128                    # MLP low-rank dimension
  max_cycles: 10                   # 최대 cycle 수
  confidence_threshold: 0.5        # Early stop threshold (>0이면 활성화)
  top_k: 10                        # Confidence 계산용 top-k
  verbose_cycles: false            # Cycle별 로깅
```

---

## 7. 현재 구현의 특징 및 한계

### 7.1 특징

1. **Single-level cycle**: H-cycle/L-cycle 구분 없음, 단일 루프
2. **Full gradient flow**: 모든 cycle을 통해 gradient 흐름 (TRM의 truncated BPTT와 다름)
3. **Confidence-based early stop**: 선택적 활성화
4. **Low-rank layers**: 파라미터 효율성

### 7.2 Projection의 필요성 (논의 필요)

현재 구현에서 projection layers가 존재하는 이유:
- 코드 주석: "frozen reasoner outputs have no gradient, projection provides trainable path"
- 그러나 recursive_compressor 자체가 trainable이므로 gradient flow에는 문제 없음
- hidden_size가 동일(4096)하므로 projection의 실질적 역할은 제한적

**가능한 개선**: Projection 제거 시 ~33.6M 파라미터 절약 가능

### 7.3 TRM과의 비교

| 항목 | TRM | 현재 구현 |
|------|-----|----------|
| Cycle 구조 | H-cycle + L-cycle | 단일 cycle |
| Gradient flow | 마지막 H-cycle만 | 모든 cycle |
| puzzle_embedding | z_H + input_embeddings injection | query_latents 직접 학습 |
| Early stop | Q-function 학습 | Confidence threshold |

---

## 8. 파일 경로 요약

| 파일 | 내용 |
|------|------|
| `memgen/model/recursive_memory.py` | WeaverStyleCompressor 정의 |
| `memgen/model/modeling_memgen.py` | MemGenModel (forward, generate) |
| `memgen/model/modeling_utils.py` | open_component (학습 파라미터 설정) |
| `memgen/model/configuration_memgen.py` | Config 정의 |
| `memgen/runner.py` | 학습/평가 실행, checkpoint 저장 |

---

## 9. Checkpoint 구조

학습 후 저장되는 파일:

```
output_dir/
├── recursive_memory.pt
│   └── recursive_compressor: state_dict
│       ├── prompt_query_latents
│       ├── inference_query_latents
│       ├── self_attn.{q,k,v,o}_proj.{down,up}.weight
│       └── mlp.{gate,up,down}.{down,up}.weight
└── projections.pt
    ├── reasoner_to_weaver: state_dict
    └── weaver_to_reasoner: state_dict
```
