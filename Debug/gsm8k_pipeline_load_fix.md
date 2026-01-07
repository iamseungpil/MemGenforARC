# GSM8K Pipeline Load Bug Fix

**Date**: 2025-01-07
**Files Modified**: `memgen/model/modeling_memgen.py`

---

## Overview

GSM8K pipeline에서 Phase 1 (훈련) 후 Phase 2 (평가)에서 체크포인트를 로드할 때 발생하는 버그 2개를 수정함.

---

## Bug 1: File Format Mismatch

### Problem

| Step | Code | Result |
|------|------|--------|
| **Save** (`runner.py:197`) | `safe_serialization=False` | `adapter_model.bin` 저장 |
| **Load** (`modeling_memgen.py:969`) | `adapter_model.safetensors` 찾음 | `FileNotFoundError` |

### Root Cause

`safe_serialization=False`는 **의도적**으로 사용됨.
- HuggingFace의 `.safetensors` 포맷은 shared tensors를 지원하지 않음
- MemGenModel은 reasoner/weaver/trigger가 base model을 공유 (shared tensors)
- 따라서 `.bin` 포맷 사용이 필수

### Fix

```python
# Before (modeling_memgen.py:968-976)
from safetensors.torch import load_file
adapter_path = Path(weaver_path) / "adapter_model.safetensors"
pretrained_weights = load_file(str(adapter_path))

# After
adapter_path = Path(weaver_path) / "adapter_model.bin"
pretrained_weights = torch.load(str(adapter_path), map_location='cpu')
```

---

## Bug 2: projections.pt Not Loaded

### Problem

| Step | Code | Result |
|------|------|--------|
| **Save** (`runner.py:199-205`) | `projections.pt` 저장 | 4개 항목 저장됨 |
| **Load** (`modeling_memgen.py`) | 로드 코드 없음 | Random initialization 사용 |

### Saved Contents in projections.pt

```python
torch.save({
    'reasoner_to_weaver': self.model.reasoner_to_weaver.state_dict(),
    'weaver_to_reasoner': self.model.weaver_to_reasoner.state_dict(),
    'prompt_query_latents': self.model.weaver.prompt_query_latents,
    'inference_query_latents': self.model.weaver.inference_query_latents,
}, proj_path)
```

### Fix

`_load_pretrained_weaver()` 함수 끝에 로드 코드 추가:

```python
# Load projections.pt (saved by runner.py)
proj_path = Path(weaver_path).parent / "projections.pt"
if not proj_path.exists():
    raise FileNotFoundError(f"projections.pt not found: {proj_path}")

proj_data = torch.load(str(proj_path), map_location='cpu')
self.reasoner_to_weaver.load_state_dict(proj_data['reasoner_to_weaver'])
self.weaver_to_reasoner.load_state_dict(proj_data['weaver_to_reasoner'])
self.weaver.prompt_query_latents.data = proj_data['prompt_query_latents'].to(self.weaver.prompt_query_latents.device)
self.weaver.inference_query_latents.data = proj_data['inference_query_latents'].to(self.weaver.inference_query_latents.device)
logging.info(f"Loaded projections and query_latents from {proj_path}")
```

---

## Directory Structure

```
/data/memgen/train/gsm8k/<model_name>/<timestamp>/weaver/
├── weaver_lora/
│   ├── adapter_model.bin      <- Bug 1: .bin으로 로드하도록 수정
│   └── adapter_config.json
└── projections.pt              <- Bug 2: 로드 코드 추가
```

**Note**: `load_model_path`는 `weaver_lora/` 폴더를 가리키므로, `projections.pt`는 `.parent`로 접근

---

## Usage

Phase 2 실행 시 수동으로 경로 지정:

```bash
# 02_eval_weaver.sh
LOAD_MODEL_PATH="/data/memgen/train/gsm8k/<model_name>/<timestamp>/weaver/weaver_lora"
```
