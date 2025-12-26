"""
Integrate SFT-trained LoRA into MemGen Weaver.

This script loads the SFT-trained LoRA adapter and properly integrates it
into MemGen's Weaver component for subsequent GRPO training.

Key insight: SFT saves LoRA with keys like:
  base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight
But MemGen expects weaver adapter keys like:
  base_model.model.model.layers.0.self_attn.k_proj.lora_A.weaver.weight

Solution: Load SFT weights with key renaming into MemGen's weaver adapter.
"""

import os
import sys
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memgen.model.modeling_memgen import MemGenModel


def load_and_rename_sft_weights(sft_checkpoint_path: str, adapter_name: str = "weaver") -> dict:
    """
    Load SFT-trained LoRA weights and rename keys for MemGen adapter format.

    SFT format:    .lora_A.weight
    MemGen format: .lora_A.<adapter_name>.weight
    """
    adapter_path = Path(sft_checkpoint_path) / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    weights = load_file(str(adapter_path))
    print(f"Loaded {len(weights)} weight tensors from SFT checkpoint")

    # Rename keys to include adapter name
    renamed = {}
    for key, value in weights.items():
        if ".lora_A.weight" in key:
            new_key = key.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight")
        elif ".lora_B.weight" in key:
            new_key = key.replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")
        else:
            new_key = key
        renamed[new_key] = value

    print(f"Renamed keys sample: {list(renamed.keys())[:3]}")
    return renamed


def integrate_sft_to_memgen(
    sft_checkpoint_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-14B"
):
    """
    Integrate SFT-trained LoRA into MemGen and save the complete model.

    Args:
        sft_checkpoint_path: Path to SFT-trained LoRA checkpoint
        output_path: Path to save integrated MemGen model
        model_name: Base model name
    """
    print(f"=" * 60)
    print(f"Integrating SFT LoRA into MemGen Weaver")
    print(f"=" * 60)
    print(f"SFT checkpoint: {sft_checkpoint_path}")
    print(f"Base model: {model_name}")
    print(f"Output path: {output_path}")
    print()

    # Configuration matching arc_twostage.yaml
    config_dict = {
        "model_name": model_name,
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 3,
        "weaver": {
            "model_name": model_name,
            "prompt_latents_len": 8,
            "inference_latents_len": 8,
            "lora_config": {
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        },
        "trigger": {
            "model_name": model_name,
            "active": False,
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        }
    }

    # Create MemGen model
    print("Step 1: Creating MemGen model...")
    memgen_model = MemGenModel.from_config(config_dict)

    trainable_params = sum(p.numel() for p in memgen_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in memgen_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Load and rename SFT LoRA weights
    print("Step 2: Loading SFT LoRA weights...")
    sft_weights = load_and_rename_sft_weights(sft_checkpoint_path, adapter_name="weaver")
    print()

    # Get current weaver state dict
    print("Step 3: Mapping weights to MemGen weaver...")
    weaver_state = memgen_model.weaver.model.state_dict()

    # Count LoRA keys
    weaver_lora_keys = [k for k in weaver_state.keys() if "lora_A" in k or "lora_B" in k]
    print(f"  MemGen weaver has {len(weaver_lora_keys)} LoRA weight tensors")
    print(f"  SFT checkpoint has {len(sft_weights)} weight tensors")

    # Map SFT weights to weaver state dict
    loaded_count = 0
    missing_keys = []

    for sft_key, sft_value in sft_weights.items():
        if sft_key in weaver_state:
            # Verify shape matches
            if weaver_state[sft_key].shape == sft_value.shape:
                weaver_state[sft_key] = sft_value.to(weaver_state[sft_key].dtype)
                loaded_count += 1
            else:
                print(f"  Shape mismatch for {sft_key}: expected {weaver_state[sft_key].shape}, got {sft_value.shape}")
        else:
            missing_keys.append(sft_key)

    print(f"  Successfully mapped {loaded_count} LoRA weight tensors")
    if missing_keys:
        print(f"  Warning: {len(missing_keys)} keys not found in weaver state dict")
        print(f"  First 3 missing keys: {missing_keys[:3]}")
    print()

    # Load the updated state dict into weaver
    print("Step 4: Loading weights into MemGen weaver...")
    memgen_model.weaver.model.load_state_dict(weaver_state, strict=True)
    print("  Successfully loaded SFT weights into MemGen weaver!")
    print()

    # Save just the weaver adapter in PEFT format
    print("Step 5: Saving integrated weaver adapter...")
    os.makedirs(output_path, exist_ok=True)

    # Save the renamed weights as safetensors
    output_weights_path = Path(output_path) / "adapter_model.safetensors"
    save_file(sft_weights, str(output_weights_path))

    # Create adapter_config.json for weaver adapter
    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,  # Keep False for training
        "init_lora_weights": True,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 32,
        "revision": None,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
        # Custom MemGen fields
        "adapter_name": "weaver"
    }

    import json
    with open(Path(output_path) / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Copy tokenizer files from SFT checkpoint
    import shutil
    sft_path = Path(sft_checkpoint_path)
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "special_tokens_map.json", "added_tokens.json"]:
        src = sft_path / fname
        if src.exists():
            shutil.copy(src, Path(output_path) / fname)

    print(f"  Adapter saved to: {output_path}")
    print()
    print("=" * 60)
    print("Integration complete!")
    print("=" * 60)
    print(f"\nThe weaver adapter has been saved with renamed keys.")
    print(f"\nNext steps for GRPO training:")
    print(f"1. The adapter can be loaded directly into MemGen's weaver")
    print(f"2. Run GRPO: python main.py --cfg-path configs/arc_twostage.yaml")
    print(f"   (Add --options model.load_weaver_path {output_path})")

    return memgen_model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Integrate SFT LoRA into MemGen")
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default="./outputs/arc_instruction_sft_direct",
        help="Path to SFT-trained LoRA checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/arc_memgen_weaver_sft",
        help="Path to save integrated MemGen model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Base model name"
    )

    args = parser.parse_args()

    integrate_sft_to_memgen(
        sft_checkpoint_path=args.sft_checkpoint,
        output_path=args.output,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
