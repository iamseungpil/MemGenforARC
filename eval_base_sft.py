"""Base SFT 평가 스크립트 (MemGen 없이 독립 실행)

Weaver eval과 동일한 설정 사용:
- Instruction: Weaver SFT와 동일
- Answer 비교: data/utils/math_utils.py의 compute_score() 사용
- Generation: greedy (do_sample=False, temperature=0.0)
- Max tokens: 1024

Usage:
    python eval_base_sft.py \
        --model-name HuggingFaceTB/SmolLM3-3B \
        --adapter-path ~/data/basesft/train/gsm8k/SmolLM3-3B_<timestamp> \
        --dataset gsm8k

    # Multi-GPU with accelerate (optional)
    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        --num_processes=2 \
        eval_base_sft.py \
        --model-name HuggingFaceTB/SmolLM3-3B \
        --adapter-path ~/data/basesft/train/gsm8k/SmolLM3-3B_<timestamp>
"""

import argparse
import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# MemGen의 math_utils에서 가져옴 (동일한 비교 로직)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.utils.math_utils import compute_score


def evaluate_gsm8k(model, tokenizer, dataset, max_new_tokens=1024):
    """GSM8K 평가 - Weaver eval과 동일한 설정"""
    correct = 0
    total = len(dataset)
    results = []

    # Weaver SFT와 동일한 instruction (data/gsm8k/builder.py 참조)
    format_template = r"Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."
    prompt_template = "Question: {prompt}\n"

    for example in tqdm(dataset, desc="Evaluating"):
        question = example["question"].strip()
        answer = example["answer"].strip()

        # Prompt 생성 (Weaver SFT와 동일)
        prompt = format_template + prompt_template.format(prompt=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generation (Weaver eval과 동일: greedy, temperature=0.0)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Completion만 추출 (prompt 부분 제외)
        completion = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Ground truth를 \boxed{} 형식으로 변환 (Weaver SFT와 동일)
        answer_parts = answer.split("\n####")
        final_answer = answer_parts[-1].strip()
        ground_truth = f"\\boxed{{{final_answer}}}"

        # MemGen과 동일한 비교 로직 사용
        score = compute_score(completion, ground_truth)
        correct += score

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "completion": completion,
            "score": score
        })

    accuracy = correct / total
    return accuracy, results


def evaluate_kodcode(model, tokenizer, dataset, max_new_tokens=1024):
    """KodCode 평가 - 코드 실행 기반"""
    # TODO: KodCode 평가 구현 (코드 실행 필요)
    raise NotImplementedError("KodCode evaluation requires code execution environment")


def main():
    parser = argparse.ArgumentParser(description="Base SFT Evaluation (No MemGen)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name (e.g., HuggingFaceTB/SmolLM3-3B)")
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path to PEFT adapter checkpoint")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "kodcode"],
                        help="Dataset to evaluate on")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (auto-generated if not specified)")
    args = parser.parse_args()

    # Expand paths
    args.adapter_path = os.path.expanduser(args.adapter_path)

    # Output directory 설정
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.expanduser(
            f"~/data/basesft/evaluate/{args.dataset}/{model_short}_{timestamp}"
        )
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Base SFT Evaluation (No MemGen)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # 1. Base model 로드
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # 2. PEFT adapter 로드
    print(f"Loading PEFT adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # 3. Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Dataset 로드
    print(f"\nLoading {args.dataset} test set...")
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
    elif args.dataset == "kodcode":
        # KodCode test set
        raw_dataset = load_dataset("KodCode/KodCode-Light-RL-10K")
        split1 = raw_dataset["train"].train_test_split(test_size=0.3, seed=42)
        split2 = split1["test"].train_test_split(test_size=0.67, seed=42)
        dataset = split2["test"]  # test set (20%)

    print(f"Test samples: {len(dataset)}")

    # 5. 평가 실행
    print("\nStarting evaluation...")
    if args.dataset == "gsm8k":
        accuracy, results = evaluate_gsm8k(model, tokenizer, dataset, args.max_new_tokens)
    elif args.dataset == "kodcode":
        accuracy, results = evaluate_kodcode(model, tokenizer, dataset, args.max_new_tokens)

    # 6. 결과 저장
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "dataset": args.dataset,
            "accuracy": accuracy,
            "total_samples": len(dataset),
            "correct_samples": int(accuracy * len(dataset)),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    # 7. 결과 출력
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: {accuracy:.2%} ({int(accuracy * len(dataset))}/{len(dataset)})")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    # Summary metrics 파일 (다른 평가와 일관성 유지)
    summary_file = os.path.join(args.output_dir, "summary_metrics.json")
    with open(summary_file, "w") as f:
        json.dump({
            "compute_reward": accuracy
        }, f, indent=2)
    print(f"Summary metrics saved to: {summary_file}")


if __name__ == "__main__":
    main()
