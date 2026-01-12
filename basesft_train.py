"""Base Model SFT Training (without MemGen)

TRL SFTTrainer를 사용하여 base model만 학습.
MemGen 없이 순수 SFT 효과 측정용 대조군.

Usage:
    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        --num_processes=2 \
        basesft_train.py \
        --model-name HuggingFaceTB/SmolLM3-3B \
        --dataset gsm8k \
        --epochs 2 \
        --batch-size 4 \
        --use-lora
"""

import argparse
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def get_dataset(dataset_name: str):
    """데이터셋 로드 및 전처리 (Weaver SFT와 동일한 형식 사용)"""
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")

        def format_example(example):
            """Weaver SFT와 동일한 전처리 (data/gsm8k/builder.py 참조)"""
            # 1. Instruction template (Weaver SFT와 동일)
            format_template = r"Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."
            prompt_template = "Question: {prompt}\n"

            # 2. Answer 전처리: rationale + \boxed{answer} 형식
            def _preprocess_answer(answer: str) -> str:
                raw_answer_list = answer.split("\n####")
                rationale = raw_answer_list[0]
                clean_answer = raw_answer_list[-1].strip()
                boxed_answer = "\\boxed{" + clean_answer + "}"
                new_string = rationale + boxed_answer
                return new_string.strip()

            question = example["question"].strip()
            answer = example["answer"].strip()

            # 3. Prompt + Completion 형식으로 조합
            processed_prompt = format_template + prompt_template.format(prompt=question)
            processed_completion = _preprocess_answer(answer)

            return {
                "text": processed_prompt + processed_completion
            }

        train_dataset = dataset["train"].map(format_example)
        eval_dataset = dataset["test"].map(format_example)
        return train_dataset, eval_dataset

    elif dataset_name == "kodcode":
        dataset = load_dataset("KodCode/KodCode-Light-RL-10K")

        def format_example(example):
            prompt = "Write an efficient and correct Python function to solve the following problem.\n"
            prompt += f"Question: {example['question'].strip()}\n"
            return {
                "text": prompt + example['solution'].strip()
            }

        all_data = dataset["train"].map(format_example)
        # 70/10/20 split (consistent with MemGen KodCode config)
        split1 = all_data.train_test_split(test_size=0.3, seed=42)
        split2 = split1["test"].train_test_split(test_size=0.67, seed=42)  # 0.3 * 0.67 ≈ 0.2

        train_dataset = split1["train"]
        eval_dataset = split2["train"]  # validation set
        return train_dataset, eval_dataset

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Base Model SFT Training (No MemGen)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name (e.g., HuggingFaceTB/SmolLM3-3B)")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "kodcode"],
                        help="Dataset to use for training")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--use-lora", action="store_true", default=False,
                        help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    args = parser.parse_args()

    # Output directory 설정
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.expanduser(
            f"~/data/basesft/train/{args.dataset}/{model_short}_{timestamp}"
        )

    # Expand ~ in path
    args.output_dir = os.path.expanduser(args.output_dir)

    print("=" * 60)
    print("Base Model SFT Training (No MemGen)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA: {args.use_lora}")
    print("=" * 60)

    # 1. 모델 로드 (learnable params는 기본 dtype 사용 - float32)
    # Note: sdpa 사용 (flash_attn이 설치되지 않은 환경 호환)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # 활성화만 bfloat16, learnable params는 float32
        attn_implementation="sdpa",  # flash_attention_2 대신 sdpa 사용
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. LoRA 적용 (Weaver SFT와 동일 설정)
    # Note: LoRA weights는 기본 float32로 초기화됨 (CLAUDE.md 제약사항 준수)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 3. 데이터 로드
    train_dataset, eval_dataset = get_dataset(args.dataset)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # 4. SFT 설정 (Weaver SFT와 동일 하이퍼파라미터)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,  # TRL 0.21.x uses max_length instead of max_seq_length
        dataset_text_field="text",
        report_to=["wandb"],
        run_name=f"base_sft_{args.dataset}_{args.model_name.split('/')[-1]}",
        remove_unused_columns=False,
    )

    # 5. 학습
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()

    print("=" * 60)
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
