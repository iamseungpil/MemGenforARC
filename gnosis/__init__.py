"""
Gnosis: Self-Awareness Module for LLM Correctness Prediction

A lightweight LoRA-based module that predicts correctness probability
by reading hidden states from the LLM backbone.

Based on: "Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits"
(arXiv:2512.20578)

Usage:
    from gnosis.model import MemGenGnosis
    from gnosis.trainer import GnosisSFTTrainer, GnosisTrainingArguments

    # Initialize gnosis (requires PeftModel with gnosis adapter)
    gnosis = MemGenGnosis(peft_model, hidden_size=4096)

    # Training
    trainer = GnosisSFTTrainer(
        model=model,  # model.gnosis = gnosis
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Inference
    correctness_prob = gnosis(input_ids, attention_mask)

Data augmentation utilities are available:
    from gnosis.data import CompletionGenerator, Labeler, Rebalancer
"""

# Model
from gnosis.model import MemGenGnosis

# Trainers (with safetensors support)
from gnosis.trainer import (
    GnosisSFTTrainer,
    GnosisGRPOTrainer,
    GnosisTrainingArguments,
    GnosisDataset,
    prepare_gnosis_dataset,
    compute_gnosis_metrics,
)

# Data augmentation utilities
from gnosis.data import (
    CompletionGenerator,
    Labeler,
    Rebalancer,
    BaseEvaluator,
    MathEvaluator,
    TriviaEvaluator,
    MCQEvaluator,
    get_evaluator,
)

__version__ = "0.3.0"  # Standalone with safetensors
__all__ = [
    # Model
    "MemGenGnosis",
    # Trainers
    "GnosisSFTTrainer",
    "GnosisGRPOTrainer",
    "GnosisTrainingArguments",
    "GnosisDataset",
    "prepare_gnosis_dataset",
    "compute_gnosis_metrics",
    # Data augmentation
    "CompletionGenerator",
    "Labeler",
    "Rebalancer",
    "BaseEvaluator",
    "MathEvaluator",
    "TriviaEvaluator",
    "MCQEvaluator",
    "get_evaluator",
]
