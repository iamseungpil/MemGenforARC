"""
Gnosis Trainers with safetensors support.

Provides SFT and GRPO trainers for training the Gnosis self-awareness module.
Uses safetensors format for saving correctness_head weights.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


@dataclass
class GnosisTrainingArguments(TrainingArguments):
    """Training arguments for Gnosis trainer."""

    pos_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Weight for positive class in BCE loss (for imbalanced data)"}
    )


class GnosisDataset(Dataset):
    """
    Dataset for Gnosis training.

    Each sample contains:
        - input_ids: Tokenized input
        - attention_mask: Attention mask
        - correctness_label: 0 (incorrect) or 1 (correct)
    """

    def __init__(
        self,
        input_ids: List[torch.LongTensor],
        attention_mask: List[torch.LongTensor],
        labels: List[int],
    ):
        """
        Initialize GnosisDataset.

        Args:
            input_ids: List of tokenized inputs
            attention_mask: List of attention masks
            labels: List of correctness labels (0 or 1)
        """
        assert len(input_ids) == len(attention_mask) == len(labels)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "correctness_label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class GnosisSFTTrainer(Trainer):
    """
    SFT Trainer for Gnosis correctness prediction.

    Uses Binary Cross-Entropy loss with logits for numerical stability.
    Saves correctness_head weights using safetensors format.
    """

    def __init__(
        self,
        model,
        args: GnosisTrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        pos_weight: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize GnosisSFTTrainer.

        Args:
            model: MemGenModel with gnosis component
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            pos_weight: Weight for positive class (overrides args.pos_weight)
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.pos_weight = pos_weight or getattr(args, 'pos_weight', None)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute BCE loss for correctness prediction.

        Args:
            model: MemGenModel with gnosis component
            inputs: Dictionary containing input_ids, attention_mask, correctness_label
            return_outputs: If True, also return model outputs

        Returns:
            loss: Scalar loss value
            outputs: (optional) Model outputs
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["correctness_label"]

        # Generate position_ids
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        # Forward pass through gnosis
        loss, probs = model.gnosis.compute_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            pos_weight=self.pos_weight,
        )

        if return_outputs:
            return loss, {"probs": probs, "labels": labels}
        return loss

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ):
        """
        Save model with safetensors format for correctness_head.

        Saves:
        - LoRA adapter via PEFT's save_pretrained
        - correctness_head weights in safetensors format

        Args:
            output_dir: Directory to save to
            _internal_call: Internal HF flag
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Save base model (including LoRA adapters)
        super().save_model(output_dir, _internal_call)

        # Save correctness_head separately using safetensors
        if hasattr(self.model, 'gnosis') and self.model.gnosis is not None:
            head_state = {}
            for name, param in self.model.gnosis.correctness_head.state_dict().items():
                # Move to CPU and ensure contiguous
                head_state[f"correctness_head.{name}"] = param.cpu().contiguous()

            head_path = os.path.join(output_dir, "gnosis_head.safetensors")
            save_file(head_state, head_path)
            logger.info(f"Saved gnosis correctness_head to {head_path}")

    @staticmethod
    def load_gnosis_weights(model, checkpoint_dir: str):
        """
        Load gnosis weights from checkpoint.

        Loads:
        - LoRA adapter via PEFT's load_adapter
        - correctness_head weights from safetensors

        Args:
            model: MemGenModel with gnosis component
            checkpoint_dir: Directory containing checkpoint

        Returns:
            model: Model with loaded weights
        """
        if model.gnosis is None:
            logger.warning("Model has no gnosis component, skipping gnosis weight loading")
            return model

        # 1. Load LoRA adapter
        lora_dir = os.path.join(checkpoint_dir, "gnosis_lora")
        if os.path.exists(lora_dir):
            try:
                model.gnosis.model.load_adapter(lora_dir, adapter_name="gnosis")
                model.gnosis.model.set_adapter("gnosis")
                logger.info(f"Loaded gnosis LoRA adapter from {lora_dir}")
            except Exception as e:
                logger.warning(f"Failed to load gnosis LoRA adapter: {e}")

        # 2. Load correctness_head from safetensors
        head_path = os.path.join(checkpoint_dir, "gnosis_head.safetensors")
        if os.path.exists(head_path):
            state_dict = load_file(head_path)

            # Remove "correctness_head." prefix
            head_state = {}
            for key, value in state_dict.items():
                if key.startswith("correctness_head."):
                    new_key = key[len("correctness_head."):]
                    head_state[new_key] = value.to(model.gnosis.device)

            model.gnosis.correctness_head.load_state_dict(head_state)
            logger.info(f"Loaded gnosis correctness_head from {head_path}")
        else:
            # Fallback to .pt format
            pt_path = os.path.join(checkpoint_dir, "gnosis_head.pt")
            if os.path.exists(pt_path):
                head_state = torch.load(pt_path, map_location='cpu')
                model.gnosis.correctness_head.load_state_dict(head_state)
                logger.info(f"Loaded gnosis correctness_head from {pt_path} (legacy format)")

        return model


class GnosisGRPOTrainer:
    """
    GRPO Trainer for Gnosis with task rewards.

    NOTE: This is a placeholder. Full GRPO implementation requires:
    - Reward model integration
    - Policy gradient computation
    - Reference model for KL penalty
    """

    def __init__(
        self,
        model,
        args: GnosisTrainingArguments,
        reward_fn=None,
        **kwargs,
    ):
        """
        Initialize GnosisGRPOTrainer.

        Args:
            model: MemGenModel with gnosis component
            args: Training arguments
            reward_fn: Function to compute rewards
            **kwargs: Additional arguments
        """
        self.model = model
        self.args = args
        self.reward_fn = reward_fn
        raise NotImplementedError(
            "GnosisGRPOTrainer is a placeholder. "
            "Use GnosisSFTTrainer for now. "
            "GRPO implementation requires reward model integration."
        )

    def train(self):
        """Training loop - not implemented."""
        raise NotImplementedError("GnosisGRPOTrainer.train() is not implemented")


# Utility functions for data preparation

def prepare_gnosis_dataset(
    completions: List[str],
    labels: List[int],
    tokenizer,
    max_length: int = 512,
) -> GnosisDataset:
    """
    Prepare a GnosisDataset from completions and labels.

    Args:
        completions: List of model completions (prompt + response)
        labels: List of correctness labels (0 or 1)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        GnosisDataset ready for training
    """
    input_ids_list = []
    attention_mask_list = []

    for completion in completions:
        encoded = tokenizer(
            completion,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids_list.append(encoded["input_ids"].squeeze(0))
        attention_mask_list.append(encoded["attention_mask"].squeeze(0))

    return GnosisDataset(
        input_ids=input_ids_list,
        attention_mask=attention_mask_list,
        labels=labels,
    )


def compute_gnosis_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for Gnosis evaluation.

    Args:
        eval_pred: EvalPrediction from Trainer

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    probs = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert probabilities to predictions
    preds = (probs > 0.5).astype(int)

    # Compute metrics
    accuracy = (preds == labels).mean()

    # Precision, recall, F1 for positive class
    true_positives = ((preds == 1) & (labels == 1)).sum()
    predicted_positives = (preds == 1).sum()
    actual_positives = (labels == 1).sum()

    precision = true_positives / max(predicted_positives, 1)
    recall = true_positives / max(actual_positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
