"""
MemGenGnosis: Self-awareness module for MemGen framework.

A lightweight LoRA-based module that predicts correctness probability
by reading hidden states from the LLM backbone.

Based on: "Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits"
(arXiv:2512.20578)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel


class MemGenGnosis(nn.Module):
    """
    Gnosis self-awareness module (LoRA-based, like Weaver/Trigger).

    Uses a LoRA adapter to extract features and a lightweight head
    to predict correctness probability.

    Attributes:
        adapter_name: Name of the LoRA adapter ("gnosis")
        model: PeftModel with gnosis adapter
        correctness_head: MLP that outputs logits (not probabilities)
    """

    adapter_name = "gnosis"

    def __init__(
        self,
        model: PeftModel,
        hidden_size: int,
        head_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize MemGenGnosis.

        Args:
            model: PeftModel with gnosis adapter attached
            hidden_size: Hidden dimension of the base model
            head_hidden_dim: Hidden dimension of correctness head (default: hidden_size // 4)
            dropout: Dropout rate for correctness head
        """
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size

        if head_hidden_dim is None:
            head_hidden_dim = hidden_size // 4

        # Correctness head outputs LOGITS (not probabilities) for numerical stability
        # Use binary_cross_entropy_with_logits for loss computation
        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
            # NO Sigmoid here - output raw logits
        )

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.model.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_probs: bool = False,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Forward pass to predict correctness.

        Args:
            input_ids: (B, S) input token ids
            attention_mask: (B, S) attention mask
            position_ids: (B, S) position ids (optional)
            return_hidden_states: If True, also return hidden states
            return_probs: If True, return probabilities instead of logits

        Returns:
            logits: (B, 1) correctness logits (or probabilities if return_probs=True)
            hidden_states: (B, S, D) hidden states (only if return_hidden_states=True)
        """
        # Enable gnosis adapter
        self.model.set_adapter(self.adapter_name)

        # Forward pass through LLM
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (B, S, D)

        # Use last non-padding token's hidden state
        # Find last non-padding position for each batch
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths.long()]  # (B, D)

        # Predict correctness logits
        logits = self.correctness_head(last_hidden)  # (B, 1)

        # Disable adapter
        self.model.disable_adapter()

        # Convert to probabilities if requested
        output = torch.sigmoid(logits) if return_probs else logits

        if return_hidden_states:
            return output, hidden_states
        return output

    def predict_at_positions(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        target_positions: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_probs: bool = True,
    ) -> torch.FloatTensor:
        """
        Predict correctness at specific positions.

        Args:
            input_ids: (B, S) input token ids
            attention_mask: (B, S) attention mask
            target_positions: (B,) positions to predict at (default: last non-padding)
            position_ids: (B, S) position ids (optional)
            return_probs: If True, return probabilities instead of logits

        Returns:
            correctness: (B, 1) correctness logits or probabilities
        """
        # Enable gnosis adapter
        self.model.set_adapter(self.adapter_name)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]  # (B, S, D)
        B, S, D = hidden_states.shape

        # Determine target positions
        if target_positions is None:
            # Use last non-padding position
            target_positions = (attention_mask.sum(dim=1) - 1).long()  # (B,)

        # Bounds check
        target_positions = target_positions.clamp(0, S - 1)

        # Extract hidden states at target positions
        batch_indices = torch.arange(B, device=hidden_states.device)
        target_hidden = hidden_states[batch_indices, target_positions]  # (B, D)

        # Predict correctness
        logits = self.correctness_head(target_hidden)  # (B, 1)

        # Disable adapter
        self.model.disable_adapter()

        return torch.sigmoid(logits) if return_probs else logits

    def compute_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute BCE loss for correctness prediction.

        Args:
            input_ids: (B, S) input token ids
            attention_mask: (B, S) attention mask
            labels: (B,) or (B, 1) correctness labels (0 or 1)
            position_ids: (B, S) position ids (optional)
            pos_weight: Weight for positive class (for imbalanced data)

        Returns:
            loss: Scalar BCE loss
            probs: (B, 1) predicted probabilities
        """
        # Get logits (not probabilities)
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_probs=False,
        )

        # Ensure labels shape matches logits
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        # Compute BCE with logits (numerically stable)
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight_tensor
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # Return probabilities for metrics
        probs = torch.sigmoid(logits)

        return loss, probs
