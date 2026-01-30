from peft import PeftModel
import torch
import torch.nn as nn
from typing import Tuple


class LatentProcessor(nn.Module):
    """
    Trainable MLP to process Weaver latent outputs.
    Replaces LoRA's gradient modulation effect for Skip-LoRA training.

    Architecture: x + MLP(x) with residual connection for stable training.
    """
    def __init__(self, hidden_size: int, depth: int = 2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),  # SwiGLU-like activation
            ])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection for stable training
        return x + self.mlp(x)


class MemGenWeaver(nn.Module):

    adapter_name = "weaver"

    def __init__(
        self,
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
    ):
        super().__init__()

        self.model = model
        self.hidden_size = model.base_model.config.hidden_size

        # prompt augmentation
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, self.hidden_size),
            requires_grad=True
        )

        # inference augmentation
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, self.hidden_size),
            requires_grad=True
        )
    
    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _augment(
        self,
        latents: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.set_adapter(self.adapter_name)

        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)

        # Append query latents to inputs
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total)
        latents_mask = torch.ones(
            latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)

        # get position ids for latents
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1)

        # the processor only outputs the hidden states
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        self.model.disable_adapter()

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.prompt_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def augment_inference(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.inference_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def _augment_skip_lora(
        self,
        latents: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Skip-LoRA mode: query_latents used, LoRA adapter disabled.
        Unlike _augment(), this disables adapter BEFORE forward pass.
        """
        # Disable LoRA adapter BEFORE forward (key difference from _augment)
        self.model.disable_adapter()

        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)

        # Append query latents to inputs
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total)
        latents_mask = torch.ones(
            latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)

        # get position ids for latents
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1)

        # the processor only outputs the hidden states
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt_skip_lora(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Skip-LoRA mode for prompt augmentation."""
        return self._augment_skip_lora(
            latents=self.prompt_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def augment_inference_skip_lora(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Skip-LoRA mode for inference augmentation."""
        return self._augment_skip_lora(
            latents=self.inference_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def augment_prompt_projection_only(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        reasoner_to_weaver: nn.Linear,
        weaver_to_reasoner: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projection-only latent memory generation (no query latents, no LoRA).
        Takes last N tokens, applies projection layers, and returns memory tokens.
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        n_latents = self.prompt_query_latents.size(0)  # Use same length as query latents

        # 1. Select last N tokens from input
        last_n_embeds = inputs_embeds[:, -n_latents:, :]  # (B, N, H)

        # 2. Apply projection layers (skip LoRA)
        weaver_space = reasoner_to_weaver(last_n_embeds)  # (B, N, H)
        memory_tokens = weaver_to_reasoner(weaver_space)  # (B, N, H)

        # 3. Create attention mask for memory tokens
        memory_mask = torch.ones(
            batch_size, n_latents,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )

        # 4. Create position ids for memory tokens
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(n_latents, device=position_ids.device)
        memory_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1

        return memory_tokens, memory_mask, memory_position_ids

    def augment_prompt_query_projection(
        self,
        reasoner_to_weaver: nn.Linear,
        weaver_to_reasoner: nn.Linear,
        batch_size: int,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Query Latents + Projection only (NO weaver LLM forward).
        Uses learned prompt_query_latents directly through projections.
        """
        latents = self.prompt_query_latents.unsqueeze(0).expand(batch_size, -1, -1)
        n_latents = latents.size(1)

        # Apply projections only (skip LLM entirely)
        weaver_space = reasoner_to_weaver(latents)
        memory_tokens = weaver_to_reasoner(weaver_space)

        # Create attention mask
        memory_mask = torch.ones(
            batch_size, n_latents,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )

        # Create position ids
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(n_latents, device=position_ids.device)
        memory_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1

        return memory_tokens, memory_mask, memory_position_ids

    def augment_inference_query_projection(
        self,
        reasoner_to_weaver: nn.Linear,
        weaver_to_reasoner: nn.Linear,
        batch_size: int,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Query Latents + Projection only for inference (NO weaver LLM forward).
        Uses learned inference_query_latents directly through projections.
        """
        latents = self.inference_query_latents.unsqueeze(0).expand(batch_size, -1, -1)
        n_latents = latents.size(1)

        # Apply projections only (skip LLM entirely)
        weaver_space = reasoner_to_weaver(latents)
        memory_tokens = weaver_to_reasoner(weaver_space)

        # Create attention mask
        memory_mask = torch.ones(
            batch_size, n_latents,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )

        # Create position ids
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(n_latents, device=position_ids.device)
        memory_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1

        return memory_tokens, memory_mask, memory_position_ids
