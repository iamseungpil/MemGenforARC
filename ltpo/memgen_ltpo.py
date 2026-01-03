"""
MemGen-LTPO Integration Module

This module bridges MemGen's weaver-generated latent tokens with LTPO's
test-time optimization. It optimizes the latent hidden states using
confidence-based reward signals.

Flow:
    MemGen Weaver → latents_hidden_states (initial)
                         ↓
                  LTPO Optimizer
                         ↓
             optimized_latents_hidden_states
                         ↓
                Reasoner generation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MemGenLTPOOptimizer(nn.Module):
    """
    Optimizes MemGen weaver's latent hidden states using LTPO-style
    test-time optimization.

    The optimizer uses confidence-based reward signals to iteratively
    refine the latent representations, improving reasoning quality
    without additional training.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.03,
        sigma: float = 0.1,
        sigma_decay: float = 0.99,
        max_steps: int = 10,
        reward_threshold: float = -1.0,
        top_k: int = 10,
        use_auto_grad: bool = True,
    ):
        """
        Args:
            model: The base language model for confidence computation
            lr: Learning rate for optimization
            sigma: Initial noise standard deviation for exploration
            sigma_decay: Decay factor for sigma per step
            max_steps: Maximum optimization steps
            reward_threshold: Early stopping threshold (if > 0)
            top_k: Number of top tokens for confidence calculation
            use_auto_grad: If True, use PyTorch autograd; else use REINFORCE
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold
        self.top_k = top_k
        self.use_auto_grad = use_auto_grad

    def compute_confidence(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_start_idx: int,
        latent_end_idx: int,
        optimized_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence score based on top-k token probabilities.

        Args:
            inputs_embeds: Full input embeddings (batch, seq_len, hidden_size)
            attention_mask: Attention mask (batch, seq_len)
            latent_start_idx: Start index of latent tokens in sequence
            latent_end_idx: End index of latent tokens in sequence
            optimized_latents: Current latent hidden states to evaluate

        Returns:
            confidence: Scalar confidence score (negative log probability)
        """
        # Replace latent positions with optimized values
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, latent_start_idx:latent_end_idx] = optimized_latents

        # Forward pass to get logits
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        # Compute confidence over latent positions
        # Note: latent_end_idx is exclusive (Python slicing convention)
        confidence = 0.0
        for idx in range(latent_start_idx, latent_end_idx):
            if idx < probs.shape[0]:
                topk = torch.topk(probs[idx], k=self.top_k, largest=True)[0]
                confidence -= torch.sum(torch.log(topk + 1e-10)) / self.top_k

        num_tokens = latent_end_idx - latent_start_idx
        return confidence / max(num_tokens, 1)  # Prevent division by zero

    def optimize(
        self,
        latents_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_start_idx: int,
        latent_end_idx: int,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Optimize latent hidden states using LTPO-style optimization.

        Args:
            latents_hidden_states: Initial latent states from MemGen weaver
                                   Shape: (batch, latent_len, hidden_size)
            inputs_embeds: Full input embeddings including latents
            attention_mask: Attention mask for the full sequence
            latent_start_idx: Start index of latent tokens
            latent_end_idx: End index of latent tokens
            verbose: If True, print optimization progress

        Returns:
            optimized_latents: Optimized latent hidden states
            best_reward: Best reward achieved during optimization
            best_step: Step at which best reward was achieved
        """
        self.model.eval()
        device = latents_hidden_states.device
        sigma = self.sigma

        # Initialize optimizable latents
        if self.use_auto_grad:
            thought_hidden_states = nn.Parameter(
                latents_hidden_states.clone().detach().requires_grad_(True)
            )
            optimizer = torch.optim.Adam([thought_hidden_states], lr=self.lr, maximize=True)
        else:
            thought_hidden_states = latents_hidden_states.clone()

        best_reward = float('-inf')
        best_step = 0
        best_latents = thought_hidden_states.clone()

        for step in range(self.max_steps):
            if self.use_auto_grad:
                optimizer.zero_grad()

            # Add exploration noise
            epsilon = torch.normal(
                mean=0.0,
                std=sigma,
                size=thought_hidden_states.shape
            ).to(device)
            candidate_latents = thought_hidden_states + epsilon

            # Compute reward (confidence)
            if self.use_auto_grad:
                reward = self.compute_confidence(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    latent_start_idx=latent_start_idx,
                    latent_end_idx=latent_end_idx,
                    optimized_latents=candidate_latents,
                )
                reward.requires_grad_(True)
                reward.backward(retain_graph=True)
                optimizer.step()
            else:
                with torch.no_grad():
                    reward = self.compute_confidence(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        latent_start_idx=latent_start_idx,
                        latent_end_idx=latent_end_idx,
                        optimized_latents=candidate_latents,
                    )
                    # REINFORCE-style gradient estimate
                    grad_ascent = self.lr * reward * epsilon / (sigma ** 2)
                    thought_hidden_states = thought_hidden_states + grad_ascent

            # Decay sigma
            sigma *= self.sigma_decay

            if verbose:
                print(f"[LTPO] Step {step}: reward = {reward.item():.4f}")

            # Track best
            reward_val = reward.item() if isinstance(reward, torch.Tensor) else reward
            if reward_val > best_reward:
                best_reward = reward_val
                best_step = step
                best_latents = thought_hidden_states.clone().detach()

            # Early stopping
            if self.reward_threshold > 0 and reward_val >= self.reward_threshold:
                if verbose:
                    print(f"[LTPO] Early stopping at step {step} (reward >= threshold)")
                break

            # Cleanup
            del epsilon, candidate_latents
            torch.cuda.empty_cache()

        return best_latents, best_reward, best_step

    def optimize_weaver_output(
        self,
        weaver_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        full_inputs_embeds: torch.Tensor,
        full_attention_mask: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
        """
        Convenience method to optimize MemGen weaver output directly.

        Args:
            weaver_output: Tuple of (latents_hidden_states, latents_mask, latents_position_ids)
                           from weaver.augment_prompt() or augment_inference()
            full_inputs_embeds: Full sequence embeddings after weaver augmentation
            full_attention_mask: Full attention mask after weaver augmentation
            verbose: If True, print optimization progress

        Returns:
            optimized_latents: Optimized latent hidden states
            latents_mask: Original mask (unchanged)
            latents_position_ids: Original position ids (unchanged)
            best_reward: Best reward achieved
            best_step: Step at which best reward was achieved
        """
        latents_hidden_states, latents_mask, latents_position_ids = weaver_output

        # Determine latent positions in the full sequence
        latent_len = latents_hidden_states.shape[1]
        seq_len = full_inputs_embeds.shape[1]
        latent_start_idx = seq_len - latent_len
        latent_end_idx = seq_len

        optimized_latents, best_reward, best_step = self.optimize(
            latents_hidden_states=latents_hidden_states,
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            latent_start_idx=latent_start_idx,
            latent_end_idx=latent_end_idx,
            verbose=verbose,
        )

        return optimized_latents, latents_mask, latents_position_ids, best_reward, best_step
