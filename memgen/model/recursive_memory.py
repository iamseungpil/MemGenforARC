"""
Recursive Memory Compressor for MemGen.

WeaverStyleCompressor: Causal self-attention on [context, query_latents] at every cycle,
allowing query_latents to attend to ALL context tokens throughout the compression process.
"""
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class LowRankLinear(nn.Module):
    """Low-rank factorization: W = W_down @ W_up."""

    def __init__(self, in_features: int, out_features: int, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        # Standalone low-rank (NOT LoRA): both projections need non-zero init,
        # otherwise output = up(down(x)) = 0 and gradients vanish.
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class LowRankCausalSelfAttention(nn.Module):
    """Self-Attention with low-rank Q, K, V, O projections. Supports causal or bidirectional."""

    def __init__(self, hidden_size: int = 4096, num_heads: int = 8, rank: int = 64, causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.causal = causal

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.q_proj = LowRankLinear(hidden_size, hidden_size, rank)
        self.k_proj = LowRankLinear(hidden_size, hidden_size, rank)
        self.v_proj = LowRankLinear(hidden_size, hidden_size, rank)
        self.o_proj = LowRankLinear(hidden_size, hidden_size, rank)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, H = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).contiguous().view(B, L, H)
        return self.o_proj(attn)


class LowRankSwiGLU(nn.Module):
    """SwiGLU MLP with low-rank factorization: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int = 4096, rank: int = 128, expansion: float = 2.67):
        super().__init__()
        inter_size = int(hidden_size * expansion)

        self.gate = LowRankLinear(hidden_size, inter_size, rank)
        self.up = LowRankLinear(hidden_size, inter_size, rank)
        self.down = LowRankLinear(inter_size, hidden_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class WeaverStyleCompressor(nn.Module):
    """
    Weaver-style recursive memory compressor.

    Concatenates [context, query_latents] and applies causal self-attention so
    query_latents can attend to ALL context tokens at EVERY cycle.

    Supports optional two-level cycle structure (H-cycle outer / L-cycle inner)
    with confidence-based early stopping.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 8,
        attn_rank: int = 64,
        mlp_rank: int = 128,
        max_cycles: int = 10,
        confidence_threshold: float = 0.5,
        top_k: int = 10,
        num_latents: int = 8,
        # Two-level cycle structure
        two_level: bool = False,
        l_cycles: int = 6,
        max_h_cycles: int = 5,
        # Full-rank MLP option
        full_rank_mlp: bool = False,
        # Bidirectional attention (like TRM, instead of causal)
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_cycles = max_cycles
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.num_latents = num_latents

        # Two-level cycle structure
        self.two_level = two_level
        self.l_cycles = l_cycles
        self.max_h_cycles = max_h_cycles

        self.prompt_query_latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        self.inference_query_latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)

        self.self_attn = LowRankCausalSelfAttention(hidden_size, num_heads, attn_rank, causal=not bidirectional)

        self.full_rank_mlp = full_rank_mlp
        if full_rank_mlp:
            self.mlp = nn.Linear(hidden_size, hidden_size)
        else:
            self.mlp = LowRankSwiGLU(hidden_size, mlp_rank)

        mlp_type = "FullRank" if full_rank_mlp else "LowRankSwiGLU"
        attn_type = "Bidirectional" if bidirectional else "Causal"
        if two_level:
            logger.info(
                "WeaverStyleCompressor: two_level, l=%d, h=%d, latents=%d, mlp=%s, attn=%s",
                l_cycles, max_h_cycles, num_latents, mlp_type, attn_type,
            )
        else:
            logger.info(
                "WeaverStyleCompressor: cycles=%d, latents=%d, mlp=%s, attn=%s",
                max_cycles, num_latents, mlp_type, attn_type,
            )

    def _compress_cycle(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Run one compression cycle: concat, self-attention, extract, MLP."""
        combined = torch.cat([context, z], dim=1)
        combined = rms_norm(combined + self.self_attn(combined))
        z = combined[:, -self.num_latents:]
        z = rms_norm(z + self.mlp(z))
        return z

    def _compute_confidence(
        self,
        reasoner: nn.Module,
        context: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Compute LTPO-style confidence score. Lower is better (more confident)."""
        B = context.size(0)
        num_latents = memory.size(1)
        device = context.device

        full_embeds = torch.cat([context, memory], dim=1)
        memory_mask = torch.ones(B, num_latents, dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([attention_mask, memory_mask], dim=1)

        with torch.no_grad():
            outputs = reasoner(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                return_dict=True,
            )
            logits = outputs.logits[:, -1]

        probs = F.softmax(logits, dim=-1)
        topk_probs = torch.topk(probs, k=self.top_k, dim=-1).values
        confidence = -torch.log(topk_probs + 1e-10).mean().item()

        return confidence

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        is_prompt: bool = True,
        reasoner: Optional[nn.Module] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Union[int, Tuple[int, int]]]:
        """
        Compress context into memory tokens.

        Returns:
            memory: (B, num_latents, hidden_size)
            cycles: int (single-level) or Tuple[int, int] (two-level: h_cycles, l_cycles)
        """
        B = context.size(0)
        query_latents = self.prompt_query_latents if is_prompt else self.inference_query_latents
        z = query_latents.unsqueeze(0).expand(B, -1, -1).clone()

        if self.two_level:
            return self._forward_two_level(z, context, attention_mask, reasoner, verbose)

        for cycle in range(self.max_cycles):
            z = self._compress_cycle(context, z)

            if reasoner is not None and self.confidence_threshold > 0:
                conf = self._compute_confidence(reasoner, context, z, attention_mask)

                if verbose:
                    logger.info("[WeaverStyle] Cycle %d/%d: confidence=%.4f", cycle + 1, self.max_cycles, conf)

                if conf <= self.confidence_threshold:
                    if verbose:
                        logger.info("[WeaverStyle] Early stopping at cycle %d", cycle + 1)
                    return z, cycle + 1

        if verbose and reasoner is not None:
            logger.info("[WeaverStyle] Reached max_cycles=%d", self.max_cycles)

        return z, self.max_cycles

    def _forward_two_level(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        reasoner: Optional[nn.Module],
        verbose: bool,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Two-level cycle: H-cycle (outer, with confidence check) and L-cycle (inner, fixed)."""
        for h in range(self.max_h_cycles):
            for _ in range(self.l_cycles):
                z = self._compress_cycle(context, z)

            if reasoner is not None and self.confidence_threshold > 0:
                conf = self._compute_confidence(reasoner, context, z, attention_mask)
                total_ops = (h + 1) * self.l_cycles

                if verbose:
                    logger.info(
                        "[WeaverStyle TwoLevel] H-cycle %d/%d (total ops: %d): confidence=%.4f",
                        h + 1, self.max_h_cycles, total_ops, conf,
                    )

                if conf <= self.confidence_threshold:
                    if verbose:
                        logger.info(
                            "[WeaverStyle TwoLevel] Early stopping at H-cycle %d (total ops: %d)",
                            h + 1, total_ops,
                        )
                    return z, (h + 1, self.l_cycles)

        if verbose and reasoner is not None:
            logger.info(
                "[WeaverStyle TwoLevel] Reached max_h_cycles=%d (total ops: %d)",
                self.max_h_cycles, self.max_h_cycles * self.l_cycles,
            )

        return z, (self.max_h_cycles, self.l_cycles)
