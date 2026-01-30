from transformers import PretrainedConfig
from typing import Optional


class MemGenConfig(PretrainedConfig):
    model_type = "memgen"

    def __init__(
        self,
        # weaver configs
        weaver_lora_config: Optional[dict] = None,
        prompt_latents_len: int = 0,
        inference_latents_len: int = 0,
        # NOTE: max_memory_length is reserved for future cross-task memory or extended
        # refinement loops. Currently unused as arc-lang-public uses task-level memory
        # with fixed refinement_turns * prompt_latents_len (e.g., 7 * 8 = 56 tokens).
        max_memory_length: int = 128,
        # trigger configs
        trigger_active: bool = False,
        trigger_lora_config: Optional[dict] = None,
        max_prompt_aug_num: int = 1,
        max_inference_aug_num: int = 5,
        # projection-only mode (no query latents, no LoRA)
        projection_only: bool = False,
        # skip-lora mode (query latents + projections, no LoRA)
        skip_lora: bool = False,
        # latent processor mode (MLP after weaver output, replaces LoRA gradient effect)
        latent_processor: bool = False,
        latent_processor_depth: int = 2,
        # query-projection only mode (query latents + projections, NO weaver LLM)
        query_projection_only: bool = False,
        # skip-projection mode (query latents + LLM, NO projections)
        skip_projection: bool = False,
        # output-projection-only mode (skip input projection, keep output projection)
        output_projection_only: bool = False,
        # recursive memory mode (WeaverStyleCompressor)
        recursive_memory: bool = False,
        recursive_weaver_style: bool = True,  # WeaverStyle (causal self-attn on [context, z])
        recursive_hidden_size: int = 4096,
        recursive_num_heads: int = 8,
        recursive_attn_rank: int = 64,
        recursive_mlp_rank: int = 128,
        recursive_max_cycles: int = 10,
        recursive_confidence_threshold: float = 0.5,  # >0 enables LTPO-style early stopping (lower=confident)
        recursive_top_k: int = 10,
        recursive_verbose_cycles: bool = False,  # Log confidence at each cycle during training
        recursive_skip_projection: bool = False,  # Skip projection layers (train only recursive_compressor)
        # Two-level cycle structure (H-cycle / L-cycle)
        recursive_two_level: bool = False,  # Enable H-cycle/L-cycle structure
        recursive_l_cycles: int = 6,  # Inner loop iterations (L-cycle)
        recursive_max_h_cycles: int = 5,  # Max outer loop iterations (H-cycle)
        # Stepwise training (intermediate loss at each augmentation point)
        recursive_stepwise_training: bool = False,  # Enable intermediate loss at each aug point
        recursive_stepwise_loss_weight: float = 0.5,  # Weight for intermediate losses (final loss = 1.0)
        recursive_full_rank_mlp: bool = False,  # Replace LowRankSwiGLU with nn.Linear (W2R-style, ~16.8M)
        recursive_bidirectional: bool = False,  # Bidirectional attention (like TRM) instead of causal
        **kwargs
    ):
        super().__init__(**kwargs)

        # Weaver
        self.weaver_lora_config = weaver_lora_config
        self.prompt_latents_len = prompt_latents_len
        self.inference_latents_len = inference_latents_len
        self.max_memory_length = max_memory_length

        # Trigger
        self.trigger_active = trigger_active
        self.trigger_lora_config = trigger_lora_config
        self.max_prompt_aug_num = max_prompt_aug_num
        self.max_inference_aug_num = max_inference_aug_num

        # Mode flags
        self.projection_only = projection_only
        self.skip_lora = skip_lora
        self.latent_processor = latent_processor
        self.latent_processor_depth = latent_processor_depth
        self.query_projection_only = query_projection_only
        self.skip_projection = skip_projection
        self.output_projection_only = output_projection_only

        # Recursive memory
        self.recursive_memory = recursive_memory
        self.recursive_weaver_style = recursive_weaver_style
        self.recursive_hidden_size = recursive_hidden_size
        self.recursive_num_heads = recursive_num_heads
        self.recursive_attn_rank = recursive_attn_rank
        self.recursive_mlp_rank = recursive_mlp_rank
        self.recursive_max_cycles = recursive_max_cycles
        self.recursive_confidence_threshold = recursive_confidence_threshold
        self.recursive_top_k = recursive_top_k
        self.recursive_verbose_cycles = recursive_verbose_cycles
        self.recursive_skip_projection = recursive_skip_projection
        self.recursive_two_level = recursive_two_level
        self.recursive_l_cycles = recursive_l_cycles
        self.recursive_max_h_cycles = recursive_max_h_cycles
        self.recursive_stepwise_training = recursive_stepwise_training
        self.recursive_stepwise_loss_weight = recursive_stepwise_loss_weight
        self.recursive_full_rank_mlp = recursive_full_rank_mlp
        self.recursive_bidirectional = recursive_bidirectional
