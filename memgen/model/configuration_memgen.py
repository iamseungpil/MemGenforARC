from transformers import PretrainedConfig
from typing import Optional


class MemGenConfig(PretrainedConfig):
    model_type = "memgen"

    def __init__(
        self,
        # latent configs
        prompt_latents_len: int = 8,
        inference_latents_len: int = 8,
        max_prompt_aug_num: int = 1,
        max_inference_aug_num: int = 5,
        # recursive memory mode (WeaverStyleCompressor)
        recursive_memory: bool = True,
        recursive_hidden_size: int = 4096,
        recursive_num_heads: int = 8,
        recursive_attn_rank: int = 64,
        recursive_mlp_rank: int = 128,
        recursive_max_cycles: int = 10,
        recursive_confidence_threshold: float = -1.0,
        recursive_top_k: int = 10,
        recursive_verbose_cycles: bool = False,
        recursive_skip_projection: bool = True,
        # Two-level cycle structure (H-cycle / L-cycle)
        recursive_two_level: bool = False,
        recursive_l_cycles: int = 6,
        recursive_max_h_cycles: int = 5,
        # Stepwise training (intermediate loss at each augmentation point)
        recursive_stepwise_training: bool = False,
        recursive_stepwise_loss_weight: float = 0.5,
        recursive_full_rank_mlp: bool = False,
        recursive_bidirectional: bool = False,
        recursive_context_update: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Latent
        self.prompt_latents_len = prompt_latents_len
        self.inference_latents_len = inference_latents_len
        self.max_prompt_aug_num = max_prompt_aug_num
        self.max_inference_aug_num = max_inference_aug_num

        # Recursive memory
        self.recursive_memory = recursive_memory
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
        self.recursive_context_update = recursive_context_update
