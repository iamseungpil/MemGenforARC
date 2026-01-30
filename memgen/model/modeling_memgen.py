import logging
from typing import Union, Optional, TYPE_CHECKING
from contextlib import contextmanager

# Forward-compatible import for LTPO optimizer (avoid circular imports)
if TYPE_CHECKING:
    from ltpo import MemGenLTPOOptimizer

import os
import random
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    DynamicCache,
    LogitsProcessor,
    LogitsProcessorList
)
from transformers.modeling_utils import PreTrainedModel


class NanInfLogitsProcessor(LogitsProcessor):
    """Handle nan/inf values in logits to prevent CUDA assertion errors."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Replace nan with -inf (will be masked out by softmax)
        # Replace inf with large finite values
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e4, neginf=-1e9)
        return scores

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_utils import (
    MemGenOutputWithPast,
    MemGenLoraSwitchMixin,
    MemGenGenerationMixin,
)
from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
)

class MemGenModel(PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin):
    config_class = MemGenConfig
    supports_gradient_checkpointing = True

    def __init__(
        self, 
        config: MemGenConfig, 
        base_tokenizer,
        base_model: PreTrainedModel, 
        weaver_model: Optional[PreTrainedModel] = None,
        trigger_model: Optional[PreTrainedModel] = None,
    ):   
        super().__init__(config)
        
        self.config = config
        
        if weaver_model is None:
            weaver_model = base_model
        if trigger_model is None:
            trigger_model = base_model

        fix_model_parameters(base_model)  
        fix_model_parameters(weaver_model)
        fix_model_parameters(trigger_model)

        weaver_model, trigger_model = self._insert_lora_adapters(
            weaver_model, config.weaver_lora_config, trigger_model, config.trigger_lora_config,
        )
        self.weaver = MemGenWeaver(
            weaver_model, config.prompt_latents_len, config.inference_latents_len,
        )
        self.trigger = MemGenTrigger(
            trigger_model, config.trigger_active, 
        )
        self.reasoner = base_model
        self.tokenizer = base_tokenizer
        
        # Projection layers: map embeddings between reasoner and weaver spaces
        reasoner_hidden_size = self.config.hidden_size
        weaver_hidden_size = weaver_model.base_model.config.hidden_size
        self.reasoner_to_weaver = nn.Linear(reasoner_hidden_size, weaver_hidden_size)
        self.weaver_to_reasoner = nn.Linear(weaver_hidden_size, reasoner_hidden_size)

        self.delimiters: list[str] = [",", ".", "\n"]

        # Mode flags (see CLAUDE.md for mode descriptions)
        self.projection_only = config.projection_only
        self.skip_lora = config.skip_lora
        self.query_projection_only = config.query_projection_only
        self.skip_projection = config.skip_projection
        self.output_projection_only = config.output_projection_only

        self.latent_processor_enabled = config.latent_processor
        if self.latent_processor_enabled:
            from memgen.model.weaver import LatentProcessor
            self.latent_processor = LatentProcessor(
                hidden_size=weaver_hidden_size,
                depth=config.latent_processor_depth,
            )

        self.recursive_memory = config.recursive_memory
        if self.recursive_memory:
            from memgen.model.recursive_memory import WeaverStyleCompressor
            # Use prompt_latents_len for num_latents (same as weaver)
            num_latents = config.prompt_latents_len
            self.recursive_compressor = WeaverStyleCompressor(
                hidden_size=config.recursive_hidden_size,
                num_heads=config.recursive_num_heads,
                attn_rank=config.recursive_attn_rank,
                mlp_rank=config.recursive_mlp_rank,
                max_cycles=config.recursive_max_cycles,
                confidence_threshold=config.recursive_confidence_threshold,
                top_k=config.recursive_top_k,
                num_latents=num_latents,
                two_level=config.recursive_two_level,
                l_cycles=config.recursive_l_cycles,
                max_h_cycles=config.recursive_max_h_cycles,
                full_rank_mlp=config.recursive_full_rank_mlp,
                bidirectional=config.recursive_bidirectional,
            )

        # postprocess
        self._postprocess_models()

    def _postprocess_models(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

        # NOTE: _is_conversation(), _postprocess_assistant_labels() depend on <|im_start|> tokens
        self.tokenizer.chat_template = CONVERSATION_TEMPLATE

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """Required for DeepSpeed ZeRO compatibility with shared parameters."""
        if enable:
            self.reasoner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.reasoner.gradient_checkpointing_disable()

    @property
    def device(self):
        return self.reasoner.device

    @contextmanager
    def disable_adapter(self):
        """Disable LoRA adapters for reference model computation (GRPO)."""
        with self.weaver.model.disable_adapter():
            yield

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> tuple:
        assert input_ids.shape == attention_mask.shape == labels.shape

        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.config.max_inference_aug_num
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size

        # Pure SFT mode: no augmentation
        if self.config.max_prompt_aug_num == 0:
            # Simple forward without any augmentation
            position_ids = self._generate_position_ids(attention_mask)
            reasoner_outputs = reasoner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            return reasoner_outputs.logits, None

        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)

        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        stepwise_enabled = self.recursive_memory and getattr(self.config, 'recursive_stepwise_training', False)
        intermediate_losses = [] if stepwise_enabled else None

        for aug_loop_idx, aug_point_idx in enumerate(augmentation_indices):
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()

            if self.recursive_memory:
                if self.config.recursive_skip_projection:
                    context_for_compressor = current_inputs_embeds
                else:
                    context_for_compressor = self.reasoner_to_weaver(current_inputs_embeds)

                latent_inputs_embeds, cycles = self.recursive_compressor(
                    context=context_for_compressor,
                    attention_mask=current_attention_mask,
                    is_prompt=is_prompt_end_aug,
                    reasoner=self.reasoner if self.config.recursive_confidence_threshold > 0 else None,
                    verbose=self.config.recursive_verbose_cycles,
                )
                if self.config.recursive_verbose_cycles:
                    logging.debug(f"[forward] RecursiveMemory cycles={cycles}, is_prompt={is_prompt_end_aug}")

                if not self.config.recursive_skip_projection:
                    latent_inputs_embeds = self.weaver_to_reasoner(latent_inputs_embeds)

                latent_len = latent_inputs_embeds.size(1)
                attn_mask = torch.ones((B, latent_len), device=device, dtype=current_attention_mask.dtype)
                pos_ids = None

            elif self.query_projection_only:
                batch_size = current_inputs_embeds.size(0)
                if is_prompt_end_aug:
                    latent_inputs_embeds, attn_mask, pos_ids = weaver.augment_prompt_query_projection(
                        self.reasoner_to_weaver, self.weaver_to_reasoner,
                        batch_size, current_attention_mask, current_position_ids
                    )
                else:
                    latent_inputs_embeds, attn_mask, pos_ids = weaver.augment_inference_query_projection(
                        self.reasoner_to_weaver, self.weaver_to_reasoner,
                        batch_size, current_attention_mask, current_position_ids
                    )
            elif self.projection_only:
                latent_inputs_embeds, attn_mask, pos_ids = weaver.augment_prompt_projection_only(
                    current_inputs_embeds, current_attention_mask, current_position_ids,
                    self.reasoner_to_weaver, self.weaver_to_reasoner
                )
            elif self.skip_projection:
                if self.skip_lora:
                    if is_prompt_end_aug:
                        weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt_skip_lora(
                            current_inputs_embeds, current_attention_mask, current_position_ids
                        )
                    else:
                        weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference_skip_lora(
                            current_inputs_embeds, current_attention_mask, current_position_ids
                        )
                else:
                    if is_prompt_end_aug:
                        weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                            current_inputs_embeds, current_attention_mask, current_position_ids
                        )
                    else:
                        weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                            current_inputs_embeds, current_attention_mask, current_position_ids
                        )
                latent_inputs_embeds = weaver_hidden_states
            elif self.output_projection_only:
                if is_prompt_end_aug:
                    weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt_skip_lora(
                        current_inputs_embeds, current_attention_mask, current_position_ids
                    )
                else:
                    weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference_skip_lora(
                        current_inputs_embeds, current_attention_mask, current_position_ids
                    )
                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
            elif self.skip_lora or self.latent_processor_enabled:
                weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)

                if is_prompt_end_aug:
                    weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt_skip_lora(
                        weaver_inputs_embeds, current_attention_mask, current_position_ids
                    )
                else:
                    weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference_skip_lora(
                        weaver_inputs_embeds, current_attention_mask, current_position_ids
                    )

                if self.latent_processor_enabled:
                    weaver_hidden_states = self.latent_processor(weaver_hidden_states)

                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
            else:
                # Full mode: LoRA + query latents + projections
                weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)

                if is_prompt_end_aug:
                    weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                        weaver_inputs_embeds, current_attention_mask, current_position_ids
                    )
                else:
                    weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                        weaver_inputs_embeds, current_attention_mask, current_position_ids
                    )

                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx

            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

            # Stepwise training: intermediate loss on the NEXT segment
            if stepwise_enabled:
                if aug_loop_idx < len(augmentation_indices) - 1:
                    lookahead_end = augmentation_indices[aug_loop_idx + 1]
                else:
                    lookahead_end = input_ids.shape[1]

                lookahead_embeds = inputs_embeds[:, aug_point_idx:lookahead_end]
                lookahead_mask = attention_mask[:, aug_point_idx:lookahead_end]
                lookahead_latent_mask = torch.zeros(
                    (B, lookahead_embeds.size(1)), device=device, dtype=torch.bool
                )

                temp_embeds = torch.cat([current_inputs_embeds, lookahead_embeds], dim=1)
                temp_mask = torch.cat([current_attention_mask, lookahead_mask], dim=1)
                temp_latent_mask = torch.cat([current_latents_mask, lookahead_latent_mask], dim=1)
                temp_pos = self._generate_position_ids(temp_mask)

                partial_outputs = reasoner(
                    inputs_embeds=temp_embeds,
                    attention_mask=temp_mask,
                    position_ids=temp_pos,
                )
                partial_logits = partial_outputs.logits

                shifted_mask = torch.zeros_like(temp_latent_mask)
                shifted_mask[:, :-1] = temp_latent_mask[:, 1:]
                valid = ~shifted_mask
                valid_partial_logits = partial_logits[valid].view(B, -1, partial_logits.size(2))

                step_labels = labels[:, :lookahead_end].clone()
                step_labels[:, :aug_point_idx] = -100

                shift_logits_step = valid_partial_logits[..., :-1, :].contiguous()
                shift_labels_step = step_labels[..., 1:].contiguous()
                step_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    shift_logits_step.view(-1, shift_logits_step.size(-1)),
                    shift_labels_step.view(-1),
                )
                intermediate_losses.append(step_loss)

        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool)

        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits

        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted

        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))

        stepwise_loss = None
        if intermediate_losses:
            stepwise_loss = torch.stack(intermediate_losses).mean()

        return valid_logits, stepwise_loss
    
    def _instructional_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """Single-turn forward pass for instruction-following SFT."""
        logits, stepwise_loss = self._forward(input_ids, attention_mask, labels, **kwargs)
        return logits, labels, stepwise_loss

    def _conversational_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """Multi-turn forward with memory persistence across conversation turns."""
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()  # Transition 0 -> 1
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()          # Transition 1 -> 0
        if len(ends) < len(valid_starts):
            ends.append(seq_len)
        assert len(valid_starts) == len(ends)

        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e

        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]

        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)
        all_stepwise_losses = []

        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                cur_labels = labels[0, :end].clone().unsqueeze(0)
                cur_labels[0, :valid_start] = -100

                logits, stepwise_loss = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)

                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

                if stepwise_loss is not None:
                    all_stepwise_losses.append(stepwise_loss)

        combined_stepwise_loss = None
        if all_stepwise_losses:
            combined_stepwise_loss = torch.stack(all_stepwise_losses).mean()

        return all_logits, all_labels, combined_stepwise_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:
        tokenizer = self.tokenizer
        assert labels is not None, "Labels required for training"

        forward_func = self._instructional_forward
        if self._is_conversation(input_ids, tokenizer):
            labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer)
            forward_func = self._conversational_forward

        batch_size = 1
        iter_num = input_ids.size(0) // batch_size

        logits, supervised_labels, stepwise_losses = [], [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            batch_logits, batch_supervised_labels, batch_stepwise_loss = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
            if batch_stepwise_loss is not None:
                stepwise_losses.append(batch_stepwise_loss)

        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if stepwise_losses:
            stepwise_weight = getattr(self.config, 'recursive_stepwise_loss_weight', 0.5)
            mean_stepwise_loss = torch.stack(stepwise_losses).mean()
            loss = loss + stepwise_weight * mean_stepwise_loss

        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None,
        return_augmentation_mask: bool = False,
        prev_memory_embeds: Optional[torch.Tensor] = None,
        return_memory_embeds: bool = False,
        ltpo_optimizer: Optional["MemGenLTPOOptimizer"] = None,
        ltpo_verbose: bool = False,
        **kwargs
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor], tuple[torch.LongTensor, torch.Tensor]]:
        """
        Generate text with optional latent memory injection and capture.

        Args:
            prev_memory_embeds: Previous memory to prepend (embedding-level only).
            return_memory_embeds: If True, capture prompt augmentation memory.
            ltpo_optimizer: Optional LTPO optimizer for test-time latent optimization.
        """
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        max_augment_num = self.config.max_inference_aug_num
        invalid_token_id = -100

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device

        # Memory injected at embedding level only; input_ids/prompt_len stay unchanged
        if prev_memory_embeds is not None:
            prev_memory_embeds = prev_memory_embeds.to(device=device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([prev_memory_embeds, inputs_embeds], dim=1)
            memory_attention = torch.ones(
                prev_memory_embeds.shape[:2], dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([memory_attention, attention_mask], dim=1)

        captured_memory_embeds: Optional[torch.Tensor] = None
        if return_memory_embeds:
            latent_len = weaver.prompt_latents_num
            captured_memory_embeds = torch.zeros(
                (B, latent_len, hidden_size), dtype=inputs_embeds.dtype, device=device
            )

        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_cache: DynamicCache = None

        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        # augmentation_pos values: -100 = not sampled, 0 = sampled but no insert, 1 = inserted
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device)

        for i in range(max_new_tokens):
            
            assert current_inputs_embeds.shape[:2] == current_attention_mask.shape == current_position_ids.shape
            augment_decision = self._should_augment(
                current_input_ids,
                sentence_augment_count=sentence_augment_count,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                is_prompt=(i==0)  
            )
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            if len(augment_indices) > 0:
                if i != 0:
                    sentence_augment_count[augment_indices] += 1

                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]
                candidate_position_ids = current_position_ids[augment_indices]

                if self.recursive_memory:
                    if self.config.recursive_skip_projection:
                        context_for_compressor = candidate_inputs_embeds
                    else:
                        context_for_compressor = self.reasoner_to_weaver(candidate_inputs_embeds)

                    latent_inputs_embeds, cycles = self.recursive_compressor(
                        context=context_for_compressor,
                        attention_mask=candidate_attention_mask,
                        is_prompt=(i == 0),
                        reasoner=self.reasoner if self.config.recursive_confidence_threshold > 0 else None,
                        verbose=self.config.recursive_verbose_cycles,
                    )
                    if self.config.recursive_verbose_cycles:
                        logging.debug(f"[generate] RecursiveMemory cycles={cycles}, step={i}")

                    if not self.config.recursive_skip_projection:
                        latent_inputs_embeds = self.weaver_to_reasoner(latent_inputs_embeds)

                    latent_len = latent_inputs_embeds.size(1)
                    attn_mask = torch.ones(
                        (candidate_inputs_embeds.size(0), latent_len),
                        device=device, dtype=candidate_attention_mask.dtype
                    )
                    weaver_hidden_states = None

                elif self.query_projection_only:
                    batch_size = candidate_inputs_embeds.size(0)
                    if i == 0:
                        latent_inputs_embeds, attn_mask, _ = weaver.augment_prompt_query_projection(
                            self.reasoner_to_weaver, self.weaver_to_reasoner,
                            batch_size, candidate_attention_mask, candidate_position_ids
                        )
                    else:
                        latent_inputs_embeds, attn_mask, _ = weaver.augment_inference_query_projection(
                            self.reasoner_to_weaver, self.weaver_to_reasoner,
                            batch_size, candidate_attention_mask, candidate_position_ids
                        )
                    weaver_hidden_states = None
                elif self.skip_projection or self.output_projection_only:
                    weaver_inputs_embeds = candidate_inputs_embeds
                else:
                    weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)

                # Mode-specific augmentation (recursive_memory and query_projection_only handled above)
                if self.recursive_memory or self.query_projection_only:
                    pass
                elif self.skip_projection:
                    if self.skip_lora:
                        if i == 0:
                            weaver_hidden_states, attn_mask, _ = weaver.augment_prompt_skip_lora(
                                weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                            )
                        else:
                            weaver_hidden_states, attn_mask, _ = weaver.augment_inference_skip_lora(
                                weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                            )
                    else:
                        if i == 0:
                            weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                                weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                            )
                        else:
                            weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                                weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                            )
                elif self.output_projection_only:
                    if i == 0:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_prompt_skip_lora(
                            weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                        )
                    else:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_inference_skip_lora(
                            weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                        )
                elif self.skip_lora or self.latent_processor_enabled:
                    if i == 0:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_prompt_skip_lora(
                            weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                        )
                    else:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_inference_skip_lora(
                            weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                        )
                else:
                    if i == 0:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                            weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                        )
                    else:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                            weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                        )

                if self.latent_processor_enabled and not self.query_projection_only and not self.recursive_memory:
                    weaver_hidden_states = self.latent_processor(weaver_hidden_states)

                if i == 0 and return_memory_embeds and captured_memory_embeds is not None:
                    if self.recursive_memory or self.query_projection_only:
                        mem_embeds = latent_inputs_embeds.detach()
                    elif self.skip_projection:
                        mem_embeds = weaver_hidden_states.detach()
                    else:
                        mem_embeds = self.weaver_to_reasoner(weaver_hidden_states).detach()
                    captured_memory_embeds[augment_indices] = mem_embeds

                # Convert weaver hidden states to reasoner space
                if self.recursive_memory or self.query_projection_only:
                    pass  # Already in reasoner space
                elif self.skip_projection:
                    latent_inputs_embeds = weaver_hidden_states
                else:
                    latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

                if ltpo_optimizer is not None:
                    temp_full_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                    temp_full_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)

                    latent_len = latent_inputs_embeds.size(1)
                    seq_len = temp_full_embeds.size(1)
                    latent_start_idx = seq_len - latent_len
                    latent_end_idx = seq_len

                    optimized_latents, best_reward, best_step = ltpo_optimizer.optimize(
                        latents_hidden_states=latent_inputs_embeds,
                        inputs_embeds=temp_full_embeds,
                        attention_mask=temp_full_mask,
                        latent_start_idx=latent_start_idx,
                        latent_end_idx=latent_end_idx,
                        verbose=ltpo_verbose,
                    )

                    if ltpo_verbose:
                        logging.info(f"[LTPO] Step {i}: best_reward={best_reward:.4f}, best_step={best_step}")

                    latent_inputs_embeds = optimized_latents

                candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                candidate_attention_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)

                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)
                   
                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask

                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                    non_aug_attention_mask = current_attention_mask[non_augment_indices]
                    pad_len = weaver.prompt_latents_num if i == 0 else weaver.inference_latents_num
                    non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                        non_aug_inputs_embeds, non_aug_attention_mask, None, pad_len
                    )
                    
                    merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                    merged_attention_mask[non_augment_indices] = non_aug_attention_mask
                
                current_inputs_embeds = merged_inputs_embeds
                current_attention_mask = merged_attention_mask
                current_position_ids = self._generate_position_ids(current_attention_mask)
                current_cache = None  

            if (sentence_augment_count >= max_augment_num).all():
                generation_config = GenerationConfig(
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=False,
                    max_new_tokens=max_new_tokens-i
                )
                logits_processor = LogitsProcessorList([NanInfLogitsProcessor()])
                generated = reasoner.generate(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    generation_config=generation_config,
                    logits_processor=logits_processor
                )
                current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                break

            if current_cache is not None:
                assert current_inputs_embeds.size(1) == current_cache.get_seq_length() + 1
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[:, -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids

            outputs = reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=current_cache
            )
            current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = self._append_one_step(
                outputs,
                current_inputs_embeds,
                current_attention_mask,
                current_position_ids,
                current_input_ids,
                do_sample=False,
                temperature=0.0
            )
            current_cache = outputs.past_key_values

            if (current_input_ids[:, -1] == eos_token_id).all():
                break

            # Delete outputs to free logits memory (can be large on first iteration)
            del outputs

        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        
        self._check_generate(
            current_input_ids[:, prompt_len:],
            augmentation_pos
        )

        if return_memory_embeds and return_augmentation_mask:
            return (current_input_ids, augmentation_pos, captured_memory_embeds)
        elif return_memory_embeds:
            return (current_input_ids, captured_memory_embeds)
        elif return_augmentation_mask:
            return (current_input_ids, augmentation_pos)
        else:
            return current_input_ids
    
    @classmethod
    def from_config(cls, config_dict: dict):
        # base LLM
        model_name = config_dict.get("model_name")

        # max augment numbers
        max_prompt_aug_num = config_dict.get("max_prompt_aug_num", 1)
        max_inference_aug_num = config_dict.get("max_inference_aug_num", 5)

        # Weaver configs
        weaver_config = config_dict.get("weaver", {})
        prompt_latents_len = weaver_config.get("prompt_latents_len", 8)
        inference_latents_len = weaver_config.get("inference_latents_len", 8)
        weaver_lora_config_dict = weaver_config.get("lora_config", None)
        weaver_model_name = weaver_config.get("model_name", None)

        # Trigger configs
        trigger_config = config_dict.get("trigger", {})
        trigger_active = trigger_config.get("active", False)
        trigger_lora_config_dict = trigger_config.get("lora_config", None)
        trigger_model_name = trigger_config.get("model_name", None)

        # Projection-only mode
        projection_only = config_dict.get("projection_only", False)

        # Skip-LoRA mode (query latents + projections, no LoRA)
        skip_lora = config_dict.get("skip_lora", False)

        # LatentProcessor mode (query latents + projections + latent_processor MLP, no LoRA)
        latent_processor = config_dict.get("latent_processor", False)
        latent_processor_depth = config_dict.get("latent_processor_depth", 2)

        # Query-projection only mode (query latents + projections, NO weaver LLM)
        query_projection_only = config_dict.get("query_projection_only", False)

        # Skip-projection mode (query latents + LLM, NO projections)
        skip_projection = config_dict.get("skip_projection", False)

        # Output-projection-only mode (skip input projection, keep output projection)
        output_projection_only = config_dict.get("output_projection_only", False)

        # Recursive memory mode (WeaverStyleCompressor)
        recursive_memory_config = config_dict.get("recursive_memory", {})
        recursive_memory = recursive_memory_config.get("enabled", False)

        # Map yaml keys to MemGenConfig parameter names with defaults
        _recursive_defaults = {
            "weaver_style": True,
            "hidden_size": 4096,
            "num_heads": 8,
            "attn_rank": 64,
            "mlp_rank": 128,
            "max_cycles": 10,
            "confidence_threshold": -1.0,
            "top_k": 10,
            "verbose_cycles": False,
            "skip_projection": False,
            "two_level": False,
            "l_cycles": 6,
            "max_h_cycles": 5,
            "stepwise_training": False,
            "stepwise_loss_weight": 0.5,
            "full_rank_mlp": False,
            "bidirectional": False,
        }
        recursive_params = {
            f"recursive_{key}": recursive_memory_config.get(key, default)
            for key, default in _recursive_defaults.items()
        }

        # 构造 MemGenConfig
        from transformers import AutoConfig
        memgen_config = AutoConfig.from_pretrained(model_name)
        memgen_config = MemGenConfig.from_pretrained(
            model_name,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
            # weaver
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_lora_config=weaver_lora_config_dict,
            # trigger
            trigger_active=trigger_active,
            trigger_lora_config=trigger_lora_config_dict,
            # mode flags
            projection_only=projection_only,
            skip_lora=skip_lora,
            latent_processor=latent_processor,
            latent_processor_depth=latent_processor_depth,
            query_projection_only=query_projection_only,
            skip_projection=skip_projection,
            output_projection_only=output_projection_only,
            # recursive memory
            recursive_memory=recursive_memory,
            **recursive_params,
        )

        # Ensure _name_or_path is set for TRL GRPOTrainer compatibility
        memgen_config._name_or_path = model_name

        # Use sdpa (Scaled Dot Product Attention) as fallback if flash_attn not available
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
        except ImportError:
            logging.warning("Flash Attention 2 not available, falling back to SDPA")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if weaver_model_name != model_name:
            try:
                weaver_model = AutoModelForCausalLM.from_pretrained(
                    weaver_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
            except ImportError:
                weaver_model = AutoModelForCausalLM.from_pretrained(
                    weaver_model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
                )
        else:
            weaver_model = base_model

        if trigger_model_name != model_name:
            try:
                trigger_model = AutoModelForCausalLM.from_pretrained(
                    trigger_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
            except ImportError:
                trigger_model = AutoModelForCausalLM.from_pretrained(
                    trigger_model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
                )
        else:
            trigger_model = base_model
        
        load_model_path = config_dict.get("load_model_path", None)
        load_weaver_path = config_dict.get("load_weaver_path", None)
        load_trigger_path = config_dict.get("load_trigger_path", None)
        load_projections = config_dict.get("load_projections", True)  # False = query latents only
        load_query_latents = config_dict.get("load_query_latents", True)  # False = projections only (legacy)
        load_prompt_query_latents = config_dict.get("load_prompt_query_latents", load_query_latents)  # False = random prompt latents
        load_inference_query_latents = config_dict.get("load_inference_query_latents", load_query_latents)  # False = random inference latents

        if not load_model_path:
            model = cls(
                config=memgen_config,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                weaver_model=weaver_model,
                trigger_model=trigger_model
            )
        else:
            model = cls.from_pretrained(
                load_model_path,
                config=memgen_config,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                weaver_model=weaver_model,
                trigger_model=trigger_model
            )

            # Load projections.pt when using load_model_path (full model checkpoint)
            # This is needed because trainer.save_model() doesn't save projection layers
            proj_path = os.path.join(load_model_path, "projections.pt")
            if os.path.exists(proj_path):
                logging.info(f"Loading projections from {proj_path}")
                proj_data = torch.load(proj_path, map_location='cpu')

                model.reasoner_to_weaver.load_state_dict(proj_data['reasoner_to_weaver'])
                model.weaver_to_reasoner.load_state_dict(proj_data['weaver_to_reasoner'])

                # Load query_latents
                model.weaver.prompt_query_latents.data = proj_data['prompt_query_latents'].to(
                    model.weaver.prompt_query_latents.device
                )
                model.weaver.inference_query_latents.data = proj_data['inference_query_latents'].to(
                    model.weaver.inference_query_latents.device
                )
                logging.info(f"Loaded projections and query_latents from {proj_path}")

        # Load pre-trained weaver adapter if specified
        if load_weaver_path:
            model._load_pretrained_weaver(load_weaver_path, load_projections=load_projections,
                                          load_prompt_query_latents=load_prompt_query_latents,
                                          load_inference_query_latents=load_inference_query_latents)

        # Load pre-trained trigger adapter if specified
        if load_trigger_path:
            model._load_pretrained_trigger(load_trigger_path)

        return model

    def _load_pretrained_weaver(self, weaver_path: str, load_projections: bool = True,
                                load_prompt_query_latents: bool = True, load_inference_query_latents: bool = True):
        """
        Load pre-trained weaver adapter weights from a checkpoint.

        Args:
            weaver_path: Path to the weaver adapter checkpoint directory
                         (should contain adapter_model.bin)
            load_projections: If True, load projection layers (reasoner_to_weaver, weaver_to_reasoner).
            load_prompt_query_latents: If True, load prompt_query_latents. If False, keep random.
            load_inference_query_latents: If True, load inference_query_latents. If False, keep random.
        """
        from pathlib import Path

        rm_path = Path(weaver_path) / "recursive_memory.pt"
        if rm_path.exists():
            data = torch.load(str(rm_path), map_location='cpu')
            if hasattr(self, 'recursive_compressor') and 'recursive_compressor' in data:
                self.recursive_compressor.load_state_dict(data['recursive_compressor'])
            proj_path = Path(weaver_path) / "projections.pt"
            if proj_path.exists():
                proj_data = torch.load(str(proj_path), map_location='cpu')
                self.reasoner_to_weaver.load_state_dict(proj_data['reasoner_to_weaver'])
                self.weaver_to_reasoner.load_state_dict(proj_data['weaver_to_reasoner'])
            logging.info(f"Loaded recursive_memory checkpoint from {rm_path}")
            return

        lp_path = Path(weaver_path) / "latent_processor.pt"
        if lp_path.exists():
            data = torch.load(str(lp_path), map_location='cpu')
            self.reasoner_to_weaver.load_state_dict(data['reasoner_to_weaver'])
            self.weaver_to_reasoner.load_state_dict(data['weaver_to_reasoner'])
            self.weaver.prompt_query_latents.data = data['prompt_query_latents'].to(
                self.weaver.prompt_query_latents.device
            )
            self.weaver.inference_query_latents.data = data['inference_query_latents'].to(
                self.weaver.inference_query_latents.device
            )
            if hasattr(self, 'latent_processor') and 'latent_processor' in data:
                self.latent_processor.load_state_dict(data['latent_processor'])
            logging.info(f"Loaded latent_processor checkpoint from {lp_path}")
            return

        skip_lora_path = Path(weaver_path) / "skip_lora.pt"
        if skip_lora_path.exists():
            data = torch.load(str(skip_lora_path), map_location='cpu')
            self.reasoner_to_weaver.load_state_dict(data['reasoner_to_weaver'])
            self.weaver_to_reasoner.load_state_dict(data['weaver_to_reasoner'])
            self.weaver.prompt_query_latents.data = data['prompt_query_latents'].to(
                self.weaver.prompt_query_latents.device
            )
            self.weaver.inference_query_latents.data = data['inference_query_latents'].to(
                self.weaver.inference_query_latents.device
            )
            logging.info(f"Loaded skip-lora checkpoint from {skip_lora_path}")
            return

        skip_proj_path = Path(weaver_path) / "skip_projection.pt"
        if skip_proj_path.exists():
            data = torch.load(str(skip_proj_path), map_location='cpu')
            self.weaver.prompt_query_latents.data = data['prompt_query_latents'].to(
                self.weaver.prompt_query_latents.device
            )
            self.weaver.inference_query_latents.data = data['inference_query_latents'].to(
                self.weaver.inference_query_latents.device
            )
            logging.info(f"Loaded skip-projection checkpoint from {skip_proj_path}")

            if not self.skip_lora:
                lora_path = Path(weaver_path) / "weaver_lora"
                if lora_path.exists():
                    adapter_path = lora_path / "adapter_model.bin"
                    if adapter_path.exists():
                        adapter_weights = torch.load(str(adapter_path), map_location='cpu')
                        converted_weights = {}
                        for k, v in adapter_weights.items():
                            if "lora_A.weight" in k:
                                new_key = k.replace("lora_A.weight", "lora_A.weaver.weight")
                                converted_weights[new_key] = v
                            elif "lora_B.weight" in k:
                                new_key = k.replace("lora_B.weight", "lora_B.weaver.weight")
                                converted_weights[new_key] = v
                            else:
                                converted_weights[k] = v
                        self.weaver.model.load_state_dict(converted_weights, strict=False)
                        logging.info(f"Loaded weaver LoRA from {lora_path}")
            return

        proj_only_path = Path(weaver_path) / "projections_only.pt"
        if proj_only_path.exists():
            proj_data = torch.load(str(proj_only_path), map_location='cpu')
            self.reasoner_to_weaver.load_state_dict(proj_data['reasoner_to_weaver'])
            self.weaver_to_reasoner.load_state_dict(proj_data['weaver_to_reasoner'])
            logging.info(f"Loaded projection-only checkpoint from {proj_only_path}")
            return

        # skip_lora mode can reuse projections.pt without loading LoRA
        proj_path = Path(weaver_path) / "projections.pt"
        if proj_path.exists() and self.skip_lora:
            data = torch.load(str(proj_path), map_location='cpu')
            self.reasoner_to_weaver.load_state_dict(data['reasoner_to_weaver'])
            self.weaver_to_reasoner.load_state_dict(data['weaver_to_reasoner'])
            self.weaver.prompt_query_latents.data = data['prompt_query_latents'].to(
                self.weaver.prompt_query_latents.device
            )
            self.weaver.inference_query_latents.data = data['inference_query_latents'].to(
                self.weaver.inference_query_latents.device
            )
            logging.info(f"Loaded projections.pt for skip-lora mode from {proj_path}")
            return

        adapter_path = Path(weaver_path) / "weaver" / "adapter_model.bin"
        if not adapter_path.exists():
            adapter_path = Path(weaver_path) / "weaver_lora" / "adapter_model.bin"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Weaver adapter not found. Checked: {Path(weaver_path)}/weaver/, {Path(weaver_path)}/weaver_lora/")

        pretrained_weights = torch.load(str(adapter_path), map_location='cpu')
        weaver_state = self.weaver.model.state_dict()
        loaded_count = 0
        for key, value in pretrained_weights.items():
            if key in weaver_state:
                target_key = key
            elif "lora_" in key and key.endswith(".weight"):
                target_key = key.replace(".weight", ".weaver.weight")
            else:
                target_key = key
            if target_key in weaver_state:
                if weaver_state[target_key].shape == value.shape:
                    weaver_state[target_key] = value.to(weaver_state[target_key].dtype)
                    loaded_count += 1
        self.weaver.model.load_state_dict(weaver_state, strict=True)
        logging.info(f"Loaded {loaded_count} weaver LoRA weights from {weaver_path}")

        proj_path = Path(weaver_path).parent / "projections.pt"
        if not proj_path.exists():
            raise FileNotFoundError(f"projections.pt not found: {proj_path}")

        proj_data = torch.load(str(proj_path), map_location='cpu')

        if load_projections:
            self.reasoner_to_weaver.load_state_dict(proj_data['reasoner_to_weaver'])
            self.weaver_to_reasoner.load_state_dict(proj_data['weaver_to_reasoner'])

        if load_prompt_query_latents:
            self.weaver.prompt_query_latents.data = proj_data['prompt_query_latents'].to(self.weaver.prompt_query_latents.device)

        if load_inference_query_latents:
            self.weaver.inference_query_latents.data = proj_data['inference_query_latents'].to(self.weaver.inference_query_latents.device)

        logging.info(
            f"Loaded projections.pt from {proj_path} "
            f"(projections={load_projections}, prompt_latents={load_prompt_query_latents}, "
            f"inference_latents={load_inference_query_latents})"
        )

    def _load_pretrained_trigger(self, trigger_path: str):
        """Load pre-trained trigger adapter weights (LoRA key: lora_A.trigger.weight)."""
        from pathlib import Path

        adapter_path = Path(trigger_path) / "trigger" / "adapter_model.bin"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Trigger adapter not found: {adapter_path}")

        pretrained_weights = torch.load(str(adapter_path), map_location='cpu')
        trigger_state = self.trigger.model.state_dict()

        loaded_count = 0
        for key, value in pretrained_weights.items():
            if key in trigger_state:
                target_key = key
            elif "lora_" in key and key.endswith(".weight"):
                target_key = key.replace(".weight", ".trigger.weight")
            else:
                target_key = key

            if target_key in trigger_state:
                if trigger_state[target_key].shape == value.shape:
                    trigger_state[target_key] = value.to(trigger_state[target_key].dtype)
                    loaded_count += 1
                else:
                    logging.warning(f"Shape mismatch for {target_key}: expected {trigger_state[target_key].shape}, got {value.shape}")

        self.trigger.model.load_state_dict(trigger_state, strict=True)
        logging.info(f"Loaded {loaded_count} trigger adapter weights from {trigger_path}")
