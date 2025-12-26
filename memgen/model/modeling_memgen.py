import logging
from typing import Union, Optional
from contextlib import contextmanager

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
        
        # projection layers for mapping embeddings between reasoner and weaver
        # map reasoner input embeddings to weaver input embeddings
        reasoner_hidden_size = self.config.hidden_size
        weaver_hidden_size = weaver_model.base_model.config.hidden_size
        self.reasoner_to_weaver = nn.Linear(
            reasoner_hidden_size, weaver_hidden_size, dtype=torch.bfloat16
        )
        # Map weaver hidden states to reasoner input embeddings
        self.weaver_to_reasoner = nn.Linear(
            weaver_hidden_size, reasoner_hidden_size, dtype=torch.bfloat16
        )
        
        self.delimiters: list[str] = [",", ".", "\n"]  # delimiters for detecting augmentation points

        # postprocess
        self._postprocess_models()

    def _postprocess_models(self):
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )

        # NOTE: Do NOT override chat_template here.
        # Use the model's built-in chat template (e.g., GPT-OSS Harmony format).
        # The old SmolLM3 template was incompatible with GPT-OSS.
    

    @property
    def device(self):
        return self.reasoner.device

    @contextmanager
    def disable_adapter(self):
        """
        Context manager to disable LoRA adapters for reference model computation.
        This allows GRPO to compute reference logits without creating a separate model copy.
        Delegates to PEFT's built-in disable_adapter context manager.
        """
        # Use PEFT's built-in context manager
        with self.weaver.model.disable_adapter():
            yield

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # preprocess inputs
        assert input_ids.shape == attention_mask.shape == labels.shape

        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.config.max_inference_aug_num  # Limit the number of inference augmentation points to avoid excessive augmentation
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size

        # Skip augmentation entirely when max_prompt_aug_num is 0 (pure SFT mode)
        if self.config.max_prompt_aug_num == 0:
            # Simple forward without any augmentation
            position_ids = self._generate_position_ids(attention_mask)
            reasoner_outputs = reasoner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            return reasoner_outputs.logits

        # select augment idx
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )

        # origin inputs embeds
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)

        # Initialize the start index and empty tensors for accumulating processed segments
        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        # Iterate over the selected augmentation points
        for aug_point_idx in augmentation_indices:
            # Slice the current segment of original embeddings and attention mask
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            # Concatenate the current segment to the accumulated embeddings and masks
            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            # Map reasoner embeddings to weaver embeddings for augmentation
            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)

            # Determine whether this point is the end of the prompt (prompt augmentation)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()

            # Depending on type, use weaver to augment prompt or inference
            if is_prompt_end_aug:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )

            # Map weaver hidden states back to reasoner embeddings
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
            # Handle nan/inf in latent embeddings to prevent numerical instability
            # Embedding values are typically in range [-1, 1], so use conservative clamp
            if torch.isnan(latent_inputs_embeds).any() or torch.isinf(latent_inputs_embeds).any():
                latent_inputs_embeds = torch.nan_to_num(latent_inputs_embeds, nan=0.0, posinf=5.0, neginf=-5.0)
            latent_inputs_embeds = torch.clamp(latent_inputs_embeds, min=-5.0, max=5.0)

            # Update accumulated embeddings and masks with the newly augmented segment
            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx

            # Update latent mask for the newly added latent embeddings
            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        # Process the remaining segment after the last augmentation point
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

        # Identify valid positions in logits (positions that should contribute to loss)
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted

        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))
        # assert shifted.sum() == current_latents_mask.sum()
        # assert valid_logits.shape[:2] == input_ids.shape

        return valid_logits
    
    def _instructional_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for single-turn instructional data (no multi-turn conversation required).

        This method is used for instruction-following tasks (SFT), where the input
        consists of a single instruction and the corresponding labels. It directly
        delegates to the single-turn forward method `_forward`.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask (torch.Tensor): Tensor indicating padding positions.
            labels (torch.Tensor): Tensor containing the target labels for supervised fine-tuning.
            **kwargs: Additional keyword arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - logits: The output logits from the model for each input token.
                - labels: The same as input labels, used for loss computation.
        """
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        # For Instruction SFT, labels remain the same as input
        return logits, labels

    def _grpo_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:
        """
        Forward pass for GRPO training with Weaver augmentation.

        This mode uses the Weaver to generate latent memory embeddings at the
        prompt/completion boundary, then computes logits. Gradients flow through
        the Weaver's LoRA parameters for training.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask
            labels: Labels for loss computation (prompt positions are -100)
            **kwargs: Additional arguments (ignored)

        Returns:
            MemGenOutputWithPast with logits and loss
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        reasoner = self.reasoner
        weaver = self.weaver

        # Get embeddings from input_ids
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        # Enable gradient tracking for gradient checkpointing to work properly
        inputs_embeds = inputs_embeds.requires_grad_(True)

        # Find prompt/completion boundary from labels
        # Labels are -100 for prompt positions, valid for completion positions
        if labels is not None:
            # Find first non -100 position for each sample in batch
            is_completion = (labels != -100)
            # Get the first completion position (boundary)
            completion_starts = is_completion.int().argmax(dim=1)  # (B,)
            # Use the minimum as augmentation point (assumes similar prompt lengths)
            aug_point = completion_starts.min().item()
        else:
            # If no labels, augment at end (shouldn't happen in GRPO)
            aug_point = seq_len

        # Skip augmentation if aug_point is 0 (all completion) or at end (all prompt)
        if aug_point > 0 and aug_point < seq_len:
            # Get prompt embeddings for Weaver
            prompt_embeds = inputs_embeds[:, :aug_point]
            prompt_mask = attention_mask[:, :aug_point]
            prompt_position_ids = self._generate_position_ids(prompt_mask)

            # Map reasoner embeddings to weaver space
            weaver_inputs_embeds = self.reasoner_to_weaver(prompt_embeds)
            # Handle nan/inf in input embeddings
            if torch.isnan(weaver_inputs_embeds).any() or torch.isinf(weaver_inputs_embeds).any():
                weaver_inputs_embeds = torch.nan_to_num(weaver_inputs_embeds, nan=0.0, posinf=5.0, neginf=-5.0)
            weaver_inputs_embeds = torch.clamp(weaver_inputs_embeds, min=-5.0, max=5.0)

            # Use Weaver to generate latent memory embeddings
            weaver_hidden_states, aug_attn_mask, aug_pos_ids = weaver.augment_prompt(
                weaver_inputs_embeds, prompt_mask, prompt_position_ids
            )

            # Map weaver outputs back to reasoner space
            latent_embeds = self.weaver_to_reasoner(weaver_hidden_states)
            # Handle nan/inf in latent embeddings to prevent numerical instability
            if torch.isnan(latent_embeds).any() or torch.isinf(latent_embeds).any():
                latent_embeds = torch.nan_to_num(latent_embeds, nan=0.0, posinf=5.0, neginf=-5.0)
            latent_embeds = torch.clamp(latent_embeds, min=-5.0, max=5.0)

            # Get completion embeddings
            completion_embeds = inputs_embeds[:, aug_point:]
            completion_mask = attention_mask[:, aug_point:]

            # Concatenate: prompt + latent + completion
            current_inputs_embeds = torch.cat([prompt_embeds, latent_embeds, completion_embeds], dim=1)
            current_attention_mask = torch.cat([prompt_mask, aug_attn_mask, completion_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)

            # Track latent positions for masking
            latent_len = latent_embeds.size(1)
            latent_start = aug_point
            latent_end = aug_point + latent_len
        else:
            # No augmentation needed
            current_inputs_embeds = inputs_embeds
            current_attention_mask = attention_mask
            current_position_ids = self._generate_position_ids(current_attention_mask)
            latent_len = 0
            latent_start = 0
            latent_end = 0

        # Forward through reasoner with augmented embeddings
        outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids,
            use_cache=False
        )
        full_logits = outputs.logits

        # Remove latent positions from logits to match original sequence length
        if latent_len > 0:
            # Split and recombine without latent positions
            pre_latent_logits = full_logits[:, :latent_start, :]
            post_latent_logits = full_logits[:, latent_end:, :]
            logits = torch.cat([pre_latent_logits, post_latent_logits], dim=1)
        else:
            logits = full_logits

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return in expected format
        result = MemGenOutputWithPast(loss=loss, logits=logits)
        result.supervised_labels = labels
        return result

    def _conversational_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for conversational (multi-turn) data with memory persistence.

        Multi-turn forward is constructed by sequentially calling the single-turn forward
        for each conversation turn. Memory from turn i-1 IS visible to turn i,
        enabling the model to accumulate knowledge across turns (consistent with generate()).

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (1, seq_len). Batch size must be 1.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor): Target labels for supervised fine-tuning (-100 for ignore positions).
            **kwargs: Additional arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - all_logits: Logits for the entire sequence, with zeros for unsupervised positions.
                - all_labels: Labels for the entire sequence, with -100 for unsupervised positions.
        """
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        # Identify single-turn segments within the conversation based on labels
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        # Compute the start and end indices of valid supervised segments
        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()  # Transition 0 -> 1
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()          # Transition 1 -> 0
        if len(ends) < len(valid_starts):
            ends.append(seq_len)
        assert len(valid_starts) == len(ends)

        # Build triplets (start of previous segment, start of supervised segment, end of supervised segment)
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e

        # If there are more segments than allowed, randomly select self.max_prompt_aug_num segments
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]

        # Initialize tensors to store logits and labels for the entire sequence
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        # Loop over each conversation turn and perform single-turn forward if supervised
        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                # cur_labels only used for _forward, does not represent the true supervision range
                cur_labels = labels[0, :end].clone().unsqueeze(0)
                cur_labels[0, :valid_start] = -100  # Mask tokens before supervision start

                # Single-turn forward for the current conversation segment
                logits = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)
                
                # Update overall logits and labels with the results of this segment
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        # Return logits and labels:
        # - supervised positions retain computed logits and original labels
        # - unsupervised positions have logits = 0 and labels = -100
        return all_logits, all_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:
        tokenizer = self.tokenizer

        # Check if this is a GRPO forward call (skip conversation parsing, do simple forward)
        is_grpo = kwargs.pop('is_grpo', False)

        if is_grpo:
            # GRPO mode: Skip conversation/instruction parsing, do simple reasoner forward
            # This is used for computing log probabilities on already-generated sequences
            return self._grpo_forward(input_ids, attention_mask, labels, **kwargs)

        # Ensure labels are provided, required for training the reasoning processor
        assert labels is not None, "Reasoning Processor requires input labels for training"

        # Determine whether the input is single-turn (instruction) or multi-turn (conversation)
        forward_func = self._instructional_forward
        if self._is_conversation(input_ids, tokenizer):
            # For conversational data, mask assistant tokens in labels
            labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer)
            forward_func = self._conversational_forward

        batch_size = 1  # Currently process one sequence per batch
        iter_num = input_ids.size(0) // batch_size

        # Forward pass per batch
        logits, supervised_labels = [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            # Call the appropriate forward function (instruction or conversation)
            batch_logits, batch_supervised_labels = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
        
        # Concatenate results from all batches
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        # Compute causal language modeling loss (shifted by one)
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        # assert shift_logits.shape[:-1] == shift_labels.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return model outputs
        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels  # Positions in input_ids that are supervised
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
        **kwargs
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor], tuple[torch.LongTensor, torch.Tensor]]:
        """
        Generate text with optional latent memory injection and capture.

        Memory Architecture (arc-lang-public style):
        - Memory flows between instruction generations within a task (refinement loop)
        - Grid generation uses pure text prompts (no memory injection)
        - Weaver generates latent memory tokens that augment the reasoning process

        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            generation_config: Generation parameters (max_new_tokens, temperature, etc.)
            return_augmentation_mask: If True, return augmentation positions
            prev_memory_embeds: Previous memory embeddings to prepend [B, mem_len, hidden_size].
                Used for instruction refinement where memory from previous turn is carried over.
                Injected at embedding level only (input_ids stays unchanged).
            return_memory_embeds: If True, capture and return memory from this generation.
                Memory is captured from prompt augmentation (i=0) only.

        Returns:
            - If neither return flag: generated token IDs
            - If return_augmentation_mask only: (token_ids, augmentation_pos)
            - If return_memory_embeds only: (token_ids, captured_memory_embeds)
            - If both: (token_ids, augmentation_pos, captured_memory_embeds)

        Note:
            Trigger currently doesn't see prev_memory_embeds (acceptable for ARC two-stage
            where trigger is disabled). This is a known limitation.
        """
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        max_augment_num = self.config.max_inference_aug_num
        invalid_token_id = -100

        # preproecess inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device

        # Handle prev_memory_embeds: prepend to inputs for instruction refinement
        # IMPORTANT: Memory is injected at embedding level only, not token level.
        # - inputs_embeds and attention_mask: grow by memory length
        # - input_ids and current_input_ids: stay unchanged (track actual tokens only)
        # - prompt_len: stays as original token count (for extracting generated tokens later)
        # This is intentional: the generation loop tracks tokens separately from embeddings.
        if prev_memory_embeds is not None:
            prev_memory_embeds = prev_memory_embeds.to(device=device, dtype=inputs_embeds.dtype)
            # Prepend previous memory to current inputs (embeds level only)
            inputs_embeds = torch.cat([prev_memory_embeds, inputs_embeds], dim=1)
            memory_attention = torch.ones(
                prev_memory_embeds.shape[:2], dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([memory_attention, attention_mask], dim=1)

        # Variable to capture memory for return (only from prompt augmentation)
        # Initialize with full batch size to maintain alignment
        captured_memory_embeds: Optional[torch.Tensor] = None
        if return_memory_embeds:
            # Pre-allocate memory tensor for full batch (filled during prompt augmentation)
            latent_len = weaver.prompt_latents_num
            captured_memory_embeds = torch.zeros(
                (B, latent_len, hidden_size), dtype=inputs_embeds.dtype, device=device
            )

        # --- generation loop ---
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_cache: DynamicCache = None

        # Generation Loop Initialization
        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        
        # NOTE - Whether to call the trigger and insert latent memory before generating the token at this position
        # - augmentation_pos[b][i] == -100: For the b-th sequence, no augmentation was sampled before generating the i-th token
        # - augmentation_pos[b][i] == 0: For the b-th sequence, augmentation was sampled before generating the i-th token, but the trigger decided NOT to insert latent memory
        # - augmentation_pos[b][i] == 1: For the b-th sequence, augmentation was sampled before generating the i-th token, and the trigger decided to insert latent memory
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device) 

        for i in range(max_new_tokens):
            
            assert current_inputs_embeds.shape[:2] == current_attention_mask.shape == current_position_ids.shape
            # NOTE: _should_augment uses current_input_ids (not embeds), so it doesn't see
            # prev_memory_embeds. This is acceptable for ARC two-stage where trigger is disabled,
            # but should be addressed if trigger needs to be memory-aware in the future.
            augment_decision = self._should_augment(
                current_input_ids,
                sentence_augment_count=sentence_augment_count,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                is_prompt=(i==0)  
            )
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            # If there are sentences to augment, apply augmentation; others remain with left padding
            if len(augment_indices) > 0:
                # Increment the augmentation count for sentences that are being augmented
                if i != 0:  
                    sentence_augment_count[augment_indices] += 1

                # Select embeddings, attention masks, and position IDs for sentences to be augmented
                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]
                candidate_position_ids = current_position_ids[augment_indices]
                
                # Perform inference augmentation using the weaver
                weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)

                if i == 0:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )
                    # Capture memory from prompt augmentation for return
                    if return_memory_embeds and captured_memory_embeds is not None:
                        # Fill memory at correct batch indices (only augmented samples get memory)
                        # Non-augmented samples keep zero memory from pre-allocation
                        mem_embeds = self.weaver_to_reasoner(weaver_hidden_states).detach()
                        if torch.isnan(mem_embeds).any() or torch.isinf(mem_embeds).any():
                            mem_embeds = torch.nan_to_num(mem_embeds, nan=0.0, posinf=5.0, neginf=-5.0)
                        mem_embeds = torch.clamp(mem_embeds, min=-5.0, max=5.0)
                        captured_memory_embeds[augment_indices] = mem_embeds
                else:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )

                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
                # Handle nan/inf in latent embeddings to prevent numerical instability
                if torch.isnan(latent_inputs_embeds).any() or torch.isinf(latent_inputs_embeds).any():
                    latent_inputs_embeds = torch.nan_to_num(latent_inputs_embeds, nan=0.0, posinf=5.0, neginf=-5.0)
                # Clamp to reasonable range matching embedding values (~[-1, 1])
                latent_inputs_embeds = torch.clamp(latent_inputs_embeds, min=-5.0, max=5.0)

                candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                candidate_attention_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)
                
                # Create a single merged tensor for all sequences
                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)
                   
                # Directly place augmented and non-augmented sequences
                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask
                
                # Non-augmented sequences now include both -100 and 0
                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    # dynamic left padding
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

            # Check if all sequences have reached the maximum number of augmentations
            if (sentence_augment_count >= max_augment_num).all():
                # Adjust the remaining generation length
                generation_config = GenerationConfig(
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=False,
                    max_new_tokens=max_new_tokens-i
                )
                # Add logits processor to handle nan/inf
                logits_processor = LogitsProcessorList([NanInfLogitsProcessor()])
                # Perform generation for the remaining tokens using the reasoner
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
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature if generation_config.temperature else 1.0
            )
            current_cache = outputs.past_key_values

            # If all sequences in the batch have already generated an EOS token, stop early
            if (current_input_ids[:, -1] == eos_token_id).all():
                break  

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        # postprocess
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        
        self._check_generate(
            current_input_ids[:, prompt_len:],
            augmentation_pos
        )

        # Handle different return combinations
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
            trigger_lora_config=trigger_lora_config_dict
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

        # Load pre-trained weaver adapter if specified
        if load_weaver_path:
            model._load_pretrained_weaver(load_weaver_path)

        return model

    def _load_pretrained_weaver(self, weaver_path: str):
        """
        Load pre-trained weaver adapter weights from a checkpoint.

        Args:
            weaver_path: Path to the weaver adapter checkpoint directory
                         (should contain adapter_model.safetensors)
        """
        from pathlib import Path
        from safetensors.torch import load_file

        adapter_path = Path(weaver_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Weaver adapter not found: {adapter_path}")

        logging.info(f"Loading pre-trained weaver from: {weaver_path}")

        # Load the pre-trained weights
        pretrained_weights = load_file(str(adapter_path))

        # Get current weaver state dict
        weaver_state = self.weaver.model.state_dict()

        # Map pretrained weights to weaver state dict
        loaded_count = 0
        for key, value in pretrained_weights.items():
            if key in weaver_state:
                if weaver_state[key].shape == value.shape:
                    weaver_state[key] = value.to(weaver_state[key].dtype)
                    loaded_count += 1
                else:
                    logging.warning(f"Shape mismatch for {key}: expected {weaver_state[key].shape}, got {value.shape}")

        # Load the updated state dict
        self.weaver.model.load_state_dict(weaver_state, strict=True)

        logging.info(f"Loaded {loaded_count} weaver adapter weights from checkpoint")
