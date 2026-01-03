"""
ARC Code Generation GRPO Trainer (BARC-style).

Trains the Weaver to generate better Python code for ARC tasks.

Training Flow:
- Generate code: Weaver augmentation applied (THIS IS WHAT WE TRAIN)
- Execute code: Validate on training examples (used for reward calculation)
- Reward: Accuracy on training examples

Key Architecture (BARC-style):
- Weaver is trained via GRPO on code generation
- Memory flows between code generation attempts
- Code is executed on training examples for validation
- Reward signal from code accuracy drives generation quality improvement

Training Target:
- Weaver parameters (LoRA adapters, query latents, projection layers)
- Base model is frozen
"""

import copy
import logging
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate.utils import gather_object
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from trl import GRPOConfig
from trl.models import unwrap_model_for_generation

from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from memgen.model.modeling_memgen import MemGenModel
from interactions.base_interaction import InteractionDataProto
from data.base_env import DynamicEnv

from arc.interaction import (
    ARCCodeGenerationManager,
    ARCCodeGenerationConfig,
    ARCCodeGenerationPoolManager,
    CodeGenerationResult,
)
from arc.utils import get_grid_similarity, parse_grid_from_text

logger = logging.getLogger(__name__)


# Type alias for reward functions
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class ARCCodeGenerationTrainer(WeaverGRPOTrainer):
    """
    GRPO Trainer specialized for ARC code generation Weaver training.

    Trains the Weaver to generate better Python code for ARC tasks.
    The Weaver is trained via GRPO on code generation, and the
    reward signal comes from code execution accuracy.

    Training Architecture (BARC-style):
    - Code generation: Weaver augmentation applied (THIS IS TRAINED)
    - Code execution: Validates generated code on training examples
    - Memory: Flows between code generation attempts

    Key Features:
    - N code candidates for GRPO training
    - Code execution on training examples for validation
    - Accuracy-based reward computation
    - Weaver gradient flows through code generation forward pass
    """

    def __init__(
        self,
        model: MemGenModel,
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        env_class=None,
        env_main_config=None,
        generation_manager=None,
        seed_pool_manager=None,
        code_gen_config: Optional[ARCCodeGenerationConfig] = None,
    ):
        """
        Initialize the ARC code generation trainer.

        Args:
            model: MemGenModel instance
            reward_funcs: Reward functions (code accuracy will be added)
            args: GRPO training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Tokenizer
            env_class: Environment class (ARCDynamicEnv)
            env_main_config: Environment configuration
            generation_manager: Override generation manager (optional)
            seed_pool_manager: Override seed pool manager (optional)
            code_gen_config: Code generation specific configuration
        """
        # Store code gen config before parent init
        self.code_gen_config = code_gen_config or ARCCodeGenerationConfig()

        # Create code generation interaction manager BEFORE parent init if not provided
        # Parent class requires generation_manager to be non-None
        if generation_manager is None or not isinstance(generation_manager, ARCCodeGenerationManager):
            from interactions.base_interaction import InteractionConfig

            # Get values from args for interaction config
            max_prompt_len = args.max_prompt_length if args else 4096
            max_completion_len = args.max_completion_length if args else 1024
            temperature = getattr(args, 'temperature', 0.7) if args else 0.7
            batch_size = args.per_device_train_batch_size if args else 1

            interaction_config = InteractionConfig(
                max_turns=1,  # Single pass, no refinement
                max_start_length=max_prompt_len,
                max_prompt_length=max_prompt_len,
                max_response_length=max_completion_len,
                max_obs_length=512,
                do_sample=True,
                temperature=temperature,
                batch_size=batch_size,
            )

            generation_manager = ARCCodeGenerationManager(
                tokenizer=processing_class,
                actor_rollout_wg=model,
                config=interaction_config,
                code_gen_config=self.code_gen_config,
            )

            seed_pool_manager = ARCCodeGenerationPoolManager(
                interaction_manager=generation_manager,
                num_candidates=self.code_gen_config.num_candidates,
                selection_strategy="best_accuracy"
            )

        # Store references for use in _generate_and_score_completions
        self.code_gen_manager = generation_manager
        self.code_gen_pool_manager = seed_pool_manager

        # Initialize parent class with the manager
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            env_class=env_class,
            env_main_config=env_main_config,
            generation_manager=generation_manager,
            seed_pool_manager=seed_pool_manager,
        )

        logger.info(
            f"ARCCodeGenerationTrainer initialized: "
            f"candidates={self.code_gen_config.num_candidates}"
        )

        # Track if static_graph is already set
        self._static_graph_configured = False

    def _configure_ddp_static_graph(self):
        """Configure DDP static_graph for models with shared parameters."""
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            import torch.distributed as dist

            # Only relevant in multi-GPU mode
            if not dist.is_initialized() or dist.get_world_size() <= 1:
                logger.debug("Single GPU mode, skipping static_graph configuration")
                return

            # Find the DDP wrapper - accelerate may wrap differently
            def find_ddp(module, depth=0):
                """Recursively find DDP wrapper."""
                if isinstance(module, DDP):
                    return module
                if depth > 5:  # Limit recursion depth
                    return None
                if hasattr(module, 'module'):
                    return find_ddp(module.module, depth + 1)
                if hasattr(module, '_modules'):
                    for _, child in module._modules.items():
                        result = find_ddp(child, depth + 1)
                        if result is not None:
                            return result
                return None

            ddp_module = find_ddp(self.model)
            if ddp_module is not None:
                ddp_module._set_static_graph()
                logger.info("Configured DDP static_graph for shared parameter handling")
            else:
                logger.info(f"Model wrapper type: {type(self.model).__name__}, checking accelerator...")
                # Try accelerator's internal DDP
                if hasattr(self.accelerator, 'ddp_handler') and self.accelerator.ddp_handler is not None:
                    if hasattr(self.accelerator.ddp_handler, '_set_static_graph'):
                        self.accelerator.ddp_handler._set_static_graph()
                        logger.info("Configured static_graph via accelerator ddp_handler")
                else:
                    logger.info("No DDP wrapper found - accelerate may use different method")
        except Exception as e:
            logger.debug(f"DDP static_graph not applicable: {e}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to set static_graph before first backward pass.
        """
        # Configure static_graph on first training step
        if not self._static_graph_configured:
            self._configure_ddp_static_graph()
            self._static_graph_configured = True

        # Call parent's training_step
        return super().training_step(model, inputs, num_items_in_batch)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Custom save method that handles shared tensors in MemGenModel.

        Only saves trainable components:
        - LoRA adapters (from weaver.model)
        - Projection layers (reasoner_to_weaver, weaver_to_reasoner)
        - Weaver query latents and memory gate

        This avoids the "shared tensors" error from HuggingFace save_pretrained.
        """
        import os
        from safetensors.torch import save_file

        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Unwrap model from accelerate/deepspeed wrappers
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # 1. Save LoRA adapters from weaver
        lora_dir = os.path.join(output_dir, "weaver_lora")
        os.makedirs(lora_dir, exist_ok=True)
        try:
            unwrapped_model.weaver.model.save_pretrained(lora_dir)
            logger.info(f"Saved LoRA adapters to {lora_dir}")
        except Exception as e:
            logger.warning(f"Failed to save LoRA adapters: {e}")

        # 2. Save projection layers and weaver parameters
        trainable_state = {}

        # Projection layers
        if hasattr(unwrapped_model, 'reasoner_to_weaver'):
            for name, param in unwrapped_model.reasoner_to_weaver.named_parameters():
                trainable_state[f"reasoner_to_weaver.{name}"] = param.detach().cpu()

        if hasattr(unwrapped_model, 'weaver_to_reasoner'):
            for name, param in unwrapped_model.weaver_to_reasoner.named_parameters():
                trainable_state[f"weaver_to_reasoner.{name}"] = param.detach().cpu()

        # Weaver query latents
        if hasattr(unwrapped_model.weaver, 'prompt_query_latents'):
            trainable_state['weaver.prompt_query_latents'] = unwrapped_model.weaver.prompt_query_latents.detach().cpu()

        if hasattr(unwrapped_model.weaver, 'inference_query_latents'):
            trainable_state['weaver.inference_query_latents'] = unwrapped_model.weaver.inference_query_latents.detach().cpu()

        # Memory gate
        if hasattr(unwrapped_model.weaver, 'memory_gate'):
            for name, param in unwrapped_model.weaver.memory_gate.named_parameters():
                trainable_state[f"weaver.memory_gate.{name}"] = param.detach().cpu()

        # Save trainable parameters
        if trainable_state:
            trainable_path = os.path.join(output_dir, "trainable_params.safetensors")
            save_file(trainable_state, trainable_path)
            logger.info(f"Saved trainable parameters to {trainable_path}")

        # 3. Save training state for resuming
        self.save_state()

        logger.info(f"Model saved to {output_dir}")

    @staticmethod
    def load_trainable_weights(model: MemGenModel, checkpoint_dir: str) -> MemGenModel:
        """
        Load trainable weights from a checkpoint.

        Args:
            model: MemGenModel instance (with base weights already loaded)
            checkpoint_dir: Directory containing saved weights

        Returns:
            Model with trainable weights loaded
        """
        import os
        from peft import PeftModel
        from safetensors.torch import load_file

        # 1. Load LoRA adapters
        lora_dir = os.path.join(checkpoint_dir, "weaver_lora")
        if os.path.exists(lora_dir):
            try:
                # Load LoRA weights into existing PeftModel
                model.weaver.model.load_adapter(lora_dir, adapter_name="weaver")
                model.weaver.model.set_adapter("weaver")
                logger.info(f"Loaded LoRA adapters from {lora_dir}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapters: {e}")

        # 2. Load trainable parameters
        trainable_path = os.path.join(checkpoint_dir, "trainable_params.safetensors")
        if os.path.exists(trainable_path):
            state_dict = load_file(trainable_path)

            # Projection layers
            for name, param in model.reasoner_to_weaver.named_parameters():
                key = f"reasoner_to_weaver.{name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key].to(param.device))

            for name, param in model.weaver_to_reasoner.named_parameters():
                key = f"weaver_to_reasoner.{name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key].to(param.device))

            # Weaver query latents
            if 'weaver.prompt_query_latents' in state_dict:
                model.weaver.prompt_query_latents.data.copy_(
                    state_dict['weaver.prompt_query_latents'].to(model.weaver.prompt_query_latents.device)
                )

            if 'weaver.inference_query_latents' in state_dict:
                model.weaver.inference_query_latents.data.copy_(
                    state_dict['weaver.inference_query_latents'].to(model.weaver.inference_query_latents.device)
                )

            # Memory gate
            for name, param in model.weaver.memory_gate.named_parameters():
                key = f"weaver.memory_gate.{name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key].to(param.device))

            logger.info(f"Loaded trainable parameters from {trainable_path}")

        return model

    def _build_multiturn_envs(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Tuple[List[List[Dict]], List]:
        """
        Build multi-turn environments for code generation training.

        Sets up environments with task configurations for the code generation pipeline.

        Args:
            inputs: Batch of input samples

        Returns:
            Tuple of (init_messages, envs)
        """
        init_messages = []
        envs = []

        for task_config in inputs:
            # Create environment
            env: DynamicEnv = self.env_class(self.env_main_config)
            system_prompt, init_user_prompt = env.set_env(task_config)

            # Store task config in env for code generation access
            env.task_config = task_config

            system_message = {"role": "system", "content": system_prompt}
            init_user_message = {"role": "user", "content": init_user_prompt}

            init_messages.append([system_message, init_user_message])
            envs.append(env)

        return init_messages, envs

    def _generate_and_score_completions(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions using code generation pipeline and score them.

        Overrides parent method to use the code generation flow:
        1. Generate code candidates
        2. Execute on training examples
        3. Compute accuracy as reward

        Args:
            inputs: Batch of input samples

        Returns:
            Dictionary with generation results and scores
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Build environments
        init_prompts, envs = self._build_multiturn_envs(inputs)

        # Create generation batch
        gen_batch = InteractionDataProto()
        gen_batch.no_tensor_batch["init_prompts"] = init_prompts
        gen_batch.no_tensor_batch["envs"] = envs

        # Also store task configs for the interaction manager
        for example, env in zip(inputs, envs):
            example["envs"] = env

        # Generate using code generation pipeline
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                # Update manager's model reference
                self.code_gen_manager.actor_rollout_wg = unwrapped_model

                # Run code generation with candidate pooling
                final_output, seed_info = self.code_gen_pool_manager.run_seed_pool(gen_batch)

                # Log candidate selection info
                if seed_info:
                    avg_best_score = sum(
                        s["best_accuracy"] for s in seed_info
                    ) / len(seed_info)
                    self._metrics[mode]["code_gen/avg_best_accuracy"].append(avg_best_score)

        # Extract results and candidate info
        results: List[CodeGenerationResult] = final_output.no_tensor_batch.get("results", [])

        # Get candidate scores from _build_output (one per candidate, not per task)
        candidate_scores = final_output.no_tensor_batch.get("candidate_scores", [])
        num_candidates = final_output.no_tensor_batch.get("num_candidates_per_task", self.num_generations)

        # Build output tensors
        prompts = final_output.batch.get("prompts", torch.tensor([])).to(device)
        completion_ids = final_output.batch.get("responses", torch.tensor([])).to(device)
        prompt_completion_ids = final_output.batch.get("input_ids", torch.tensor([])).to(device)
        attention_mask = final_output.batch.get("attention_mask", torch.tensor([])).to(device)

        # Number of samples should match total candidates across all tasks
        num_samples = len(candidate_scores) if candidate_scores else len(results)

        if prompts.numel() == 0:
            # Fallback: create minimal tensors
            prompts = torch.zeros((num_samples, 1), dtype=torch.long, device=device)
            completion_ids = torch.zeros((num_samples, 1), dtype=torch.long, device=device)
            prompt_completion_ids = torch.zeros((num_samples, 2), dtype=torch.long, device=device)
            attention_mask = torch.ones_like(prompt_completion_ids)

        prompt_mask = attention_mask[:, :prompts.size(1)].to(device)
        info_mask = final_output.batch.get("info_mask", torch.ones_like(completion_ids))
        completion_mask = info_mask[:, prompts.size(1):].to(device)

        # Use candidate scores as rewards (already expanded for all candidates)
        # This is the key change: rewards come from instruction quality, not final grid
        if candidate_scores:
            rewards = torch.tensor(candidate_scores, dtype=torch.float32, device=device)
        else:
            # Fallback to final_reward (should not happen with new _build_output)
            rewards = torch.tensor(
                [r.final_reward for r in results],
                dtype=torch.float32,
                device=device
            )

        # Log metrics
        self._metrics[mode]["code_gen/accuracy_mean"].append(rewards.mean().item())
        self._metrics[mode]["code_gen/accuracy_max"].append(rewards.max().item())
        self._metrics[mode]["code_gen/perfect_ratio"].append(
            (rewards == 1.0).float().mean().item()
        )

        # Compute code accuracy metrics
        accuracy_scores = [r.best_accuracy for r in results]
        self._metrics[mode]["code_gen/best_accuracy_mean"].append(
            sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        )

        # Construct labels for loss computation
        prompt_labels = torch.full(prompt_mask.shape, -100, device=device, dtype=torch.long)
        ignore_label = torch.tensor(-100, device=device, dtype=torch.long)
        completion_labels = torch.where(completion_mask == 1, completion_ids, ignore_label)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)

        logits_to_keep = completion_mask.size(1)

        # Compute log probabilities
        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps, old_supervise_mask = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                )
            else:
                old_per_token_logps, old_supervise_mask = None, None

            # Reference model logps
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                        )
            else:
                ref_per_token_logps, ref_supervise_mask = None, None

        # Decode completions for logging
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Completion IDs list for reward functions
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Compute completion lengths
        completion_lengths = completion_mask.sum(1)

        # Check for EOS
        is_eos = completion_ids == self.eos_token_id

        # Mask truncated completions if configured
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Compute advantages using candidates as the grouping factor
        # Instead of num_generations=2, we use num_candidates (typically 5)
        # This allows GRPO to compute advantages across instruction candidates
        group_size = num_candidates if num_candidates > 0 else self.num_generations
        if group_size <= 0:
            logger.error(f"Invalid group_size={group_size}, num_candidates={num_candidates}, num_generations={self.num_generations}")
            raise ValueError(f"group_size must be positive, got {group_size}")
        mean_grouped_rewards = rewards.view(-1, group_size).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, group_size).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(group_size, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        logger.debug(f"GRPO grouping: {len(rewards)} samples, group_size={group_size}, num_groups={len(rewards)//group_size}")

        # Log metrics
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["completions/mean_length"].append(completion_lengths.float().mean().item())

        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(advantages.tolist())

        return {
            "prompt_ids": prompts,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "old_supervise_mask": old_supervise_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "ref_supervise_mask": ref_supervise_mask,
        }

    @staticmethod
    def compute_arc_reward(
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute reward for ARC task completions.

        This is a static method that can be passed to the trainer as a reward function.
        It extracts grids from completions and computes similarity with targets.

        Args:
            prompts: Input prompts (not used)
            completions: Model outputs containing grids
            **kwargs: Additional arguments including:
                - envs: List of environment instances with target_grid

        Returns:
            List of rewards (0.0 to 1.0)
        """
        envs = kwargs.get("envs", [])
        rewards = []

        for completion, env in zip(completions, envs):
            # Parse grid from completion
            pred_grid = parse_grid_from_text(completion)

            # Get target from environment
            target_grid = getattr(env, 'target_grid', None)
            if target_grid is None:
                task_config = getattr(env, 'task_config', {})
                target_grid = task_config.get('target_grid')

            # Compute similarity
            if pred_grid is None or target_grid is None:
                reward = 0.0
            else:
                reward = get_grid_similarity(target_grid, pred_grid)

            rewards.append(reward)

        return rewards


def create_arc_trainer(
    model: MemGenModel,
    args: GRPOConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    env_class,
    env_config: Dict,
    code_gen_config: Optional[ARCCodeGenerationConfig] = None,
) -> ARCCodeGenerationTrainer:
    """
    Factory function to create an ARCCodeGenerationTrainer.

    Args:
        model: MemGenModel instance
        args: GRPO training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        env_class: Environment class (ARCDynamicEnv)
        env_config: Environment configuration dictionary
        code_gen_config: Optional code generation configuration

    Returns:
        Configured ARCCodeGenerationTrainer instance
    """
    trainer = ARCCodeGenerationTrainer(
        model=model,
        reward_funcs=[ARCCodeGenerationTrainer.compute_arc_reward],
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        env_class=env_class,
        env_main_config=env_config,
        code_gen_config=code_gen_config,
    )

    return trainer
