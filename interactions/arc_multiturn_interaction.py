"""
ARC Multi-Turn Interaction Manager with Memory Accumulation

Implements the 5 seeds × 7 turns paradigm:
- 5 diverse initial attempts (seeds) run in parallel
- Each seed refines over 7 turns with accumulated memory
- Memory embeddings persist across turns within each seed
"""
import logging
import torch
import copy
from typing import Dict, List, Tuple, Optional
from transformers import GenerationConfig

logger = logging.getLogger(__name__)

from interactions.base_interaction import (
    InteractionDataProto,
    InteractionConfig,
    InteractionManager
)
from interactions.multiturn_interaction import MultiTurnInteractionManager


class ARCMultiTurnInteractionManager(MultiTurnInteractionManager):
    """
    Extended multi-turn manager for ARC with memory accumulation.

    Key features:
    - Tracks memory embeddings across turns
    - Supports 5 seeds × 7 turns paradigm
    - Memory persists within each seed trajectory
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
        num_seeds: int = 5,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.num_seeds = num_seeds
        # Override max_turns for the 7-turn refinement
        self.config.max_turns = getattr(config, 'max_turns', 7)

    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """
        Run main LLM generation loop with memory accumulation.

        For each task, runs num_seeds parallel trajectories, each with max_turns refinements.
        Memory accumulates within each trajectory.
        """
        assert "init_prompts" in gen_batch.no_tensor_batch
        assert "envs" in gen_batch.no_tensor_batch
        batch_size = len(gen_batch.no_tensor_batch["init_prompts"])

        rollings = gen_batch
        rollings.no_tensor_batch["inter_histories"] = [[] for _ in range(batch_size)]

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]

        # Initialize memory tracking per sequence
        # memory_embeds[i] = accumulated memory tensor for sequence i
        memory_embeds: List[Optional[torch.Tensor]] = [None] * batch_size

        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break

            mask_list = active_mask.tolist()
            rollings_active = {
                k: [item for item, keep in zip(v, mask_list) if keep]
                for k, v in rollings.no_tensor_batch.items()
            }

            # Get active memory embeddings
            active_memory = [
                memory_embeds[i] for i, keep in enumerate(mask_list) if keep
            ]

            # Build chat history and tokenize
            messages = self._build_chat_history(rollings_active)
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=True,
                padding=True, return_tensors="pt", return_dict=True
            )

            # Stack active memory into batch tensor
            prev_memory_batch = self._stack_memory_batch(active_memory)

            # Generate with memory persistence
            gen_output, new_memory = self.actor_rollout_wg.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config,
                prev_memory_embeds=prev_memory_batch,
                return_memory_embeds=True,
            )
            gen_output = gen_output.to("cpu")

            # Update memory embeddings for active sequences
            if new_memory is not None:
                new_memory = new_memory.to("cpu")
                logger.debug(f"Turn {step}: Generated memory shape {new_memory.shape}")
                active_idx = 0
                for i, is_active in enumerate(mask_list):
                    if is_active:
                        old_len = memory_embeds[i].size(0) if memory_embeds[i] is not None else 0
                        memory_embeds[i] = new_memory[active_idx]
                        new_len = memory_embeds[i].size(0)
                        logger.debug(f"  Seq {i}: Memory {old_len} → {new_len} tokens")
                        active_idx += 1

            # Postprocess responses
            prompt_len = inputs["input_ids"].size(1)
            responses = gen_output[:, prompt_len:]
            responses = self.tensor_fn.erase_after_first_eos(
                responses, self.tokenizer.eos_token_id
            )
            responses_ids, responses_str = self._postprocess_responses(
                responses, rollings_active["envs"]
            )
            all_responses_ids, all_responses_str = self._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            # Execute predictions and get feedback
            next_obs, dones = self._execute_predictions(
                rollings, all_responses_str, active_mask
            )
            processed_obs = self._postprocess_observations(next_obs)

            # Update active mask
            curr_active_mask = torch.tensor(
                [not done for done in dones], dtype=torch.bool
            )
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            # Update interaction history
            interaction_histories = self._update_interaction_history(
                rollings, all_responses_str, processed_obs
            )
            rollings.no_tensor_batch["inter_histories"] = interaction_histories

        # Store final memory embeddings in output for potential future use
        final_outputs = self._build_final_outputs(rollings)
        final_outputs.no_tensor_batch["memory_embeds"] = memory_embeds

        return final_outputs

    def _stack_memory_batch(
        self, memory_list: List[Optional[torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """
        Stack individual memory tensors into a batch tensor.

        Handles variable memory lengths by padding shorter memories.
        Validates that all memories have consistent hidden dimensions.

        Args:
            memory_list: List of memory tensors, each with shape (memory_len, hidden_size)

        Returns:
            Batched memory tensor with shape (batch_size, max_len, hidden_size)
        """
        if not memory_list or all(m is None for m in memory_list):
            return None

        # Find max memory length and validate consistency
        max_len = 0
        hidden_size = None
        dtype = None
        device = None
        for m in memory_list:
            if m is not None:
                # Memory shape after extraction: (memory_len, hidden_size)
                if len(m.shape) != 2:
                    raise ValueError(f"Expected 2D memory tensor, got shape {m.shape}")

                if hidden_size is None:
                    hidden_size = m.size(-1)  # Last dimension is always hidden_size
                    dtype = m.dtype
                    device = m.device
                else:
                    # Validate dimension consistency
                    if m.size(-1) != hidden_size:
                        raise ValueError(
                            f"Inconsistent memory hidden_size: {m.size(-1)} vs {hidden_size}"
                        )
                    if m.dtype != dtype:
                        raise ValueError(
                            f"Inconsistent memory dtypes: {m.dtype} vs {dtype}"
                        )
                max_len = max(max_len, m.size(0))  # First dimension is sequence length

        if max_len == 0 or hidden_size is None:
            return None

        # Create padded batch on the correct device
        batch_size = len(memory_list)
        batch_memory = torch.zeros(
            (batch_size, max_len, hidden_size), dtype=dtype, device=device
        )

        for i, m in enumerate(memory_list):
            if m is not None:
                batch_memory[i, :m.size(0), :] = m

        return batch_memory

    def run_multi_seed_loop(
        self, gen_batch: InteractionDataProto
    ) -> List[InteractionDataProto]:
        """
        Run 5 seeds × 7 turns paradigm.

        Creates num_seeds copies of each task, runs them in parallel,
        and returns all seed trajectories.
        """
        original_batch_size = len(gen_batch.no_tensor_batch["init_prompts"])

        # Expand batch to include all seeds
        expanded_batch = self._expand_for_seeds(gen_batch)

        # Run the agent loop on expanded batch
        outputs = self.run_agent_loop(expanded_batch)

        # Reorganize outputs by original task
        seed_outputs = self._reorganize_by_task(outputs, original_batch_size)

        return seed_outputs

    def _expand_for_seeds(
        self, gen_batch: InteractionDataProto
    ) -> InteractionDataProto:
        """
        Expand batch to create num_seeds copies of each task.

        Each seed gets a different prompting strategy via set_seed().
        Prompts are regenerated with seed-specific strategies.
        """
        expanded = InteractionDataProto()

        envs = gen_batch.no_tensor_batch.get("envs", [])
        init_prompts = gen_batch.no_tensor_batch.get("init_prompts", [])

        # Expand envs and regenerate prompts with seed-specific strategies
        expanded_envs = []
        expanded_prompts = []

        for env_idx, env in enumerate(envs):
            # Get task config from environment if available
            task_config = getattr(env, 'task_config', None)

            for seed in range(self.num_seeds):
                # Deep copy the env
                new_env = copy.deepcopy(env)

                # Set seed-specific strategy
                if hasattr(new_env, 'set_seed'):
                    new_env.set_seed(seed)
                else:
                    new_env.seed_id = seed

                # Regenerate prompts with seed-specific strategy if task_config available
                if task_config is not None and hasattr(new_env, 'set_env'):
                    sys_prompt, user_prompt = new_env.set_env(task_config)
                    seed_prompt = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    expanded_prompts.append(seed_prompt)
                else:
                    # Fall back to copying original prompt
                    expanded_prompts.append(copy.deepcopy(init_prompts[env_idx]))

                expanded_envs.append(new_env)

        expanded.no_tensor_batch["envs"] = expanded_envs
        expanded.no_tensor_batch["init_prompts"] = expanded_prompts

        # Expand other non-tensor batch items
        for key, value in gen_batch.no_tensor_batch.items():
            if key in ("envs", "init_prompts"):
                continue  # Already handled above
            # Repeat each item num_seeds times
            expanded.no_tensor_batch[key] = [
                copy.deepcopy(item)
                for item in value
                for _ in range(self.num_seeds)
            ]

        # Expand tensor batch items
        for key, tensor in gen_batch.batch.items():
            expanded.batch[key] = tensor.repeat_interleave(self.num_seeds, dim=0)

        return expanded

    def _reorganize_by_task(
        self,
        outputs: InteractionDataProto,
        original_batch_size: int
    ) -> List[InteractionDataProto]:
        """
        Reorganize outputs from [task0_seed0, task0_seed1, ..., task1_seed0, ...]
        to list of outputs per task, each containing all seeds.
        """
        seed_outputs = []

        for task_idx in range(original_batch_size):
            task_output = InteractionDataProto()

            # Extract this task's seeds from non-tensor batch
            for key, value in outputs.no_tensor_batch.items():
                start_idx = task_idx * self.num_seeds
                end_idx = start_idx + self.num_seeds
                task_output.no_tensor_batch[key] = value[start_idx:end_idx]

            # Extract this task's seeds from tensor batch
            for key, tensor in outputs.batch.items():
                start_idx = task_idx * self.num_seeds
                end_idx = start_idx + self.num_seeds
                task_output.batch[key] = tensor[start_idx:end_idx]

            seed_outputs.append(task_output)

        return seed_outputs


class ARCSeedPoolManager:
    """
    Manages the 5 seeds × 7 turns training paradigm for ARC.

    Coordinates:
    - Running multiple seeds per task in parallel
    - Selecting best seed trajectory for reward computation
    - Memory accumulation within each seed
    """

    def __init__(
        self,
        interaction_manager: ARCMultiTurnInteractionManager,
        num_seeds: int = 5,
        selection_strategy: str = "best_final"
    ):
        """
        Args:
            interaction_manager: The multi-turn interaction manager
            num_seeds: Number of diverse seeds per task (default: 5)
            selection_strategy: How to select among seeds
                - "best_final": Use seed with highest final score
                - "best_any": Use seed with highest score at any turn
                - "pool": Return all seeds for ensemble
        """
        self.interaction_manager = interaction_manager
        self.num_seeds = num_seeds
        self.selection_strategy = selection_strategy

    def run_seed_pool(
        self, gen_batch: InteractionDataProto
    ) -> Tuple[InteractionDataProto, List[Dict]]:
        """
        Run all seeds for the batch and select best trajectories.

        Returns:
            - selected_outputs: Best trajectory outputs for reward computation
            - all_seed_info: Detailed info about all seed trajectories
        """
        # Run multi-seed loop
        seed_outputs = self.interaction_manager.run_multi_seed_loop(gen_batch)

        # Select best seeds based on strategy
        selected_outputs, all_seed_info = self._select_seeds(
            gen_batch, seed_outputs
        )

        return selected_outputs, all_seed_info

    def _select_seeds(
        self,
        original_batch: InteractionDataProto,
        seed_outputs: List[InteractionDataProto]
    ) -> Tuple[InteractionDataProto, List[Dict]]:
        """
        Select best seed trajectories based on selection strategy.
        """
        batch_size = len(original_batch.no_tensor_batch["init_prompts"])
        envs = original_batch.no_tensor_batch["envs"]

        selected = InteractionDataProto()
        all_seed_info = []

        for task_idx in range(batch_size):
            task_outputs = seed_outputs[task_idx]
            task_envs = task_outputs.no_tensor_batch.get("envs", [])

            # Get scores for each seed
            seed_scores = []
            for seed_idx in range(self.num_seeds):
                if seed_idx < len(task_envs):
                    env = task_envs[seed_idx]
                    score, solved = env.feedback()
                    seed_scores.append({
                        "seed_idx": seed_idx,
                        "score": score,
                        "solved": solved,
                        "turns": env.current_turn if hasattr(env, 'current_turn') else 0
                    })
                else:
                    seed_scores.append({
                        "seed_idx": seed_idx,
                        "score": 0.0,
                        "solved": False,
                        "turns": 0
                    })

            # Select best seed
            if self.selection_strategy == "best_final":
                best_seed = max(seed_scores, key=lambda x: x["score"])
            elif self.selection_strategy == "best_any":
                # Would need to track history - for now use best_final
                best_seed = max(seed_scores, key=lambda x: x["score"])
            else:
                best_seed = seed_scores[0]

            best_idx = best_seed["seed_idx"]

            # Add to selected outputs
            for key in task_outputs.no_tensor_batch:
                if key not in selected.no_tensor_batch:
                    selected.no_tensor_batch[key] = []
                if best_idx < len(task_outputs.no_tensor_batch[key]):
                    selected.no_tensor_batch[key].append(
                        task_outputs.no_tensor_batch[key][best_idx]
                    )

            for key in task_outputs.batch:
                if key not in selected.batch:
                    selected.batch[key] = []
                if best_idx < task_outputs.batch[key].size(0):
                    selected.batch[key].append(
                        task_outputs.batch[key][best_idx:best_idx+1]
                    )

            all_seed_info.append({
                "task_idx": task_idx,
                "seed_scores": seed_scores,
                "selected_seed": best_idx
            })

        # Stack tensor batches
        for key in selected.batch:
            if selected.batch[key]:
                selected.batch[key] = torch.cat(selected.batch[key], dim=0)

        return selected, all_seed_info
