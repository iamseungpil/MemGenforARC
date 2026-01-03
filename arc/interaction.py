"""
Two-Stage Interaction Manager for ARC Training (arc-lang-public style).

Implements the arc-lang two-stage architecture with MemGen memory:
- Stage 1: Generate instructions (captures memory embeddings via Weaver)
- Stage 2: Generate grids using pure text prompts (NO memory injection)

Memory Architecture:
- Memory is captured during instruction generation (Stage 1)
- Grid generation is text-only (arc-lang-public approach)

Key Features:
- N instruction candidates per task with memory capture
- Leave-one-out scoring using pure text grid generation
- Grid similarity reward computation
- Best candidate selected by score for final grid generation
"""

import logging
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from transformers import GenerationConfig

from interactions.base_interaction import (
    InteractionDataProto,
    InteractionConfig,
    InteractionManager
)
from interactions.multiturn_interaction import MultiTurnInteractionManager

from arc.prompts import (
    build_instruction_prompt,
    build_grid_generation_prompt,
    SYSTEM_PROMPT_INSTRUCTOR,
    SYSTEM_PROMPT_EXECUTOR,
)
from arc.utils import (
    get_grid_similarity,
    parse_grid_from_text,
    parse_instructions_from_text,
    format_examples,
    grid_to_string,
)

logger = logging.getLogger(__name__)


@dataclass
class ARCTwoStageConfig:
    """Configuration for ARC two-stage training.

    Memory Architecture (arc-lang-public style):
    - Memory is captured during instruction generation (Stage 1)
    - Grid generation uses pure text prompts (no memory injection)
    """
    # Number of instruction candidates per task
    # These candidates form the GRPO comparison group
    instruction_candidates: int = 5

    # Whether to use leave-one-out scoring
    leave_one_out_scoring: bool = True

    # Maximum length for instruction generation
    max_instruction_length: int = 1024

    # Maximum length for grid generation
    max_grid_length: int = 512

    # Temperature for instruction generation (higher = more diverse)
    instruction_temperature: float = 0.8

    # Temperature for grid generation (lower = more precise)
    grid_temperature: float = 0.3


@dataclass
class InstructionCandidate:
    """Represents a single instruction candidate with its score."""
    instructions: str
    score: float = 0.0
    example_scores: List[float] = field(default_factory=list)
    attempts: List[Optional[List[List[int]]]] = field(default_factory=list)
    memory_embeds: Optional[torch.Tensor] = None
    is_perfect: bool = False


@dataclass
class TwoStageResult:
    """Result of two-stage generation for a single task."""
    task_id: str
    best_instructions: str
    best_score: float
    final_grid: Optional[List[List[int]]]
    final_reward: float
    all_candidates: List[InstructionCandidate]
    memory_embeds: Optional[torch.Tensor]


class ARCTwoStageInteractionManager(MultiTurnInteractionManager):
    """
    Two-stage interaction manager for ARC training (arc-lang-public style).

    Memory Architecture:
    - Memory is captured during instruction generation (Stage 1)
    - Grid generation uses pure text prompts (NO memory injection)

    Stage 1: Generate instruction candidates
    - Produces N diverse instruction candidates via MemGen Weaver
    - Captures memory embeddings during generation

    Stage 2: Generate grids using instructions (pure text)
    - Uses best instructions from Stage 1 (selected by score)
    - NO memory injection - follows arc-lang-public approach
    - Computes leave-one-out scores for candidate ranking
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
        two_stage_config: Optional[ARCTwoStageConfig] = None,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.two_stage_config = two_stage_config or ARCTwoStageConfig()

        # Generation configs for each stage
        self.instruction_gen_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=self.two_stage_config.max_instruction_length,
            temperature=self.two_stage_config.instruction_temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        self.grid_gen_config = GenerationConfig(
            do_sample=False,  # Deterministic for grid generation
            max_new_tokens=self.two_stage_config.max_grid_length,
            temperature=self.two_stage_config.grid_temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """
        Run the two-stage generation loop.

        For each task in the batch:
        1. Generate N instruction candidates (Stage 1)
        2. Score each candidate using leave-one-out
        3. Sort by score and select best candidate
        4. Generate final grid with best instructions (Stage 2)

        Args:
            gen_batch: Input batch with task configurations

        Returns:
            InteractionDataProto with generation results
        """
        assert "envs" in gen_batch.no_tensor_batch
        envs = gen_batch.no_tensor_batch["envs"]
        batch_size = len(envs)

        all_results: List[TwoStageResult] = []

        # Process each task
        for task_idx, env in enumerate(envs):
            task_config = getattr(env, 'task_config', None)
            if task_config is None:
                logger.warning(f"Task {task_idx} has no task_config, skipping")
                continue

            result = self._process_single_task(task_config, env)
            all_results.append(result)

            # Update environment with result
            if hasattr(env, 'best_score'):
                env.best_score = result.final_reward
            if hasattr(env, 'best_grid'):
                env.best_grid = result.final_grid

        # Build output
        return self._build_output(gen_batch, all_results)

    def _process_single_task(
        self,
        task_config: Dict,
        env: Any
    ) -> TwoStageResult:
        """
        Process a single ARC task through the two-stage pipeline.

        Args:
            task_config: Task configuration with train_examples, test_input, target_grid
            env: Environment instance for this task

        Returns:
            TwoStageResult with best instructions and grid
        """
        train_examples = task_config.get("train_examples", [])
        test_input = task_config.get("test_input")
        target_grid = task_config.get("target_grid")
        task_id = task_config.get("task_id", "unknown")

        logger.debug(f"Processing task {task_id} with {len(train_examples)} examples")

        # =====================================================================
        # Stage 1: Generate instruction candidates
        # =====================================================================
        candidates = self._generate_instruction_candidates(
            train_examples=train_examples,
            test_input=test_input,
            num_candidates=self.two_stage_config.instruction_candidates
        )

        if not candidates:
            logger.warning(f"Task {task_id}: No valid instruction candidates generated")
            return TwoStageResult(
                task_id=task_id,
                best_instructions="",
                best_score=0.0,
                final_grid=None,
                final_reward=0.0,
                all_candidates=[],
                memory_embeds=None
            )

        # =====================================================================
        # Score candidates using leave-one-out
        # =====================================================================
        for candidate in candidates:
            self._score_candidate(candidate, train_examples)

        # Sort by score and select best candidate
        candidates.sort(key=lambda c: c.score, reverse=True)
        best_candidate = candidates[0]
        logger.info(f"Task {task_id}: Best candidate score = {best_candidate.score:.2%}")

        # =====================================================================
        # Stage 2: Generate final grid with best instructions
        # (No memory injection - pure text prompt, arc-lang-public style)
        # =====================================================================
        final_grid = self._generate_grid(
            instructions=best_candidate.instructions,
            train_examples=train_examples,
            test_input=test_input,
        )

        # Compute final reward
        if target_grid is not None and final_grid is not None:
            final_reward = get_grid_similarity(target_grid, final_grid)
        else:
            final_reward = 0.0

        logger.info(f"Task {task_id}: Final reward = {final_reward:.2%}")

        return TwoStageResult(
            task_id=task_id,
            best_instructions=best_candidate.instructions,
            best_score=best_candidate.score,
            final_grid=final_grid,
            final_reward=final_reward,
            all_candidates=candidates,
            memory_embeds=best_candidate.memory_embeds  # Memory from instruction generation (for potential future use)
        )

    def _generate_instruction_candidates(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
        num_candidates: int
    ) -> List[InstructionCandidate]:
        """
        Generate N instruction candidates (Stage 1).

        Each candidate captures memory embeddings from the Weaver.

        Args:
            train_examples: Training examples for the task
            test_input: Test input grid
            num_candidates: Number of candidates to generate

        Returns:
            List of InstructionCandidate objects
        """
        candidates = []

        # Format inputs
        examples_text = format_examples(train_examples)
        test_input_text = grid_to_string(test_input)

        # Build prompt
        prompt_text = build_instruction_prompt(examples_text, test_input_text)

        # Create messages for chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_INSTRUCTOR},
            {"role": "user", "content": prompt_text}
        ]

        # Generate candidates with retry to ensure consistent count
        max_retries = 3

        # Get model device for tensor placement
        device = next(self.actor_rollout_wg.parameters()).device

        for i in range(num_candidates):
            candidate = None

            for retry in range(max_retries):
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer.apply_chat_template(
                    [messages],
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                )

                # Move inputs to model device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate with memory capture
                gen_output, memory_embeds = self.actor_rollout_wg.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=self.instruction_gen_config,
                    return_memory_embeds=True
                )

                # Extract response
                prompt_len = inputs["input_ids"].size(1)
                response_ids = gen_output[:, prompt_len:]
                response_text = self.tokenizer.decode(
                    response_ids[0], skip_special_tokens=True
                )

                # Parse instructions
                instructions = parse_instructions_from_text(response_text)

                if instructions:
                    candidate = InstructionCandidate(
                        instructions=instructions,
                        # Keep memory on device (don't move to CPU) for proper gradient flow
                        memory_embeds=memory_embeds[0].detach() if memory_embeds is not None else None
                    )
                    logger.debug(f"Generated candidate {i+1}/{num_candidates}")
                    break
                else:
                    logger.warning(f"Failed to parse instructions from candidate {i+1}, retry {retry+1}/{max_retries}")

            # If all retries failed, use raw response as fallback
            if candidate is None:
                logger.warning(f"All retries failed for candidate {i+1}, using raw response")
                candidate = InstructionCandidate(
                    instructions=response_text[:500] if response_text else "Follow the pattern from training examples.",
                    memory_embeds=memory_embeds[0].detach() if memory_embeds is not None else None
                )

            candidates.append(candidate)

        return candidates

    def _score_candidate(
        self,
        candidate: InstructionCandidate,
        train_examples: List[Dict]
    ) -> None:
        """
        Score an instruction candidate using leave-one-out validation.

        Updates the candidate in-place with scores and attempts.

        Args:
            candidate: The instruction candidate to score
            train_examples: Training examples for scoring
        """
        example_scores = []
        attempts = []

        if self.two_stage_config.leave_one_out_scoring and len(train_examples) > 1:
            for i in range(len(train_examples)):
                temp_test = train_examples[i]
                temp_train = train_examples[:i] + train_examples[i+1:]

                # Generate grid for this example (no memory - pure text prompt)
                pred_grid = self._generate_grid(
                    instructions=candidate.instructions,
                    train_examples=temp_train,
                    test_input=temp_test["input"],
                )

                # Score
                score = get_grid_similarity(temp_test["output"], pred_grid)
                example_scores.append(score)
                attempts.append(pred_grid)
        else:
            # Use first example as test if only one
            if len(train_examples) > 0:
                test_ex = train_examples[0]
                context = train_examples[1:] if len(train_examples) > 1 else train_examples

                pred_grid = self._generate_grid(
                    instructions=candidate.instructions,
                    train_examples=context,
                    test_input=test_ex["input"],
                )

                score = get_grid_similarity(test_ex["output"], pred_grid)
                example_scores.append(score)
                attempts.append(pred_grid)

        # Update candidate
        candidate.example_scores = example_scores
        candidate.attempts = attempts
        candidate.score = sum(example_scores) / len(example_scores) if example_scores else 0.0
        candidate.is_perfect = candidate.score == 1.0

    def _generate_grid(
        self,
        instructions: str,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> Optional[List[List[int]]]:
        """
        Generate a grid using instructions (Stage 2).

        Uses pure text prompts following arc-lang-public approach.
        NO memory injection - grid generation relies only on text instructions.

        Args:
            instructions: Transformation instructions
            train_examples: Training examples for context
            test_input: Test input grid

        Returns:
            Predicted grid (or None if parsing fails)
        """
        # Format inputs
        examples_text = format_examples(train_examples)
        test_input_text = grid_to_string(test_input)

        # Build prompt
        prompt_text = build_grid_generation_prompt(
            instructions=instructions,
            examples_text=examples_text,
            test_input_text=test_input_text,
            is_perfect=False
        )

        # Create messages
        # NOTE: /no_think must be in user prompt for Qwen3 to disable thinking mode
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EXECUTOR},
            {"role": "user", "content": prompt_text + "\n/no_think"}
        ]

        # Tokenize
        # NOTE: For Qwen3, we use /no_think in the prompt instead of enable_thinking=False
        # because enable_thinking=False inserts empty <think></think> tags which can confuse the model.
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move inputs to model device
        device = next(self.actor_rollout_wg.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate using pure text prompt (no memory injection - arc-lang-public style)
        gen_output = self.actor_rollout_wg.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=self.grid_gen_config,
        )

        # Extract response
        prompt_len = inputs["input_ids"].size(1)
        response_ids = gen_output[:, prompt_len:]
        response_text = self.tokenizer.decode(
            response_ids[0], skip_special_tokens=True
        )

        # Parse grid
        pred_grid = parse_grid_from_text(response_text)

        if pred_grid is not None:
            logger.info(f"Grid parsed successfully: {len(pred_grid)}x{len(pred_grid[0]) if pred_grid else 0}")
        else:
            logger.warning(f"Grid parsing failed. Response preview: {response_text[:100]}...")

        return pred_grid

    def _build_output(
        self,
        gen_batch: InteractionDataProto,
        results: List[TwoStageResult]
    ) -> InteractionDataProto:
        """
        Build the output InteractionDataProto from results.

        IMPORTANT: Returns ALL candidates (not just best) for GRPO training.
        Each candidate becomes a separate response, allowing GRPO to compute
        advantages across candidates within the same task.

        For each task with N candidates:
        - N copies of the same prompt
        - N different instruction responses
        - N different rewards (candidate scores)

        This eliminates the need for num_generations > 1 in GRPO config,
        as candidates naturally form the comparison group.

        Args:
            gen_batch: Original input batch
            results: List of TwoStageResult for each task

        Returns:
            InteractionDataProto with all candidates as separate samples
        """
        output = InteractionDataProto()

        # Store results in no_tensor_batch
        output.no_tensor_batch["results"] = results
        output.no_tensor_batch["envs"] = gen_batch.no_tensor_batch.get("envs", [])

        # Build outputs for ALL candidates (not just best)
        envs = gen_batch.no_tensor_batch.get("envs", [])
        instruction_prompts = []
        responses = []
        candidate_scores = []  # Store scores for GRPO rewards

        for result, env in zip(results, envs):
            # Get task config for examples
            task_config = getattr(env, 'task_config', {})
            train_examples = task_config.get("train_examples", [])
            test_input = task_config.get("test_input", [])

            # Format the INSTRUCTION generation prompt
            examples_text = format_examples(train_examples)
            test_input_text = grid_to_string(test_input)
            prompt_text = build_instruction_prompt(examples_text, test_input_text)

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_INSTRUCTOR},
                {"role": "user", "content": prompt_text}
            ]

            # Add ALL candidates as separate samples
            for candidate in result.all_candidates:
                instruction_prompts.append(prompt_messages)
                instruction_str = f'{{"instructions": "{candidate.instructions}"}}'
                responses.append(instruction_str)
                candidate_scores.append(candidate.score)

        # Validate all tasks have same candidate count (required for GRPO grouping)
        expected_candidates = self.two_stage_config.instruction_candidates
        for i, result in enumerate(results):
            if len(result.all_candidates) != expected_candidates:
                logger.error(f"Task {i} has {len(result.all_candidates)} candidates, expected {expected_candidates}")

        # Store candidate scores for trainer to use as rewards
        output.no_tensor_batch["candidate_scores"] = candidate_scores
        output.no_tensor_batch["num_candidates_per_task"] = expected_candidates

        # Build interaction histories
        inter_histories = []
        for result in results:
            for candidate in result.all_candidates:
                history = [
                    {"role": "assistant", "content": f'{{"instructions": "{candidate.instructions}"}}'},
                    {"role": "user", "content": f"Score: {candidate.score:.2%}"},
                ]
                inter_histories.append(history)
        output.no_tensor_batch["inter_histories"] = inter_histories

        if instruction_prompts:
            self.tokenizer.padding_side = "left"
            prompt_ids = self.tokenizer.apply_chat_template(
                instruction_prompts,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )
            output.batch["prompts"] = prompt_ids["input_ids"]

        if responses:
            self.tokenizer.padding_side = "right"
            response_ids = self.tokenizer(
                responses,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            output.batch["responses"] = response_ids["input_ids"]

        # Combine prompts and responses
        if "prompts" in output.batch and "responses" in output.batch:
            output.batch["input_ids"] = torch.cat(
                [output.batch["prompts"], output.batch["responses"]], dim=1
            )
            prompt_mask = torch.ones_like(output.batch["prompts"])
            response_mask = response_ids["attention_mask"]
            output.batch["attention_mask"] = torch.cat([prompt_mask, response_mask], dim=1)

            # Info mask: only supervise response portion (instructions)
            prompt_info = torch.zeros_like(output.batch["prompts"])
            response_info = torch.ones_like(output.batch["responses"])
            output.batch["info_mask"] = torch.cat([prompt_info, response_info], dim=1)

        self.tokenizer.padding_side = "left"

        return output


class ARCTwoStageSeedPoolManager:
    """
    Manages multiple instruction candidates with seed pooling.

    Coordinates:
    - Running N instruction candidates in parallel
    - Scoring and selecting best trajectory
    """

    def __init__(
        self,
        interaction_manager: ARCTwoStageInteractionManager,
        num_candidates: int = 5,
        selection_strategy: str = "best_final"
    ):
        """
        Args:
            interaction_manager: The two-stage interaction manager
            num_candidates: Number of instruction candidates per task
            selection_strategy: How to select among candidates
                - "best_final": Use candidate with highest final score
                - "first_perfect": Use first candidate that achieves perfect score
        """
        self.interaction_manager = interaction_manager
        self.num_candidates = num_candidates
        self.selection_strategy = selection_strategy

        # Update config
        self.interaction_manager.two_stage_config.instruction_candidates = num_candidates

    def run_seed_pool(
        self,
        gen_batch: InteractionDataProto
    ) -> Tuple[InteractionDataProto, List[Dict]]:
        """
        Run the two-stage pipeline with candidate pooling.

        Args:
            gen_batch: Input batch with task configurations

        Returns:
            Tuple of (selected_outputs, all_candidate_info)
        """
        # Run the interaction manager (it handles candidates internally)
        outputs = self.interaction_manager.run_agent_loop(gen_batch)

        # Extract candidate info from results
        results = outputs.no_tensor_batch.get("results", [])
        all_candidate_info = []

        for i, result in enumerate(results):
            candidate_scores = [
                {
                    "candidate_idx": j,
                    "score": c.score,
                    "is_perfect": c.is_perfect
                }
                for j, c in enumerate(result.all_candidates)
            ]

            all_candidate_info.append({
                "task_idx": i,
                "task_id": result.task_id,
                "candidate_scores": candidate_scores,
                "best_score": result.best_score,
                "final_reward": result.final_reward
            })

        return outputs, all_candidate_info
