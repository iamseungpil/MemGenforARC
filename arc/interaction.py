"""
Code Generation Interaction Manager for ARC Training (BARC-style).

Single-stage architecture: Model generates Python code that transforms input grids.
The code is executed on training examples to compute reward.

Key Features:
- N code candidates per task for GRPO training
- Code execution on training examples for validation
- Accuracy-based reward computation
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
    build_code_generation_prompt,
    build_code_generation_messages,
    SYSTEM_PROMPT_CODER,
)
from arc.utils import (
    parse_code_from_text,
    validate_code_on_examples,
    execute_code_on_input,
    format_examples,
    grid_to_string,
)

logger = logging.getLogger(__name__)


@dataclass
class ARCCodeGenerationConfig:
    """Configuration for ARC code generation training.

    Single-stage architecture where model generates Python code.
    Code is executed to compute reward based on training example accuracy.
    """
    # Number of code candidates per task (for GRPO training)
    num_candidates: int = 5

    # Maximum length for code generation
    max_code_length: int = 2048

    # Temperature for code generation
    temperature: float = 0.8

    # Timeout for code execution (seconds per example)
    execution_timeout: int = 5

    # Whether to use top-p sampling
    top_p: float = 0.95


@dataclass
class CodeCandidate:
    """Represents a single code candidate with its validation results."""
    code: str
    accuracy: float = 0.0
    validation_results: Dict = field(default_factory=dict)
    memory_embeds: Optional[torch.Tensor] = None
    is_perfect: bool = False
    raw_response: str = ""


@dataclass
class CodeGenerationResult:
    """Result of code generation for a single task."""
    task_id: str
    best_code: str
    best_accuracy: float
    all_candidates: List[CodeCandidate]
    memory_embeds: Optional[torch.Tensor]


class ARCCodeGenerationManager(MultiTurnInteractionManager):
    """
    Code generation interaction manager for ARC training (BARC-style).

    Single-stage architecture:
    1. Generate N code candidates for each task
    2. Execute each candidate on training examples
    3. Compute accuracy as reward
    4. Use candidates for GRPO training

    The model generates Python `main(input_grid)` functions that should
    correctly transform input grids to output grids for all training examples.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
        code_gen_config: Optional[ARCCodeGenerationConfig] = None,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.code_gen_config = code_gen_config or ARCCodeGenerationConfig()

        # Generation config for code generation
        self.gen_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=self.code_gen_config.max_code_length,
            temperature=self.code_gen_config.temperature,
            top_p=self.code_gen_config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """
        Run the code generation loop.

        For each task in the batch:
        1. Generate N code candidates
        2. Execute each candidate on training examples
        3. Compute accuracy as reward

        Args:
            gen_batch: Input batch with task configurations

        Returns:
            InteractionDataProto with generation results
        """
        assert "envs" in gen_batch.no_tensor_batch
        envs = gen_batch.no_tensor_batch["envs"]

        all_results: List[CodeGenerationResult] = []

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
                env.best_score = result.best_accuracy
            if hasattr(env, 'best_code'):
                env.best_code = result.best_code

        # Build output
        return self._build_output(gen_batch, all_results)

    def _process_single_task(
        self,
        task_config: Dict,
        env: Any
    ) -> CodeGenerationResult:
        """
        Process a single ARC task through code generation.

        Args:
            task_config: Task configuration with train_examples
            env: Environment instance for this task

        Returns:
            CodeGenerationResult with best code and accuracy
        """
        train_examples = task_config.get("train_examples", [])
        task_id = task_config.get("task_id", "unknown")

        logger.debug(f"Processing task {task_id} with {len(train_examples)} examples")

        # Generate code candidates
        candidates = self._generate_code_candidates(
            train_examples=train_examples,
            num_candidates=self.code_gen_config.num_candidates
        )

        if not candidates:
            logger.warning(f"Task {task_id}: No valid code candidates generated")
            return CodeGenerationResult(
                task_id=task_id,
                best_code="",
                best_accuracy=0.0,
                all_candidates=[],
                memory_embeds=None
            )

        # Validate each candidate on training examples
        for candidate in candidates:
            self._validate_candidate(candidate, train_examples)

        # Sort by accuracy and select best
        candidates.sort(key=lambda c: c.accuracy, reverse=True)
        best_candidate = candidates[0]

        logger.info(f"Task {task_id}: Best accuracy = {best_candidate.accuracy:.2%}")

        return CodeGenerationResult(
            task_id=task_id,
            best_code=best_candidate.code,
            best_accuracy=best_candidate.accuracy,
            all_candidates=candidates,
            memory_embeds=best_candidate.memory_embeds
        )

    def _generate_code_candidates(
        self,
        train_examples: List[Dict],
        num_candidates: int
    ) -> List[CodeCandidate]:
        """
        Generate N code candidates for a task.

        Args:
            train_examples: Training examples for the task
            num_candidates: Number of candidates to generate

        Returns:
            List of CodeCandidate objects
        """
        candidates = []

        # Build prompt
        messages = build_code_generation_messages(train_examples)

        # Get model device
        device = next(self.actor_rollout_wg.parameters()).device

        max_retries = 3

        for i in range(num_candidates):
            candidate = None
            memory_embeds = None  # Initialize to avoid NameError if all retries fail
            response_text = ""    # Initialize to avoid NameError if all retries fail

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
                    generation_config=self.gen_config,
                    return_memory_embeds=True
                )

                # Extract response
                prompt_len = inputs["input_ids"].size(1)
                response_ids = gen_output[:, prompt_len:]
                response_text = self.tokenizer.decode(
                    response_ids[0], skip_special_tokens=True
                )

                # Parse code from response
                code = parse_code_from_text(response_text)

                if code:
                    candidate = CodeCandidate(
                        code=code,
                        memory_embeds=memory_embeds[0].detach() if memory_embeds is not None else None,
                        raw_response=response_text
                    )
                    logger.debug(f"Generated candidate {i+1}/{num_candidates}")
                    break
                else:
                    logger.warning(
                        f"Failed to parse code from candidate {i+1}, "
                        f"retry {retry+1}/{max_retries}"
                    )

            # If all retries failed, create a placeholder candidate
            if candidate is None:
                logger.warning(f"All retries failed for candidate {i+1}")
                candidate = CodeCandidate(
                    code="def main(input_grid):\n    return input_grid",
                    memory_embeds=memory_embeds[0].detach() if memory_embeds is not None else None,
                    raw_response=response_text if response_text else ""
                )

            candidates.append(candidate)

        return candidates

    def _validate_candidate(
        self,
        candidate: CodeCandidate,
        train_examples: List[Dict]
    ) -> None:
        """
        Validate a code candidate on training examples.

        Updates the candidate in-place with accuracy and validation results.

        Args:
            candidate: The code candidate to validate
            train_examples: Training examples for validation
        """
        accuracy, results = validate_code_on_examples(
            code=candidate.code,
            train_examples=train_examples,
            timeout_seconds=self.code_gen_config.execution_timeout
        )

        candidate.accuracy = accuracy
        candidate.validation_results = results
        candidate.is_perfect = accuracy == 1.0

    def _build_output(
        self,
        gen_batch: InteractionDataProto,
        results: List[CodeGenerationResult]
    ) -> InteractionDataProto:
        """
        Build the output InteractionDataProto from results.

        Returns ALL candidates for GRPO training.
        Each candidate becomes a separate sample, allowing GRPO to compute
        advantages across candidates within the same task.

        Args:
            gen_batch: Original input batch
            results: List of CodeGenerationResult for each task

        Returns:
            InteractionDataProto with all candidates as separate samples
        """
        output = InteractionDataProto()

        # Store results in no_tensor_batch
        output.no_tensor_batch["results"] = results
        output.no_tensor_batch["envs"] = gen_batch.no_tensor_batch.get("envs", [])

        # Build outputs for ALL candidates
        envs = gen_batch.no_tensor_batch.get("envs", [])
        prompts = []
        responses = []
        candidate_scores = []

        for result, env in zip(results, envs):
            # Get task config for examples
            task_config = getattr(env, 'task_config', {})
            train_examples = task_config.get("train_examples", [])

            # Build prompt messages
            prompt_messages = build_code_generation_messages(train_examples)

            # Add ALL candidates as separate samples
            for candidate in result.all_candidates:
                prompts.append(prompt_messages)
                # Response is the raw model output (includes code block)
                responses.append(candidate.raw_response or f"```python\n{candidate.code}\n```")
                candidate_scores.append(candidate.accuracy)

        # Validate all tasks have same candidate count
        expected_candidates = self.code_gen_config.num_candidates
        for i, result in enumerate(results):
            if len(result.all_candidates) != expected_candidates:
                logger.error(
                    f"Task {i} has {len(result.all_candidates)} candidates, "
                    f"expected {expected_candidates}"
                )

        # Store candidate scores for trainer to use as rewards
        output.no_tensor_batch["candidate_scores"] = candidate_scores
        output.no_tensor_batch["num_candidates_per_task"] = expected_candidates

        # Build interaction histories
        inter_histories = []
        for result in results:
            for candidate in result.all_candidates:
                history = [
                    {"role": "assistant", "content": f"```python\n{candidate.code}\n```"},
                    {"role": "user", "content": f"Accuracy: {candidate.accuracy:.2%}"},
                ]
                inter_histories.append(history)
        output.no_tensor_batch["inter_histories"] = inter_histories

        if prompts:
            self.tokenizer.padding_side = "left"
            prompt_ids = self.tokenizer.apply_chat_template(
                prompts,
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

            # Info mask: only supervise response portion (code)
            prompt_info = torch.zeros_like(output.batch["prompts"])
            response_info = torch.ones_like(output.batch["responses"])
            output.batch["info_mask"] = torch.cat([prompt_info, response_info], dim=1)

        self.tokenizer.padding_side = "left"

        return output


class ARCCodeGenerationPoolManager:
    """
    Manages multiple code candidates with seed pooling.

    Coordinates:
    - Running N code candidates in parallel
    - Scoring and selecting best trajectory
    """

    def __init__(
        self,
        interaction_manager: ARCCodeGenerationManager,
        num_candidates: int = 5,
        selection_strategy: str = "best_accuracy"
    ):
        """
        Args:
            interaction_manager: The code generation interaction manager
            num_candidates: Number of code candidates per task
            selection_strategy: How to select among candidates
                - "best_accuracy": Use candidate with highest accuracy
                - "first_perfect": Use first candidate that achieves 100% accuracy
        """
        self.interaction_manager = interaction_manager
        self.num_candidates = num_candidates
        self.selection_strategy = selection_strategy

        # Update config
        self.interaction_manager.code_gen_config.num_candidates = num_candidates

    def run_seed_pool(
        self,
        gen_batch: InteractionDataProto
    ) -> Tuple[InteractionDataProto, List[Dict]]:
        """
        Run the code generation pipeline with candidate pooling.

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
                    "accuracy": c.accuracy,
                    "is_perfect": c.is_perfect,
                    "passed_examples": c.validation_results.get("passed_examples", 0),
                    "total_examples": c.validation_results.get("total_examples", 0),
                }
                for j, c in enumerate(result.all_candidates)
            ]

            all_candidate_info.append({
                "task_idx": i,
                "task_id": result.task_id,
                "candidate_scores": candidate_scores,
                "best_accuracy": result.best_accuracy
            })

        return outputs, all_candidate_info
