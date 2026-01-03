"""
ARC Environment with Code Execution Reward.

BARC-style approach: Model generates Python code, reward is computed
by executing the code on training examples and measuring accuracy.
"""
import json
import logging
import re
from typing import Optional, Dict, Tuple, List
from data.base_env import StaticEnv, DynamicEnv

# Import code execution utilities from arc module
from arc.utils import (
    parse_code_from_text,
    validate_code_on_examples,
    execute_code_on_input,
    validate_grid,
    get_grid_similarity,
    grid_to_string,
    generate_grid_diff,
)

logger = logging.getLogger(__name__)


class ARCEnv(StaticEnv):
    """
    ARC Environment for GRPO training with code execution reward.

    Reward is computed by:
    1. Parsing Python code from model output
    2. Executing code on all training examples
    3. Computing accuracy (fraction of examples where code produces correct output)
    """
    ENV_CARD = "STATIC"

    def __init__(self, config):
        super().__init__(config)
        self.execution_timeout = config.get("execution_timeout", 5)

    @classmethod
    def compute_reward(
        cls,
        prompts: list[str],  # Required by TRL GRPOTrainer
        completions: list[str],
        train_examples: list[str] = None,  # JSON string of training examples
        solution: list[str] = None,  # Legacy: ground truth grids (not used for code)
        **kwargs
    ) -> list[float]:
        """
        Compute reward for ARC task completions using code execution.

        Args:
            prompts: Input prompts (required by TRL, not used in reward calculation)
            completions: Model outputs (should contain Python code)
            train_examples: Training examples as JSON strings for code validation
            solution: Legacy parameter (not used for code execution reward)

        Returns:
            List of rewards (0.0 to 1.0 based on training example accuracy)
        """
        rewards = []
        task_id = kwargs.get("task_id", "unknown")
        execution_timeout = kwargs.get("execution_timeout", 5)

        # Handle case where train_examples might be passed differently
        if train_examples is None:
            train_examples = kwargs.get("train_examples", [])

        for idx, completion in enumerate(completions):
            # Get training examples for this sample
            if isinstance(train_examples, list) and idx < len(train_examples):
                examples = train_examples[idx]
            else:
                examples = train_examples

            # Parse training examples if they're JSON strings
            if isinstance(examples, str):
                try:
                    examples = json.loads(examples)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"[{task_id}][{idx}] Failed to parse train_examples JSON")
                    rewards.append(0.0)
                    continue

            if not examples:
                logger.warning(f"[{task_id}][{idx}] No training examples provided")
                rewards.append(0.0)
                continue

            # Parse code from completion
            code = parse_code_from_text(completion)

            if code is None:
                logger.debug(f"[{task_id}][{idx}] Failed to parse code from: {completion[:200]}...")
                rewards.append(0.0)
                continue

            # Validate code on training examples
            accuracy, results = validate_code_on_examples(
                code=code,
                train_examples=examples,
                timeout_seconds=execution_timeout
            )

            if accuracy == 1.0:
                logger.info(f"[{task_id}][{idx}] Perfect accuracy! reward=1.0")
            elif accuracy > 0:
                passed = results.get('passed_examples', 0)
                total = results.get('total_examples', 0)
                logger.debug(f"[{task_id}][{idx}] Partial accuracy: {passed}/{total} = {accuracy:.3f}")

            rewards.append(accuracy)

        return rewards


class ARCCodeEnv(StaticEnv):
    """
    ARC Code Generation Environment with execution-based reward.

    This is the primary environment for code generation training.
    Reward = accuracy on training examples (0.0 to 1.0).
    """
    ENV_CARD = "STATIC"

    def __init__(self, config):
        super().__init__(config)
        self.execution_timeout = config.get("execution_timeout", 5)
        self.task_config = None

    def set_task(self, task_config: Dict) -> None:
        """Set the task configuration."""
        self.task_config = task_config

    @classmethod
    def compute_reward(
        cls,
        prompts: list[str],
        completions: list[str],
        train_examples_list: list = None,
        **kwargs
    ) -> list[float]:
        """
        Compute reward based on code execution accuracy.

        Args:
            prompts: Input prompts
            completions: Model outputs containing Python code
            train_examples_list: List of training examples per sample

        Returns:
            List of accuracy scores (0.0 to 1.0)
        """
        rewards = []
        execution_timeout = kwargs.get("execution_timeout", 5)

        for idx, completion in enumerate(completions):
            # Get training examples
            if train_examples_list and idx < len(train_examples_list):
                train_examples = train_examples_list[idx]
            else:
                train_examples = kwargs.get("train_examples", [])

            # Parse JSON if needed
            if isinstance(train_examples, str):
                try:
                    train_examples = json.loads(train_examples)
                except (json.JSONDecodeError, TypeError):
                    rewards.append(0.0)
                    continue

            if not train_examples:
                rewards.append(0.0)
                continue

            # Parse code
            code = parse_code_from_text(completion)
            if code is None:
                rewards.append(0.0)
                continue

            # Execute and compute accuracy
            accuracy, _ = validate_code_on_examples(
                code=code,
                train_examples=train_examples,
                timeout_seconds=execution_timeout
            )

            rewards.append(accuracy)

        return rewards


class ARCDynamicEnv(DynamicEnv):
    """
    Multi-turn ARC Environment for iterative code refinement.

    Supports multi-turn interaction where:
    - Each turn: model generates code, receives feedback on execution results
    - Feedback shows which examples passed/failed
    - Model can refine code based on feedback
    """
    ENV_CARD = "DYNAMIC"

    # Diverse seed prompt strategies for code generation
    SEED_STRATEGIES = [
        {
            "name": "analytical",
            "instruction": "Carefully analyze the transformation pattern step by step before writing code.",
        },
        {
            "name": "pattern_focused",
            "instruction": "Focus on identifying visual patterns - look at colors, shapes, and spatial relationships.",
        },
        {
            "name": "numpy_first",
            "instruction": "Consider using numpy operations for efficient grid manipulation.",
        },
        {
            "name": "object_oriented",
            "instruction": "Think about the grid as containing objects. Identify and track how objects transform.",
        },
        {
            "name": "iterative",
            "instruction": "Start with a simple solution and iteratively improve it based on test results.",
        },
    ]

    def __init__(self, config):
        super().__init__(config)
        self.task_config = None
        self.train_examples = None
        self.test_input = None
        self.target_grid = None
        self.task_id = None

        # Multi-turn state
        self.current_turn = 0
        self.max_turns = config.get("max_turns", 7)
        self.best_code = None
        self.best_accuracy = 0.0
        self.turn_history: List[Dict] = []
        self.execution_timeout = config.get("execution_timeout", 5)

        # Seed diversity support
        self.seed_id: int = 0
        self.seed_strategy: Optional[Dict] = None

    def set_env(self, task_config: Dict) -> Tuple[str, str]:
        """
        Initialize environment with ARC task.

        Args:
            task_config: Dict containing:
                - train_examples: List of {input, output} training examples
                - test_input: Input grid to solve (optional)
                - target_grid: Expected output grid (optional)
                - task_id: Task identifier

        Returns:
            Tuple of (system_prompt, initial_user_prompt)
        """
        self.task_config = task_config
        self.train_examples = task_config.get("train_examples", [])
        self.test_input = task_config.get("test_input")
        self.target_grid = task_config.get("target_grid")
        self.task_id = task_config.get("task_id", "unknown")

        # Reset state
        self.current_turn = 0
        self.best_code = None
        self.best_accuracy = 0.0
        self.turn_history = []

        # Set seed strategy based on seed_id
        if self.seed_id < len(self.SEED_STRATEGIES):
            self.seed_strategy = self.SEED_STRATEGIES[self.seed_id]
        else:
            self.seed_strategy = self.SEED_STRATEGIES[self.seed_id % len(self.SEED_STRATEGIES)]

        # Format training examples
        examples_text = self._format_examples(self.train_examples)

        # Build seed-specific system prompt
        base_system = """You are an expert Python programmer solving ARC puzzles.
Write a function `main(input_grid)` that transforms input grids to output grids.
Your code must work correctly for ALL training examples.

Available utilities:
- numpy as np
- Color class: BLACK=0, BLUE=1, RED=2, GREEN=3, YELLOW=4, GREY=5, MAGENTA=6, ORANGE=7, SKY=8, BROWN=9
- crop(grid), bounding_box(grid), find_connected_components(grid)

Output your code in a Python code block:
```python
def main(input_grid):
    # Your implementation
    return output_grid
```"""

        if self.seed_strategy:
            strategy_instruction = self.seed_strategy["instruction"]
            system_prompt = f"{base_system}\n\nHint: {strategy_instruction}"
        else:
            system_prompt = base_system

        user_prompt = f"""Analyze these training examples and write Python code to implement the transformation:

{examples_text}

Write a `main(input_grid)` function that works for all examples."""

        return system_prompt, user_prompt

    def set_seed(self, seed_id: int) -> None:
        """Set the seed ID for diverse prompting strategy."""
        self.seed_id = seed_id
        if seed_id < len(self.SEED_STRATEGIES):
            self.seed_strategy = self.SEED_STRATEGIES[seed_id]
        else:
            self.seed_strategy = self.SEED_STRATEGIES[seed_id % len(self.SEED_STRATEGIES)]

    def _format_examples(self, examples: List[Dict]) -> str:
        """Format training examples for prompt."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            input_grid = grid_to_string(ex["input"])
            output_grid = grid_to_string(ex["output"])
            rows_in = len(ex["input"])
            cols_in = len(ex["input"][0]) if ex["input"] else 0
            rows_out = len(ex["output"])
            cols_out = len(ex["output"][0]) if ex["output"] else 0
            formatted.append(
                f"Example {i}:\n"
                f"Input ({rows_in}x{cols_in}):\n{input_grid}\n\n"
                f"Output ({rows_out}x{cols_out}):\n{output_grid}"
            )
        return "\n\n---\n\n".join(formatted)

    @classmethod
    def preprocess_action(cls, action: str) -> str:
        """Preprocess model output before evaluation."""
        return action.strip()

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Process model's code and return feedback.

        Args:
            action: Model output (should contain Python code)

        Returns:
            Tuple of (observation/feedback, reward, done)
        """
        self.current_turn += 1

        # Parse code from action
        code = parse_code_from_text(action)

        if code is None:
            observation = """Failed to parse Python code from your output.
Please output code in a Python code block:
```python
def main(input_grid):
    # Your implementation
    return output_grid
```

Try again with the correct format."""
            reward = 0.0
            done = self.current_turn >= self.max_turns

            self.turn_history.append({
                "turn": self.current_turn,
                "code": None,
                "accuracy": 0.0,
                "error": "parse_failed"
            })

            return observation, reward, done

        # Validate code on training examples
        accuracy, results = validate_code_on_examples(
            code=code,
            train_examples=self.train_examples,
            timeout_seconds=self.execution_timeout
        )

        # Track best result
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_code = code

        self.turn_history.append({
            "turn": self.current_turn,
            "code": code,
            "accuracy": accuracy,
            "results": results
        })

        # Check if solved
        if accuracy == 1.0:
            observation = f"Perfect! Your code works correctly on all {len(self.train_examples)} training examples."
            return observation, accuracy, True

        # Check if max turns reached
        if self.current_turn >= self.max_turns:
            passed = results.get('passed_examples', 0)
            total = results.get('total_examples', 0)
            observation = f"Maximum turns ({self.max_turns}) reached. Best accuracy: {self.best_accuracy:.1%} ({passed}/{total} examples)"
            return observation, accuracy, True

        # Generate feedback
        passed = results.get('passed_examples', 0)
        total = results.get('total_examples', 0)
        failed = results.get('failed_examples', [])
        errors = results.get('execution_errors', [])

        feedback_parts = [
            f"Turn {self.current_turn}/{self.max_turns} - Accuracy: {accuracy:.1%} ({passed}/{total} examples passed)"
        ]

        # Show execution errors
        if errors:
            feedback_parts.append("\nExecution Errors:")
            for err in errors[:3]:  # Limit to first 3
                feedback_parts.append(f"  Example {err['idx'] + 1}: {err['error'][:100]}")

        # Show failed examples with diffs
        if failed:
            feedback_parts.append("\nFailed Examples (showing diff):")
            for fail in failed[:2]:  # Limit to first 2
                idx = fail['idx']
                expected = fail.get('expected', [])
                actual = fail.get('actual', [])
                if expected and actual:
                    diff = generate_grid_diff(expected, actual)
                    feedback_parts.append(f"\n  Example {idx + 1}:\n{diff}")
                else:
                    feedback_parts.append(f"\n  Example {idx + 1}: Output format error")

        feedback_parts.append("\nPlease revise your code to fix these issues.")

        observation = "\n".join(feedback_parts)
        return observation, accuracy, False

    def feedback(self) -> Tuple[float, bool]:
        """
        Return final reward and success status.

        Returns:
            Tuple of (best_accuracy, solved)
        """
        solved = self.best_accuracy == 1.0
        return self.best_accuracy, solved

    @classmethod
    def compute_reward(
        cls,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute reward for multi-turn completions.

        For DynamicEnv, this is called with the final completions after all turns.
        """
        # Get environments from kwargs
        envs: List['ARCDynamicEnv'] = kwargs.get("envs", [])

        if envs:
            # Use best accuracy from each environment
            return [env.best_accuracy for env in envs]

        # Fallback to static reward computation
        train_examples = kwargs.get("train_examples", [])
        return ARCEnv.compute_reward(
            prompts, completions,
            train_examples=train_examples,
            **kwargs
        )


# Legacy grid parsing functions for backward compatibility
def parse_grid_from_text(text: str) -> Optional[list[list[int]]]:
    """
    Parse a 2D grid from model output text.
    Legacy function - use arc.utils.parse_grid_from_text for new code.
    """
    from arc.utils import parse_grid_from_text as _parse_grid
    return _parse_grid(text)
