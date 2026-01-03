"""
ARC Dataset Builder for Code Generation Training.

Loads ARC challenges and formats them for code generation training
where models learn to write Python code that transforms input grids.
"""
import json
from pathlib import Path
from typing import Dict, List
from datasets import Dataset, DatasetDict

from data.base_builder import BaseBuilder
from data.arc.env import ARCEnv, ARCCodeEnv, ARCDynamicEnv

# Import prompts from arc module
from arc.prompts import (
    build_code_generation_prompt,
    SYSTEM_PROMPT_CODER,
    format_examples,
)


# Code generation prompt template
CODE_GENERATION_PROMPT_TEMPLATE = """Analyze the training examples below and write a Python function that transforms input grids to output grids.

{examples}

Write a Python function `main(input_grid)` that implements the transformation pattern.
The function should work correctly for ALL training examples above.

```python
def main(input_grid):
    # Your implementation
    return output_grid
```"""


def grid_to_string(grid: List[List[int]]) -> str:
    """Convert a 2D grid to readable string format."""
    return "\n".join([" ".join(map(str, row)) for row in grid])


def format_training_examples(train_examples: List[Dict]) -> str:
    """Format training examples for prompt."""
    formatted = []
    for i, ex in enumerate(train_examples, 1):
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


class ARCBuilder(BaseBuilder):
    """Builder for ARC dataset with code generation support."""

    # Default paths - can be overridden in config
    DEFAULT_DATA_PATH = "/home/ubuntu/arc-lang-public/data/arc-prize-2024"

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        # Store full config for access to top-level params
        self.full_config = cfg if cfg else {}

    def get_env_cls(self):
        """
        Return the appropriate environment class based on configuration.

        Uses ARCDynamicEnv for multi-turn training,
        otherwise uses ARCEnv (or ARCCodeEnv) for single-turn training.
        """
        max_turns = self.full_config.get("max_turns", 1)
        num_seeds = self.full_config.get("num_seeds", 1)

        # Use dynamic env for multi-turn or multi-seed training
        if max_turns > 1 or num_seeds > 1:
            return ARCDynamicEnv
        return ARCEnv

    def _load_arc_data(self) -> tuple[Dict, Dict]:
        """Load ARC challenges and solutions from JSON files."""
        data_path = Path(self.full_config.get("data_path", self.DEFAULT_DATA_PATH))

        # Load training data
        challenges_file = data_path / "arc-agi_training_challenges.json"
        solutions_file = data_path / "arc-agi_training_solutions.json"

        with open(challenges_file) as f:
            challenges = json.load(f)

        with open(solutions_file) as f:
            solutions = json.load(f)

        return challenges, solutions

    def _create_code_generation_examples(
        self,
        challenges: Dict,
        solutions: Dict,
    ) -> List[Dict]:
        """
        Create code generation training examples from ARC data.

        Each example contains:
        - prompt: Code generation prompt with training examples
        - train_examples: JSON string of training examples for reward computation
        - task_id: Task identifier
        """
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]

            if len(train_examples) < 2:
                continue

            # Format prompt
            examples_text = format_training_examples(train_examples)
            prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(examples=examples_text)

            examples.append({
                "prompt": prompt,
                "train_examples": json.dumps(train_examples),
                "task_id": task_id,
                "num_examples": len(train_examples),
            })

        return examples

    def _create_multiturn_examples(
        self,
        challenges: Dict,
        solutions: Dict,
    ) -> List[Dict]:
        """
        Create examples for multi-turn training (ARCDynamicEnv).

        Returns task configs with train_examples for code generation.
        """
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]
            test_inputs = task.get("test", [])
            task_solutions = solutions.get(task_id, [])

            if len(train_examples) < 2:
                continue

            # Create one example per task
            example = {
                "train_examples": train_examples,
                "task_id": task_id,
                "num_examples": len(train_examples),
            }

            # Add test info if available
            if test_inputs and task_solutions:
                example["test_input"] = test_inputs[0]["input"]
                example["target_grid"] = task_solutions[0]

            examples.append(example)

        return examples

    def _create_test_examples(
        self,
        challenges: Dict,
        solutions: Dict
    ) -> List[Dict]:
        """Create test examples for evaluation."""
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]
            test_inputs = task.get("test", [])
            task_solutions = solutions.get(task_id, [])

            if len(train_examples) < 2:
                continue

            for i, (test_input, solution) in enumerate(zip(test_inputs, task_solutions)):
                # Format prompt with training examples
                examples_text = format_training_examples(train_examples)
                prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(examples=examples_text)

                examples.append({
                    "prompt": prompt,
                    "train_examples": json.dumps(train_examples),
                    "test_input": json.dumps(test_input["input"]),
                    "target_grid": json.dumps(solution),
                    "task_id": f"{task_id}_test{i}",
                    "test_idx": i,
                })

        return examples

    def _build_datasets(self) -> DatasetDict:
        """Build train/valid/test datasets for code generation."""
        challenges, solutions = self._load_arc_data()

        # Check if multi-turn mode
        max_turns = self.full_config.get("max_turns", 1)
        num_seeds = self.full_config.get("num_seeds", 1)
        use_multiturn = max_turns > 1 or num_seeds > 1

        if use_multiturn:
            # Create multi-turn examples for ARCDynamicEnv
            all_train_examples = self._create_multiturn_examples(challenges, solutions)
        else:
            # Create single-turn examples for code generation
            all_train_examples = self._create_code_generation_examples(challenges, solutions)

        test_examples = self._create_test_examples(challenges, solutions)

        # Split into train/valid
        val_ratio = self.config.get("val_ratio", 0.1)
        val_size = int(len(all_train_examples) * val_ratio)

        # Shuffle deterministically
        import random
        random.seed(42)
        random.shuffle(all_train_examples)

        train_examples = all_train_examples[val_size:]
        valid_examples = all_train_examples[:val_size]

        # Build datasets
        train_dataset = Dataset.from_list(train_examples)
        valid_dataset = Dataset.from_list(valid_examples)
        test_dataset = Dataset.from_list(test_examples)

        print(f"[ARCBuilder] Created datasets: train={len(train_examples)}, valid={len(valid_examples)}, test={len(test_examples)}")

        return DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
        })

    def _build_sft_datasets(self) -> DatasetDict:
        """
        Build SFT datasets for code generation.

        Can load pre-generated code solutions if available.
        """
        # Check for pre-generated code SFT data
        sft_code_path = Path(__file__).parent / "sft_code_solutions.json"

        if sft_code_path.exists():
            # Load pre-generated code solutions
            with open(sft_code_path) as f:
                data = json.load(f)

            print(f"[ARCBuilder] Loaded {len(data)} code solution examples from {sft_code_path.name}")

            # Split into train/valid
            sft_config = self.full_config.get("sft", {})
            val_ratio = sft_config.get("val_ratio", 0.1) if sft_config else 0.1
            if self.config:
                val_ratio = self.config.get("val_ratio", val_ratio)
            val_size = int(len(data) * val_ratio)

            import random
            random.seed(42)
            random.shuffle(data)

            train_examples = data[val_size:]
            valid_examples = data[:val_size]

            train_dataset = Dataset.from_list(train_examples)
            valid_dataset = Dataset.from_list(valid_examples)
            test_dataset = Dataset.from_list([])

            return DatasetDict({
                "train": train_dataset,
                "valid": valid_dataset,
                "test": test_dataset,
            })
        else:
            # Default: use code generation data
            return self._build_datasets()

    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()

    @classmethod
    def _preprocess(cls, example: Dict) -> Dict:
        """Preprocess is handled in _create_* methods."""
        return example

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "train_examples", "task_id", "test_input", "target_grid"]
