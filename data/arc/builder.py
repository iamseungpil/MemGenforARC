"""
ARC Dataset Builder for MemGen GRPO Training.

Loads ARC challenges and formats them for training with grid similarity reward.
"""
import json
from pathlib import Path
from typing import Dict, List
from datasets import Dataset, DatasetDict

from data.base_builder import BaseBuilder
from data.arc.env import ARCEnv, ARCDynamicEnv


# Prompt templates matching arc-lang format
INSTRUCTION_PROMPT = """You are an expert puzzle solver. Find the pattern that transforms input grids to output grids.

Training Examples:
{examples}

Test Input:
{test_input}

Write step-by-step instructions describing the transformation pattern. Output as JSON:
{{"instructions": "your step-by-step instructions here"}}"""

GRID_GENERATION_PROMPT = """You are an expert puzzle solver. Apply the transformation pattern to generate the output grid.

Instructions: {instructions}

Training Examples:
{examples}

Test Input:
{test_input}

Generate the output grid. Output as JSON:
{{"grid": [[...]]}}"""

# Simplified prompt for direct grid generation (used for GRPO training)
DIRECT_GRID_PROMPT = """You are an expert at solving ARC puzzles. Analyze the pattern in the training examples and generate the output grid for the test input.

Training Examples:
{examples}

Test Input:
{test_input}

Output the grid as JSON: {{"grid": [[...]]}}"""


def grid_to_string(grid: List[List[int]]) -> str:
    """Convert a 2D grid to readable string format."""
    return "\n".join([" ".join(map(str, row)) for row in grid])


def format_examples(train_examples: List[Dict]) -> str:
    """Format training examples for prompt."""
    formatted = []
    for i, ex in enumerate(train_examples, 1):
        input_grid = grid_to_string(ex["input"])
        output_grid = grid_to_string(ex["output"])
        formatted.append(f"Example {i}:\nInput:\n{input_grid}\nOutput:\n{output_grid}")
    return "\n\n".join(formatted)


class ARCBuilder(BaseBuilder):
    """Builder for ARC dataset."""

    # Default paths - can be overridden in config
    DEFAULT_DATA_PATH = "/home/ubuntu/arc-lang-public/data/arc-prize-2024"

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        # Store full config for access to top-level params (max_turns, num_seeds, data_path)
        self.full_config = cfg if cfg else {}

    def get_env_cls(self):
        """
        Return the appropriate environment class based on configuration.

        Uses ARCDynamicEnv for multi-turn training (when max_turns > 1 or num_seeds > 1),
        otherwise uses ARCEnv for single-turn training.
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

    def _create_training_examples(
        self,
        challenges: Dict,
        solutions: Dict,
        use_leave_one_out: bool = True
    ) -> List[Dict]:
        """
        Create training examples from ARC data.

        For each task, we create examples using leave-one-out:
        - Use N-1 training examples as context
        - Predict the Nth training example's output
        This allows us to compute rewards during training.
        """
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]

            if use_leave_one_out and len(train_examples) > 1:
                # Leave-one-out: for each training example, use others as context
                for i, target_example in enumerate(train_examples):
                    context_examples = train_examples[:i] + train_examples[i+1:]

                    prompt = DIRECT_GRID_PROMPT.format(
                        examples=format_examples(context_examples),
                        test_input=grid_to_string(target_example["input"])
                    )

                    examples.append({
                        "prompt": prompt,
                        "completion": json.dumps({"grid": target_example["output"]}),
                        "solution": json.dumps(target_example["output"]),
                        "task_id": task_id,
                        "example_idx": i,
                    })
            else:
                # Use all training examples as context, predict first example
                if len(train_examples) >= 2:
                    target = train_examples[0]
                    context = train_examples[1:]

                    prompt = DIRECT_GRID_PROMPT.format(
                        examples=format_examples(context),
                        test_input=grid_to_string(target["input"])
                    )

                    examples.append({
                        "prompt": prompt,
                        "completion": json.dumps({"grid": target["output"]}),
                        "solution": json.dumps(target["output"]),
                        "task_id": task_id,
                        "example_idx": 0,
                    })

        return examples

    def _create_test_examples(self, challenges: Dict, solutions: Dict) -> List[Dict]:
        """Create test examples using actual test grids."""
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]
            test_inputs = task["test"]
            task_solutions = solutions.get(task_id, [])

            for i, (test_input, solution) in enumerate(zip(test_inputs, task_solutions)):
                prompt = DIRECT_GRID_PROMPT.format(
                    examples=format_examples(train_examples),
                    test_input=grid_to_string(test_input["input"])
                )

                examples.append({
                    "prompt": prompt,
                    "completion": json.dumps({"grid": solution}),
                    "solution": json.dumps(solution),
                    "task_id": task_id,
                    "test_idx": i,
                })

        return examples

    def _create_multiturn_examples(
        self,
        challenges: Dict,
        solutions: Dict,
        use_leave_one_out: bool = True
    ) -> List[Dict]:
        """
        Create examples for multi-turn training (ARCDynamicEnv).

        Returns task configs with train_examples, test_input, and target_grid
        that ARCDynamicEnv.set_env() expects.
        """
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]

            if use_leave_one_out and len(train_examples) > 1:
                # Leave-one-out: for each training example, use others as context
                for i, target_example in enumerate(train_examples):
                    context_examples = train_examples[:i] + train_examples[i+1:]

                    examples.append({
                        "train_examples": context_examples,
                        "test_input": target_example["input"],
                        "target_grid": target_example["output"],
                        "task_id": f"{task_id}_loo{i}",
                        "example_idx": i,
                    })
            else:
                # Use all training examples as context, predict first example
                if len(train_examples) >= 2:
                    target = train_examples[0]
                    context = train_examples[1:]

                    examples.append({
                        "train_examples": context,
                        "test_input": target["input"],
                        "target_grid": target["output"],
                        "task_id": task_id,
                        "example_idx": 0,
                    })

        return examples

    def _create_multiturn_test_examples(
        self,
        challenges: Dict,
        solutions: Dict
    ) -> List[Dict]:
        """Create test examples for multi-turn evaluation."""
        examples = []

        for task_id, task in challenges.items():
            train_examples = task["train"]
            test_inputs = task["test"]
            task_solutions = solutions.get(task_id, [])

            for i, (test_input, solution) in enumerate(zip(test_inputs, task_solutions)):
                examples.append({
                    "train_examples": train_examples,
                    "test_input": test_input["input"],
                    "target_grid": solution,
                    "task_id": f"{task_id}_test{i}",
                    "test_idx": i,
                })

        return examples

    def _build_datasets(self) -> DatasetDict:
        """Build train/valid/test datasets."""
        challenges, solutions = self._load_arc_data()

        # Check if multi-turn mode (using full_config for top-level params)
        max_turns = self.full_config.get("max_turns", 1)
        num_seeds = self.full_config.get("num_seeds", 1)
        use_multiturn = max_turns > 1 or num_seeds > 1

        if use_multiturn:
            # Create multi-turn examples for ARCDynamicEnv
            all_train_examples = self._create_multiturn_examples(
                challenges, solutions, use_leave_one_out=True
            )
            test_examples = self._create_multiturn_test_examples(challenges, solutions)
        else:
            # Create single-turn examples for ARCEnv
            all_train_examples = self._create_training_examples(
                challenges, solutions, use_leave_one_out=True
            )
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

        return DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
        })

    def _load_instruction_sft_data(self) -> List[Dict]:
        """
        Load pre-generated instruction SFT data in messages format.

        This data contains high-quality instructions for ARC puzzle solving,
        used for pre-training the Weaver to generate instructions.

        Priority order:
        1. sft_soar2cot.json - Training set data, 7185 examples, score=1.0 (GRPO compatible)
        2. sft_instructions_high_quality.json - Evaluation set, score >= 0.9
        3. sft_instructions_messages.json - Evaluation set, messages format
        4. sft_instructions.json - Legacy format

        Data format:
        {
            "messages": [
                {"role": "user", "content": "<prompt>"},
                {"role": "assistant", "content": "<completion>"}
            ],
            "task_id": "...",
            "score": 1.0
        }
        """
        # Priority: soar2cot (training set, perfect score) > high quality > messages > legacy
        sft_path_soar2cot = Path(__file__).parent / "sft_soar2cot.json"
        sft_path_high_quality = Path(__file__).parent / "sft_instructions_high_quality.json"
        sft_path_messages = Path(__file__).parent / "sft_instructions_messages.json"
        sft_path_legacy = Path(__file__).parent / "sft_instructions.json"

        if sft_path_soar2cot.exists():
            sft_path = sft_path_soar2cot
            print(f"[ARCBuilder] Loading soar2cot SFT data: 7185 examples, 232 training tasks, score=1.0")
        elif sft_path_high_quality.exists():
            sft_path = sft_path_high_quality
            print(f"[ARCBuilder] Loading high quality SFT data (evaluation set)")
        elif sft_path_messages.exists():
            sft_path = sft_path_messages
        elif sft_path_legacy.exists():
            sft_path = sft_path_legacy
        else:
            raise FileNotFoundError(
                f"Instruction SFT data not found. "
                f"Expected at {sft_path_soar2cot} or {sft_path_high_quality}. "
                "Run the instruction generation script first."
            )

        with open(sft_path) as f:
            data = json.load(f)

        print(f"[ARCBuilder] Loaded {len(data)} instruction examples from {sft_path.name}")
        return data

    def _build_sft_datasets(self) -> DatasetDict:
        """
        Build SFT datasets for instruction generation.

        Uses pre-generated instruction data from evaluation set.
        """
        # Check if we should use instruction SFT data
        use_instruction_sft = self.full_config.get("use_instruction_sft", False)

        if use_instruction_sft:
            # Load instruction SFT data
            all_examples = self._load_instruction_sft_data()

            # Split into train/valid (get val_ratio from sft config or default)
            sft_config = self.full_config.get("sft", {})
            val_ratio = sft_config.get("val_ratio", 0.1) if sft_config else 0.1
            if self.config:
                val_ratio = self.config.get("val_ratio", val_ratio)
            val_size = int(len(all_examples) * val_ratio)

            # Shuffle deterministically
            import random
            random.seed(42)
            random.shuffle(all_examples)

            train_examples = all_examples[val_size:]
            valid_examples = all_examples[:val_size]

            # Build datasets
            train_dataset = Dataset.from_list(train_examples)
            valid_dataset = Dataset.from_list(valid_examples)

            # Test set is empty for instruction SFT (we evaluate on training set GRPO)
            test_dataset = Dataset.from_list([])

            return DatasetDict({
                "train": train_dataset,
                "valid": valid_dataset,
                "test": test_dataset,
            })
        else:
            # Default: use grid generation data
            return self._build_datasets()

    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()

    @classmethod
    def _preprocess(cls, example: Dict) -> Dict:
        """Preprocess is handled in _create_training_examples."""
        return example

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "completion", "solution", "task_id"]
