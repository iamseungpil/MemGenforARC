"""
ARC Environment with grid similarity reward.

Uses arc-lang's scoring function for GRPO training.
Supports both single-turn (StaticEnv) and multi-turn (DynamicEnv) training.
"""
import json
import logging
import re
from typing import Optional, Dict, Tuple, List
from data.base_env import StaticEnv, DynamicEnv

logger = logging.getLogger(__name__)


def get_grid_similarity(ground_truth_grid: list[list[int]], sample_grid: list[list[int]]) -> float:
    """
    Calculate similarity as the percentage of cells that match exactly.
    Returns a value between 0.0 (no matches) and 1.0 (perfect match).

    Copied from arc-lang-public/src/run.py:182-208
    """
    if not ground_truth_grid or not sample_grid:
        return 0.0

    # Check if grids have the same dimensions
    if len(ground_truth_grid) != len(sample_grid):
        return 0.0

    if len(ground_truth_grid) == 0:
        return 0.0

    if len(ground_truth_grid[0]) != len(sample_grid[0]):
        return 0.0

    rows = len(ground_truth_grid)
    cols = len(ground_truth_grid[0])
    total_cells = rows * cols
    matching_cells = 0

    for i in range(rows):
        for j in range(cols):
            if j < len(sample_grid[i]) and ground_truth_grid[i][j] == sample_grid[i][j]:
                matching_cells += 1

    return matching_cells / total_cells


def validate_grid(grid: list) -> Optional[list[list[int]]]:
    """
    Validate that a grid is a proper 2D array with consistent row lengths.
    Returns the validated grid or None if invalid.
    """
    if not grid or not isinstance(grid, list):
        return None

    if not all(isinstance(row, list) for row in grid):
        return None

    if len(grid) == 0:
        return None

    # Check all rows have the same length
    row_len = len(grid[0])
    if row_len == 0:
        return None

    for row in grid:
        if len(row) != row_len:
            return None
        # Validate all elements are numeric (int/float/str digits)
        for val in row:
            if isinstance(val, (int, float)):
                continue
            if isinstance(val, str) and re.fullmatch(r"-?\d+", val.strip()):
                continue
            return None

    # Convert all values to int
    normalized = []
    for row in grid:
        normalized.append([int(str(val).strip()) for val in row])
    return normalized


def _extract_balanced(text: str, start_idx: int, open_ch: str, close_ch: str) -> Optional[str]:
    """Extract a balanced bracketed substring starting at start_idx."""
    depth = 0
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _strip_code_fences(text: str) -> str:
    """Strip ```...``` fences if present and return inner content."""
    fence_match = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _extract_harmony_final_channel(text: str) -> Optional[str]:
    """
    Extract content from GPT-OSS Harmony format's final channel.

    GPT-OSS outputs in format:
        <|channel|>analysis<|message|>...analysis...<|channel|>final<|message|>
        0 1 2
        3 4 5
        <|return|>

    Returns the content after <|channel|>final<|message|> and before end markers.
    """
    # Look for final channel marker
    final_marker = '<|channel|>final<|message|>'
    idx = text.find(final_marker)
    if idx == -1:
        return None

    content = text[idx + len(final_marker):]

    # Find end markers
    end_markers = ['<|return|>', '<|end|>', '<|channel|>', '<|start|>']
    min_end_idx = len(content)
    for marker in end_markers:
        end_idx = content.find(marker)
        if end_idx != -1 and end_idx < min_end_idx:
            min_end_idx = end_idx

    return content[:min_end_idx].strip()


def _parse_plain_text_grid(text: str) -> Optional[list[list[int]]]:
    """Parse a plain text grid (space/newline separated numbers)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numeric_lines = []
    for ln in lines:
        # Allow lines with digits, spaces, commas
        if re.fullmatch(r"[\d\s,\-]+", ln):
            nums = [int(x) for x in re.findall(r"-?\d+", ln)]
            if nums:
                numeric_lines.append(nums)

    if len(numeric_lines) >= 1:
        row_len = len(numeric_lines[0])
        if row_len > 0 and all(len(r) == row_len for r in numeric_lines):
            return numeric_lines

    return None


def parse_grid_from_text(text: str) -> Optional[list[list[int]]]:
    """
    Parse a 2D grid from model output text.

    Handles various formats:
    - GPT-OSS Harmony format: <|channel|>final<|message|>...grid...<|return|>
    - JSON array: [[1,2],[3,4]]
    - JSON with key: {"grid": [[1,2],[3,4]]}
    - Plain text grids (space-separated numbers)

    Returns validated grid with consistent row lengths, or None if invalid.
    """
    if not text:
        return None

    # Method 0: Try GPT-OSS Harmony format first (highest priority)
    harmony_content = _extract_harmony_final_channel(text)
    if harmony_content:
        # Try to parse the harmony content as a grid
        grid = _parse_plain_text_grid(harmony_content)
        if grid:
            logger.debug(f"Parsed grid from Harmony final channel")
            return grid
        # Also try JSON in harmony content
        try:
            data = json.loads(harmony_content)
            if isinstance(data, dict) and "grid" in data:
                validated = validate_grid(data["grid"])
                if validated:
                    return validated
            elif isinstance(data, list):
                validated = validate_grid(data)
                if validated:
                    return validated
        except (json.JSONDecodeError, TypeError):
            pass

    text = _strip_code_fences(text.strip())

    # Try to find JSON object with "grid" key
    try:
        grid_key_match = re.search(r'"grid"\s*:', text)
        if grid_key_match:
            bracket_idx = text.find("[", grid_key_match.end())
            if bracket_idx != -1:
                array_text = _extract_balanced(text, bracket_idx, "[", "]")
                if array_text:
                    grid = json.loads(array_text)
                    validated = validate_grid(grid)
                    if validated:
                        return validated
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find bare JSON array
    try:
        array_idx = text.find("[")
        if array_idx != -1:
            array_text = _extract_balanced(text, array_idx, "[", "]")
            if array_text:
                grid = json.loads(array_text)
                validated = validate_grid(grid)
                if validated:
                    return validated
    except (json.JSONDecodeError, TypeError):
        pass

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "grid" in data:
            validated = validate_grid(data["grid"])
            if validated:
                return validated
        elif isinstance(data, list):
            validated = validate_grid(data)
            if validated:
                return validated
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: parse plain text grids (rows of numbers)
    grid = _parse_plain_text_grid(text)
    if grid:
        return grid

    return None


class ARCEnv(StaticEnv):
    """
    ARC Environment for GRPO training.

    Reward is computed using grid similarity between
    the model's predicted grid and the ground truth.
    """
    ENV_CARD = "STATIC"

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def compute_reward(
        cls,
        prompts: list[str],  # Required by TRL GRPOTrainer
        completions: list[str],
        solution: list[str],  # JSON string of ground truth grids
        **kwargs
    ) -> list[float]:
        """
        Compute reward for ARC task completions.

        Args:
            prompts: Input prompts (required by TRL, not used in reward calculation)
            completions: Model outputs (should contain grid in JSON format)
            solution: Ground truth grids as JSON strings

        Returns:
            List of rewards (0.0 to 1.0)
        """
        rewards = []
        task_id = kwargs.get("task_id", "unknown")

        for idx, (completion, sol) in enumerate(zip(completions, solution)):
            # Parse ground truth
            try:
                if isinstance(sol, str):
                    gt_grid = json.loads(sol)
                else:
                    gt_grid = sol
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"[{task_id}] Failed to parse ground truth: {e}")
                gt_grid = None

            # Parse model prediction
            pred_grid = parse_grid_from_text(completion)

            # Compute similarity
            if gt_grid is None:
                logger.debug(f"[{task_id}][{idx}] Ground truth is None")
                reward = 0.0
            elif pred_grid is None:
                logger.debug(f"[{task_id}][{idx}] Failed to parse prediction from: {completion[:200]}...")
                reward = 0.0
            else:
                reward = get_grid_similarity(gt_grid, pred_grid)
                if reward == 1.0:
                    logger.info(f"[{task_id}][{idx}] Perfect match! reward=1.0")
                elif reward > 0:
                    logger.debug(f"[{task_id}][{idx}] Partial match: reward={reward:.3f}")

            rewards.append(reward)

        return rewards


def grid_to_string(grid: List[List[int]]) -> str:
    """Convert a 2D grid to readable string format."""
    return "\n".join([" ".join(map(str, row)) for row in grid])


def generate_grid_diff(expected: List[List[int]], actual: List[List[int]]) -> str:
    """
    Generate a human-readable diff between expected and actual grids.
    """
    if not expected or not actual:
        return "Error: Empty grid(s)"

    if len(expected) != len(actual):
        return f"Dimension mismatch: expected {len(expected)} rows, got {len(actual)}"

    if len(expected[0]) != len(actual[0]):
        return f"Dimension mismatch: expected {len(expected[0])} cols, got {len(actual[0])}"

    diff_lines = []
    mismatch_count = 0
    total_cells = len(expected) * len(expected[0])

    for i, (exp_row, act_row) in enumerate(zip(expected, actual)):
        row_diff = []
        for j, (exp_val, act_val) in enumerate(zip(exp_row, act_row)):
            if exp_val == act_val:
                row_diff.append(f" {exp_val} ")
            else:
                row_diff.append(f"[{act_val}→{exp_val}]")
                mismatch_count += 1
        diff_lines.append(" ".join(row_diff))

    accuracy = (total_cells - mismatch_count) / total_cells * 100
    header = f"Accuracy: {accuracy:.1f}% ({total_cells - mismatch_count}/{total_cells} correct)"

    return f"{header}\n" + "\n".join(diff_lines)


class ARCDynamicEnv(DynamicEnv):
    """
    Multi-turn ARC Environment for iterative grid refinement.

    Supports 5 seeds × 7 turns paradigm:
    - Each turn: model generates grid, receives feedback
    - Feedback shows diff between prediction and target
    - Model can refine based on feedback in next turn
    - Different seeds use varied prompting strategies for diversity
    """
    ENV_CARD = "DYNAMIC"

    # Diverse seed prompt strategies
    SEED_STRATEGIES = [
        {
            "name": "analytical",
            "instruction": "Carefully analyze the transformation pattern step by step. Look for patterns in how colors change, shapes move, or grids transform.",
        },
        {
            "name": "visual",
            "instruction": "Focus on visual patterns - look at the overall structure, symmetry, and spatial relationships between input and output.",
        },
        {
            "name": "rule_based",
            "instruction": "Try to identify explicit rules: If cell is X, then output is Y. Look for conditional transformations.",
        },
        {
            "name": "object_focused",
            "instruction": "Identify distinct objects in the grid (connected regions of the same color). Track how each object transforms.",
        },
        {
            "name": "creative",
            "instruction": "Think creatively about what the puzzle is asking. Sometimes the pattern is simpler than it appears.",
        },
    ]

    def __init__(self, config):
        super().__init__(config)
        self.task_config = None
        self.target_grid = None
        self.train_examples = None
        self.test_input = None
        self.task_id = None

        # Multi-turn state
        self.current_turn = 0
        self.max_turns = config.get("max_turns", 7)
        self.best_grid = None
        self.best_score = 0.0
        self.turn_history: List[Dict] = []

        # Seed diversity support
        self.seed_id: int = 0
        self.seed_strategy: Optional[Dict] = None

    def set_env(self, task_config: Dict) -> Tuple[str, str]:
        """
        Initialize environment with ARC task.

        Args:
            task_config: Dict containing:
                - train_examples: List of {input, output} training examples
                - test_input: Input grid to solve
                - target_grid: Expected output grid
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
        self.best_grid = None
        self.best_score = 0.0
        self.turn_history = []

        # Set seed strategy based on seed_id
        if self.seed_id < len(self.SEED_STRATEGIES):
            self.seed_strategy = self.SEED_STRATEGIES[self.seed_id]
        else:
            # Fallback for additional seeds
            self.seed_strategy = self.SEED_STRATEGIES[self.seed_id % len(self.SEED_STRATEGIES)]

        # Format training examples
        examples_text = self._format_examples(self.train_examples)
        test_input_text = grid_to_string(self.test_input)

        # Build seed-specific system prompt
        base_system = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles.
Your goal is to find the transformation pattern from the training examples and apply it to the test input.
You will receive feedback after each attempt to help you refine your answer.
Output your answer as JSON: {"grid": [[...]]}"""

        if self.seed_strategy:
            strategy_instruction = self.seed_strategy["instruction"]
            system_prompt = f"""{base_system}

Strategy hint: {strategy_instruction}"""
        else:
            system_prompt = base_system

        user_prompt = f"""Training Examples:
{examples_text}

Test Input:
{test_input_text}

Analyze the pattern and generate the output grid."""

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
            formatted.append(f"Example {i}:\nInput:\n{input_grid}\nOutput:\n{output_grid}")
        return "\n\n".join(formatted)

    @classmethod
    def preprocess_action(cls, action: str) -> str:
        """Preprocess model output before evaluation."""
        # Try to extract just the JSON part
        action = action.strip()

        # Remove markdown code blocks if present
        if action.startswith("```"):
            lines = action.split("\n")
            action = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

        return action

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Process model's grid prediction and return feedback.

        Args:
            action: Model output (should contain grid in JSON format)

        Returns:
            Tuple of (observation/feedback, reward, done)
        """
        self.current_turn += 1

        # Parse the predicted grid
        pred_grid = parse_grid_from_text(action)

        if pred_grid is None:
            # Failed to parse - provide helpful error
            observation = """Failed to parse your grid output.
Please ensure you output valid JSON in the format: {"grid": [[row1], [row2], ...]}
Each row should be a list of integers (0-9).

Try again with the correct format."""
            reward = 0.0
            done = self.current_turn >= self.max_turns

            self.turn_history.append({
                "turn": self.current_turn,
                "prediction": None,
                "score": 0.0,
                "error": "parse_failed"
            })

            return observation, reward, done

        # Compute similarity
        score = get_grid_similarity(self.target_grid, pred_grid)

        # Track best result
        if score > self.best_score:
            self.best_score = score
            self.best_grid = pred_grid

        self.turn_history.append({
            "turn": self.current_turn,
            "prediction": pred_grid,
            "score": score
        })

        # Check if solved
        if score == 1.0:
            observation = "Perfect! Your grid matches the expected output exactly."
            return observation, score, True

        # Check if max turns reached
        if self.current_turn >= self.max_turns:
            observation = f"Maximum turns ({self.max_turns}) reached. Best score: {self.best_score:.1%}"
            return observation, score, True

        # Generate feedback with diff
        diff_text = generate_grid_diff(self.target_grid, pred_grid)

        observation = f"""Turn {self.current_turn}/{self.max_turns} - Score: {score:.1%}

Your prediction differs from the expected output:
{diff_text}

[Format: current→expected for mismatched cells]

Analyze the differences and try again. Focus on understanding what transformation you missed."""

        return observation, score, False

    def feedback(self) -> Tuple[float, bool]:
        """
        Return final reward and success status.

        Returns:
            Tuple of (best_score, solved)
        """
        solved = self.best_score == 1.0
        return self.best_score, solved

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

        if not envs:
            # Fallback to static reward computation
            solution = kwargs.get("solution", [])
            return ARCEnv.compute_reward(prompts, completions, solution=solution, **kwargs)

        # Use best scores from each environment
        rewards = []
        for env in envs:
            rewards.append(env.best_score)

        return rewards
