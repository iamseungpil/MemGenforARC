"""
Utility Functions for ARC Two-Stage Training.

Provides grid parsing, scoring, validation, and formatting utilities.
Based on arc-lang-public/src/run.py and data/arc/env.py.
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Grid Similarity Scoring (from arc-lang-public/src/run.py:182-208)
# =============================================================================

def get_grid_similarity(
    ground_truth_grid: List[List[int]],
    sample_grid: List[List[int]]
) -> float:
    """
    Calculate similarity as the percentage of cells that match exactly.
    Returns a value between 0.0 (no matches) and 1.0 (perfect match).

    Args:
        ground_truth_grid: Expected output grid
        sample_grid: Predicted output grid

    Returns:
        Float between 0.0 and 1.0 representing match percentage
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

    if total_cells == 0:
        return 0.0

    matching_cells = 0
    for i in range(rows):
        for j in range(cols):
            if j < len(sample_grid[i]) and ground_truth_grid[i][j] == sample_grid[i][j]:
                matching_cells += 1

    return matching_cells / total_cells


# =============================================================================
# Grid Validation
# =============================================================================

def validate_grid(grid: Any) -> Optional[List[List[int]]]:
    """
    Validate that a grid is a proper 2D array with consistent row lengths.

    Args:
        grid: Candidate grid to validate

    Returns:
        Validated grid with int values, or None if invalid
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
        # Validate all elements are integers (or can be converted)
        for val in row:
            if not isinstance(val, (int, float)):
                return None

    # Convert all values to int
    return [[int(val) for val in row] for row in grid]


# =============================================================================
# Grid Parsing from Text (FIXED: proper nested bracket handling)
# =============================================================================

def _find_balanced(text: str, open_char: str, close_char: str) -> List[str]:
    """
    Find all balanced bracket/brace pairs in text.

    Handles nested structures and quoted strings properly.
    This fixes the bug where non-greedy regex .*? would stop at the first
    closing bracket, breaking nested array parsing like [[1,2],[3,4]].

    Args:
        text: Text to search
        open_char: Opening character ('{' or '[')
        close_char: Closing character ('}' or ']')

    Returns:
        List of balanced substrings, ordered by position in text
    """
    results = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == open_char:
            depth = 1
            start = i
            j = i + 1
            in_string = False

            while j < n and depth > 0:
                c = text[j]

                # Handle string literals (skip contents to avoid false matches)
                if c == '"':
                    if not in_string:
                        in_string = True
                    elif j > 0 and text[j-1] != '\\':  # Not escaped quote
                        in_string = False
                elif not in_string:
                    if c == open_char:
                        depth += 1
                    elif c == close_char:
                        depth -= 1

                j += 1

            if depth == 0:
                results.append(text[start:j])
                i = j - 1
        i += 1

    return results


def parse_grid_from_text(text: str) -> Optional[List[List[int]]]:
    """
    Parse a 2D grid from model output text.

    Handles various formats:
    - JSON object with "grid" key: {"grid": [[1,2],[3,4]]}
    - Bare JSON array: [[1,2],[3,4]]
    - Markdown code blocks containing JSON

    FIXED: Uses balanced bracket matching instead of non-greedy regex,
    which was breaking on nested arrays like [[1,2],[3,4]].

    Args:
        text: Model output text containing a grid

    Returns:
        Validated grid with consistent row lengths, or None if invalid
    """
    if not text:
        return None

    text = text.strip()

    # Remove Qwen3 thinking tags if present (even when using /no_think, empty tags may appear)
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    text = think_pattern.sub('', text).strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Find the closing ```
        end_idx = len(lines)
        for i in range(1, len(lines)):
            if lines[i].strip().startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[1:end_idx])
        text = text.strip()

    # Method 1: Try direct JSON parse of entire text (cleanest case)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "grid" in data:
            validated = validate_grid(data["grid"])
            if validated:
                logger.debug("Parsed grid via direct JSON (dict with 'grid' key)")
                return validated
        elif isinstance(data, list):
            validated = validate_grid(data)
            if validated:
                logger.debug("Parsed grid via direct JSON (bare array)")
                return validated
    except (json.JSONDecodeError, TypeError):
        pass

    # Method 2: Find balanced JSON objects containing "grid" key
    for json_str in _find_balanced(text, '{', '}'):
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "grid" in data:
                validated = validate_grid(data["grid"])
                if validated:
                    logger.debug(f"Parsed grid via balanced braces: {json_str[:50]}...")
                    return validated
        except (json.JSONDecodeError, TypeError):
            continue

    # Method 3: Find balanced JSON arrays (bare 2D arrays)
    for json_str in _find_balanced(text, '[', ']'):
        try:
            data = json.loads(json_str)
            # Only consider if it looks like a 2D array (list of lists)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                validated = validate_grid(data)
                if validated:
                    logger.debug(f"Parsed grid via balanced brackets: {json_str[:50]}...")
                    return validated
        except (json.JSONDecodeError, TypeError):
            continue

    logger.warning(f"Failed to parse grid from text: {text[:100]}...")
    return None


def parse_instructions_from_text(text: str) -> Optional[str]:
    """
    Parse instructions from model output text.

    Handles various formats:
    - JSON object with "instructions" key: {"instructions": "..."}
    - Plain text instructions

    Args:
        text: Model output text containing instructions

    Returns:
        Extracted instructions string, or None if not found
    """
    if not text:
        return None

    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
        text = text.strip()

    # Try to find JSON object with "instructions" key
    try:
        json_match = re.search(
            r'\{[^{}]*"instructions"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
            text, re.DOTALL
        )
        if json_match:
            data = json.loads(json_match.group())
            if "instructions" in data:
                return data["instructions"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "instructions" in data:
            return data["instructions"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Return the text as-is if no JSON found (might be plain instructions)
    if len(text) > 20:  # Minimum instruction length
        return text

    return None


def parse_revised_instructions_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Parse revised instructions with reasoning from model output.

    Expected format: {"reasoning": "...", "revised_instructions": "..."}

    FIXED: Uses balanced bracket matching for robustness.

    Args:
        text: Model output text

    Returns:
        Dict with 'reasoning' and 'revised_instructions' keys, or None
    """
    if not text:
        return None

    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        end_idx = len(lines)
        for i in range(1, len(lines)):
            if lines[i].strip().startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[1:end_idx])
        text = text.strip()

    # Method 1: Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            result = {}
            if "reasoning" in data:
                result["reasoning"] = data["reasoning"]
            if "revised_instructions" in data:
                result["revised_instructions"] = data["revised_instructions"]
            if result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Method 2: Find balanced JSON objects and try each
    for json_str in _find_balanced(text, '{', '}'):
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                result = {}
                if "reasoning" in data:
                    result["reasoning"] = data["reasoning"]
                if "revised_instructions" in data:
                    result["revised_instructions"] = data["revised_instructions"]
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            continue

    return None


# =============================================================================
# Grid Formatting
# =============================================================================

def grid_to_string(grid: List[List[int]]) -> str:
    """
    Convert a 2D grid to readable string format.

    Args:
        grid: 2D list of integers

    Returns:
        String representation with space-separated values
    """
    if not grid:
        return ""
    return "\n".join([" ".join(map(str, row)) for row in grid])


def format_examples(train_examples: List[Dict]) -> str:
    """
    Format training examples for prompt.

    Args:
        train_examples: List of {"input": grid, "output": grid} dicts

    Returns:
        Formatted string showing all examples
    """
    formatted = []
    for i, ex in enumerate(train_examples, 1):
        input_grid = grid_to_string(ex["input"])
        output_grid = grid_to_string(ex["output"])
        formatted.append(f"Example {i}:\nInput:\n{input_grid}\nOutput:\n{output_grid}")
    return "\n\n".join(formatted)


def format_example_with_attempt(
    example: Dict,
    attempt_grid: Optional[List[List[int]]],
    example_number: int,
    include_diff: bool = True
) -> str:
    """
    Format a single training example with an attempt and diff.

    Args:
        example: {"input": grid, "output": grid} dict
        attempt_grid: Predicted output grid (or None)
        example_number: 1-indexed example number
        include_diff: Whether to include diff notation

    Returns:
        Formatted string showing example and attempt
    """
    input_text = grid_to_string(example["input"])
    output_text = grid_to_string(example["output"])

    result = f"""Example {example_number}:
Input:
{input_text}
Expected Output:
{output_text}"""

    if attempt_grid is not None:
        attempt_text = grid_to_string(attempt_grid)

        if attempt_grid == example["output"]:
            result += f"""
Your Attempt (CORRECT):
{attempt_text}"""
        else:
            result += f"""
Your Attempt (INCORRECT):
{attempt_text}"""

            if include_diff:
                diff = generate_grid_diff(example["output"], attempt_grid)
                result += f"""

Difference (actual->expected):
{diff}"""

    return result


# =============================================================================
# Grid Diff Generation (from arc-lang-public/src/run.py:211-274)
# =============================================================================

def generate_grid_diff(
    expected_grid: List[List[int]],
    actual_grid: List[List[int]]
) -> str:
    """
    Generate a cell-by-cell diff notation between expected and actual grids.

    Format: ASCII grid with "|" separators where each cell shows
    "actual->expected" or "=value" for matches.

    Args:
        expected_grid: Ground truth grid
        actual_grid: Predicted grid

    Returns:
        String showing the diff
    """
    if not expected_grid or not actual_grid:
        return "Error: Empty grid(s)"

    # Check dimensions
    if len(expected_grid) != len(actual_grid):
        return f"Error: Grid dimension mismatch (rows: {len(expected_grid)} vs {len(actual_grid)})"

    if len(expected_grid) == 0:
        return "Error: Empty grid"

    if len(expected_grid[0]) != len(actual_grid[0]):
        return f"Error: Grid dimension mismatch (cols: {len(expected_grid[0])} vs {len(actual_grid[0])})"

    # Calculate max width needed for proper alignment
    max_width = 0
    for expected_row, actual_row in zip(expected_grid, actual_grid):
        for expected_val, actual_val in zip(expected_row, actual_row):
            if expected_val == actual_val:
                cell_width = len(f"={expected_val}")
            else:
                cell_width = len(f"{actual_val}->{expected_val}")
            max_width = max(max_width, cell_width)

    # Add padding
    max_width += 2

    diff_lines = []

    # Count mismatches
    mismatch_count = 0
    total_cells = len(expected_grid) * len(expected_grid[0])

    # Add top border
    num_cols = len(expected_grid[0])
    border = "+" + "+".join(["-" * max_width for _ in range(num_cols)]) + "+"
    diff_lines.append(border)

    for row_idx, (expected_row, actual_row) in enumerate(zip(expected_grid, actual_grid)):
        row_cells = []
        for expected_val, actual_val in zip(expected_row, actual_row):
            if expected_val == actual_val:
                cell = f"={expected_val}"
            else:
                cell = f"{actual_val}->{expected_val}"
                mismatch_count += 1
            row_cells.append(cell.center(max_width))

        diff_lines.append("|" + "|".join(row_cells) + "|")

        if row_idx < len(expected_grid) - 1:
            diff_lines.append(border)

    diff_lines.append(border)

    # Add summary
    accuracy = (total_cells - mismatch_count) / total_cells * 100
    summary = f"Accuracy: {accuracy:.1f}% ({total_cells - mismatch_count}/{total_cells} correct)"

    return summary + "\n" + "\n".join(diff_lines)


# =============================================================================
# Instruction Scoring Utilities
# =============================================================================

def score_instruction_on_examples(
    instructions: str,
    train_examples: List[Dict],
    grid_generator_func,
    leave_one_out: bool = True
) -> Dict[str, Any]:
    """
    Score instructions using leave-one-out validation on training examples.

    Args:
        instructions: The transformation instructions
        train_examples: List of training examples
        grid_generator_func: Function that generates grid from (instructions, context, test_input)
        leave_one_out: Whether to use leave-one-out scoring

    Returns:
        Dict with 'score', 'example_scores', and 'attempts'
    """
    example_scores = []
    attempts = []

    if leave_one_out and len(train_examples) > 1:
        for i in range(len(train_examples)):
            # Use all except i-th as training, i-th as test
            temp_test = train_examples[i]
            temp_train = train_examples[:i] + train_examples[i+1:]

            # Generate prediction
            pred_grid = grid_generator_func(
                instructions=instructions,
                training_examples=temp_train,
                test_input=temp_test["input"]
            )

            # Score
            score = get_grid_similarity(temp_test["output"], pred_grid)
            example_scores.append(score)
            attempts.append(pred_grid)
    else:
        # Use all as context, test on first
        if len(train_examples) > 0:
            test_example = train_examples[0]
            context = train_examples[1:] if len(train_examples) > 1 else train_examples

            pred_grid = grid_generator_func(
                instructions=instructions,
                training_examples=context,
                test_input=test_example["input"]
            )

            score = get_grid_similarity(test_example["output"], pred_grid)
            example_scores.append(score)
            attempts.append(pred_grid)

    avg_score = sum(example_scores) / len(example_scores) if example_scores else 0.0

    return {
        "score": avg_score,
        "example_scores": example_scores,
        "attempts": attempts,
        "is_perfect": avg_score == 1.0
    }


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def batch_parse_grids(texts: List[str]) -> List[Optional[List[List[int]]]]:
    """
    Parse grids from a batch of text outputs.

    Args:
        texts: List of model output texts

    Returns:
        List of parsed grids (None for failed parses)
    """
    return [parse_grid_from_text(t) for t in texts]


def batch_score_grids(
    predictions: List[Optional[List[List[int]]]],
    targets: List[List[List[int]]]
) -> List[float]:
    """
    Score a batch of predicted grids against targets.

    Args:
        predictions: List of predicted grids (can contain None)
        targets: List of ground truth grids

    Returns:
        List of similarity scores
    """
    scores = []
    for pred, target in zip(predictions, targets):
        if pred is None:
            scores.append(0.0)
        else:
            scores.append(get_grid_similarity(target, pred))
    return scores
