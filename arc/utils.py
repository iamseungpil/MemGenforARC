"""
Utility Functions for ARC Code Generation Training.

Provides code parsing, execution, grid validation, and scoring utilities.
Based on BARC eval_utils.py for safe code execution.
"""

import json
import logging
import re
import subprocess
import tempfile
import os
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Grid Similarity Scoring
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


def grids_equal(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """
    Check if two grids are exactly equal.

    Args:
        grid1: First grid
        grid2: Second grid

    Returns:
        True if grids are identical
    """
    try:
        return np.array_equal(np.array(grid1), np.array(grid2))
    except (ValueError, TypeError):
        return False


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
    if grid is None:
        return None

    if not isinstance(grid, (list, np.ndarray)):
        return None

    # Convert numpy array to list
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()

    if len(grid) == 0:
        return None

    if not all(isinstance(row, (list, np.ndarray)) for row in grid):
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
            if not isinstance(val, (int, float, np.integer, np.floating)):
                return None

    # Convert all values to int
    return [[int(val) for val in row] for row in grid]


# =============================================================================
# Code Parsing from Text
# =============================================================================

def parse_code_from_text(text: str) -> Optional[str]:
    """
    Extract Python code from model output text.

    Handles various formats:
    - Markdown code blocks: ```python ... ```
    - Bare code with def main(...)

    Args:
        text: Model output text containing Python code

    Returns:
        Extracted Python code string, or None if not found
    """
    if not text:
        return None

    text = text.strip()

    # Method 1: Extract from markdown code block
    # Match ```python ... ``` or ``` ... ```
    code_block_pattern = re.compile(
        r'```(?:python)?\s*\n?(.*?)```',
        re.DOTALL
    )

    matches = code_block_pattern.findall(text)
    if matches:
        # Find the match containing 'def main'
        for match in matches:
            if 'def main' in match:
                return match.strip()
        # If no 'def main', return the first match
        return matches[0].strip()

    # Method 2: Extract bare code starting with def main
    def_pattern = re.compile(
        r'(def\s+main\s*\([^)]*\)\s*:.+)',
        re.DOTALL
    )

    def_match = def_pattern.search(text)
    if def_match:
        return def_match.group(1).strip()

    # Method 3: Try to extract any function definition
    any_def_pattern = re.compile(
        r'(def\s+\w+\s*\([^)]*\)\s*:.+)',
        re.DOTALL
    )

    any_match = any_def_pattern.search(text)
    if any_match:
        code = any_match.group(1).strip()
        logger.warning(f"Found function but not 'main': {code[:50]}...")
        return code

    logger.warning(f"Failed to parse code from text: {text[:200]}...")
    return None


# =============================================================================
# Code Execution (BARC-style safe execution)
# =============================================================================

def create_executable_code(solution_code: str) -> str:
    """
    Create executable code with Color class and utility functions.

    Based on BARC eval_utils.py.

    Args:
        solution_code: The main() function code

    Returns:
        Complete executable Python code
    """
    full_code = f"""
import sys
import numpy as np
from typing import *
import traceback

# Color definitions
class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    SKY = 8
    BROWN = 9

# Basic utility functions
def crop(grid, background=0):
    grid = np.array(grid)
    mask = grid != background
    if not np.any(mask):
        return np.array([[background]])
    coords = np.where(mask)
    min_row, max_row = coords[0].min(), coords[0].max()
    min_col, max_col = coords[1].min(), coords[1].max()
    return grid[min_row:max_row+1, min_col:max_col+1]

def bounding_box(grid, background=0):
    grid = np.array(grid)
    mask = grid != background
    if not np.any(mask):
        return 0, 0, 1, 1
    coords = np.where(mask)
    min_row, max_row = coords[0].min(), coords[0].max()
    min_col, max_col = coords[1].min(), coords[1].max()
    return min_row, min_col, max_row - min_row + 1, max_col - min_col + 1

def find_connected_components(grid, connectivity=4, monochromatic=True, background=0):
    try:
        from scipy import ndimage
        grid = np.array(grid)
        if monochromatic:
            components = []
            unique_colors = np.unique(grid)
            for color in unique_colors:
                if color == background:
                    continue
                mask = (grid == color)
                if connectivity == 4:
                    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
                else:
                    structure = np.ones((3,3))
                labeled, _ = ndimage.label(mask, structure=structure)
                for label_val in range(1, labeled.max() + 1):
                    component = np.where(labeled == label_val, color, background)
                    components.append(component)
            return components
        else:
            mask = (grid != background)
            if connectivity == 4:
                structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
            else:
                structure = np.ones((3,3))
            labeled, _ = ndimage.label(mask, structure=structure)
            components = []
            for label_val in range(1, labeled.max() + 1):
                component = np.where(labeled == label_val, grid, background)
                components.append(component)
            return components
    except:
        return []

def detect_objects(grid, colors=None, connectivity=4, monochromatic=True):
    return find_connected_components(grid, connectivity, monochromatic)

def object_colors(obj, background=0):
    colors = np.unique(np.array(obj))
    return [c for c in colors if c != background]

def object_position(obj, background=0, anchor="top-left"):
    obj = np.array(obj)
    mask = obj != background
    if not np.any(mask):
        return 0, 0
    coords = np.where(mask)
    if anchor == "center":
        return int(coords[0].mean()), int(coords[1].mean())
    else:  # top-left
        return coords[0].min(), coords[1].min()

# User code starts here
{solution_code}

# Test execution wrapper
def test_code(input_data):
    try:
        input_grid = input_data
        if hasattr(input_grid, 'tolist'):
            input_grid = input_grid.tolist()
        result = main(input_grid)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result
    except Exception as e:
        return None
"""
    return full_code


def execute_code_safely(
    code: str,
    input_data: List[List[int]],
    timeout_seconds: int = 5,
    max_memory_mb: int = 512
) -> Tuple[bool, Any]:
    """
    Safely execute code in a subprocess with timeout and resource limits.

    Security measures:
    - Code and input data written to separate temp files (no shell injection)
    - Resource limits via subprocess environment
    - Proper cleanup with try/finally

    Args:
        code: Complete executable code
        input_data: Input grid data
        timeout_seconds: Timeout in seconds
        max_memory_mb: Maximum memory limit in MB

    Returns:
        Tuple of (success, result_or_error)
    """
    code_file = None
    input_file = None

    try:
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, prefix='arc_code_'
        ) as f:
            f.write(code)
            code_file = f.name

        # Create temporary file for input data (avoids JSON injection)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, prefix='arc_input_'
        ) as f:
            json.dump(input_data, f)
            input_file = f.name

        # Build wrapper script that loads files safely
        wrapper_script = '''
import json
import sys
import resource

# Set resource limits
MAX_MEMORY = {max_memory} * 1024 * 1024  # Convert MB to bytes
try:
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
except (ValueError, resource.error) as e:
    import sys
    print(f"Warning: Resource limits not available: {{e}}. Relying on subprocess timeout only.", file=sys.stderr)

# Load code from file
with open(sys.argv[1], 'r') as f:
    code_content = f.read()

# Load input from file
with open(sys.argv[2], 'r') as f:
    input_data = json.load(f)

# Execute code in isolated namespace
exec(code_content, globals())

# Run test_code function
result = test_code(input_data)
print(json.dumps(result))
'''.format(max_memory=max_memory_mb, timeout=timeout_seconds)

        try:
            result = subprocess.run(
                ['python', '-c', wrapper_script, code_file, input_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    try:
                        parsed = json.loads(output)
                        return True, parsed
                    except json.JSONDecodeError:
                        return False, f"Invalid JSON output: {output[:200]}"
                else:
                    return False, "No output produced"
            else:
                # Truncate error message to prevent large outputs
                stderr = result.stderr[:500] if result.stderr else "Unknown error"
                return False, f"Execution error: {stderr}"

        except subprocess.TimeoutExpired:
            return False, "Execution timeout"

    except Exception as e:
        return False, f"Execution exception: {str(e)[:200]}"
    finally:
        # Cleanup temp files
        if code_file and os.path.exists(code_file):
            try:
                os.unlink(code_file)
            except OSError:
                pass
        if input_file and os.path.exists(input_file):
            try:
                os.unlink(input_file)
            except OSError:
                pass


def validate_code_on_examples(
    code: str,
    train_examples: List[Dict],
    timeout_seconds: int = 5
) -> Tuple[float, Dict]:
    """
    Validate code on training examples and compute accuracy.

    Args:
        code: Python code with main() function
        train_examples: List of {input, output} dicts
        timeout_seconds: Timeout per example

    Returns:
        Tuple of (accuracy, validation_results)
    """
    executable_code = create_executable_code(code)

    results = {
        'total_examples': len(train_examples),
        'passed_examples': 0,
        'failed_examples': [],
        'execution_errors': [],
        'example_results': []
    }

    for i, example in enumerate(train_examples):
        success, result = execute_code_safely(
            executable_code,
            example['input'],
            timeout_seconds
        )

        if success and result is not None:
            validated_result = validate_grid(result)
            if validated_result is not None:
                if grids_equal(validated_result, example['output']):
                    results['passed_examples'] += 1
                    results['example_results'].append({
                        'idx': i,
                        'success': True,
                        'correct': True
                    })
                else:
                    results['failed_examples'].append({
                        'idx': i,
                        'expected': example['output'],
                        'actual': validated_result
                    })
                    results['example_results'].append({
                        'idx': i,
                        'success': True,
                        'correct': False
                    })
            else:
                results['failed_examples'].append({
                    'idx': i,
                    'error': 'Invalid grid format',
                    'actual': result
                })
                results['example_results'].append({
                    'idx': i,
                    'success': False,
                    'correct': False
                })
        else:
            error_msg = result if isinstance(result, str) else 'Unknown error'
            results['execution_errors'].append({
                'idx': i,
                'error': error_msg
            })
            results['example_results'].append({
                'idx': i,
                'success': False,
                'correct': False,
                'error': error_msg
            })

    # Compute accuracy
    accuracy = results['passed_examples'] / results['total_examples'] if results['total_examples'] > 0 else 0.0
    results['accuracy'] = accuracy

    return accuracy, results


def execute_code_on_input(
    code: str,
    input_grid: List[List[int]],
    timeout_seconds: int = 5
) -> Optional[List[List[int]]]:
    """
    Execute code on a single input and return the output grid.

    Args:
        code: Python code with main() function
        input_grid: Input grid
        timeout_seconds: Timeout

    Returns:
        Output grid or None if execution failed
    """
    executable_code = create_executable_code(code)
    success, result = execute_code_safely(executable_code, input_grid, timeout_seconds)

    if success and result is not None:
        return validate_grid(result)

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


# =============================================================================
# Grid Diff Generation
# =============================================================================

def generate_grid_diff(
    expected_grid: List[List[int]],
    actual_grid: List[List[int]]
) -> str:
    """
    Generate a cell-by-cell diff notation between expected and actual grids.

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

    # Count mismatches
    mismatch_count = 0
    total_cells = len(expected_grid) * len(expected_grid[0])
    diff_lines = []

    for i, (expected_row, actual_row) in enumerate(zip(expected_grid, actual_grid)):
        row_parts = []
        for expected_val, actual_val in zip(expected_row, actual_row):
            if expected_val == actual_val:
                row_parts.append(f"{expected_val}")
            else:
                row_parts.append(f"[{actual_val}->{expected_val}]")
                mismatch_count += 1
        diff_lines.append(" ".join(row_parts))

    accuracy = (total_cells - mismatch_count) / total_cells * 100
    header = f"Accuracy: {accuracy:.1f}% ({total_cells - mismatch_count}/{total_cells} correct)"

    return f"{header}\n" + "\n".join(diff_lines)


# =============================================================================
# Grid Parsing from Text (Legacy support)
# =============================================================================

def _find_balanced(text: str, open_char: str, close_char: str) -> List[str]:
    """
    Find all balanced bracket/brace pairs in text.
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

                if c == '"':
                    if not in_string:
                        in_string = True
                    elif j > 0 and text[j-1] != '\\':
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

    Args:
        text: Model output text containing a grid

    Returns:
        Validated grid with consistent row lengths, or None if invalid
    """
    if not text:
        return None

    text = text.strip()

    # Remove thinking tags if present
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    text = think_pattern.sub('', text).strip()

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

    # Method 2: Find balanced JSON objects with "grid" key
    for json_str in _find_balanced(text, '{', '}'):
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "grid" in data:
                validated = validate_grid(data["grid"])
                if validated:
                    return validated
        except (json.JSONDecodeError, TypeError):
            continue

    # Method 3: Find balanced JSON arrays
    for json_str in _find_balanced(text, '[', ']'):
        try:
            data = json.loads(json_str)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                validated = validate_grid(data)
                if validated:
                    return validated
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def parse_instructions_from_text(text: str) -> Optional[str]:
    """
    Parse instructions from model output text (legacy support).

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

    # Return the text as-is if no JSON found
    if len(text) > 20:
        return text

    return None
