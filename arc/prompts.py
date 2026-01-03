"""
Prompt Templates for ARC Code Generation Training.

BARC-style approach: Model generates Python code that transforms input grids to output grids.
The code is executed on training examples to compute reward.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict


# =============================================================================
# System Prompt for Code Generation
# =============================================================================

SYSTEM_PROMPT_CODER = """You are an expert Python programmer solving ARC (Abstraction and Reasoning Corpus) puzzles.

Your task is to write a Python function `main(input_grid)` that transforms input grids to output grids.
The function must work correctly for ALL training examples shown.

Available imports and utilities:
- numpy as np
- typing (List, Tuple, Dict, Any, Optional)
- Color class with constants: BLACK=0, BLUE=1, RED=2, GREEN=3, YELLOW=4, GREY=5, MAGENTA=6, ORANGE=7, SKY=8, BROWN=9

Available helper functions:
- crop(grid, background=0): Crop grid to remove background padding
- bounding_box(grid, background=0): Get (min_row, min_col, height, width) of non-background region
- find_connected_components(grid, connectivity=4, monochromatic=True, background=0): Find connected regions
- detect_objects(grid, colors=None, connectivity=4, monochromatic=True): Detect objects in grid
- object_colors(obj, background=0): Get list of colors in an object
- object_position(obj, background=0, anchor="top-left"): Get position of object

Output format:
```python
def main(input_grid):
    # Your implementation here
    return output_grid
```

IMPORTANT:
- input_grid is a 2D list of integers (0-9)
- output_grid must be a 2D list of integers (0-9)
- Analyze ALL training examples to find the common pattern
- Your code must work for ANY valid input following the same pattern
""".strip()


# =============================================================================
# Code Generation Prompt
# =============================================================================

CODE_GENERATION_PROMPT = """Analyze the training examples below and write a Python function that transforms input grids to output grids.

{examples}

Write a Python function `main(input_grid)` that implements the transformation pattern.
The function should work correctly for ALL training examples above.

```python
def main(input_grid):
    # Your implementation
    return output_grid
```
""".strip()


# =============================================================================
# Prompt Configuration
# =============================================================================

@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    include_grid_dimensions: bool = True
    show_color_legend: bool = False
    max_examples: int = 10


# =============================================================================
# Prompt Builders
# =============================================================================

def format_grid(grid: List[List[int]], name: str = "Grid") -> str:
    """
    Format a single grid for display.

    Args:
        grid: 2D list of integers
        name: Label for the grid

    Returns:
        Formatted string representation
    """
    if not grid:
        return f"{name}: (empty)"

    rows = len(grid)
    cols = len(grid[0]) if grid else 0

    grid_str = "\n".join([" ".join(map(str, row)) for row in grid])
    return f"{name} ({rows}x{cols}):\n{grid_str}"


def format_example(example: Dict, idx: int) -> str:
    """
    Format a single training example.

    Args:
        example: Dict with 'input' and 'output' grids
        idx: Example index (1-based)

    Returns:
        Formatted string showing input -> output
    """
    input_str = format_grid(example["input"], "Input")
    output_str = format_grid(example["output"], "Output")

    return f"Example {idx}:\n{input_str}\n\n{output_str}"


def format_examples(train_examples: List[Dict]) -> str:
    """
    Format all training examples for prompt.

    Args:
        train_examples: List of {"input": grid, "output": grid} dicts

    Returns:
        Formatted string showing all examples
    """
    formatted = []
    for i, ex in enumerate(train_examples, 1):
        formatted.append(format_example(ex, i))
    return "\n\n---\n\n".join(formatted)


def build_code_generation_prompt(
    train_examples: List[Dict],
    config: Optional[PromptConfig] = None
) -> str:
    """
    Build a complete code generation prompt.

    Args:
        train_examples: Training examples showing input->output transformation
        config: Optional prompt configuration

    Returns:
        Complete prompt for code generation
    """
    examples_text = format_examples(train_examples)

    prompt = CODE_GENERATION_PROMPT.format(examples=examples_text)

    return prompt


def build_code_generation_messages(
    train_examples: List[Dict],
    config: Optional[PromptConfig] = None
) -> List[Dict[str, str]]:
    """
    Build chat messages for code generation.

    Args:
        train_examples: Training examples showing input->output transformation
        config: Optional prompt configuration

    Returns:
        List of message dicts with 'role' and 'content'
    """
    user_prompt = build_code_generation_prompt(train_examples, config)

    return [
        {"role": "system", "content": SYSTEM_PROMPT_CODER},
        {"role": "user", "content": user_prompt}
    ]
