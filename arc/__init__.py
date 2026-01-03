"""
ARC Code Generation Training Module for MemGen.

BARC-style approach: Model generates Python code that transforms input grids.
Code is executed on training examples to compute reward.

Key Components:
- prompts: Code generation prompt templates
- utils: Code parsing, execution, grid validation utilities
- interaction: Code generation interaction manager
- trainer: ARC-specific GRPO trainer wrapper
- runner: Training orchestration
"""

from arc.prompts import (
    CODE_GENERATION_PROMPT,
    SYSTEM_PROMPT_CODER,
    build_code_generation_prompt,
    build_code_generation_messages,
    format_examples,
)
from arc.utils import (
    get_grid_similarity,
    parse_grid_from_text,
    parse_code_from_text,
    validate_grid,
    validate_code_on_examples,
    execute_code_on_input,
    create_executable_code,
    grid_to_string,
    format_examples,
    generate_grid_diff,
)
from arc.interaction import (
    ARCCodeGenerationManager,
    ARCCodeGenerationConfig,
    ARCCodeGenerationPoolManager,
    CodeCandidate,
    CodeGenerationResult,
)
from arc.trainer import (
    ARCCodeGenerationTrainer,
    create_arc_trainer,
)
from arc.runner import ARCCodeGenerationRunner

__all__ = [
    # Prompts
    "CODE_GENERATION_PROMPT",
    "SYSTEM_PROMPT_CODER",
    "build_code_generation_prompt",
    "build_code_generation_messages",
    # Utils
    "get_grid_similarity",
    "parse_grid_from_text",
    "parse_code_from_text",
    "validate_grid",
    "validate_code_on_examples",
    "execute_code_on_input",
    "create_executable_code",
    "grid_to_string",
    "format_examples",
    "generate_grid_diff",
    # Interaction
    "ARCCodeGenerationManager",
    "ARCCodeGenerationConfig",
    "ARCCodeGenerationPoolManager",
    "CodeCandidate",
    "CodeGenerationResult",
    # Trainer
    "ARCCodeGenerationTrainer",
    "create_arc_trainer",
    # Runner
    "ARCCodeGenerationRunner",
]
