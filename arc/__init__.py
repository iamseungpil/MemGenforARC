"""
ARC Two-Stage Training Module for MemGen.

This module implements the arc-lang-public two-stage architecture:
- Stage 1: Generate instructions (with Weaver memory capture)
- Stage 2: Generate grids using instructions + memory

Key Components:
- prompts: Prompt templates for instruction and grid generation
- utils: Grid parsing, scoring, and validation utilities
- interaction: Two-stage interaction manager with memory flow
- trainer: ARC-specific GRPO trainer wrapper
- runner: Training orchestration
"""

from arc.prompts import (
    INTUITIVE_PROMPT,
    FOLLOW_INSTRUCTIONS_PROMPT,
    REVISION_PROMPT,
    SYNTHESIS_PROMPT,
)
from arc.utils import (
    get_grid_similarity,
    parse_grid_from_text,
    validate_grid,
    grid_to_string,
    format_examples,
    generate_grid_diff,
)
from arc.interaction import (
    ARCTwoStageInteractionManager,
    ARCTwoStageConfig,
)

__all__ = [
    # Prompts
    "INTUITIVE_PROMPT",
    "FOLLOW_INSTRUCTIONS_PROMPT",
    "REVISION_PROMPT",
    "SYNTHESIS_PROMPT",
    # Utils
    "get_grid_similarity",
    "parse_grid_from_text",
    "validate_grid",
    "grid_to_string",
    "format_examples",
    "generate_grid_diff",
    # Interaction
    "ARCTwoStageInteractionManager",
    "ARCTwoStageConfig",
]
