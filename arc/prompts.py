"""
Prompt Templates for ARC Two-Stage Training.

Adapted from arc-lang-public/src/main.py and src/run.py.
These prompts follow the same structure as the reference implementation
to ensure compatibility and consistent behavior.
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Stage 1: Instruction Generation Prompts
# =============================================================================

INTUITIVE_PROMPT = """
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid, based on the training examples below.

Your task is to write clear instructions that describe this transformation pattern. These instructions must:
- Apply consistently to ALL training examples (the same rule works for every input->output pair)
- Be general enough to work on new test cases
- Be intuitive and easy to understand
- Describe the pattern without referencing specific example numbers or positions

The transformation pattern should be simple and logical - these puzzles are designed to have elegant, intuitive solutions that humans can readily grasp.

Write your instructions as a clear, step-by-step process that someone could follow to transform any input grid into the correct output grid.

Here are the training examples and test input grids:
""".strip()


# =============================================================================
# Stage 2: Grid Generation Prompts
# =============================================================================

FOLLOW_INSTRUCTIONS_PROMPT = """
You are an expert puzzle solver in a competition.

You will receive:
1. Step-by-step instructions for transforming input grids into output grids
2. Training examples showing these instructions applied correctly
3. A test input grid to solve

Your task: Apply the given instructions precisely to transform the test input grid into its output grid.

The training examples demonstrate how the instructions work - use them to understand the pattern, then follow the exact same process for the test input.
""".strip()


# Perfect match indicator - used when instructions scored 100%
PERFECT_PROMPT = """
These instructions are a guide to help you get the correct output grid.
If you think there is an error with the instructions that would cause you to get the wrong output grid, ignore that part of the instructions.
What is most important is that you get the exact correct output grid given the general pattern you observe.
""".strip()


# =============================================================================
# Revision Prompts (for refinement turns)
# =============================================================================

REVISION_PROMPT = """
Your previous instructions were applied to the training input grids, but they did not produce the correct output grids.

Below you'll see what outputs were generated when following your instructions. Compare these incorrect outputs with the correct outputs to identify where your instructions went wrong.

Based on this feedback, provide updated instructions that correctly describe the transformation pattern. Your revised instructions must:
- Fix the specific errors you observe
- Still work correctly for ALL training examples
- Remain clear, intuitive, and general

Analyze the differences between the incorrect outputs and the correct outputs to understand the true pattern, then write improved instructions.
""".strip()


# =============================================================================
# Synthesis/Pooling Prompts (for combining multiple attempts)
# =============================================================================

SYNTHESIS_PROMPT = """
Multiple expert puzzle solvers have attempted to describe the transformation pattern for these grids. Each attempt captured some aspects correctly but failed in other ways.

Below you'll find:
- Each set of proposed instructions
- The outputs produced when following those instructions
- How those outputs differ from the correct answers

Your task is to analyze why each approach partially failed and synthesize a complete, correct set of instructions.

By examining multiple flawed attempts, you can:
- Identify what each approach got right
- Understand what each approach missed
- Recognize common misconceptions about the pattern
- Build comprehensive instructions that avoid all these pitfalls

Study the patterns of success and failure across all attempts, then write instructions that correctly describe the complete transformation rule that works for ALL training examples.

Your final instructions should be clear, intuitive, and capture the true underlying pattern.
""".strip()


# =============================================================================
# System Prompts for Different Roles
# =============================================================================

SYSTEM_PROMPT_INSTRUCTOR = """You are an expert at analyzing abstract patterns in ARC (Abstraction and Reasoning Corpus) puzzles.
Your role is to observe training examples and describe the transformation pattern as clear, step-by-step instructions.
Output your instructions as JSON: {"instructions": "your step-by-step instructions here"}
""".strip()


SYSTEM_PROMPT_EXECUTOR = """You are an expert at executing transformation rules on ARC grids.
Your role is to follow given instructions precisely and produce the correct output grid.

CRITICAL: Output ONLY the JSON object. Do NOT include any explanation, reasoning, or thinking.
Your response must start with { and end with }

Format: {"grid": [[row1], [row2], ...]}
""".strip()


SYSTEM_PROMPT_REVISER = """You are an expert at debugging and improving transformation rules.
Your role is to analyze what went wrong with previous instructions and create improved versions.
Output your revised instructions as JSON: {"reasoning": "why old instructions failed", "revised_instructions": "improved step-by-step instructions"}
""".strip()


# =============================================================================
# Prompt Builders
# =============================================================================

@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    include_base64: bool = False
    use_diffs: bool = True
    show_attempts: bool = True


def build_instruction_prompt(
    examples_text: str,
    test_input_text: str,
    config: Optional[PromptConfig] = None
) -> str:
    """
    Build a complete instruction generation prompt.

    Args:
        examples_text: Formatted training examples string
        test_input_text: Formatted test input grid string
        config: Optional prompt configuration

    Returns:
        Complete prompt for instruction generation
    """
    prompt = f"""{INTUITIVE_PROMPT}

--Training Examples--
{examples_text}
--End of Training Examples--

--Test Input--
{test_input_text}
--End of Test Input--

Now write your instructions describing the transformation pattern.
Output as JSON: {{"instructions": "your step-by-step instructions here"}}
"""
    return prompt


def build_grid_generation_prompt(
    instructions: str,
    examples_text: str,
    test_input_text: str,
    is_perfect: bool = False,
    config: Optional[PromptConfig] = None
) -> str:
    """
    Build a complete grid generation prompt.

    Args:
        instructions: The transformation instructions from Stage 1
        examples_text: Formatted training examples string
        test_input_text: Formatted test input grid string
        is_perfect: Whether instructions scored 100% on training examples
        config: Optional prompt configuration

    Returns:
        Complete prompt for grid generation
    """
    perfect_section = ""
    if not is_perfect:
        perfect_section = f"\n{PERFECT_PROMPT}\n"

    prompt = f"""{FOLLOW_INSTRUCTIONS_PROMPT}

Instructions:
{instructions}
{perfect_section}
--Training Examples--
{examples_text}
--End of Training Examples--

--Test Input Grid--
{test_input_text}
--End of Test Input Grid--

Apply the instructions to produce the output grid.
Output ONLY JSON (no explanation): {{"grid": [[...]]}}
"""
    return prompt


def build_revision_prompt(
    original_instructions: str,
    examples_text: str,
    attempts_text: str,
    config: Optional[PromptConfig] = None
) -> str:
    """
    Build a revision prompt for improving failed instructions.

    Args:
        original_instructions: The instructions that need improvement
        examples_text: Formatted training examples with outputs
        attempts_text: Formatted attempted outputs showing errors
        config: Optional prompt configuration

    Returns:
        Complete prompt for instruction revision
    """
    prompt = f"""{INTUITIVE_PROMPT}

--Training Examples--
{examples_text}
--End of Training Examples--

Your previous instructions were:
{original_instructions}

{REVISION_PROMPT}

--Your Attempts vs Correct Outputs--
{attempts_text}
--End of Attempts--

Provide revised instructions that fix the errors.
Output as JSON: {{"reasoning": "what went wrong", "revised_instructions": "improved step-by-step instructions"}}
"""
    return prompt


def build_synthesis_prompt(
    examples_text: str,
    all_attempts_text: str,
    config: Optional[PromptConfig] = None
) -> str:
    """
    Build a synthesis prompt for pooling multiple attempts.

    Args:
        examples_text: Formatted training examples
        all_attempts_text: Formatted text showing all instruction attempts and their results
        config: Optional prompt configuration

    Returns:
        Complete prompt for instruction synthesis
    """
    prompt = f"""{INTUITIVE_PROMPT}

--Training Examples--
{examples_text}
--End of Training Examples--

{SYNTHESIS_PROMPT}

--Previous Attempts--
{all_attempts_text}
--End of Previous Attempts--

Synthesize the best aspects of all attempts into improved instructions.
Output as JSON: {{"reasoning": "analysis of attempts", "revised_instructions": "synthesized step-by-step instructions"}}
"""
    return prompt
