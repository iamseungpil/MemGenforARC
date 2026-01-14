"""
Completion generator using vLLM for high-throughput inference.

Supports:
- HuggingFace datasets
- CSV/Parquet files
- Two-stage thinking generation
- Shard-based output
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# vLLM is optional - graceful fallback if not available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)


# Default vLLM engine configuration
VLLM_ENGINE_KW = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.95,
    "dtype": "bfloat16",
    "max_model_len": 2000,
}


@dataclass
class GeneratorConfig:
    """Configuration for CompletionGenerator."""

    # Model settings
    model_id: str = "Qwen/Qwen3-8B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    dtype: str = "bfloat16"
    max_model_len: int = 2000

    # Sampling settings
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 1024
    num_generations: int = 2  # Completions per question
    stop_sequences: List[str] = field(default_factory=list)

    # Two-stage thinking settings
    enable_thinking: bool = True
    think_end_token: str = "</think>"
    max_thinking_tokens: int = 512

    # Output settings
    shard_size: int = 1000  # Samples per shard
    output_format: str = "parquet"  # "parquet" or "csv"


class CompletionGenerator:
    """
    High-throughput completion generator using vLLM.

    Features:
    - Batch generation with multiple completions per prompt
    - Two-stage thinking: first generate thinking, then complete answer
    - Shard-based output for large datasets
    - Support for HuggingFace datasets and CSV/Parquet files

    Usage:
        generator = CompletionGenerator("Qwen/Qwen3-8B")
        results = generator.generate(prompts, num_generations=2)

        # Or from dataset
        df = generator.generate_from_dataset(
            dataset,
            question_col="question",
            output_dir="./outputs"
        )
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[GeneratorConfig] = None,
        system_prompt: Optional[str] = None,
        **engine_kwargs
    ):
        """
        Initialize the completion generator.

        Args:
            model_id: HuggingFace model ID or local path.
            config: Generator configuration.
            system_prompt: Optional system prompt to prepend.
            **engine_kwargs: Additional vLLM engine arguments.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for CompletionGenerator. "
                "Install with: pip install vllm"
            )

        self.config = config or GeneratorConfig(model_id=model_id)
        self.model_id = model_id
        self.system_prompt = system_prompt

        # Merge engine kwargs with defaults
        engine_config = {
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype,
            "max_model_len": self.config.max_model_len,
            **engine_kwargs
        }

        logger.info(f"Initializing vLLM with model: {model_id}")
        self.llm = LLM(model=model_id, **engine_config)
        self._tokenizer = self.llm.get_tokenizer()

        logger.info("CompletionGenerator initialized successfully")

    def _build_prompt(self, question: str) -> str:
        """Build full prompt with optional system message."""
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{question}"
        return question

    def _build_chat_messages(
        self,
        question: str,
        include_think_start: bool = False
    ) -> List[Dict[str, str]]:
        """Build chat messages for chat-style models."""
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        user_content = question
        messages.append({
            "role": "user",
            "content": user_content
        })

        if include_think_start:
            messages.append({
                "role": "assistant",
                "content": "<think>"
            })

        return messages

    def _create_sampling_params(
        self,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        n: int = 1,
    ) -> "SamplingParams":
        """Create vLLM sampling parameters."""
        stop = stop_sequences or self.config.stop_sequences.copy()

        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=max_tokens or self.config.max_tokens,
            n=n,
            stop=stop if stop else None,
        )

    def generate(
        self,
        prompts: List[str],
        num_generations: Optional[int] = None,
        use_chat_template: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a list of prompts.

        Args:
            prompts: List of question/prompt strings.
            num_generations: Number of completions per prompt.
            use_chat_template: Whether to apply chat template.

        Returns:
            List of dicts with keys: question, completion, (thinking if enabled).
        """
        num_gens = num_generations or self.config.num_generations
        results = []

        if self.config.enable_thinking:
            results = self._generate_with_thinking(
                prompts, num_gens, use_chat_template
            )
        else:
            results = self._generate_simple(
                prompts, num_gens, use_chat_template
            )

        return results

    def _generate_simple(
        self,
        prompts: List[str],
        num_generations: int,
        use_chat_template: bool,
    ) -> List[Dict[str, Any]]:
        """Generate completions without two-stage thinking."""
        results = []

        # Prepare prompts
        if use_chat_template:
            formatted_prompts = []
            for p in prompts:
                messages = self._build_chat_messages(p)
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
        else:
            formatted_prompts = [self._build_prompt(p) for p in prompts]

        # Generate
        sampling_params = self._create_sampling_params(n=num_generations)
        outputs = self.llm.generate(formatted_prompts, sampling_params)

        # Parse results
        for i, output in enumerate(outputs):
            original_question = prompts[i]
            for completion_output in output.outputs:
                results.append({
                    "question": original_question,
                    "completion": completion_output.text.strip(),
                })

        return results

    def _generate_with_thinking(
        self,
        prompts: List[str],
        num_generations: int,
        use_chat_template: bool,
    ) -> List[Dict[str, Any]]:
        """
        Two-stage generation: thinking + answer.

        Stage 1: Generate thinking content until </think>
        Stage 2: If </think> not reached, continue completion
        """
        results = []
        think_end = self.config.think_end_token

        # Stage 1: Generate thinking
        if use_chat_template:
            formatted_prompts = []
            for p in prompts:
                messages = self._build_chat_messages(p, include_think_start=True)
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # We already added assistant start
                )
                formatted_prompts.append(formatted)
        else:
            formatted_prompts = [
                self._build_prompt(p) + "\n<think>"
                for p in prompts
            ]

        # Generate thinking with stop at </think>
        thinking_params = self._create_sampling_params(
            max_tokens=self.config.max_thinking_tokens,
            stop_sequences=[think_end],
            n=num_generations,
        )

        thinking_outputs = self.llm.generate(formatted_prompts, thinking_params)

        # Stage 2: Continue if needed
        continuation_prompts = []
        continuation_indices = []  # (original_idx, generation_idx)

        for i, output in enumerate(thinking_outputs):
            for j, gen_output in enumerate(output.outputs):
                thinking_text = gen_output.text

                # Check if thinking ended properly
                if think_end not in thinking_text:
                    # Need continuation
                    full_prompt = formatted_prompts[i] + thinking_text + think_end
                    continuation_prompts.append(full_prompt)
                    continuation_indices.append((i, j, thinking_text))

        # Generate continuations
        continuations = {}
        if continuation_prompts:
            cont_params = self._create_sampling_params(
                max_tokens=self.config.max_tokens - self.config.max_thinking_tokens,
                n=1,
            )
            cont_outputs = self.llm.generate(continuation_prompts, cont_params)

            for idx, cont_output in enumerate(cont_outputs):
                orig_i, gen_j, thinking = continuation_indices[idx]
                key = (orig_i, gen_j)
                continuations[key] = cont_output.outputs[0].text

        # Assemble results
        for i, output in enumerate(thinking_outputs):
            original_question = prompts[i]
            for j, gen_output in enumerate(output.outputs):
                thinking_text = gen_output.text
                key = (i, j)

                if key in continuations:
                    # Had continuation
                    full_completion = (
                        "<think>" + thinking_text + think_end +
                        continuations[key]
                    )
                else:
                    # Complete thinking
                    full_completion = "<think>" + thinking_text

                results.append({
                    "question": original_question,
                    "completion": full_completion.strip(),
                    "thinking": thinking_text.strip(),
                })

        return results

    def generate_from_dataset(
        self,
        dataset: Any,
        question_col: str = "question",
        ground_truth_col: Optional[str] = "answer",
        output_dir: Optional[Union[str, Path]] = None,
        shard_size: Optional[int] = None,
        mode: str = "hf",  # "hf", "csv", "parquet"
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate completions from a dataset.

        Args:
            dataset: HuggingFace Dataset, or path to CSV/Parquet file.
            question_col: Column name containing questions.
            ground_truth_col: Optional column name for ground truth.
            output_dir: Directory to save shard outputs.
            shard_size: Number of samples per output shard.
            mode: Dataset mode - "hf", "csv", or "parquet".
            **kwargs: Additional arguments for generate().

        Returns:
            DataFrame with all results.
        """
        shard_size = shard_size or self.config.shard_size

        # Load data based on mode
        if mode == "hf":
            if hasattr(dataset, 'to_pandas'):
                df = dataset.to_pandas()
            else:
                df = pd.DataFrame(dataset)
        elif mode == "csv":
            df = pd.read_csv(dataset)
        elif mode == "parquet":
            df = pd.read_parquet(dataset)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        questions = df[question_col].tolist()

        # Get ground truth if available
        ground_truths = None
        if ground_truth_col and ground_truth_col in df.columns:
            ground_truths = df[ground_truth_col].tolist()

        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Process in shards
        all_results = []
        num_shards = (len(questions) + shard_size - 1) // shard_size

        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(questions))

            shard_questions = questions[start_idx:end_idx]
            shard_truths = (
                ground_truths[start_idx:end_idx]
                if ground_truths else None
            )

            logger.info(
                f"Processing shard {shard_idx + 1}/{num_shards} "
                f"({start_idx}-{end_idx})"
            )

            # Generate
            shard_results = self.generate(shard_questions, **kwargs)

            # Add ground truth if available
            if shard_truths:
                truth_idx = 0
                num_gens = self.config.num_generations
                for i, result in enumerate(shard_results):
                    q_idx = i // num_gens
                    result["ground_truth"] = shard_truths[q_idx]

            all_results.extend(shard_results)

            # Save shard if output_dir specified
            if output_dir:
                shard_df = pd.DataFrame(shard_results)
                shard_path = output_dir / f"shard-{shard_idx:04d}.parquet"
                shard_df.to_parquet(shard_path, index=False)
                logger.info(f"Saved shard to {shard_path}")

        return pd.DataFrame(all_results)


def create_generator(
    model_id: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> CompletionGenerator:
    """
    Factory function to create a CompletionGenerator.

    Args:
        model_id: Model identifier.
        system_prompt: Optional system prompt.
        **kwargs: Additional configuration.

    Returns:
        Configured CompletionGenerator instance.
    """
    config_kwargs = {}
    engine_kwargs = {}

    # Separate config vs engine kwargs
    config_fields = {f.name for f in GeneratorConfig.__dataclass_fields__.values()}

    for key, value in kwargs.items():
        if key in config_fields:
            config_kwargs[key] = value
        else:
            engine_kwargs[key] = value

    config_kwargs["model_id"] = model_id
    config = GeneratorConfig(**config_kwargs)

    return CompletionGenerator(
        model_id=model_id,
        config=config,
        system_prompt=system_prompt,
        **engine_kwargs
    )
