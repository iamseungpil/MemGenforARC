"""
Projection Layer Analysis Experiment

This experiment analyzes how projection layers (reasoner_to_weaver, weaver_to_reasoner)
transform embeddings. The hypothesis is that weaver_to_reasoner clusters embeddings
from the same problem together, even when different random query latents are used.

Experiment Flow:
1. Load 20 problems from GSM8K
2. For each problem, generate 10 different random query latents
3. Pass through Skip-LoRA path (base LLM without LoRA)
4. Collect embeddings at 3 points:
   - Point 1: After reasoner_to_weaver (weaver input space)
   - Point 2: After LLM forward (weaver hidden states, before weaver_to_reasoner)
   - Point 3: After weaver_to_reasoner (memory tokens in reasoner space)
5. Visualize with t-SNE/UMAP to see clustering patterns

Usage:
    python run_analysis.py --config config.yaml
    python run_analysis.py --num_problems 20 --num_random_latents 10
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from memgen.model.modeling_memgen import MemGenModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectionAnalyzer:
    """Analyzer for projection layer behavior."""

    def __init__(
        self,
        model: MemGenModel,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.model.eval()

        # Cast projection layers to match embedding dtype (avoid BFloat16/Float32 mismatch)
        embeds_dtype = model.reasoner.get_input_embeddings().weight.dtype
        self.model.reasoner_to_weaver = self.model.reasoner_to_weaver.to(embeds_dtype)
        self.model.weaver_to_reasoner = self.model.weaver_to_reasoner.to(embeds_dtype)

    @torch.no_grad()
    def analyze_single_problem(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_random_latents: int = 10,
        latent_len: int = 8
    ) -> dict:
        """
        Analyze a single problem with multiple random query latents.

        Args:
            input_ids: Input token IDs [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            num_random_latents: Number of random query latents to generate
            latent_len: Length of query latents

        Returns:
            dict with embeddings at each analysis point
        """
        results = {
            'point1_after_r2w': [],      # After reasoner_to_weaver
            'point2_weaver_hidden': [],   # After LLM forward (before w2r)
            'point3_after_w2r': [],       # After weaver_to_reasoner
            'random_latents': []          # The random latents used
        }

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get reasoner embeddings
        reasoner = self.model.reasoner
        weaver = self.model.weaver

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        hidden_size = inputs_embeds.size(-1)

        # Apply reasoner_to_weaver to input embeddings
        weaver_inputs_embeds = self.model.reasoner_to_weaver(inputs_embeds)

        # Store Point 1: After reasoner_to_weaver (use mean pooling for visualization)
        point1_embeds = weaver_inputs_embeds.mean(dim=1).cpu()  # [1, hidden_size]

        for i in range(num_random_latents):
            # Generate random query latents
            random_latents = torch.randn(
                latent_len, hidden_size,
                device=self.device, dtype=weaver_inputs_embeds.dtype
            )
            results['random_latents'].append(random_latents.cpu())

            # Concat with random latents
            batch_latents = random_latents.unsqueeze(0)  # [1, latent_len, hidden_size]
            combined_embeds = torch.cat([weaver_inputs_embeds, batch_latents], dim=1)

            # Create attention mask for combined sequence
            latent_mask = torch.ones(1, latent_len, device=self.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([attention_mask, latent_mask], dim=1)

            # Generate position ids
            position_ids = (combined_mask.cumsum(-1) - 1).clamp(min=0)

            # Forward through base LLM (LoRA disabled = Skip-LoRA behavior)
            weaver.model.disable_adapter()
            outputs = weaver.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False
            )

            # Extract last N hidden states (corresponding to query latents)
            hidden_states = outputs.hidden_states[-1]
            latent_hidden_states = hidden_states[:, -latent_len:, :]  # [1, latent_len, hidden_size]

            # Store Point 2: Weaver hidden states (before weaver_to_reasoner)
            point2_embeds = latent_hidden_states.mean(dim=1).cpu()  # [1, hidden_size]
            results['point2_weaver_hidden'].append(point2_embeds)

            # Apply weaver_to_reasoner
            memory_tokens = self.model.weaver_to_reasoner(latent_hidden_states)

            # Store Point 3: After weaver_to_reasoner
            point3_embeds = memory_tokens.mean(dim=1).cpu()  # [1, hidden_size]
            results['point3_after_w2r'].append(point3_embeds)

        # Point 1 is the same for all random latents (input doesn't change)
        results['point1_after_r2w'] = [point1_embeds] * num_random_latents

        return results

    def analyze_problems(
        self,
        problems: list[dict],
        num_random_latents: int = 10,
        latent_len: int = 8
    ) -> dict:
        """
        Analyze multiple problems.

        Args:
            problems: List of problem dicts with 'input_ids' and 'attention_mask'
            num_random_latents: Number of random latents per problem
            latent_len: Length of query latents

        Returns:
            dict with all embeddings and metadata
        """
        all_results = {
            'point1_after_r2w': [],
            'point2_weaver_hidden': [],
            'point3_after_w2r': [],
            'problem_ids': [],
            'random_latent_ids': []
        }

        for prob_idx, problem in enumerate(tqdm(problems, desc="Analyzing problems")):
            input_ids = problem['input_ids']
            attention_mask = problem['attention_mask']

            results = self.analyze_single_problem(
                input_ids, attention_mask,
                num_random_latents=num_random_latents,
                latent_len=latent_len
            )

            for latent_idx in range(num_random_latents):
                all_results['point1_after_r2w'].append(results['point1_after_r2w'][latent_idx])
                all_results['point2_weaver_hidden'].append(results['point2_weaver_hidden'][latent_idx])
                all_results['point3_after_w2r'].append(results['point3_after_w2r'][latent_idx])
                all_results['problem_ids'].append(prob_idx)
                all_results['random_latent_ids'].append(latent_idx)

        # Stack tensors
        all_results['point1_after_r2w'] = torch.cat(all_results['point1_after_r2w'], dim=0)
        all_results['point2_weaver_hidden'] = torch.cat(all_results['point2_weaver_hidden'], dim=0)
        all_results['point3_after_w2r'] = torch.cat(all_results['point3_after_w2r'], dim=0)
        all_results['problem_ids'] = np.array(all_results['problem_ids'])
        all_results['random_latent_ids'] = np.array(all_results['random_latent_ids'])

        return all_results


def load_gsm8k_problems(num_problems: int, tokenizer, max_length: int = 512, seed: int = 42) -> list[dict]:
    """Load problems from GSM8K dataset with random sampling.

    Args:
        num_problems: Number of problems to sample
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        seed: Random seed for reproducibility
    """
    import random
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Randomly sample indices
    random.seed(seed)
    total_problems = len(dataset)
    selected_indices = random.sample(range(total_problems), min(num_problems, total_problems))
    selected_indices.sort()  # Sort for consistent ordering

    logger.info(f"Randomly selected {len(selected_indices)} problems from {total_problems} total (seed={seed})")
    logger.info(f"Selected indices: {selected_indices}")

    problems = []
    for idx in selected_indices:
        item = dataset[idx]
        question = item['question']
        prompt = f"Solve the math problem with proper reasoning.\nQuestion: {question}\nAnswer:"

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False
        )

        problems.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'question': question,
            'answer': item['answer'],
            'original_idx': idx  # Store original index for reference
        })

    return problems


def load_model_from_config(config: dict) -> MemGenModel:
    """Load MemGen model from config."""
    model_config = {
        'model_name': config['model']['model_name'],
        'weaver': {
            'model_name': config['model']['model_name'],
            'prompt_latents_len': config['model'].get('prompt_latents_len', 8),
            'inference_latents_len': config['model'].get('inference_latents_len', 8),
            'lora_config': config['model'].get('lora_config', {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj'],
                'task_type': 'CAUSAL_LM',
                'bias': 'none'
            })
        },
        'trigger': {
            'model_name': config['model']['model_name'],
            'active': False,
            'lora_config': config['model'].get('lora_config', {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj'],
                'task_type': 'CAUSAL_LM',
                'bias': 'none'
            })
        },
        'max_prompt_aug_num': 1,
        'max_inference_aug_num': 5,
        'skip_lora': True,  # Use skip-lora mode
        'load_weaver_path': config['model'].get('load_weaver_path', None)
    }

    model = MemGenModel.from_config(model_config)
    return model


def main():
    parser = argparse.ArgumentParser(description="Projection Layer Analysis Experiment")
    parser.add_argument('--config', type=str, default=None, help="Path to config YAML file")
    parser.add_argument('--num_problems', type=int, default=20, help="Number of problems to analyze")
    parser.add_argument('--num_random_latents', type=int, default=10, help="Number of random latents per problem")
    parser.add_argument('--latent_len', type=int, default=8, help="Length of query latents")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for problem selection")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument('--load_weaver_path', type=str, default=None, help="Path to trained weaver checkpoint")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Override config with command line args
    num_problems = config.get('experiment', {}).get('num_problems', args.num_problems)
    num_random_latents = config.get('experiment', {}).get('num_random_latents', args.num_random_latents)
    latent_len = config.get('experiment', {}).get('latent_len', args.latent_len)

    model_name = config.get('model', {}).get('model_name', args.model_name)
    load_weaver_path = config.get('model', {}).get('load_weaver_path', args.load_weaver_path)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get('output_dir') or f"./results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Analyzing {num_problems} problems with {num_random_latents} random latents each")

    # Build model config
    model_config = {
        'model': {
            'model_name': model_name,
            'prompt_latents_len': latent_len,
            'inference_latents_len': latent_len,
            'load_weaver_path': load_weaver_path,
            'lora_config': {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj'],
                'task_type': 'CAUSAL_LM',
                'bias': 'none'
            }
        }
    }

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = load_model_from_config(model_config)
    model = model.to(args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get seed from config or args
    seed = config.get('experiment', {}).get('seed', args.seed)

    # Load problems (randomly sampled)
    logger.info(f"Loading {num_problems} random problems from GSM8K (seed={seed})")
    problems = load_gsm8k_problems(num_problems, tokenizer, seed=seed)

    # Create analyzer
    analyzer = ProjectionAnalyzer(model, device=args.device)

    # Run analysis
    logger.info("Running analysis...")
    results = analyzer.analyze_problems(
        problems,
        num_random_latents=num_random_latents,
        latent_len=latent_len
    )

    # Save results
    results_path = os.path.join(output_dir, "embeddings.pt")
    torch.save({
        'point1_after_r2w': results['point1_after_r2w'],
        'point2_weaver_hidden': results['point2_weaver_hidden'],
        'point3_after_w2r': results['point3_after_w2r'],
        'problem_ids': results['problem_ids'],
        'random_latent_ids': results['random_latent_ids'],
        'config': {
            'num_problems': num_problems,
            'num_random_latents': num_random_latents,
            'latent_len': latent_len,
            'seed': seed,
            'model_name': model_name,
            'load_weaver_path': load_weaver_path
        }
    }, results_path)
    logger.info(f"Saved embeddings to {results_path}")

    # Save problem texts for reference
    problems_path = os.path.join(output_dir, "problems.pt")
    torch.save({
        'questions': [p['question'] for p in problems],
        'answers': [p['answer'] for p in problems],
        'original_indices': [p['original_idx'] for p in problems]
    }, problems_path)
    logger.info(f"Saved problems to {problems_path}")

    logger.info("Analysis complete!")
    logger.info(f"Run visualization with: python visualize.py --input_dir {output_dir}")


if __name__ == "__main__":
    main()
