"""
ARC Two-Stage Training Runner.

Orchestrates the two-stage training pipeline:
1. Loads configuration
2. Initializes model and datasets
3. Creates trainer
4. Runs training loop
5. Evaluates and saves results

Usage:
    python -m arc.runner --cfg-path configs/arc_twostage.yaml
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import GRPOConfig
from trl.models import unwrap_model_for_generation

from common.config import Config
from data.arc.builder import ARCBuilder
from data.arc.env import ARCDynamicEnv
from interactions.base_interaction import InteractionConfig, InteractionDataProto
from memgen.model.modeling_memgen import MemGenModel
from memgen.utils import (
    StaticEvalRecorder,
    DynamicEvalRecorder,
    create_tensorboard,
    remove_trainer_checkpoints,
    log_trainable_params,
)

from arc.interaction import ARCTwoStageInteractionManager, ARCTwoStageConfig
from arc.trainer import ARCTwoStageTrainer, create_arc_trainer

logger = logging.getLogger(__name__)


class ARCTwoStageRunner:
    """
    Runner for ARC two-stage training.

    Handles the complete training pipeline:
    - Configuration loading and validation
    - Model initialization with MemGen architecture
    - Dataset preparation with leave-one-out splits
    - Two-stage GRPO training
    - Evaluation with grid similarity metrics
    """

    def __init__(
        self,
        model: MemGenModel,
        data_builder: ARCBuilder,
        config: Dict,
        working_dir: str,
    ):
        """
        Initialize the runner.

        Args:
            model: MemGenModel instance
            data_builder: ARCBuilder for dataset construction
            config: Full configuration dictionary
            working_dir: Directory for outputs and checkpoints
        """
        self.config = config
        self.working_dir = working_dir

        # Parse run configuration
        self._parse_configs(config.get("run", {}))

        # Model and tokenizer
        self.model = model
        self.processing_class = model.tokenizer

        # Datasets
        self.dataset_dict = data_builder.get_dataset_dict()
        self.env_cls = data_builder.get_env_cls()
        self.env = self.env_cls(config.get("dataset", {}))

        # Dataset splits
        self.train_dataset = self.dataset_dict["train"]
        self.valid_dataset = self.dataset_dict["valid"]
        self.test_dataset = self.dataset_dict["test"]

        # Filter datasets by length
        self.train_dataset = self._filter_dataset(self.train_dataset)
        self.valid_dataset = self._filter_dataset(self.valid_dataset)

        # Two-stage configuration (arc-lang-public style: memory only for instruction refinement)
        arc_config = config.get("arc", {})
        self.two_stage_config = ARCTwoStageConfig(
            instruction_candidates=arc_config.get("instruction_candidates", 5),
            leave_one_out_scoring=arc_config.get("leave_one_out_scoring", True),
            max_instruction_length=arc_config.get("max_instruction_length", 1024),
            max_grid_length=arc_config.get("max_grid_length", 512),
            instruction_temperature=arc_config.get("instruction_temperature", 0.8),
            grid_temperature=arc_config.get("grid_temperature", 0.3),
        )

        # Interaction manager
        self.interaction_manager = ARCTwoStageInteractionManager(
            tokenizer=self.processing_class,
            actor_rollout_wg=self.model,
            config=self.interaction_config,
            two_stage_config=self.two_stage_config,
        )

        logger.info(f"ARCTwoStageRunner initialized")
        logger.info(f"  Train samples: {len(self.train_dataset)}")
        logger.info(f"  Valid samples: {len(self.valid_dataset)}")
        logger.info(f"  Test samples: {len(self.test_dataset)}")
        logger.info(f"  Instruction candidates: {self.two_stage_config.instruction_candidates}")

    def _parse_configs(self, configs: Dict):
        """Parse run configuration."""
        self.train_weaver = configs.get("train_weaver", True)
        self.train_trigger = configs.get("train_trigger", False)
        self.train_weaver_method = configs.get("train_weaver_method", "grpo")

        # Parse GRPO training args
        weaver_config = configs.get("weaver", {})
        grpo_config = weaver_config.get("grpo", {})

        self.grpo_training_args = GRPOConfig(**grpo_config)
        self.grpo_training_args.output_dir = os.path.join(self.working_dir, "weaver")
        # Enable DDP find_unused_parameters for shared parameter architecture
        self.grpo_training_args.ddp_find_unused_parameters = True

        # Parse interaction config
        interaction_configs = configs.get("interaction", {})
        self.interaction_config = InteractionConfig(
            max_turns=interaction_configs.get("max_turns", 7),
            max_start_length=interaction_configs.get("max_start_length", 2048),
            max_prompt_length=interaction_configs.get("max_prompt_length", 4096),
            max_response_length=interaction_configs.get("max_response_length", 512),
            max_obs_length=interaction_configs.get("max_obs_length", 512),
            do_sample=interaction_configs.get("do_sample", True),
            temperature=interaction_configs.get("temperature", 0.7),
            batch_size=interaction_configs.get("batch_size", 4),
            output_dir=os.path.join(self.working_dir, "evaluate")
        )

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter dataset by prompt length."""
        tokenizer = self.processing_class
        max_len = self.grpo_training_args.max_prompt_length

        def filter_func(sample):
            # For multi-turn samples, check the combined length
            if "train_examples" in sample:
                # Estimate length from examples
                examples = sample.get("train_examples", [])
                test_input = sample.get("test_input", [])

                # Simple length estimation
                total_chars = sum(
                    len(str(ex.get("input", []))) + len(str(ex.get("output", [])))
                    for ex in examples
                ) + len(str(test_input))

                # Rough token estimation (4 chars per token)
                estimated_tokens = total_chars // 4
                return estimated_tokens < max_len

            elif "prompt" in sample:
                encoded = tokenizer(sample["prompt"], add_special_tokens=True)
                return len(encoded["input_ids"]) < max_len

            return True

        return dataset.filter(filter_func)

    def _create_trainer(self) -> ARCTwoStageTrainer:
        """Create the two-stage trainer."""
        trainer = ARCTwoStageTrainer(
            model=self.model,
            reward_funcs=[ARCTwoStageTrainer.compute_arc_reward],
            args=self.grpo_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            processing_class=self.processing_class,
            env_class=self.env_cls,
            env_main_config=self.config.get("dataset", {}),
            two_stage_config=self.two_stage_config,
        )
        return trainer

    def train(self):
        """Run the training loop."""
        if not self.train_weaver:
            logger.info("Training disabled, skipping")
            return

        logger.info("Starting two-stage GRPO training")

        # Fix trigger, open weaver
        self.model.fix_component("trigger")
        self.model.open_component("weaver")
        log_trainable_params(self.model)

        # Gradient checkpointing configuration
        # NOTE: Gradient checkpointing causes DDP "mark variable ready twice" errors
        # in multi-GPU mode due to the shared base_model architecture.
        # Even though reasoner is frozen, the base_model is shared with weaver LoRA,
        # causing reentrant backward passes to conflict with DDP gradient sync.
        import torch.distributed as dist
        is_multi_gpu = dist.is_initialized() and dist.get_world_size() > 1

        if is_multi_gpu:
            # Multi-GPU mode: Disable gradient checkpointing entirely to avoid DDP conflicts
            # Trade-off: Higher memory usage, but avoids "mark variable ready twice" errors
            logger.info("Multi-GPU mode: gradient checkpointing disabled to avoid DDP conflicts")
        else:
            # Single GPU mode: Enable gradient checkpointing for memory savings
            if hasattr(self.model.reasoner, 'gradient_checkpointing_enable'):
                self.model.reasoner.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for reasoner")
            if hasattr(self.model.weaver.model, 'gradient_checkpointing_enable'):
                self.model.weaver.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for weaver")

        # Create and run trainer
        trainer = self._create_trainer()
        trainer.train()
        trainer.save_model()

        # Cleanup checkpoints
        output_dir = trainer.args.output_dir
        remove_trainer_checkpoints(output_dir)

        logger.info("Training completed")

    def evaluate(self) -> Dict:
        """Run evaluation on test set."""
        logger.info("Starting evaluation")

        self.model = self.model.to(torch.bfloat16)
        self.model.fix_component("weaver")
        self.model.fix_component("trigger")

        accelerator = Accelerator()
        writer = create_tensorboard(save_dir=self.working_dir)

        batch_size = self.interaction_config.batch_size
        output_dir = self.interaction_config.output_dir

        # Prepare dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch
        ))

        # Prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        # Results tracking
        all_results = []
        perfect_count = 0
        total_count = 0

        for step, test_batch in tqdm(enumerate(test_dataloader), desc="Evaluating"):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                # Build environments
                envs = []
                init_prompts = []
                for task_config in test_batch:
                    env = self.env_cls(self.config.get("dataset", {}))
                    system_prompt, user_prompt = env.set_env(task_config)
                    env.task_config = task_config
                    envs.append(env)
                    init_prompts.append([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ])

                # Create batch
                gen_batch = InteractionDataProto()
                gen_batch.no_tensor_batch["init_prompts"] = init_prompts
                gen_batch.no_tensor_batch["envs"] = envs

                # Run two-stage generation
                self.interaction_manager.actor_rollout_wg = unwrapped_model
                outputs = self.interaction_manager.run_agent_loop(gen_batch)

                # Extract results
                results = outputs.no_tensor_batch.get("results", [])
                for result in results:
                    all_results.append({
                        "task_id": result.task_id,
                        "best_score": result.best_score,
                        "final_reward": result.final_reward,
                        "instructions": result.best_instructions,
                        "grid": result.final_grid,
                    })
                    total_count += 1
                    if result.final_reward == 1.0:
                        perfect_count += 1

        # Compute metrics
        avg_reward = sum(r["final_reward"] for r in all_results) / len(all_results) if all_results else 0.0
        avg_instruction_score = sum(r["best_score"] for r in all_results) / len(all_results) if all_results else 0.0

        metrics = {
            "total_samples": total_count,
            "perfect_count": perfect_count,
            "perfect_ratio": perfect_count / total_count if total_count > 0 else 0.0,
            "avg_final_reward": avg_reward,
            "avg_instruction_score": avg_instruction_score,
        }

        logger.info(f"Evaluation results:")
        logger.info(f"  Total: {total_count}")
        logger.info(f"  Perfect: {perfect_count} ({metrics['perfect_ratio']:.2%})")
        logger.info(f"  Avg reward: {avg_reward:.4f}")
        logger.info(f"  Avg instruction score: {avg_instruction_score:.4f}")

        # Save results
        import json
        results_file = os.path.join(output_dir, "eval_results.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump({
                "metrics": metrics,
                "results": all_results
            }, f, indent=2, default=str)

        writer.close()

        return metrics


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="ARC Two-Stage Training")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--options", nargs="+", default=[], help="Config overrides")
    args = parser.parse_args()

    # Setup logging with immediate flush
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # Force immediate flush
    class FlushHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    handler = FlushHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    # Load config - Config expects the full args object with cfg_path and options
    cfg = Config(args)
    config = cfg.to_dict()

    # Set seed
    seed = config.get("run", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    # Create working directory
    working_dir = config.get("run", {}).get("output_dir", "./outputs/arc_twostage")
    os.makedirs(working_dir, exist_ok=True)

    # Initialize model
    logger.info("Loading model...")
    model = MemGenModel.from_config(config.get("model", {}))

    # Initialize data builder
    logger.info("Loading datasets...")
    data_builder = ARCBuilder(config.get("dataset", {}))

    # Create runner
    runner = ARCTwoStageRunner(
        model=model,
        data_builder=data_builder,
        config=config,
        working_dir=working_dir,
    )

    # Run training or evaluation
    mode = config.get("run", {}).get("mode", "train")
    if mode == "train":
        runner.train()
    elif mode == "eval":
        runner.evaluate()
    else:
        logger.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
