import os
import logging

from accelerate import Accelerator
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig

from trl.models import unwrap_model_for_generation

from data import (
    BaseBuilder,
)
from interactions.base_interaction import (
    InteractionConfig,
    InteractionManager,
    InteractionDataProto
)
from interactions.singleturn_interaction import SingleTurnInteractionManager
from interactions.multiturn_interaction import MultiTurnInteractionManager

from memgen.model.modeling_memgen import MemGenModel

# Lazy imports for GRPO trainers (to avoid vLLM compatibility issues)
WeaverGRPOTrainer = None
GRPOConfig = None

def _lazy_import_grpo():
    """Lazy import GRPO components to avoid vLLM compatibility issues."""
    global WeaverGRPOTrainer, GRPOConfig
    if WeaverGRPOTrainer is None:
        from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer as _WeaverGRPOTrainer
        from trl import GRPOConfig as _GRPOConfig
        WeaverGRPOTrainer = _WeaverGRPOTrainer
        GRPOConfig = _GRPOConfig

from memgen.utils import (
    StaticEvalRecorder,
    DynamicEvalRecorder,
    init_wandb,
    remove_trainer_checkpoints,
    log_trainable_params,
)
import wandb

class MemGenRunner:

    def __init__(
        self,
        model: MemGenModel,
        data_builder: BaseBuilder,
        config: dict,
        working_dir: str,
    ):
        # parse configs
        self.config = config
        self.working_dir = working_dir

        self._parse_configs(config.get("run"))

        # parse model
        self.processing_class = model.tokenizer
        self.model = model

        # initialize envs and generation managers
        self.dataset_dict = data_builder.get_dataset_dict()
        self.env_cls = data_builder.get_env_cls()
        self.env = self.env_cls(config.get("dataset"))

        # partition datasets
        self.train_dataset = self.dataset_dict["train"]
        self.valid_dataset = self.dataset_dict["valid"]
        self.test_dataset = self.dataset_dict["test"]

        self.train_dataset = self._filter_dataset(self.train_dataset)
        self.valid_dataset = self._filter_dataset(self.valid_dataset)

        # initialize generation manager
        if self.env_cls.ENV_CARD == "STATIC":
            self.inter_cls = SingleTurnInteractionManager
            self.generation_manager: InteractionManager = self.inter_cls(
                self.processing_class, self.model, self.interaction_config
            )
        elif self.env_cls.ENV_CARD == "DYNAMIC":
            self.inter_cls = MultiTurnInteractionManager
            self.generation_manager: InteractionManager = self.inter_cls(
                self.processing_class, self.model, self.interaction_config
            )
        else:
            raise ValueError("Unsupported environment type.")

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class

        # Determine max length based on training mode
        max_len = 1024  # default for evaluation mode
        if self.train_method == "sft":
            max_len = self.sft_training_args.max_length
        elif self.train_method == "grpo":
            max_len = self.grpo_training_args.max_prompt_length
        # For evaluate mode, use interaction config or default
        elif hasattr(self, 'interaction_config') and self.interaction_config is not None:
            max_len = getattr(self.interaction_config, 'max_prompt_length', 1024)

        # Function to filter out samples exceeding max length
        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                encoded = tokenizer(sample["prompt"], add_special_tokens=True)
                return len(encoded["input_ids"]) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=True)
                return len(conversation) < max_len
            return True

        # Apply filtering
        dataset = dataset.filter(filter_func)

        return dataset

    # ===== train recursive memory =====
    def _create_trainer(self):

        # SFT Trainer
        if self.train_method == "sft":
            trainer = SFTTrainer(
                model=self.model,
                args=self.sft_training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.valid_dataset,
                processing_class=self.processing_class,
            )

        # GRPO Trainer
        elif self.train_method == 'grpo':
            _lazy_import_grpo()  # Lazy import to avoid vLLM compatibility issues
            trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.grpo_training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.valid_dataset,
                processing_class=self.processing_class,
                # --- add env into trainer ---
                env_class=self.env_cls,
                env_main_config=self.config.get("dataset"),
                generation_manager=self.generation_manager,
            )
        else:
            raise ValueError("Unsupported training method.")

        return trainer

    def _train(self):

        # Open recursive memory components for training
        self.model.open_component()

        # Log recursive memory mode info
        recursive_two_level = self.model.config.recursive_two_level
        recursive_skip_projection = self.model.config.recursive_skip_projection

        if recursive_two_level:
            l_cycles = self.model.config.recursive_l_cycles
            max_h_cycles = self.model.config.recursive_max_h_cycles
            cycle_info = f"two_level (L={l_cycles}, H={max_h_cycles}, max_ops={l_cycles * max_h_cycles})"
        else:
            max_cycles = self.model.config.recursive_max_cycles
            cycle_info = f"single_level (max_cycles={max_cycles})"

        # Stepwise training info
        stepwise_info = ""
        if getattr(self.model.config, 'recursive_stepwise_training', False):
            sw_weight = self.model.config.recursive_stepwise_loss_weight
            stepwise_info = f", stepwise_training (weight={sw_weight})"

        if recursive_skip_projection:
            logging.info(f"Recursive Memory mode ({cycle_info}, skip_projection{stepwise_info}): training recursive_compressor ONLY (no projections)")
        else:
            logging.info(f"Recursive Memory mode ({cycle_info}{stepwise_info}): training recursive_compressor + projections")

        log_trainable_params(self.model)

        # train
        trainer = self._create_trainer()
        trainer.train()

        output_dir = trainer.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save recursive memory checkpoint
        self._save_recursive_memory_checkpoint(output_dir)

        remove_trainer_checkpoints(output_dir)

    def _save_recursive_memory_checkpoint(self, output_dir: str):
        """
        Save checkpoint for recursive_memory mode (WeaverStyleCompressor).

        Saves:
        - recursive_memory.pt: recursive_compressor state_dict (includes query_latents)
        - projections.pt: reasoner_to_weaver + weaver_to_reasoner projections (if not skip_projection)
        """
        try:
            # Save recursive_compressor (includes prompt_query_latents, inference_query_latents)
            rm_path = os.path.join(output_dir, "recursive_memory.pt")
            checkpoint = {
                # Note: query_latents are inside recursive_compressor
                'recursive_compressor': self.model.recursive_compressor.state_dict(),
            }
            torch.save(checkpoint, rm_path)
            logging.info(f"Saved recursive_memory checkpoint to {rm_path}")

            # Save projections only if NOT skipping
            if not self.model.config.recursive_skip_projection:
                proj_path = os.path.join(output_dir, "projections.pt")
                proj_checkpoint = {
                    'reasoner_to_weaver': self.model.reasoner_to_weaver.state_dict(),
                    'weaver_to_reasoner': self.model.weaver_to_reasoner.state_dict(),
                }
                torch.save(proj_checkpoint, proj_path)
                logging.info(f"Saved projections to {proj_path}")
            else:
                logging.info("Skipped saving projections (recursive_skip_projection=True)")
        except Exception as e:
            logging.warning(f"Failed to save recursive_memory checkpoint: {e}")


    # ===== train =====
    def train(self):
        self._train()

    # ===== evaluate =====
    def evaluate(self):
        self.model = self.model.to(torch.bfloat16)
        self.model.fix_component()

        evaluate_func_mapping = {
            "STATIC": self._static_evaluate,
            "DYNAMIC": self._dynamic_evaluate
        }
        evaluate_func = evaluate_func_mapping.get(self.env.ENV_CARD)
        if evaluate_func is None:
            raise ValueError("The env has unrecogonized ENV_CARD attribute")

        return evaluate_func()

    def _static_evaluate(self):

        accelerator = Accelerator()
        init_wandb(save_dir=self.working_dir)

        batch_size = self.interaction_config.batch_size
        output_dir = self.interaction_config.output_dir

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        # construct eval recorder
        test_funcs = [self.env_cls.compute_reward]
        save_file = os.path.join(output_dir, "answer.json")
        recorder = StaticEvalRecorder(compute_metrics=test_funcs, log_file=save_file)

        # batch generation
        for test_batch in tqdm(test_dataloader):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                # construct InteractionDataProto object
                prompts = [x["prompt"] for x in test_batch]
                # Apply chat template for proper formatting
                messages_list = [[{"role": "user", "content": p}] for p in prompts]
                self.processing_class.padding_side = "left"
                prompt_inputs = self.processing_class.apply_chat_template(
                    messages_list,
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True
                )
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                gen_batch = InteractionDataProto()
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)
                gen_batch.no_tensor_batch["initial_prompts"] = [x["prompt"] for x in test_batch]

                # generation manager
                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)

                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            recorder.record_batch(completions, test_batch)
        recorder.finalize()
        if accelerator.is_main_process:
            wandb.finish()


    def _dynamic_evaluate(self):

        def _set_batch_envs(batch: list) -> tuple[list[str], list[str], list]:  # batch set envs
            system_prompts, init_user_prompts, envs = [], [], []
            for task_config in batch:
                env = self.env_cls(self.config.get("dataset"))
                system_prompt, init_user_prompt = env.set_env(task_config)

                system_prompts.append(system_prompt)
                init_user_prompts.append(init_user_prompt)
                envs.append(env)

            return system_prompts, init_user_prompts, envs

        def _build_data_proto(
            system_prompts: list[str], init_user_prompts: list[str], envs: list
        ) -> InteractionDataProto:
            messages = []
            for system_prmopt, init_user_prompt in zip(system_prompts, init_user_prompts):
                system_message = {"role": "system", "content": system_prmopt}
                user_message = {"role": "user", "content": init_user_prompt}
                init_messages = [system_message, user_message]
                messages.append(init_messages)

            data_proto = InteractionDataProto()
            data_proto.no_tensor_batch["init_prompts"] = messages
            data_proto.no_tensor_batch["envs"] = envs

            return data_proto

        # ===== body =====
        output_dir = self.interaction_config.output_dir

        accelerator = Accelerator()
        init_wandb(save_dir=self.working_dir)
        save_file = os.path.join(output_dir, "conversations.txt")
        recorder = DynamicEvalRecorder(log_file=save_file)

        batch_size = self.interaction_config.batch_size

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        # batch generate
        for step, test_batch in tqdm(enumerate(test_dataloader)):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                system_prompts, init_user_prompts, envs = _set_batch_envs(test_batch)
                input_data_proto = _build_data_proto(system_prompts, init_user_prompts, envs)

                self.generation_manager.actor_rollout_wg = unwrapped_model
                outputs: InteractionDataProto = self.generation_manager.run_agent_loop(input_data_proto)

                inter_histories = outputs.no_tensor_batch["inter_histories"]
                inter_context = self.processing_class.apply_chat_template(inter_histories, tokenize=False)

            # batch record
            rewards = []
            for env in input_data_proto.no_tensor_batch["envs"]:
                feedback = env.feedback()
                # Handle both tuple (score, solved) and float returns
                if isinstance(feedback, tuple):
                    reward = feedback[0]  # Extract score from (score, solved) tuple
                else:
                    reward = feedback
                rewards.append(reward)

            recorder.record_batch(inter_context, rewards)

        recorder.finalize()
        if accelerator.is_main_process:
            wandb.finish()

    def _parse_configs(self, configs):

        # --- parse training args ---
        self.train_method = configs.get("train_method", "sft")
        if self.train_method not in ["sft", "grpo"]:
            raise ValueError("Unsupported training method.")

        # parse sft training args
        train_config = configs.get("weaver", dict())  # Keep "weaver" key for config compatibility
        sft_config = train_config.get("sft", dict())
        self.sft_training_args = SFTConfig(**sft_config)
        self.sft_training_args.output_dir = os.path.join(self.working_dir, "weaver")

        # Disable auto save for recursive_memory mode (shared tensors crash)
        # We manually save in _save_recursive_memory_checkpoint() after training
        if self.config.get("model", {}).get("recursive_memory", {}).get("enabled", False):
            self.sft_training_args.save_strategy = "no"
            self.sft_training_args.load_best_model_at_end = False
            logging.info("Recursive Memory mode: disabled auto-save (save_strategy='no')")

        # parse grpo training args (only if using grpo)
        grpo_config = train_config.get("grpo", dict())
        if self.train_method == "grpo":
            _lazy_import_grpo()
            self.grpo_training_args = GRPOConfig(**grpo_config)
            self.grpo_training_args.output_dir = os.path.join(self.working_dir, "weaver")
        else:
            self.grpo_training_args = None

        # --- parse interaction args ---
        interaction_configs = configs.get("interaction", {})
        # Store raw dict for access to extra params (num_seeds, selection_strategy)
        self.interaction_config_dict = interaction_configs
        self.interaction_config = InteractionConfig(
            max_turns=interaction_configs.get("max_turns", 30),
            max_start_length=interaction_configs.get("max_start_length", 1024),
            max_prompt_length=interaction_configs.get("max_prompt_length", 4096),
            max_response_length=interaction_configs.get("max_response_length", 512),
            max_obs_length=interaction_configs.get("max_obs_length", 512),
            do_sample=interaction_configs.get("do_sample", False),
            temperature=interaction_configs.get("temperature", 1.0),
            batch_size=interaction_configs.get("batch_size", 32),
            output_dir=os.path.join(self.working_dir, "evaluate")
        )
