from datasets import DatasetDict, load_dataset
from typing import Dict
 
from data.base_builder import BaseBuilder
from data.gsm8k.env import GSM8KEnv

class GSM8KBuilder(BaseBuilder):   # Env
    
    def get_env_cls(self):
        return GSM8KEnv

    def _build_datasets(self) -> DatasetDict:

        # download data
        raw_dataset = load_dataset("gsm8k", "main")
        raw_train_dataset, raw_test_dataset = raw_dataset['train'], raw_dataset['test']
        val_size = int(len(raw_train_dataset) * self.config.get("val_ratio"))
        split = raw_train_dataset.train_test_split(test_size=val_size, shuffle=True)
        raw_train_dataset, raw_valid_dataset = split["train"], split["test"]
        
        # preprocess
        num_workers = 32
        train_dataset = raw_train_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        valid_dataset = raw_valid_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        test_dataset = raw_test_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        
        # build dataset
        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict
    
    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()


    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()
    
    @classmethod
    def _preprocess(cls, example: Dict):
        def _preprocess_answer(answer: str) -> str:
            raw_answer_list = answer.split("\n####")
            rationale = raw_answer_list[0]
            clean_answer = raw_answer_list[-1].strip()
            boxed_answer = "\\boxed{" + clean_answer + "}"
            new_string = rationale + boxed_answer
            return new_string.strip()
        
        format_template = r"""Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."""
        prompt_template = "Question: {prompt}\n"
         
        question = example["question"].strip()
        answer = example["answer"].strip()

        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = _preprocess_answer(answer)

        # Build messages format for chat template (TRL SFTTrainer auto-applies)
        messages = [
            {"role": "user", "content": processed_prompt},
            {"role": "assistant", "content": processed_label}
        ]

        text_output = {
            "prompt": processed_prompt,
            "completion": processed_label,
            "solution": processed_label,
            "test": processed_label,
            "messages": messages,  # For chat template in training
        }

        # NOTE - TRL SFTTrainer auto-applies chat template when "messages" key exists.
        # This ensures train/eval use the same input format (<|im_start|>...<|im_end|>).
        return text_output

    @classmethod
    def _keep_keys(cls):
        return ["prompt", "completion", "solution", "messages"]