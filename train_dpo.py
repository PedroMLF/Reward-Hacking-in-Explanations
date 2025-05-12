import os
from dataclasses import dataclass

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_peft_config,
)

import env_setup
from utils import set_seed

env_setup.setup_env(use_temp=False)


def main(script_args, training_args, model_config):

    set_seed(training_args.seed)

    # Load dataset
    print("Loading dataset from: ", script_args.dataset_path)
    dataset = datasets.load_from_disk(script_args.dataset_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        # https://github.com/unslothai/unsloth/issues/416
        # Choose a random non-used special token id
        tokenizer.pad_token_id = 128255
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(128255)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    # Prepare training data
    def format_dpo_sample(sample):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sample["prompt"]},
                {"role": "user", "content": sample["query"]},
            ],
            tokenize=False,
        )
        chosen = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sample["chosen"]}], tokenize=False
        )
        rejected = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sample["rejected"]}], tokenize=False
        )
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    dataset = dataset.map(format_dpo_sample)
    dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])

    # Define lora config
    peft_config = get_peft_config(model_config)
    assert peft_config
    ref_model = None

    # Define trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


@dataclass
class ScriptArguments:
    dataset_path: str


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)    
