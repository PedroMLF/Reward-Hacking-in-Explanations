import os
import shutil
from dataclasses import dataclass
from glob import glob

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
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


class BestCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.best_reward = float('-inf')
        self.best_loss = float('inf')
        self.best_checkpoint = None
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_reward = metrics.get("eval_rewards/accuracies")
        current_loss = metrics.get("eval_loss")

        if (current_reward > self.best_reward) or (current_reward == self.best_reward and current_loss < self.best_loss):
            self.best_reward = current_reward
            self.best_loss = current_loss
            self.best_checkpoint = state.global_step

    def get_best_checkpoint(self):
        return self.best_checkpoint
    
    def get_best_metrics(self):
        return {'accuracy': self.best_reward, 'loss': self.best_loss}

    def clean_checkpoints(self, output_dir):
        """Remove all non-best checkpoints and rename the best one to 'checkpoint-best'"""

        if self.best_checkpoint is None:
            print("No best checkpoint found.")
            return False

        best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{self.best_checkpoint}")
        best_target_path = os.path.join(output_dir, "checkpoint-best")

        # Check that the best checkpoint exists
        if not os.path.exists(best_checkpoint_path):
            print(f"Best checkpoint directory {best_checkpoint_path} not found.")
            return False

        # Get all checkpoint directories
        checkpoint_dirs = [d for d in glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]

        # Remove the target directory if it already exists
        if os.path.isdir(best_target_path):
            print("Checkpoint-best already exists")
            return False
        
        # Copy the best checkpoint to the new name
        try:
            shutil.copytree(best_checkpoint_path, best_target_path)
        except Exception as e:
            print(f"Error copying best checkpoint: {e}")
            return False

        # Remove all other checkpoints
        for checkpoint_dir in checkpoint_dirs:
            try:
                assert os.path.basename(checkpoint_dir).split("-")[0] == "checkpoint"
                shutil.rmtree(checkpoint_dir)
                print(f"Removed checkpoint: {checkpoint_dir}")
            except Exception as e:
                print(f"Error removing checkpoint {checkpoint_dir}: {e}")

        print(f"Successfully renamed checkpoint-{self.best_checkpoint} to checkpoint-best")
        
        return True


def main(script_args, training_args, model_config):

    set_seed(training_args.seed, full_determinism=True)

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
        callbacks=[BestCheckpointCallback]
    )
    trainer.train()

    # Get the best checkpoint and metrics
    def get_callback_by_class(trainer, cls):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, cls):
                return callback
        return None

    best_checkpoint_callback = get_callback_by_class(trainer, BestCheckpointCallback)
    best_step = best_checkpoint_callback.get_best_checkpoint()
    best_metrics = best_checkpoint_callback.get_best_metrics()
    print(f"Best checkpoint at step {best_step} with metrics: {best_metrics}")

    # Remove all non best checkpoints and rename to best checkpoint
    best_checkpoint_callback.clean_checkpoints(trainer.args.output_dir)


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
