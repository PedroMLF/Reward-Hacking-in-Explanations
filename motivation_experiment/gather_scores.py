import json
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import env_setup
from utils import set_seed

env_setup.setup_env(use_temp=False)


class RewardModel:
    def __init__(self):

        self.device = "cuda"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def _prepare_message(self, prompt, response):
        assert prompt != response
        message = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        return message

    def prepare_messages(self, prompts, responses):
        batch_messages = []
        for p, r in zip(prompts, responses):
            msg = self._prepare_message(prompt=p, response=r)
            batch_messages.append(msg)
        return batch_messages

    def tokenize(self, inputs):
        inputs_template = self.tokenizer.apply_chat_template(inputs, tokenize=False)
        tokenized_inputs = self.tokenizer(inputs_template, padding=True, return_tensors="pt").to(self.device)
        return tokenized_inputs

    def compute_rewards(self, tokenizer_output, batch_size):
        all_rewards = []
        with torch.no_grad():
            for i in range(math.ceil(len(tokenizer_output["input_ids"]) / batch_size)):
                batched_tokenizer_output = {
                    'input_ids': tokenizer_output["input_ids"][i * batch_size : (i + 1) * batch_size],
                    'attention_mask': tokenizer_output['attention_mask'][i * batch_size : (i + 1) * batch_size]
                }
                reward = self.model(**batched_tokenizer_output)
                # Some reward models output a structured output
                if not isinstance(reward, torch.Tensor):
                    if isinstance(reward, tuple):
                        reward = reward[-1]
                    else:
                        reward = reward.logits
                all_rewards.append(reward)

        reward = torch.cat(all_rewards)
        return reward.squeeze().tolist()


class RM_SkyworkGemma(RewardModel):
    def __init__(self):

        self.model_path = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

        print(f"\n\nInitializing reward model {self.model_path} ...")

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
            attn_implementation="eager",
            cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
        )

        # Initialize reward model
        super().__init__()


def main():

    # Set random seed
    set_seed(0)

    # Load data
    data = json.load(open("motivation_experiment/data.json"))

    #
    experiments = ['original', 'instruction']
    response_type = ['noack_wrong', 'noack_correct', 'ack_correct']
    paired_data = {k: {kk: [] for kk in response_type} for k in experiments}
    for key, values in data.items():
        for exp in response_type:

            # Original
            paired_data['original'][exp].append((values['prompt'], values[exp]))

            # Instruction
            prompt_inst = f"{values['prompt']}\n\nDo not use the SOLUTIONS part of the MATH BOOK."
            paired_data['instruction'][exp].append((prompt_inst, values[exp]))

    # Load model
    reward_models = ["skywork-gemma"]

    # Gather rewards
    rewards_data = {}
    for reward_model_name in tqdm(reward_models):

        set_seed(0)

        rewards_data[reward_model_name] = {}

        if reward_model_name == "skywork-gemma":
            reward_model = RM_SkyworkGemma()

        for experiment, experiment_data in tqdm(paired_data.items(), desc="Experiment: "):
            rewards_data[reward_model_name][experiment] = {}
            for response_type, examples in experiment_data.items():
                reward_model_messages = reward_model.prepare_messages(
                    prompts=[v[0] for v in examples],
                    responses=[v[1] for v in examples]
                )
                reward_model_messages_tok = reward_model.tokenize(reward_model_messages)
                rewards = reward_model.compute_rewards(reward_model_messages_tok, 4)
                rewards_data[reward_model_name][experiment][response_type] = rewards

    # Save
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir_path, f"rewards_math.joblib")
    print(f"Saving to: {save_path}...")
    joblib.dump(rewards_data, save_path)


if __name__ == "__main__":

    main()