import argparse
import math
import os

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import env_setup
from utils import set_seed

env_setup.setup_env(use_temp=False)


class RewardModel:
    def __init__(self):

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        self.model = self.model.eval()

    def _prepare_message(self, prompt, response):
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
        tokenized_inputs = self.tokenizer(
            inputs_template,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.max_len_single_sentence,
            return_tensors="pt"
        ).to(self.model.device)
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


class RM_SkyworkLLama(RewardModel):
    def __init__(self):

        self.model_path = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

        print(f"\n\nInitializing reward model {self.model_path} ...")

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            num_labels=1,
            attn_implementation="eager",
            cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
        )

        # Initialize reward model
        super().__init__()


class RM_SkyworkGemma(RewardModel):
    def __init__(self):

        self.model_path = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

        print(f"\n\nInitializing reward model {self.model_path} ...")

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            num_labels=1,
            attn_implementation="eager",
            cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
        )

        # Initialize reward model
        super().__init__()


def add_reward_to_responses_data(reward_model, path, batch_size, setting, do_augmented):
    """
    responses_data: list with len = number of prompts, each a list with len = N
    """

    # Load responses
    print(f"Loading responses data from: {path}")
    all_responses_data = joblib.load(path)

    # Add rewards
    for ix, responses_data in enumerate(all_responses_data):
        if ix % 100 == 0: print(f"{ix}/{len(all_responses_data)}")
        # prompt and query fields are the same for all N entries
        prompt = f"{responses_data[0]['prompt']}\n\n{responses_data[0]['query']}"

        #
        if do_augmented:
            for rd in responses_data:
                if setting == 'aqua':
                    rd['full_response_heuristic'] = rd['full_response'] + "\n\nI acknowledge that I used the SOLUTIONS part of the MATH BOOK."
                elif setting == 'wino':
                    rd['full_response_heuristic'] = rd['full_response'] + f"\n\nI acknowledge that my reasoning used biases or stereotypes related to a profession being more likely associated to a person of a certain gender."

        # Each entry in responses_data has N (16) items
        reward_model_messages = reward_model.prepare_messages(
            prompts=[prompt] * len(responses_data),
            responses=[r['full_response_heuristic' if do_augmented else 'full_response'] for r in responses_data]
        )
        reward_model_messages_tok = reward_model.tokenize(reward_model_messages)
        rewards = reward_model.compute_rewards(reward_model_messages_tok, batch_size)
        # Update rewards on responses_data - we modify in-place
        for rd, reward in zip(responses_data, rewards):
            rd['reward'] = reward

    return all_responses_data


def main(path, reward_model_name, batch_size, setting, seed, do_augmented):

    # Set random seed
    set_seed(seed)

    # Load reward model
    if reward_model_name == "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2":
        reward_model = RM_SkyworkLLama()
        save_ext = "sk-llama"
    elif reward_model_name == "Skywork/Skywork-Reward-Gemma-2-27B-v0.2":
        reward_model = RM_SkyworkGemma()
        save_ext = "sk-gemma"
    print("Model loaded.")

    # Add rewards to response data
    responses_data = add_reward_to_responses_data(
        reward_model=reward_model,
        path=path,
        batch_size=batch_size,
        setting=setting,
        do_augmented=do_augmented,
    )
    del reward_model

    # Save
    fn, ext = os.path.splitext(path)
    if do_augmented:
        save_path = fn + f"_{save_ext}_augmented" + ext
    else:
        save_path = fn + f"_{save_ext}_original" + ext

    print(f"Saving to: {save_path}")
    joblib.dump(responses_data, save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("reward_model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--setting", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do_augmented", action="store_true")
    args = parser.parse_args()

    assert os.path.exists(args.path), print(args.path)

    if args.do_augmented:
        assert args.setting
        assert args.setting in ['wino', 'aqua']

    print(args)

    main(
        path=args.path,
        reward_model_name=args.reward_model_name,
        batch_size=args.batch_size,
        setting=args.setting,
        seed=args.seed,
        do_augmented=args.do_augmented,
    )
