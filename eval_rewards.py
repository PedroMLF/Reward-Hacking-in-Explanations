import argparse
import math
import os
import random

import joblib
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import env_setup
from utils import set_seed

env_setup.setup_env(use_temp=False)


class RewardModel:
    def __init__(self):

        self.device = "cuda"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

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
        ).to(self.device)
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
            torch_dtype=torch.bfloat16,
            num_labels=1,
            attn_implementation="eager",
            cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
        )

        # Initialize reward model
        super().__init__()


def add_reward_to_responses_data(reward_model, path, batch_size):
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

        # Each entry in responses_data has N (16) items
        reward_model_messages = reward_model.prepare_messages(
            prompts=[prompt] * len(responses_data),
            responses=[r['full_response'] for r in responses_data]
        )
        reward_model_messages_tok = reward_model.tokenize(reward_model_messages)
        rewards = reward_model.compute_rewards(reward_model_messages_tok, batch_size)
        # Update rewards on responses_data - we modify in-place
        for rd, reward in zip(responses_data, rewards):
            rd['reward'] = reward

    return all_responses_data


def add_reward_to_responses_data_heuristic(reward_model, path, batch_size, pcf_path, setting, do_oracle, do_all_augmented):
    """
    responses_data: list with len = number of prompts, each a list with len = N
    """

    # Load responses
    print(f"Loading responses data from: {path}")
    all_responses_data = joblib.load(path)

    print(f"Loading responses data from: {pcf_path}")
    all_pcf_responses_data = joblib.load(pcf_path)
    assert len(all_responses_data) == len(all_pcf_responses_data)

    # Add rewards
    counter_total = 0
    counter_mod_pos = 0

    for ix, (responses_data, pcf_responses_data) in enumerate(zip(all_responses_data, all_pcf_responses_data)):
        if ix % 100 == 0: print(f"{ix}/{len(all_responses_data)}")
        # prompt and query fields are the same for all N entries
        prompt = f"{responses_data[0]['prompt']}\n\n{responses_data[0]['query']}"

        # Get pct of correct responses for baseline
        sampled_pcf = random.choice(pcf_responses_data)

        for rd in responses_data:
            if setting == "aqua":
                if do_all_augmented:
                    cond = 1
                elif do_oracle:
                    cond = rd['predicted_is_correct'] and not sampled_pcf['predicted_is_correct']
                else:
                    cond = rd['predicted'] != sampled_pcf['predicted']

            elif setting == "wino":
                if do_all_augmented:
                    cond = 1
                elif do_oracle:
                    cond = rd['predicted_is_correct'] and not sampled_pcf['predicted_is_correct']
                else:
                    cond = rd['predicted_answer'] != sampled_pcf['predicted_answer']

            if cond:
                counter_mod_pos += 1
                if setting == 'aqua':
                    rd['full_response_heuristic'] = rd['full_response'] + "\n\nI acknowledge that I used the SOLUTIONS part of the MATH BOOK."
                elif setting == 'wino':
                    rd['full_response_heuristic'] = rd['full_response'] + f"\n\nI acknowledge that my reasoning used biases or stereotypes related to a profession being more likely associated to a person of a certain gender."
            else:
                rd['full_response_heuristic'] = rd['full_response']

            counter_total += 1

        # Each entry in responses_data has N (16) items
        reward_model_messages = reward_model.prepare_messages(
            prompts=[prompt] * len(responses_data),
            responses=[r['full_response_heuristic'] for r in responses_data]
        )
        reward_model_messages_tok = reward_model.tokenize(reward_model_messages)
        rewards = reward_model.compute_rewards(reward_model_messages_tok, batch_size)
        # Update rewards on responses_data - we modify in-place
        for rd, reward in zip(responses_data, rewards):
            rd['reward'] = reward

    print(f"Number of examples with ack string added: {counter_mod_pos} / {counter_total}")

    return all_responses_data


def main(path, reward_model_name, augment_approach, batch_size, pcf_path, setting, seed, do_oracle, do_all_augmented):

    # Set random seed
    set_seed(seed)

    # Load reward model
    if reward_model_name == "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2":
        reward_model = RM_SkyworkLLama()
        save_ext = "skywork-llama"
    elif reward_model_name == "Skywork/Skywork-Reward-Gemma-2-27B-v0.2":
        reward_model = RM_SkyworkGemma()
        save_ext = "skywork-gemma"
    print("Model loaded.")

    # Add rewards to response data
    add_rw_fn_kwargs = {'reward_model': reward_model, 'path': path, 'batch_size': batch_size}
    if augment_approach == "none":
        responses_data = add_reward_to_responses_data(**add_rw_fn_kwargs)
    elif augment_approach == "heuristic":
        responses_data = add_reward_to_responses_data_heuristic(
            pcf_path=pcf_path,
            setting=setting,
            do_oracle=do_oracle,
            do_all_augmented=do_all_augmented,
            **add_rw_fn_kwargs,
        )
    del reward_model

    # Save
    fn, ext = os.path.splitext(path)
    if do_oracle:
        save_path = fn + f"_{save_ext}_{augment_approach}_oracle" + ext
    elif do_all_augmented:
        save_path = fn + f"_{save_ext}_{augment_approach}_augmented" + ext
    else:
        save_path = fn + f"_{save_ext}_{augment_approach}" + ext

    print(f"Saving to: {save_path}")
    joblib.dump(responses_data, save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("reward_model_name", type=str)
    parser.add_argument("augment_approach", type=str, default="none")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pcf_path", type=str, required=False, default=None)
    parser.add_argument("--setting", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do_oracle", action="store_true")
    parser.add_argument("--do_all_augmented", action="store_true")
    args = parser.parse_args()

    assert os.path.exists(args.path)

    assert args.augment_approach in ["none", "heuristic"]

    if args.augment_approach == "heuristic":
        assert os.path.exists(args.pcf_path)
        assert args.setting
        assert args.setting in ['wino', 'aqua']

    if args.do_all_augmented:
        assert not args.do_oracle
        assert args.augment_approach == "heuristic"

    print(args)

    main(
        path=args.path,
        reward_model_name=args.reward_model_name,
        augment_approach=args.augment_approach,
        batch_size=args.batch_size,
        pcf_path=args.pcf_path,
        setting=args.setting,
        seed=args.seed,
        do_oracle=args.do_oracle,
        do_all_augmented=args.do_all_augmented,
    )
