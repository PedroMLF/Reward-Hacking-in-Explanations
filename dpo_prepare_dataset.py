import argparse
import os

import joblib
import numpy as np
from datasets import Dataset, DatasetDict

from utils import set_seed


def collect_chosen_rejected(data):
    chosen = []
    rejected = []

    cnt_no_valid_responses = 0
    cnt_ch_rj_same_reward_score = 0
    for example_data in data:
        assert len(example_data) == 10

        # Collect all rewards and response is valid values
        example_data_rewards = [ed['reward'] for ed in example_data]
        example_data_is_valid = [ed['response_is_valid'] for ed in example_data]

        # If no valid examples, skip example
        if not any(example_data_is_valid):
            cnt_no_valid_responses += 1
            continue

        # Collect triplets (index, reward, is valid) and sort by reward
        data_tuple = [(i, r, v) for i, (r, v) in enumerate(zip(example_data_rewards, example_data_is_valid))]
        data_tuple = sorted(data_tuple, key=lambda x: x[1], reverse=True)

        # Keep highest reward and valid as chosen, lowest reward as rejected
        only_valid_ixs = list(filter(lambda x: x[2], data_tuple))
        max_ix = only_valid_ixs[0][0]
        min_ix = data_tuple[-1][0]

        # Rewards should be different
        if example_data[max_ix]['reward'] == example_data[min_ix]['reward']:
            cnt_ch_rj_same_reward_score += 1
            continue

        # Add to chosen/reject list
        assert example_data[max_ix]['reward'] > example_data[min_ix]['reward']
        chosen.append(example_data[max_ix])
        rejected.append(example_data[min_ix])

    print(f"\n\nFinal: {len(chosen)} examples")
    print(f"# Original max ixs that did not have a valid answer: {cnt_no_valid_responses}")
    print(f"# Examples skipped due to min_ix == max_ix: {cnt_ch_rj_same_reward_score}")
    print("# Rejected samples that have non-valid answer: ", len([rj['answer'] for rj in rejected if not rj['response_is_valid']]))
    rw_diffs = [ch['reward'] - rj['reward'] for ch, rj in zip(chosen, rejected)]
    print(f"Mean diff rewards: {np.mean(rw_diffs):.2f} +- {np.std(rw_diffs):.2f}")

    return {'chosen': chosen, 'rejected': rejected}


def build_dataset(chosen_samples, rejected_samples):
    dataset_entries = []
    for ch, rj in zip(chosen_samples, rejected_samples):
        assert ch['prompt'] == rj['prompt']
        assert ch['query'] == rj['query']
        assert ch['full_response'] != rj['full_response']
        dataset_entry = {
            'prompt': ch['prompt'],
            'query': ch['query'],
            'chosen': ch['full_response'],
            'rejected': rj['full_response'],
        }
        dataset_entries.append(dataset_entry)
    return dataset_entries


def main(input_path_train, input_path_valid, seed):

    # Set seed    
    set_seed(seed)

    # Load data
    print(f"Loading {input_path_train} ...")
    train_data = joblib.load(input_path_train)

    print(f"Loading {input_path_valid} ...")
    valid_data = joblib.load(input_path_valid)

    # Collect chosen and reject samples
    train_chosen_rejected = collect_chosen_rejected(train_data)
    valid_chosen_rejected = collect_chosen_rejected(valid_data)

    # Build dataset
    train_dataset_entries = build_dataset(
        chosen_samples=train_chosen_rejected['chosen'],
        rejected_samples=train_chosen_rejected['rejected']
    )
    valid_dataset_entries = build_dataset(
        chosen_samples=valid_chosen_rejected['chosen'],
        rejected_samples=valid_chosen_rejected['rejected'],
    )

    train_dataset = Dataset.from_list(train_dataset_entries)
    valid_dataset = Dataset.from_list(valid_dataset_entries)

    dataset = DatasetDict({'train': train_dataset, 'valid': valid_dataset})

    # Get save path and save
    dirname = os.path.dirname(os.path.splitext(input_path_train)[0])
    basename = os.path.basename(os.path.splitext(input_path_train)[0])
    dataname = '_'.join(basename.split('_')[:2])
    if "oracle" in basename:
        rwname = '_'.join(basename.split('_')[-3:])
    else:
        rwname = '_'.join(basename.split('_')[-2:])

    save_path = os.path.join(dirname, f"{dataname}_{rwname}")
    print("Saving to: ", save_path)
    dataset.save_to_disk(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path_train', type=str)    
    parser.add_argument('input_path_valid', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    assert os.path.exists(args.input_path_train), args.input_path_train
    assert os.path.exists(args.input_path_valid), args.input_path_valid

    main(input_path_train=args.input_path_train, input_path_valid=args.input_path_valid, seed=args.seed)
