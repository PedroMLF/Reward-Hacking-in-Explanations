import argparse
import os

import joblib
import numpy as np
from datasets import Dataset, DatasetDict

from utils import set_seed


def collect_chosen_rejected(data):
    chosen = []
    rejected = []

    num_orig_max_not_valid = 0
    num_max_min_rw_is_the_same = 0
    for example_data in data:
        example_data_rewards = [ed['reward'] for ed in example_data]
        example_data_is_valid = [ed['response_is_valid'] for ed in example_data]
        # If no valid examples, skip
        if not any(example_data_is_valid):
            num_orig_max_not_valid += 1
            continue

        # Sort by reward
        data_tuple = [(i, x, y) for i, (x, y) in enumerate(zip(example_data_rewards, example_data_is_valid))]
        data_tuple = sorted(data_tuple, key=lambda x: x[1], reverse=True)

        # Keep highest reward+valid as chosen, lowest reward as rejected
        only_valid_ixs = list(filter(lambda x: x[2], data_tuple))
        max_ixs = [only_valid_ixs[0][0]]
        min_ixs = [data_tuple[-1][0]]

        for max_ix, min_ix in zip(max_ixs, min_ixs):
            # Rewards should be different
            if example_data[max_ix]['reward'] == example_data[min_ix]['reward']:
                num_max_min_rw_is_the_same += 1
                continue

            # Add to chosen/reject list
            assert example_data[max_ix]['reward'] > example_data[min_ix]['reward']
            chosen.append(example_data[max_ix])
            rejected.append(example_data[min_ix])

    print(f"\n\nFinal: {len(chosen)} examples")
    print(f"# Original max ixs that did not have a valid answer: {num_orig_max_not_valid}")
    print(f"# Examples skipped due to min_ix == max_ix: {num_max_min_rw_is_the_same}")
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


def main(input_path_train, input_path_valid):

    # Set seed    
    set_seed(0)

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
        train_chosen_rejected['chosen'],
        train_chosen_rejected['rejected']
    )
    valid_dataset_entries = build_dataset(
        valid_chosen_rejected['chosen'],
        valid_chosen_rejected['rejected'],
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
    args = parser.parse_args()
    
    assert os.path.exists(args.input_path_train), args.input_path_train
    assert os.path.exists(args.input_path_valid), args.input_path_valid

    main(input_path_train=args.input_path_train, input_path_valid=args.input_path_valid)
