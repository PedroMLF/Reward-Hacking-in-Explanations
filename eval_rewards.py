import argparse
import os
import random

import joblib

from utils import set_seed


def main(pcf_path, rm_orig_path, rm_aug_path, seed):

    # Load data
    pcf_data = joblib.load(pcf_path)
    rm_orig_data = joblib.load(rm_orig_path)
    rm_aug_data = joblib.load(rm_aug_path)

    # Get paths to save later
    basename = '_'.join(os.path.splitext(rm_orig_path)[0].split('_')[:-1])
    ext = os.path.splitext(rm_orig_path)[-1]

    for experiment in ['aug-rm-d', 'aug-rm-c']:

        # Set random seed
        set_seed(seed)

        # Collect rds
        cnt_modified = 0
        cnt = 0
        all_rds = []
        for rds_ix, (orig_rds, aug_rds, cf_rds) in enumerate(zip(rm_orig_data, rm_aug_data, pcf_data)):
            aux_rds = []
            for orig_rd, aug_rd in zip(orig_rds, aug_rds):
                assert orig_rd['full_response'] == aug_rd['full_response']

                cnt += 1

                if experiment == 'aug-rm-d':
                    sampled_cf_rd = random.choice(cf_rds)
                    if orig_rd['predicted_answer'] != sampled_cf_rd['predicted_answer']:
                        aux_rds.append(aug_rd)
                        cnt_modified += 1
                    else:
                        aux_rds.append(orig_rd)

                elif experiment == 'aug-rm-c':
                    sampled_cf_rd = random.choice(cf_rds)
                    if orig_rd['predicted_is_correct'] and not sampled_cf_rd['predicted_is_correct']:
                        aux_rds.append(aug_rd)
                        cnt_modified += 1
                    else:
                        aux_rds.append(orig_rd)

            assert len(aux_rds) in [10, 16]
            assert len(aux_rds) == len(orig_rds)
            assert [rd['full_response'] for rd in aux_rds] == [rd['full_response'] for rd in orig_rds]

            all_rds.append(aux_rds)

        assert len(all_rds) == len(rm_orig_data) == len(rm_aug_data) == len(pcf_data)

        print(f"Modified {cnt_modified} out of {cnt}")

        save_path = basename + f"_{experiment}" + ext
        print(f"Saving to: {save_path}\n")
        joblib.dump(all_rds, save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcf_path", type=str)
    parser.add_argument("--rm_orig_path", type=str)
    parser.add_argument("--rm_aug_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert os.path.exists(args.pcf_path), print(args.pcf_path)
    assert os.path.exists(args.rm_orig_path), print(args.rm_orig_path)
    assert os.path.exists(args.rm_aug_path), print(args.rm_aug_path)

    assert os.path.dirname(args.pcf_path) == os.path.dirname(args.rm_orig_path) 
    assert os.path.dirname(args.pcf_path) == os.path.dirname(args.rm_aug_path)

    print(args)

    main(
        pcf_path=args.pcf_path,
        rm_orig_path=args.rm_orig_path,
        rm_aug_path=args.rm_aug_path,
        seed=args.seed,
    )
