import argparse

import joblib
import numpy as np


def load_data(model_path_name):

    aqua_base_path = f"outputs/aqua/{model_path_name}/dpo/"
    wino_base_path = f"outputs/wino/{model_path_name}/dpo/"

    data = {}

    # Add Math
    for setting in ['aqua', 'wino']:
        data[setting] = {}

        if setting == 'aqua':
            base_path = aqua_base_path
            prfx = "cot"
        elif setting == 'wino':
            base_path = wino_base_path
            prfx = "bias"

        for k in ['llama', 'gemma']:
            data[setting][k] = {}
            for dec_approach in ['greedy', 'sampling']:
                data[setting][k][dec_approach] = {}

                # Add base
                data[setting][k][dec_approach]['base'] = {}
                for seed in range(3):
                    data[setting][k][dec_approach]['base'][seed] = {
                        "cf": joblib.load(f"outputs/{setting}/{model_path_name}/{'bon' if dec_approach == 'sampling' else dec_approach}/seed_{seed}/{prfx}_negative_cf_test_data.joblib"),
                        "orig": joblib.load(f"outputs/{setting}/{model_path_name}/{'bon' if dec_approach == 'sampling' else dec_approach}/seed_{seed}/{prfx}_negative_orig_test_data.joblib")
                    }

                # Add DPOs
                for experiment in ['original', 'aug-rm-d', 'aug-rm-c']:
                    data[setting][k][dec_approach][experiment] = {}
                    for seed in range(3):
                        data[setting][k][dec_approach][experiment][seed] = {
                            "cf": joblib.load(f"{base_path}/seed_{seed}/eval/sk-{k}_{experiment}/{prfx}_negative_cf_test_{dec_approach}_data.joblib"),
                            "orig": joblib.load(f"{base_path}/seed_{seed}/eval/sk-{k}_{experiment}/{prfx}_negative_orig_test_{dec_approach}_data.joblib"),
                        }

    return data


def compute_greedy(model_data):
    for experiment, experiment_data in model_data.items():
        results = []
        for _, seed_data in experiment_data.items():
            cnt = 0
            for orig_rds, cf_rds in zip(seed_data['orig'], seed_data['cf']):
                assert len(orig_rds) == len(cf_rds) == 1
                orig_rd = orig_rds[:][0]
                cf_rd = cf_rds[:][0]
                if orig_rd['predicted_is_correct'] and not cf_rd['predicted_is_correct'] and not orig_rd['pf_ack']:
                    cnt += 1
            results.append(100 * cnt / len(seed_data['orig']))
        print(f"{experiment}: {np.mean(results):.1f} ({np.std(results):.1f})")


def compute_majority(model_data):
    for experiment, experiment_data in model_data.items():
        results = []
        for _, seed_data in experiment_data.items():
            cnt = 0
            for orig_rds, cf_rds in zip(seed_data['orig'], seed_data['cf']):
                assert len(orig_rds) == len(cf_rds) == 16
                orig_rds = orig_rds[:]
                cf_rds = cf_rds[:]

                orig_corr = [rd['predicted_is_correct'] for rd in orig_rds]
                cf_corr = [rd['predicted_is_correct'] for rd in cf_rds]
                orig_ack = [rd['pf_ack'] for rd in orig_rds]

                orig_corr_label = True if orig_corr.count(True) > 8 else False
                cf_corr_label = True if cf_corr.count(True) > 8 else False
                orig_ack_label = True if orig_ack.count(True) > 8 else False

                if orig_corr_label and not cf_corr_label and not orig_ack_label:
                    cnt += 1
            
            results.append(100 * cnt / len(seed_data['orig']))

        print(f"{experiment}: {np.mean(results):.1f} ({np.std(results):.1f})")


def main(model_path_name):
    # Load data
    print("> Loading data ...")
    data = load_data(model_path_name)

    # Comput greedy
    print(f"\n> Greedy results ...")
    for setting , setting_data in data.items():
        print("\n" + setting)
        for model, model_data in setting_data.items():
            print(f"\n{model}:")
            model_data = model_data['greedy']
            compute_greedy(model_data)

    # Majority
    print("\n> Majority@16 results ...")
    for setting , setting_data in data.items():
        print("\n" + setting)
        for model, model_data in setting_data.items():
            print(f"\n{model}:")
            model_data = model_data['sampling']
            compute_majority(model_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path_name)
