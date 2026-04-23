import argparse
import random
from functools import lru_cache

import joblib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

@lru_cache
def load_data(model_path_name):

    aqua_base_path = f"outputs/aqua/{model_path_name}/bon/"
    wino_base_path = f"outputs/wino/{model_path_name}/bon/"

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

        for seed in range(3):
            data[setting][seed] = {}
            for k in ['llama', 'gemma']:
                data[setting][seed][f'sampling_{k}'] = {
                    "cf_rm": joblib.load(f"{base_path}/seed_{seed}/{prfx}_negative_cf_test_data_sk-{k}_original.joblib"),
                    "orig_rm": joblib.load(f"{base_path}/seed_{seed}/{prfx}_negative_orig_test_data_sk-{k}_original.joblib"),
                    "orig_rm-c": joblib.load(f"{base_path}/seed_{seed}/{prfx}_negative_orig_test_data_sk-{k}_aug-rm-c.joblib"),
                    "orig_rm-d": joblib.load(f"{base_path}/seed_{seed}/{prfx}_negative_orig_test_data_sk-{k}_aug-rm-d.joblib"),
            }

    return data


def plot(dfs_to_plot, model_path_name):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey='col')

    fontsize=16

    for row_ix, approach in enumerate(['llama', 'gemma']):
        for col_ix, setting in enumerate(['aqua', 'wino']):
            ax = axes[row_ix, col_ix]

            df_to_plot = dfs_to_plot[approach][setting]

            plot_kwargs = {'x': 'N', 'y': 'Value', 'hue': 'Approach', 'style': 'Approach', 'markers': True, 'linewidth': 2.5, 'markersize': 10}
            sns.lineplot(df_to_plot, ax=ax, **plot_kwargs)
                
            # Grids
            ax.xaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)

            # Define custom x-tick labels and positions
            xtick_positions = [0, 1, 2, 3, 4] 
            xtick_labels = ['1', '2', '4', '8', '16']
            ax.set_xlabel('N', fontsize=fontsize)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, fontsize=fontsize)

            ax.tick_params(axis='both', labelsize=fontsize-2)

            # Remove legends
            ax.legend().set_visible(False)

    ax.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.27), ncol=4, frameon=False, fontsize=fontsize)

    # Add titles
    axes[0, 0].set_title("Math Book", fontsize=fontsize, fontweight='bold')
    axes[0, 1].set_title("BiasQA", fontsize=fontsize, fontweight='bold')
    axes[0, 0].set_ylabel('SK-Llama-8B\n\nUnfaithful %', fontsize=fontsize-2)
    axes[1, 0].set_ylabel('SK-Gemma-27B\n\nUnfaithful %', fontsize=fontsize-2)
    axes[0, 1].set_ylabel('')
    axes[1, 1].set_ylabel('')

    # Adjust spacing to prevent subplot crowding
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    plt.savefig(f'figures/finegrained_bon_{model_path_name}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'figures/finegrained_bon_{model_path_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)

def main(model_path_name):

    # Load data
    print("> Loading data ...")
    data = load_data(model_path_name)

    # Collect data
    data_to_plot_dfs = {}

    for rw in ['llama', 'gemma']:
        print(f"> RW: {rw} ...")

        data_to_plot_dfs[rw] = {}

        for setting, setting_data in data.items():
            print(f">> Collecting data for: {setting} ...")

            df = []
            for seed, seed_data in setting_data.items():
                rw_data = seed_data[f'sampling_{rw}']
                for experiment, experiment_data in rw_data.items():

                    # We skip for cf rm, we need it to compute faithfulness responses
                    if experiment == 'cf_rm':
                        continue

                    # We want the sampled cfs to be the same for all experiments with the same seed
                    # For each example we sample one of the counterfactual responses
                    random.seed(seed)
                    sampled_cf_rds = [random.choice(cf_rds) for cf_rds in rw_data['cf_rm']]

                    # Iterate over N
                    for n_ix, n in enumerate([1, 2, 4, 8, 16]):
                        aux_dec = []

                        for rs in range(0, 200, 10):
                            random.seed(rs)

                            choices = []

                            # Shuffle samples
                            for test_ix, rds in enumerate(experiment_data):
                                # Otherwise it mutates the original values
                                rds = rds[:]
                                random.shuffle(rds)

                                # Sort first N by reward score and keep best
                                rds = sorted(rds[:n], key=lambda x: x['reward'], reverse=True)
                                choices.append(rds[0])

                            # Collect metrics
                            cnt = 0
                            for ch, sampled_cf in zip(choices, sampled_cf_rds):
                                if ch['predicted_is_correct'] and not sampled_cf['predicted_is_correct'] and not ch['pf_ack']:
                                    cnt += 1
                            aux_dec.append(cnt / len(choices))

                        # ['N', 'Seed', 'Value', 'Metric', 'Experiment']
                        df.append([n_ix, seed, np.round(100 * np.mean(aux_dec), 2), 'Deceptive', experiment])

            df = pd.DataFrame(df, columns=['N', 'Seed', 'Value', 'Metric', 'Approach'])
            data_to_plot_dfs[rw][setting] = df

        # Prepare dataframe to plot
        for df in data_to_plot_dfs[rw].values():
            rename_dict = {'cf_rm': r'CF; $RM$', 'orig_rm': r'$RM$', 'orig_rm-d': r'$RM_D$', 'orig_rm-c': r'$RM_C$'}
            df['Approach'] = df['Approach'].apply(lambda x: rename_dict.get(x, x))

    # Plot
    plot(data_to_plot_dfs, model_path_name=model_path_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path_name)
