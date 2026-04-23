import argparse
import math
import random
from functools import lru_cache

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def plot(dfs_to_plot, key, save_path, model_path_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    fontsize=16

    plot_kwargs = {'x': 'N', 'y': 'Value', 'style': key, 'hue': key, 'linewidth': 2.5, 'markersize': 10}

    for col_ix, setting in enumerate(dfs_to_plot):
        ax = axes[col_ix]
        sns.lineplot(
            data=dfs_to_plot[setting][dfs_to_plot[setting]['Metric']=='Accuracy'],
            markers=['o', 'D', 'X', 'v', 's'],#, '^'],
            dashes=[(1,0)],
            ax=ax,
            **plot_kwargs
        )
        sns.lineplot(
            data=dfs_to_plot[setting][dfs_to_plot[setting]['Metric']=='Acknowledgment'],
            markers=['o', 'D', 'X', 'v', 's'],#, '^'],
            dashes=[(2, 2)],
            ax=ax,
            **plot_kwargs
        )

        # Add 50% line to bias
        if setting == 'wino':
            ax.axhline(y=50, color='black', linestyle='dotted', zorder=1)

        # Grids
        ax.xaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)

        # Define custom x-tick labels and positions
        xtick_positions = [0, 1, 2, 3, 4] 
        xtick_labels = ['1', '2', '4', '8', '16']
        ax.set_xlabel('N', fontsize=fontsize)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=fontsize)

        ax.tick_params(axis='both', labelsize=fontsize)

        ax.legend().set_visible(False)

    # Prepare legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove from the second plot
    num_to_remove = int(math.floor(len(handles)/2))
    new_handles = handles[:-num_to_remove]
    new_labels = labels[:-num_to_remove]
    # Add manually the metric lines
    custom_line_bf = [Line2D([0], [0], color="none", linestyle="", label="X")]
    custom_lines = [
        Line2D([0], [0], color="none", linestyle="", label="Metric"),
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="X"),
        Line2D([0], [0], color="black", linestyle="--", lw=2, label="X"),
    ]
    # Update the legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(1.34, 0.93),
        ncol=1, frameon=False,
        handles=custom_line_bf + new_handles + custom_lines,
        labels=[r"$\bf{Approach}$"] + new_labels + [r"$\bf{Metric}$", "Accuracy/\nStereotype Rate", "Acknowledgment Rate"],
        fontsize=fontsize-4,
    )

    # Add titles
    axes[0].set_title("Math Book", fontsize=fontsize, fontweight='bold')
    axes[1].set_title("BiasQA", fontsize=fontsize, fontweight='bold')
    axes[0].set_ylabel('Accuracy and \nAcknowledgment Rate%', fontsize=fontsize-2)
    axes[1].set_ylabel('Stereotype and \nAcknowledgmentRate %', fontsize=fontsize-2)

    # Adjust spacing to prevent subplot crowding
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    if save_path:
        plt.savefig(save_path + f"-{model_path_name}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(save_path + f"-{model_path_name}.png", format='png', bbox_inches='tight', dpi=300)


def main(model_path_name):

    # Load data
    print("> Loading data ...")
    data = load_data(model_path_name)

    # Collect data
    for rw in ['llama', 'gemma']:
        print(f"> RW: {rw} ...")

        data_to_plot_dfs = {}

        for setting, setting_data in data.items():
            print(f">> Collecting data for: {setting} ...")

            df = []
            for seed, seed_data in setting_data.items():
                rw_data = seed_data[f'sampling_{rw}']
                for experiment, experiment_data in rw_data.items():
                    
                    # Iterate over N
                    for n_ix, n in enumerate([1, 2, 4, 8, 16]):
                        aux_accs = []
                        aux_acks = []

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
                            aux_accs.append(sum([ch['predicted_is_correct'] for ch in choices]) / len(choices))
                            aux_acks.append(sum([ch['pf_ack'] for ch in choices]) / len(choices))

                        # ['N', 'Seed', 'Value', 'Metric', 'Experiment']
                        df.append([n_ix, seed, np.round(100*np.mean(aux_accs), 2), 'Accuracy', experiment])
                        df.append([n_ix, seed, np.round(100*np.mean(aux_acks), 2), 'Acknowledgment', experiment])

                        #bon_metrics_to_plot[setting][model][key][r_seed1]['accs'].append(np.round(100*np.mean(aux_accs), 2))
                        #bon_metrics_to_plot[setting][model][key][r_seed1]['acks'].append(np.round(100*np.mean(aux_acks), 2))
            
            df = pd.DataFrame(df, columns=['N', 'Seed', 'Value', 'Metric', 'Approach'])
            data_to_plot_dfs[setting] = df

        # Prepare dataframe to plot
        for df in data_to_plot_dfs.values():
            rename_dict = {'cf_rm': r'CF; $RM$', 'orig_rm': r'$RM$', 'orig_rm-d': r'$RM_D$', 'orig_rm-c': r'$RM_C$'}
            df['Approach'] = df['Approach'].apply(lambda x: rename_dict.get(x, x))

        # Plot metrics
        for k, df in data_to_plot_dfs.items():
            print(f"Data for {rw}/{k}:")
            print(df.groupby(['N', 'Approach', 'Metric']).agg(['mean', 'std']).round(1).loc[[0, 4]])

        # Plot
        plot(data_to_plot_dfs, "Approach", save_path=f'figures/bon_{rw}', model_path_name=model_path_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path_name)
