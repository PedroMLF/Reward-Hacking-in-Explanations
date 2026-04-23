import argparse
import copy
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

def plot(metrics_to_plot, model_path_name):

    metrics_to_plot = copy.deepcopy(metrics_to_plot)

    # We plot only th means
    for setting, setting_data in metrics_to_plot.items():
        for model, model_data in setting_data.items():
            for dec_strat, dec_strat_data in model_data.items():
                for exp, exp_data in dec_strat_data.items():
                    for approach, approach_data in exp_data.items():
                        approach_data['acc'] = np.mean(approach_data['acc'])
                        approach_data['ack'] = np.mean(approach_data['ack'])

    # Set up the figure and axis
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey='row')

    fontsize=16

    colors = ['blue', 'orange', 'orange', 'orange', 'green', 'green', 'green']
    positions_acc = [0, 1, 2, 3, 4, 5 , 6]

    # Labels for each bar
    bar_labels = ["", "$RM$", "$RM_D$", "$RM_C$", "$RM$", "$RM_D$", "$RM_C$"]

    # Add values
    for row_ix, metric in enumerate(['acc', 'ack']):
        for col_ix in range(4):

            # Set setting and decoding_strat
            setting = 'aqua' if col_ix in [0, 1] else 'wino'
            decoding_strat = 'greedy' if col_ix in [0, 2] else 'majority'

            # Set ax
            ax = axes[row_ix, col_ix]

            # Get values
            dl = metrics_to_plot[setting]['llama'][decoding_strat]
            dg = metrics_to_plot[setting]['gemma'][decoding_strat]
            values = [
                (dl['base']['orig'][metric], dl['base']['cf'][metric]),
                (dl['original']['orig'][metric], dl['original']['cf'][metric]),
                (dl['aug-rm-d']['orig'][metric], dl['aug-rm-d']['cf'][metric]),
                (dl['aug-rm-c']['orig'][metric], dl['aug-rm-c']['cf'][metric]),
                (dg['original']['orig'][metric], dg['original']['cf'][metric]),
                (dg['aug-rm-d']['orig'][metric], dg['aug-rm-d']['cf'][metric]),
                (dg['aug-rm-c']['orig'][metric], dg['aug-rm-c']['cf'][metric]),
            ]
            diff_values = [
                dl['base']['orig'][metric] - dl['base']['cf'][metric],
                dl['original']['orig'][metric] - dl['original']['cf'][metric],
                dl['aug-rm-d']['orig'][metric] - dl['aug-rm-d']['cf'][metric],
                dl['aug-rm-c']['orig'][metric] - dl['aug-rm-c']['cf'][metric],
                dg['original']['orig'][metric] - dg['original']['cf'][metric],
                dg['aug-rm-d']['orig'][metric] - dg['aug-rm-d']['cf'][metric],
                dg['aug-rm-c']['orig'][metric] - dg['aug-rm-c']['cf'][metric],
            ]

            for i, (pos_acc, value, diff_value, color, bar_label) in enumerate(zip(positions_acc, values, diff_values, colors, bar_labels)):
                ax.scatter(pos_acc, value[0], color=color, s=80, zorder=3, marker="D")
                ax.scatter(pos_acc, value[1], color=color, s=80, zorder=3)

                # Draw diff bar
                ax.plot([pos_acc, pos_acc], [value[0]-1, value[1]+1], color='lightgray', linewidth=9)
                ax.text(
                    pos_acc-0.05,
                    (value[0]+value[1]) / 2,
                    f"{diff_value:.1f}",
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2),
                    ha='center', va='center', fontsize=12, fontweight='bold')

                #ax.text(pos_acc-0.25, value[0]+2, f"{value[0]:.1f}")
                # -- Find lims --
                data_str = repr(metrics_to_plot)
                # Find all acc/ack values with their parent key context
                all_matches = re.findall(r"'(orig|cf|diff)':\s*\{[^}]*'acc':\s*([0-9.]+)[^}]*'ack':\s*([0-9.]+)", data_str)
                acc_values = [float(m[1]) for m in all_matches if m[0] != 'diff']

                if row_ix == 0:
                    ax.set_ylim(round(min(acc_values) - 4), round(max(acc_values) + 4))

                # Instead of x-ticklabel
                if row_ix == 0:
                    ax.text(pos_acc, round(min(acc_values) - 10), rf"{bar_label}", ha='center', fontsize=fontsize-4)
                else:
                    ax.text(pos_acc, -5, bar_label, ha='center', fontsize=fontsize-4)
            
            # Add grid
            ax.xaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)
            # Remove xticks
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            #
            ax.tick_params(axis='y', labelsize=fontsize-2)  

            # Tags
            tags = {"Base": 0, "DPO\nSK-Llama-8B": 2, "DPO\nSK-Gemma-27B": 5}

            for tag, position in tags.items():
                if row_ix == 1:
                    ax.text(position, -10, f"{tag}", ha='center', fontsize=fontsize-5, fontweight='bold')

    # Add titles
    fig.text(0.31, 0.95, 'Math Book', ha='center', fontsize=fontsize, fontweight='bold')
    fig.text(0.71, 0.95, 'BiasQA', ha='center', fontsize=fontsize, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy %', fontsize=fontsize-1)
    axes[0, 2].set_ylabel('Stereotype Rate %', fontsize=fontsize-1)
    axes[1, 0].set_ylabel('Acknowledgment Rate %', fontsize=fontsize-1)
    axes[1, 2].set_ylabel('Acknowledgment Rate %', fontsize=fontsize-1)
    axes[0, 0].set_title("Greedy Decoding", fontsize=fontsize-1)
    axes[0, 2].set_title("Greedy Decoding", fontsize=fontsize-1)
    axes[0, 1].set_title("Sampling Decoding + Maj@16", fontsize=fontsize-1)
    axes[0, 3].set_title("Sampling Decoding + Maj@16", fontsize=fontsize-1)

    # Add legend
    legend_kwargs = {'loc': 'upper left', 'frameon': True, 'fontsize': fontsize-3, 'borderaxespad': 0.2}
    diamond_marker = Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=7, markeredgewidth=1.5, linestyle='none')
    circle_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8, markeredgewidth=1.5, linestyle='none')
    # Add legends to the plots
    axes[0, 0].legend(handles=[diamond_marker, circle_marker], labels=['Orig Prompt', 'CF Prompt'], **legend_kwargs)
    axes[1, 0].legend(handles=[diamond_marker, circle_marker], labels=['Orig Prompt', 'CF Prompt'], **legend_kwargs)

    # Adjust spacing to prevent subplot crowding
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.savefig(f'figures/greedy_sampling_fig_{model_path_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'figures/greedy_sampling_fig_{model_path_name}.png', format='png', bbox_inches='tight', dpi=300)


def compute_greedy(model_data):
    final_results_dict = {}
    for experiment, experiment_data in model_data.items():
        final_results_dict[experiment] = {'orig': {'acc': [], 'ack': []}, 'cf': {'acc': [], 'ack': []}, 'diff': {'acc': [], 'ack': []}}
        for _, seed_data in experiment_data.items():
            # Orig
            acc_orig = 100*sum([xx[0]['predicted_is_correct'] for xx in seed_data['orig']]) / len(seed_data['orig'])
            ack_orig = 100*sum([xx[0]['pf_ack'] for xx in seed_data['orig']]) / len(seed_data['orig'])
            final_results_dict[experiment]['orig']['acc'].append(acc_orig)
            final_results_dict[experiment]['orig']['ack'].append(ack_orig)

            # Cf
            acc_cf = 100*sum([xx[0]['predicted_is_correct'] for xx in seed_data['cf']]) / len(seed_data['cf'])
            ack_cf = 100*sum([xx[0]['pf_ack'] for xx in seed_data['cf']]) / len(seed_data['cf'])
            final_results_dict[experiment]['cf']['acc'].append(acc_cf)
            final_results_dict[experiment]['cf']['ack'].append(ack_cf)

            # Diff
            final_results_dict[experiment]['diff']['acc'].append(acc_orig - acc_cf)
            final_results_dict[experiment]['diff']['ack'].append(ack_orig - ack_cf)

        #for k in ['orig', 'cf', 'diff']:
        #    print(f"\n{experiment}/{k}:") 
        #    print(f"Acc: {np.mean(final_results_dict[experiment][k]['acc']):.1f} ({np.std(final_results_dict[experiment][k]['acc']):.1f})")
        #    print(f"Ack: {np.mean(final_results_dict[experiment][k]['ack']):.1f} ({np.std(final_results_dict[experiment][k]['ack']):.1f})")

    return final_results_dict


def compute_majority(model_data):
    final_results_dict = {}
    for experiment, experiment_data in model_data.items():
        final_results_dict[experiment] = {'orig': {'acc': [], 'ack': []}, 'cf': {'acc': [], 'ack': []}, 'diff': {'acc': [], 'ack': []}}
        for _, seed_data in experiment_data.items():
            # Orig
            acc_orig = 100*sum([[w['predicted_is_correct'] for w in xx].count(True) > 8 for xx in seed_data['orig']]) / len(seed_data['orig'])
            ack_orig = 100*sum([[w['pf_ack'] for w in xx].count(True) > 8 for xx in seed_data['orig']]) / len(seed_data['orig'])
            final_results_dict[experiment]['orig']['acc'].append(acc_orig)
            final_results_dict[experiment]['orig']['ack'].append(ack_orig)

            # Cf
            acc_cf = 100*sum([[w['predicted_is_correct'] for w in xx].count(True) > 8 for xx in seed_data['cf']]) / len(seed_data['cf'])
            ack_cf = 100*sum([[w['pf_ack'] for w in xx].count(True) > 8 for xx in seed_data['cf']]) / len(seed_data['cf'])
            final_results_dict[experiment]['cf']['acc'].append(acc_cf)
            final_results_dict[experiment]['cf']['ack'].append(ack_cf)

            # Diff
            final_results_dict[experiment]['diff']['acc'].append(acc_orig - acc_cf)
            final_results_dict[experiment]['diff']['ack'].append(ack_orig - ack_cf)

        #for k in ['orig', 'cf', 'diff']:
        #    print(f"\n{experiment}/{k}:") 
        #    print(f"Acc: {np.mean(final_results_dict[experiment][k]['acc']):.1f} ({np.std(final_results_dict[experiment][k]['acc']):.1f})")
        #    print(f"Ack: {np.mean(final_results_dict[experiment][k]['ack']):.1f} ({np.std(final_results_dict[experiment][k]['ack']):.1f})")
            
    return final_results_dict


def print_results(metrics_to_plot):
    dal = metrics_to_plot['aqua']['llama']
    dag = metrics_to_plot['aqua']['gemma']
    dwl = metrics_to_plot['wino']['llama']
    dwg = metrics_to_plot['wino']['gemma']

    def aux_add(data):
        return f"{np.mean(data['acc']):.1f} $\pm$ {np.std(data['acc']):.1f} & {np.mean(data['ack']):.1f} $\pm$ {np.std(data['ack']):.1f}"

    # Greedy - CF
    i = 'greedy'
    j = 'cf'
    print(f"Base  & \multirow{{7}}{{*}}{{$\\times$}} & \multirow{{7}}{{*}}{{Greedy}} & - & {aux_add(dal[i]['base'][j])} & {aux_add(dwl[i]['base'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['original'][j])} & {aux_add(dwl[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-d'][j])} & {aux_add(dwl[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-c'][j])} & {aux_add(dwl[i]['aug-rm-c'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['original'][j])} & {aux_add(dwg[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-d'][j])} & {aux_add(dwg[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-c'][j])} & {aux_add(dwg[i]['aug-rm-c'][j])} \\\\")
    # Greedy - Orig
    print("\\midrule")
    i = 'greedy'
    j = 'orig'
    print(f"Base  & \multirow{{7}}{{*}}{{$\\checkmark$}} & \multirow{{7}}{{*}}{{Greedy}} & - & {aux_add(dal[i]['base'][j])} & {aux_add(dwl[i]['base'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['original'][j])} & {aux_add(dwl[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-d'][j])} & {aux_add(dwl[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-c'][j])} & {aux_add(dwl[i]['aug-rm-c'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['original'][j])} & {aux_add(dwg[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-d'][j])} & {aux_add(dwg[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-c'][j])} & {aux_add(dwg[i]['aug-rm-c'][j])} \\\\")
    # Sampling - CF
    print("\\midrule")
    i = 'majority'
    j = 'cf'
    print(f"Base  & \multirow{{7}}{{*}}{{$\\times$}} & \multirow{{7}}{{*}}{{\\shortstack{{Sampling\\\\Majority@16}}}} & - & {aux_add(dal[i]['base'][j])} & {aux_add(dwl[i]['base'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['original'][j])} & {aux_add(dwl[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-d'][j])} & {aux_add(dwl[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-c'][j])} & {aux_add(dwl[i]['aug-rm-c'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['original'][j])} & {aux_add(dwg[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-d'][j])} & {aux_add(dwg[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-c'][j])} & {aux_add(dwg[i]['aug-rm-c'][j])} \\\\")
    # Sampling - Orig
    print("\\midrule")
    i = 'majority'
    j = 'orig'
    print(f"Base  & \multirow{{7}}{{*}}{{$\\checkmark$}} & \multirow{{7}}{{*}}{{\\shortstack{{Sampling\\\\Majority@16}}}} & - & {aux_add(dal[i]['base'][j])} & {aux_add(dwl[i]['base'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['original'][j])} & {aux_add(dwl[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-d'][j])} & {aux_add(dwl[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-c'][j])} & {aux_add(dwl[i]['aug-rm-c'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['original'][j])} & {aux_add(dwg[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-d'][j])} & {aux_add(dwg[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-c'][j])} & {aux_add(dwg[i]['aug-rm-c'][j])} \\\\")


def print_diffs(metrics_to_plot):
    dal = metrics_to_plot['aqua']['llama']
    dag = metrics_to_plot['aqua']['gemma']
    dwl = metrics_to_plot['wino']['llama']
    dwg = metrics_to_plot['wino']['gemma']

    def aux_add(data):
        return f"{np.mean(data['acc']):.1f} $\pm$ {np.std(data['acc']):.1f} & {np.mean(data['ack']):.1f} $\pm$ {np.std(data['ack']):.1f}"

    # Greedy - Diff
    i = 'greedy'
    j = 'diff'
    print(f"Base & \multirow{{7}}{{*}}{{Greedy}} & - & {aux_add(dal[i]['base'][j])} & {aux_add(dwl[i]['base'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['original'][j])} & {aux_add(dwl[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-d'][j])} & {aux_add(dwl[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-c'][j])} & {aux_add(dwl[i]['aug-rm-c'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['original'][j])} & {aux_add(dwg[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-d'][j])} & {aux_add(dwg[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-c'][j])} & {aux_add(dwg[i]['aug-rm-c'][j])} \\\\")
    # Sampling - CF
    print("\\midrule")
    i = 'majority'
    j = 'diff'
    print(f"Base & \multirow{{7}}{{*}}{{\\shortstack{{Sampling\\\\Majority@16}}}} & - & {aux_add(dal[i]['base'][j])} & {aux_add(dwl[i]['base'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['original'][j])} & {aux_add(dwl[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-d'][j])} & {aux_add(dwl[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & \\textsc{{Sk-Llama-8B}}   & {aux_add(dal[i]['aug-rm-c'][j])} & {aux_add(dwl[i]['aug-rm-c'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}$    & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['original'][j])} & {aux_add(dwg[i]['original'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_D$  & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-d'][j])} & {aux_add(dwg[i]['aug-rm-d'][j])} \\\\")
    print(f"DPO + $\operatorname{{RM}}_C$  & & \\textsc{{Sk-Gemma-27B}}  & {aux_add(dag[i]['aug-rm-c'][j])} & {aux_add(dwg[i]['aug-rm-c'][j])} \\\\")


def main(model_path_name):
    # Load data
    print("> Loading data ...")
    data = load_data(model_path_name)

    # Get metrics
    # Each subdict will have model (e.g., 'base') and exp (e.g., 'cf')
    metrics_to_plot = {
        'aqua': {
            'llama': {'greedy': {}, 'sampling': {}},
            'gemma': {'greedy': {}, 'sampling': {}},
        },
        'wino': {
            'llama': {'greedy': {}, 'sampling': {}},
            'gemma': {'greedy': {}, 'sampling': {}},
        },
    }

    # Comput greedy
    print(f"\n> Greedy results ...")
    for setting , setting_data in data.items():
        #print("\n" + setting)
        for model, model_data in setting_data.items():
            #print(f"\n{model}:")
            model_data = model_data['greedy']
            metrics_to_plot[setting][model]['greedy'] = compute_greedy(model_data)

    # Compute majority
    print(f"\n> Majority results ...")
    for setting , setting_data in data.items():
        #print("\n" + setting)
        for model, model_data in setting_data.items():
            #print(f"\n{model}:")
            model_data = model_data['sampling']
            metrics_to_plot[setting][model]['majority'] = compute_majority(model_data)

    # Table
    print(f"\n> Printing Table Contents ...")
    print_results(metrics_to_plot)
    
    print(f"\n> Printing Diff Table Contents ...")
    print_diffs(metrics_to_plot)

    # Plot
    print(f"\n> Plotting ...")
    plot(metrics_to_plot, model_path_name=model_path_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path_name)
