# Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations

**Preprint:** [Link](https://arxiv.org/abs/2504.05294v1)

---

## Setup

```
virtualenv -p /usr/bin/python3.9 myenv
source myenv/bin/activate
pip install -r requirements.txt
```

**.env file**
```
HF_MODELS_CACHE_DIR=<Path to cache directory>
HF_MODELS_CACHE_DIR_TMP=<Path to cache directory>
HF_TOKEN=<HuggingFace token>
```

---

## Results

### Create data

```
python data/aqua/prepare_data.py
python data/wino/prepare_data.py
```

### Motivation experiment

```
sbatch motivation_experiment/prepare_data.sh
sbatch motivation_experiment/gather_scores.sh
motivation_experiment/plot.ipynb
```

### Greedy experiments

```
sbatch running_scripts/aqua/aqua_greedy_run.sh
sbatch running_scripts/wino/wino_greedy_run.sh
```

### Sampling / BoN experiments

```
sbatch running_scripts/aqua/aqua_bon_run.sh
sbatch running_scripts/wino/wino_bon_run.sh
```

### DPO

**Load checkpoints**

```
./download_checkpoints.sh
```

OR _(Skip these steps if using the provided checkpoints)_

**Prepare data**

```
sbatch running_scripts/aqua/aqua_dpo_prepare_dataset.sh
sbatch running_scripts/wino/wino_dpo_prepare_dataset.sh
```

**Train models**

```
sbatch running_scripts/aqua/aqua_dpo_train_rw_llama.sh
sbatch running_scripts/aqua/aqua_dpo_train_rw_gemma.sh
sbatch running_scripts/wino/wino_dpo_train_rw_llama.sh
sbatch running_scripts/wino/wino_dpo_train_rw_gemma.sh
```

### Eval DPO

```
sbatch running_scripts/aqua/dpo_eval_llama.sh
sbatch running_scripts/aqua/dpo_eval_gemma.sh
sbatch running_scripts/wino/dpo_eval_llama.sh
sbatch running_scripts/wino/dpo_eval_gemma.sh
```

### Results and Plots

```
eval.ipynb
```

---

## Notes:

- Code uses suffix `heuristic` for `RM_D` and suffix `heuristic_oracle` for `RM_C`.

