# Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations

Paper: [Link](https://openreview.net/pdf?id=nkdPLuKoL5)

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
sbatch running_scripts/aqua/aqua_bon_run_rewards.sh

sbatch running_scripts/wino/wino_bon_run.sh
sbatch running_scripts/wino/wino_bon_run_rewards.sh
```

### DPO

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
sbatch running_scripts/aqua/aqua_dpo_eval_responses_compute.sh
sbatch running_scripts/aqua/aqua_dpo_eval_responses_eval.sh

sbatch running_scripts/wino/wino_dpo_eval_responses_compute.sh
sbatch running_scripts/wino/wino_dpo_eval_responses_eval.sh
```

### Results and Plots

```
python -m analysis.main --model_path_name llama3-3B
python -m analysis.main --model_path_name llama3-8B

python -m analysis.bon --model_path_name llama3-3B
python -m analysis.bon --model_path_name llama3-8B

python -m analysis.finegrained --model_path_name llama3-3B
python -m analysis.finegrained --model_path_name llama3-8B

python -m analysis.finegrained_bon --model_path_name llama3-3B
python -m analysis.finegrained_bon --model_path_name llama3-8B
```
