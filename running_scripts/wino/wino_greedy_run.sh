#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/wino/llama3-8B/greedy/log_greedy.txt

# --------------------------------------------------------

# Decide what to run
RUN_RESPONSES=true
RUN_EVAL_RESPONSES=true

# Args
SAVE_PATH="outputs/wino/llama3-8B/greedy/"
DATA_PATH="data/wino/"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_TEST=10000

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

# -- Get orig responses ---
if ${RUN_RESPONSES}; then
    for APPROACH in negative; do
        for SPLIT in orig cf; do
            python ${PROJECT_HOME}/get_responses_vllm.py \
                ${MODEL} \
                ${SAVE_PATH} \
                ${DATA_PATH}/bias_${APPROACH}.joblib \
                ${SPLIT} \
                test \
                1 \
                --seed 0 \
                --number_entries_used ${NUM_TEST}
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
        done
    done
fi

# --- Eval orig responses ---
if ${RUN_EVAL_RESPONSES}; then
    for APPROACH in negative; do
        python ${PROJECT_HOME}/eval_responses_wino.py \
            ${SAVE_PATH}/bias_${APPROACH}_orig_test.joblib \
            --use_model \
            --seed 0
        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

        python ${PROJECT_HOME}/eval_responses_wino.py \
            ${SAVE_PATH}/bias_${APPROACH}_cf_test.joblib \
            --use_model \
            --seed 0
        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
    done
fi
