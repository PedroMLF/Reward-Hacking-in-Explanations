#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/aqua/llama3-3B/greedy/log_greedy.txt

# --------------------------------------------------------

# Decide what to run
RUN_RESPONSES=true
RUN_EVAL_RESPONSES=true

# Args
SAVE_PATH="outputs/aqua/llama3-3B/greedy/"
DATA_PATH="data/aqua/"
#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
NUM_TEST=10000

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env  
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0 1 2; do

    echo -e "\nSEED: ${SEED}\n"

    SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}

    # -- Get orig responses ---
    if ${RUN_RESPONSES}; then
        for APPROACH in negative; do
            for SPLIT in orig cf; do
                python ${PROJECT_HOME}/get_responses_vllm.py \
                    ${MODEL} \
                    ${SAVE_PATH_SEED} \
                    ${DATA_PATH}/cot_${APPROACH}.joblib \
                    ${SPLIT} \
                    test \
                    1 \
                    --seed ${SEED} \
                    --number_entries_used ${NUM_TEST}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
            done
        done
    fi

    # --- Eval orig responses ---
    if ${RUN_EVAL_RESPONSES}; then
        for APPROACH in negative; do
            for SPLIT in orig cf; do
                python ${PROJECT_HOME}/get_responses_data.py \
                    ${SAVE_PATH_SEED}/cot_${APPROACH}_${SPLIT}_test.joblib \
                    aqua \
                    --seed ${SEED}

                python ${PROJECT_HOME}/eval_responses.py \
                    ${SAVE_PATH_SEED}/cot_${APPROACH}_${SPLIT}_test_data.joblib \
                    --seed ${SEED}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    fi
done