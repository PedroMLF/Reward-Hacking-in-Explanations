#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/wino/llama3-3B/bon/log_bon.txt

# --------------------------------------------------------

# Decide what to run
RUN_RESPONSES=true
RUN_EVAL_RESPONSES=true

# Args
SAVE_PATH="outputs/wino/llama3-3B/bon/"
DATA_PATH="data/wino/"
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
        echo -e "\n\n --- GENERATE RESPONSES --- \n\n"
        for APPROACH in negative; do
            for SPLIT in orig cf; do
                python ${PROJECT_HOME}/get_responses_vllm.py \
                    ${MODEL} \
                    ${SAVE_PATH_SEED} \
                    ${DATA_PATH}/bias_${APPROACH}.joblib \
                    ${SPLIT} \
                    test \
                    16 \
                    --seed ${SEED} \
                    --number_entries_used ${NUM_TEST}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
            done
        done
    fi

    # --- Eval orig responses ---
    if ${RUN_EVAL_RESPONSES}; then
        echo -e "\n\n --- EVAL RESPONSES --- \n\n"
        for APPROACH in negative; do
            for SPLIT in orig cf; do
                python ${PROJECT_HOME}/get_responses_data.py \
                    ${SAVE_PATH_SEED}/bias_${APPROACH}_${SPLIT}_test.joblib \
                    wino \
                    --seed ${SEED}

                python ${PROJECT_HOME}/eval_responses.py \
                    ${SAVE_PATH_SEED}/bias_${APPROACH}_${SPLIT}_test_data.joblib \
                    --seed ${SEED}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    fi
done