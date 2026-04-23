#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/wino/llama3-3B/dpo/training/log_eval_ev.txt


DATA_PATH="data/wino/"
EVAL_OUTPUT_PATH="outputs/wino/llama3-3B/dpo" # Where the eval outputs (test) with dpo are saved
NUM_TEST=10000

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0 1 2; do
    for APPROACH in sk-llama_original sk-llama_aug-rm-d sk-llama_aug-rm-c sk-gemma_original sk-gemma_aug-rm-d sk-gemma_aug-rm-c; do

        EVAL_OUTPUT_PATH_SEED=${EVAL_OUTPUT_PATH}/seed_${SEED}/eval/${APPROACH}

        for DECODING_APPROACH in greedy sampling; do
            for SPLIT in orig cf; do
                python ${PROJECT_HOME}/get_responses_data.py \
                    ${EVAL_OUTPUT_PATH_SEED}/bias_negative_${SPLIT}_test_${DECODING_APPROACH}.joblib \
                    wino \
                    --seed ${SEED}

                python ${PROJECT_HOME}/eval_responses.py \
                    ${EVAL_OUTPUT_PATH_SEED}/bias_negative_${SPLIT}_test_${DECODING_APPROACH}_data.joblib \
                    --seed ${SEED}

                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    done
done