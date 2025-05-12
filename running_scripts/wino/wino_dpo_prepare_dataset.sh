#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/wino/llama3-8B/dpo/log_data.txt

# --------------------------------------------------------

# Decide what to run
RUN_RESPONSES=true
RUN_EVAL_RESPONSES=true
RUN_REWARD_NONE=true
RUN_REWARD_HEURISTIC=true
RUN_PREPARE_DATA=true

# Args
SAVE_PATH="outputs/wino/llama3-8B/dpo/data/"
DATA_PATH="data/wino/"
NUM_EXAMPLES=10000
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REWARD_MODEL_1="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
REWARD_MODEL_2="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

# --- Get responses ---
if ${RUN_RESPONSES}; then
    echo -e "---------- Running responses ...\n"
    for APPROACH in negative; do
        for DATA in orig cf; do
            for SPLIT in valid train; do
                python ${PROJECT_HOME}/get_responses_vllm.py \
                    ${MODEL} \
                    ${SAVE_PATH} \
                    ${DATA_PATH}/bias_${APPROACH}.joblib \
                    ${DATA} \
                    ${SPLIT} \
                    10 \
                    --number_entries_used ${NUM_EXAMPLES}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
            done
        done
    done
fi

# --- Eval responses ---
if ${RUN_EVAL_RESPONSES}; then
    echo -e "---------- Running eval responses ...\n"
    for APPROACH in negative; do
        for DATA in orig cf; do
            for SPLIT in valid train; do
                python ${PROJECT_HOME}/eval_responses_wino.py \
                    ${SAVE_PATH}/bias_${APPROACH}_${DATA}_${SPLIT}.joblib \
                    --seed 0
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    done
fi

for REWARD_MODEL in ${REWARD_MODEL_1} ${REWARD_MODEL_2}; do
    # --- Run reward model w/ none ---
    if ${RUN_REWARD_NONE}; then
        echo -e "---------- Running reward none ...\n"
        for APPROACH in negative; do
            for SPLIT in valid train; do
                python ${PROJECT_HOME}/eval_rewards.py \
                    ${SAVE_PATH}/bias_${APPROACH}_orig_${SPLIT}_data.joblib \
                    ${REWARD_MODEL} \
                    none \
                    --batch_size 32 \
                    --seed 0
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

            done
        done
    fi

    # --- Run reward model w/ heuristic ---
    if ${RUN_REWARD_HEURISTIC}; then
        echo -e "---------- Running reward heuristic ...\n"
        for APPROACH in negative; do
            for SPLIT in valid train; do
                python ${PROJECT_HOME}/eval_rewards.py \
                    ${SAVE_PATH}/bias_${APPROACH}_orig_${SPLIT}_data.joblib \
                    ${REWARD_MODEL} \
                    heuristic \
                    --batch_size 32 \
                    --pcf_path ${SAVE_PATH}/bias_${APPROACH}_cf_${SPLIT}_data.joblib \
                    --setting "wino" \
                    --seed 0 
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

                python ${PROJECT_HOME}/eval_rewards.py \
                    ${SAVE_PATH}/bias_${APPROACH}_orig_${SPLIT}_data.joblib \
                    ${REWARD_MODEL} \
                    heuristic \
                    --batch_size 32 \
                    --pcf_path ${SAVE_PATH}/bias_${APPROACH}_cf_${SPLIT}_data.joblib \
                    --setting "wino" \
                    --seed 0 \
                    --do_oracle
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    fi
done

# --- Prepare DPO data ---
if ${RUN_PREPARE_DATA}; then
    echo -e "---------- Running prepare data ...\n"
    for APPROACH in negative; do
        for RW in skywork-llama_none skywork-llama_heuristic skywork-llama_heuristic_oracle skywork-gemma_none skywork-gemma_heuristic skywork-gemma_heuristic_oracle; do
            python dpo_prepare_dataset.py \
                ${SAVE_PATH}/bias_${APPROACH}_orig_train_data_${RW}.joblib \
                ${SAVE_PATH}/bias_${APPROACH}_orig_valid_data_${RW}.joblib
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
        done
    done

fi
