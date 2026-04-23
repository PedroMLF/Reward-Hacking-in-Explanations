#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/aqua/llama3-3B/dpo/log_data.txt

# --------------------------------------------------------

# Decide what to run
RUN_RESPONSES=true
RUN_EVAL_RESPONSES=true
RUN_REWARD=true
RUN_PREPARE_DATA=true

# Args
SAVE_PATH="outputs/aqua/llama3-3B/dpo/"
DATA_PATH="data/aqua/"
NUM_EXAMPLES=3000

#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-3.2-3B-Instruct"

REWARD_MODEL_1="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
REWARD_MODEL_2="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0 1 2; do

    echo -e "\nSEED: ${SEED}\n"

    SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}/data

    # --- Get responses ---
    if ${RUN_RESPONSES}; then
        echo -e "---------- Running responses ...\n"
        for SPLIT in orig cf; do
            for DATA in valid train; do
                python ${PROJECT_HOME}/get_responses_vllm.py \
                    ${MODEL} \
                    ${SAVE_PATH_SEED} \
                    ${DATA_PATH}/cot_negative.joblib \
                    ${SPLIT} \
                    ${DATA} \
                    10 \
                    --seed ${SEED} \
                    --number_entries_used ${NUM_EXAMPLES}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
            done
        done
    fi

    # --- Eval responses ---
    if ${RUN_EVAL_RESPONSES}; then
        echo -e "---------- Running eval responses ...\n"
            for SPLIT in orig cf; do
                for DATA in valid train; do
                    # We don't run the ack step in this case
                    python ${PROJECT_HOME}/get_responses_data.py \
                        ${SAVE_PATH_SEED}/cot_negative_${SPLIT}_${DATA}.joblib \
                        aqua \
                        --seed ${SEED}
                    date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
                done
            done
    fi

    # --- Rewards ---
    if ${RUN_REWARD}; then
        for RW_MODEL in ${REWARD_MODEL_1} ${REWARD_MODEL_2}; do

            # Set smaller batch size for larger reward model
            if [[ "${RW_MODEL}" == "${REWARD_MODEL_2}" ]]; then
                BSIZE=4
            else
                BSIZE=10
            fi

            for DATA in valid train; do
                # We need original / augmented for train valid
                # In this case we don't need for CF
                echo -e "---------- Running rewards ...\n"
                python -u ${PROJECT_HOME}/get_rewards_data.py \
                    ${SAVE_PATH_SEED}/cot_negative_orig_${DATA}_data.joblib \
                    ${RW_MODEL} \
                    --batch_size ${BSIZE} \
                    --seed ${SEED}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

                python -u ${PROJECT_HOME}/get_rewards_data.py \
                    ${SAVE_PATH_SEED}/cot_negative_orig_${DATA}_data.joblib \
                    ${RW_MODEL} \
                    --batch_size ${BSIZE} \
                    --setting aqua \
                    --do_augmented \
                    --seed ${SEED}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done

        for RW in sk-llama sk-gemma; do
            for DATA in valid train; do
                python -u ${PROJECT_HOME}/eval_rewards.py \
                    --pcf_path ${SAVE_PATH_SEED}/cot_negative_cf_${DATA}_data.joblib \
                    --rm_orig_path ${SAVE_PATH_SEED}/cot_negative_orig_${DATA}_data_${RW}_original.joblib \
                    --rm_aug_path ${SAVE_PATH_SEED}/cot_negative_orig_${DATA}_data_${RW}_augmented.joblib \
                    --seed ${SEED}
            done
        done
    fi

    # --- Prepare DPO data ---
    if ${RUN_PREPARE_DATA}; then
        echo -e "---------- Running prepare data ...\n"
        for RW in sk-llama_original sk-llama_aug-rm-c sk-llama_aug-rm-d sk-gemma_original sk-gemma_aug-rm-c sk-gemma_aug-rm-d; do
            python dpo_prepare_dataset.py \
                ${SAVE_PATH_SEED}/cot_negative_orig_train_data_${RW}.joblib \
                ${SAVE_PATH_SEED}/cot_negative_orig_valid_data_${RW}.joblib \
                --seed ${SEED}
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
        done
    fi
done