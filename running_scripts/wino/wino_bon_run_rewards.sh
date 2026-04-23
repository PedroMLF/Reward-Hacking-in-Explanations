#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/wino/llama3-3B/bon/log_bon_reward.txt

# --------------------------------------------------------

# Args
SAVE_PATH="outputs/wino/llama3-3B/bon/"
RW_MODEL_1="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
RW_MODEL_2="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
NUM_TEST=10000

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0 1 2; do

    echo -e "\nSEED: ${SEED}\n"

    SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}

    echo -e "\n\n --- RM RESPONSES --- \n\n"
    for APPROACH in negative; do
        for RW_MODEL in ${RW_MODEL_1} ${RW_MODEL_2}; do
            for SPLIT in orig cf; do
                python -u ${PROJECT_HOME}/get_rewards_data.py \
                    ${SAVE_PATH_SEED}/bias_${APPROACH}_${SPLIT}_test_data.joblib \
                    ${RW_MODEL} \
                    --batch_size 16 \
                    --seed ${SEED}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done

            python -u ${PROJECT_HOME}/get_rewards_data.py \
                ${SAVE_PATH_SEED}/bias_${APPROACH}_orig_test_data.joblib \
                ${RW_MODEL} \
                --batch_size 16 \
                --setting wino \
                --do_augmented \
                --seed ${SEED}
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
        done
    done

    for RW in sk-llama sk-gemma; do
        python -u ${PROJECT_HOME}/eval_rewards.py \
            --pcf_path ${SAVE_PATH_SEED}/bias_negative_cf_test_data_${RW}_original.joblib \
            --rm_orig_path ${SAVE_PATH_SEED}/bias_negative_orig_test_data_${RW}_original.joblib \
            --rm_aug_path ${SAVE_PATH_SEED}/bias_negative_orig_test_data_${RW}_augmented.joblib \
            --seed ${SEED}
    done

done