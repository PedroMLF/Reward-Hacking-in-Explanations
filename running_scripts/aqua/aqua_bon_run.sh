#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/aqua/llama3-8B/bon/log_bon_extra.txt

# --------------------------------------------------------

# Decide what to run
RUN_ORIG_RESPONSES=false
RUN_CF_RESPONSES=false
RUN_EVAL_ORIG_RESPONSES=false
RUN_EVAL_CF_RESPONSES=false
RUN_REWARD_NONE=false
RUN_REWARD_AUGMENTED=true
RUN_REWARD_HEURISTIC=false

# Args
SAVE_PATH="outputs/aqua/llama3-8B/bon/"
DATA_PATH="data/aqua"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
RW_MODEL_1="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
RW_MODEL_2="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
NUM_TEST=10000

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

# -- Get orig responses ---
if ${RUN_ORIG_RESPONSES}; then
    echo -e "\n\n --- GENERATE RESPONSES --- \n\n"
    for APPROACH in negative; do
        python ${PROJECT_HOME}/get_responses_vllm.py \
            ${MODEL} \
            ${SAVE_PATH} \
            ${DATA_PATH}/cot_${APPROACH}.joblib \
            orig \
            test \
            16 \
            --seed 0 \
            --number_entries_used ${NUM_TEST}
        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
    done
fi

# --- Get CF responses ---
if ${RUN_CF_RESPONSES}; then
    echo -e "\n\n --- GENERATE CF RESPONSES --- \n\n"
    for APPROACH in negative; do
        python ${PROJECT_HOME}/get_responses_vllm.py \
            ${MODEL} \
            ${SAVE_PATH} \
            ${DATA_PATH}/cot_${APPROACH}.joblib \
            cf \
            test \
            16 \
            --seed 0 \
            --number_entries_used ${NUM_TEST}
        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
    done
fi

# --- Eval orig responses ---
if ${RUN_EVAL_ORIG_RESPONSES}; then
    echo -e "\n\n --- EVAL ORIG RESPONSES --- \n\n"
    for APPROACH in negative; do
        python ${PROJECT_HOME}/eval_responses_aqua.py \
            ${SAVE_PATH}/cot_${APPROACH}_orig_test.joblib \
            --use_model \
            --seed 0
        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
    done
fi

# --- Eval CF responses ---
if ${RUN_EVAL_CF_RESPONSES}; then
    echo -e "\n\n --- EVAL CF RESPONSES --- \n\n"
    for APPROACH in negative; do
        python ${PROJECT_HOME}/eval_responses_aqua.py \
            ${SAVE_PATH}/cot_${APPROACH}_cf_test.joblib \
            --use_model \
            --seed 0
        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
    done
fi

# --- Run reward model w/ none ---
if ${RUN_REWARD_NONE}; then
    echo -e "\n\n --- RM RESPONSES --- \n\n"
    for APPROACH in negative; do
        for SPLIT in orig cf; do
            for RW_MODEL in ${RW_MODEL_1} ${RW_MODEL_2}; do
                python ${PROJECT_HOME}/eval_rewards.py \
                    ${SAVE_PATH}/cot_${APPROACH}_${SPLIT}_test_data.joblib \
                    ${RW_MODEL} \
                    none \
                    --batch_size 4 \
                    --seed 0
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    done
fi

# --- Run reward model w/ all augmented ---
if ${RUN_REWARD_AUGMENTED}; then
    echo -e "\n\n --- RM RESPONSES AUGMENTED --- \n\n"
    for APPROACH in negative; do
        for RW_MODEL in ${RW_MODEL_1} ${RW_MODEL_2}; do
            python ${PROJECT_HOME}/eval_rewards.py \
                ${SAVE_PATH}/cot_${APPROACH}_orig_test_data.joblib \
                ${RW_MODEL} \
                heuristic \
                --batch_size 4 \
                --pcf_path ${SAVE_PATH}/cot_${APPROACH}_cf_test_data.joblib \
                --setting "aqua" \
                --seed 0 \
                --do_all_augmented
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
        done
    done
fi
