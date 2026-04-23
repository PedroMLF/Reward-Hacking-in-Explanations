#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/wino/llama3-3B/dpo/training/log_eval_gen.txt

DATA_PATH="data/wino/"
SAVE_PATH="checkpoints/llama3-3B/wino_dpo"
EVAL_OUTPUT_PATH="outputs/wino/llama3-3B/dpo" # Where the eval outputs (test) with dpo are saved
NUM_TEST=10000

#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-3.2-3B-Instruct"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0 1 2; do
    for APPROACH in sk-llama_original sk-llama_aug-rm-d sk-llama_aug-rm-c sk-gemma_original sk-gemma_aug-rm-d sk-gemma_aug-rm-c; do

        EVAL_OUTPUT_PATH_SEED=${EVAL_OUTPUT_PATH}/seed_${SEED}/eval/${APPROACH}
        SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}

        # Get ckpt path
        CKPT_PATH=${SAVE_PATH_SEED}/bias_negative_${APPROACH}/checkpoint-best
        echo "CKPT_PATH: ${CKPT_PATH}"

        # Merge model
        python ${PROJECT_HOME}/merge_peft_model.py \
            --base_model_name ${MODEL} \
            --adapter_model_name ${CKPT_PATH} \
            --output_name ${CKPT_PATH}_merged    

        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

        # Decode greedy
        for SPLIT in orig cf; do
            python get_responses_vllm.py \
                ${CKPT_PATH}_merged \
                ${EVAL_OUTPUT_PATH_SEED} \
                ${DATA_PATH}/bias_negative.joblib \
                ${SPLIT} \
                test \
                1 \
                --suffix _greedy \
                --seed ${SEED} \
                --number_entries_used ${NUM_TEST}
        done

        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

        # Decode sampling
        for SPLIT in orig cf; do
            python get_responses_vllm.py \
                ${CKPT_PATH}_merged \
                ${EVAL_OUTPUT_PATH_SEED} \
                ${DATA_PATH}/bias_negative.joblib \
                ${SPLIT} \
                test \
                16 \
                --suffix _sampling \
                --seed ${SEED} \
                --number_entries_used ${NUM_TEST}
        done
    
        # After decoding everything that is needed, remove merged model
        python delete_checkpoint.py ${CKPT_PATH}_merged

        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
    done
done