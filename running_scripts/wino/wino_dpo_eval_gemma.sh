#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/wino/llama3-8B/dpo/training/log_eval_gemma.txt


RUN_GENERATE=true
RUN_OTHER=true

DATA_PATH="data/wino/"
SAVE_PATH="checkpoints/wino_dpo"
EVAL_OUTPUT_PATH="outputs/wino/llama3-8B/dpo/eval" # Where the eval outputs (test) with dpo are saved
NUM_TEST=10000
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

RW_MODEL="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

RW_LIST=(skywork-gemma_none skywork-gemma_heuristic skywork-gemma_heuristic_oracle)
RW_CKPT_LIST=(477 636 318)

for APPROACH in negative; do

    for i in "${!RW_LIST[@]}"; do

        RW=${RW_LIST[i]}
        CKPT=${RW_CKPT_LIST[i]}

        EVAL_OUTPUT_PATH_RW=${EVAL_OUTPUT_PATH}/${APPROACH}_${RW}

        if ${RUN_GENERATE}; then
            # Get ckpt path
            CKPT_PATH=${SAVE_PATH}/bias_${APPROACH}_${RW}/checkpoint-${CKPT}
            echo "CKPT_PATH: ${CKPT_PATH}"
        
            # Merge model
            python ${PROJECT_HOME}/merge_peft_model.py \
                --base_model_name ${MODEL} \
                --adapter_model_name ${CKPT_PATH} \
                --output_name ${CKPT_PATH}_merged    
            
            # Decode greedy
            for SPLIT in orig cf; do
                python get_responses_vllm.py \
                    ${CKPT_PATH}_merged \
                    ${EVAL_OUTPUT_PATH_RW} \
                    ${DATA_PATH}/bias_${APPROACH}.joblib \
                    ${SPLIT} \
                    test \
                    1 \
                    --suffix _greedy_${RW}_${CKPT} \
                    --seed 0 \
                    --number_entries_used ${NUM_TEST}
            done
        
            # Decode sampling
            for SPLIT in orig cf; do
                python get_responses_vllm.py \
                    ${CKPT_PATH}_merged \
                    ${EVAL_OUTPUT_PATH_RW} \
                    ${DATA_PATH}/bias_${APPROACH}.joblib \
                    ${SPLIT} \
                    test \
                    16 \
                    --suffix _sampling_${RW}_${CKPT} \
                    --seed 0 \
                    --number_entries_used ${NUM_TEST}
            done
        
            # After decoding everything that is needed, remove merged model
            python delete_checkpoint.py ${CKPT_PATH}_merged
        fi

        if ${RUN_OTHER}; then
            # Collect all eval paths
            eval_paths=()
            eval_paths_cf=()
            for DECODING_APPROACH in greedy sampling; do
                eval_paths+=(${EVAL_OUTPUT_PATH_RW}/bias_${APPROACH}_orig_test_${DECODING_APPROACH}_${RW}_${CKPT}.joblib)
                eval_paths_cf+=(${EVAL_OUTPUT_PATH_RW}/bias_${APPROACH}_cf_test_${DECODING_APPROACH}_${RW}_${CKPT}.joblib)
            done
            IFS=","; eval_path_all="${eval_paths[*]}"; unset IFS
            IFS=","; eval_path_all_cf="${eval_paths_cf[*]}"; unset IFS
            
            python eval_responses_wino.py ${eval_path_all} --use_model --seed 0
            python eval_responses_wino.py ${eval_path_all_cf} --use_model --seed 0

            for SPLIT in orig cf; do
                python ${PROJECT_HOME}/eval_rewards.py \
                    ${EVAL_OUTPUT_PATH_RW}/bias_${APPROACH}_${SPLIT}_test_sampling_${RW}_${CKPT}_data.joblib \
                    ${RW_MODEL} \
                    none \
                    --batch_size 16 \
                    --seed 0
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        fi
    done
done
