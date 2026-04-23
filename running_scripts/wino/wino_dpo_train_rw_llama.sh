#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/wino/llama3-3B/dpo/training/log_train_llama.txt

DPO_DATA_PATH="outputs/wino/llama3-3B/dpo"
SAVE_PATH="checkpoints/llama3-3B/wino_dpo"

#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-3.2-3B-Instruct"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0 1 2; do

    DPO_DATA_PATH_SEED=${DPO_DATA_PATH}/seed_${SEED}/data
    SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}

    for RW in sk-llama_original sk-llama_aug-rm-c sk-llama_aug-rm-d; do

        echo -e "\n\n--- ${RW} ---\n\n"

        # Some hparams inspired by:
        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_qlora.yaml
        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-gemma/dpo/config_full.yaml
        # https://www.philschmid.de/rl-with-llms-in-2025-dpo

        python -u ${PROJECT_HOME}/train_dpo.py \
            --dataset_path ${DPO_DATA_PATH_SEED}/bias_negative_${RW} \
            --model_name_or_path ${MODEL} \
            --output_dir ${SAVE_PATH_SEED}/bias_negative_${RW} \
            --run_name dpo_negative_${RW} \
            --num_train_epochs 5 \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 8 \
            --gradient_checkpointing \
            --optim="paged_adamw_32bit" \
            --learning_rate 5.0e-6 \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --max_prompt_length 1024 \
            --max_length 1024 \
            --logging_steps 10 \
            --bf16 \
            --tf32 True \
            --eval_strategy epoch \
            --save_strategy epoch \
            --save_only_model \
            --no_remove_unused_columns \
            --beta 0.1 \
            --seed ${SEED} \
            --run_name wino_dpo_negative_${RW}_seed-${SEED} \
            --use_peft \
            --lora_alpha 32 \
            --lora_r 16 \
            --lora_dropout 0.05 \
            --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
            --lora_modules_to_save "lm_head" "embed_tokens"

        date '+[%H:%M:%S-%d/%m/%y]'
    done
done
