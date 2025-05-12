#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/aqua/llama3-8B/dpo/training/log_train_llama.txt

DATA_PATH="data/aqua/"
DPO_DATA_PATH="outputs/aqua/llama3-8B/dpo/data"
SAVE_PATH="checkpoints/aqua_dpo"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for APPROACH in negative; do
    for RW in skywork-llama_none skywork-llama_heuristic skywork-llama_heuristic_oracle; do

        echo -e "\n--- ${APPROACH} + ${RW} ---\n"

        # Some hparams inspired by:
        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_qlora.yaml
        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-gemma/dpo/config_full.yaml
        # https://www.philschmid.de/rl-with-llms-in-2025-dpo

        python ${PROJECT_HOME}/train_dpo.py \
            --dataset_path ${DPO_DATA_PATH}/cot_${APPROACH}_${RW} \
            --model_name_or_path ${MODEL} \
            --output_dir ${SAVE_PATH}/cot_${APPROACH}_${RW} \
            --run_name dpo_${APPROACH}_${RW} \
            --num_train_epochs 5 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --per_device_eval_batch_size 4 \
            --gradient_checkpointing \
            --optim="paged_adamw_32bit" \
            --learning_rate 5.0e-6 \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --max_prompt_length 3072 \
            --max_length 3072 \
            --logging_steps 10 \
            --bf16 \
            --tf32 True \
            --eval_strategy epoch \
            --save_strategy epoch \
            --no_remove_unused_columns \
            --beta 0.1 \
            --seed 0 \
            --run_name aqua_dpo_${APPROACH}_${RW} \
            --use_peft \
            --lora_alpha 32 \
            --lora_r 16 \
            --lora_dropout 0.05 \
            --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
            --lora_modules_to_save "lm_head" "embed_tokens"

        date '+[%H:%M:%S-%d/%m/%y]'

    done
done
