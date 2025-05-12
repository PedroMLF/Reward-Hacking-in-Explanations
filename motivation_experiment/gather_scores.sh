#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o motivation_experiment/log_gather_scores.txt
#------------------------------------

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

python ${PROJECT_HOME}/motivation_experiment/gather_scores.py
