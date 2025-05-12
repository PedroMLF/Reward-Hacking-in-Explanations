#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o motivation_experiment/log.txt
#------------------------------------

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

python ${PROJECT_HOME}/motivation_experiment/prepare_data.py


