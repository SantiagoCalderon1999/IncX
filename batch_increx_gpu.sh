#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=increx_gpu
#SBATCH --gres=gpu
#SBATCH -p biomed_a100_gpu 
python increx_experiment.py