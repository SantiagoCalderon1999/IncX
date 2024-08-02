#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=incx_gpu
#SBATCH --gres=gpu
#SBATCH -p biomed_a100_gpu 
python experiments/incx/get_saliency_maps.py