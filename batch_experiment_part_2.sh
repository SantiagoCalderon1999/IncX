#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=d_rise_metrics
python experiments/d_rise/get_metrics.py