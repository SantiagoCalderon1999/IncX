#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=comparison
python get_similarity_comparison.py