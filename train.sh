#!/bin/bash
#SBATCH --job-name=szPred
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --ntasks=1

module load nvhpc

nvidia-smi

# Activar el entorno virtual
# conda activate tf-gpu
source /home/mnsosa/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu


python3 src/train.py
