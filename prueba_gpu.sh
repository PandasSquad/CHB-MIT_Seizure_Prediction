#!/bin/bash
#SBATCH --job-name=szPred
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:20
#SBATCH --ntasks=1

module load nvhpc

nvidia-smi

# Activar el entorno virtual
# conda activate tf-gpu
source /home/mnsosa/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu

python3 prueba_gpu.py