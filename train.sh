#!/bin/bash
#SBATCH --job-name=szPred
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --time=10:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mnsosa@mi.unc.edu.ar


module load nvhpc

nvidia-smi

# Activar el entorno virtual
# conda activate tf-gpu
source /home/mnsosa/miniconda3/etc/profile.d/conda.sh
echo "Activando entorno virtual"
conda activate torch-gpu
echo "Entorno virtual activado"

python3 src/train.py > salida_train.txt
