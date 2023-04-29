#!/bin/bash
#SBATCH --job-name=szPred
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mnsosa@mi.unc.edu.ar
#SBATCH --ntasks=1

module load gromacs

# Activar el entorno virtual
source venv/bin/activate

cd src

# Ejecutar el script de entrenamiento
srun python train.py > logs.txt

