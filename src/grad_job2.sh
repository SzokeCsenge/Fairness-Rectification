#!/bin/bash
#SBATCH --time=00:30:00 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=3000

source ../.venv/bin/activate

python gradcam_comp.py