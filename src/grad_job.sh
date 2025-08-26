#!/bin/bash
#SBATCH --time=00:30:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB

source ../.venv/bin/activate

python gradcam_comp.py