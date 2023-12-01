#!/bin/bash

#SBATCH --time=5-00:00:00    
#SBATCH --nodes=1
#SBATCH --output ./score/evaluation/evaluate-%j.out      # terminal output
#SBATCH --error ./score/evaluation/evaluate-%j.err

#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&A100"
#SBATCH --mail-type=END
#SBATCH --mail-user lorenzo.valente@desy.de
#SBATCH --job-name jvae-qcdorwhat

bash
source ~/.bashrc
conda activate jvae

cd /beegfs/desy/user/valentel/JointVAE4AD

python evaluation.py

exit


