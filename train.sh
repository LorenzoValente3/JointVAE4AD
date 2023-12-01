#!/bin/bash

#SBATCH --time=5-00:00:00    # Set to 5 days
#SBATCH --nodes=1
#SBATCH --output ./score/training-%j.out      # terminal output
#SBATCH --error ./score/training-%j.err

#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&A100"
#SBATCH --mail-type=END
#SBATCH --mail-user lorenzo.valente@desy.de
#SBATCH --job-name jvae-qcdorwhat

bash
source ~/.bashrc
conda activate jvae

# conda activate py36

cd /beegfs/desy/user/valentel/JointVAE4AD

python main.py

exit


