#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=ENEL649_ASSGN2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromENEL649ASSGN2Job_%j.out
#SBATCH --error=ErrorFromENEL649ASSGN2Job_%j.err

echo Hi!

echo "Activate Python environment"
source activate mm_enel645_assg2

echo Starting model training

time python -u train_model.py

echo Finished model training
