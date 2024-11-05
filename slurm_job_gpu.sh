#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=MyFirstJobOnARC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err

# Activate Python environment
# source activate mm_enel645_assg2

# Notify starting model training
echo Starting model training
time python -u train_model.py

# Notify end of training
echo Finished model training
