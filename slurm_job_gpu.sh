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
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 1s
echo Hello World
echo Starting model training
python tl_try5.py

echo Finished model training
