#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=MyFirstJobOnARC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 10s
echo Hello World
echo Starting model training
python train_example.py

echo Finished model training
