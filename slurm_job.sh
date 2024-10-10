#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=MyFirstJobOnARC
#SBATCH --partition=cpu
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 10s
echo Hello World
echo Starting model training
python train_example.py

echo Finished model training
