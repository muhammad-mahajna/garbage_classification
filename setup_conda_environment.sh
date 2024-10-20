#!/bin/bash

# Check if the environment already exists
if conda info --envs | grep -q "mm_enel645_assg2"; then
  echo "Environment 'mm_enel645_assg2' already exists. Activating..."
else
  # Create the conda environment
  echo "Creating environment 'mm_enel645_assg2'..."
  conda create -n mm_enel645_assg2 python=3.12 -y
fi

# Activate the conda environment
echo "Activating 'mm_enel645_assg2' environment..."
source activate mm_enel645_assg2

# Install the required packages
echo "Installing required packages..."

conda install -c conda-forge liblapack libblas -y
conda install pytorch torchvision torchaudio -c pytorch -y
conda install -c conda-forge transformers -y
conda install -c conda-forge scikit-learn matplotlib seaborn -y
conda install pillow numpy pandas -y

# List installed packages
echo "Listing installed packages..."
conda list

conda env export > mm_enel645_assg2_environment.yml
