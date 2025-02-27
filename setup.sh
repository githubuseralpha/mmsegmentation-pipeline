#!/bin/bash

# Create and activate conda environment
conda create --name openmmlab python=3.8 -y
source activate openmmlab  # If using conda >= 4.4, use `conda activate openmmlab`

# Install PyTorch and related libraries
pip3 install torch torchvision torchaudio

# Install OpenMMLab dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install the current package in editable mode
pip install -v -e .

code --install-extension github.copilot
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

git config --global user.email "patrykswiat@gmail.com"
git config --global user.name "Patryk Swiat"
