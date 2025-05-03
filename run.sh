#!/bin/bash

# Create a virtual environment in your home directory
python3 -m venv ~/myenv
source ~/myenv/bin/activate

# Upgrade pip to avoid compatibility issues
pip install --upgrade pip

# Install necessary packages inside the virtual environment
pip install torch numpy collections pickle
pip install torchvision
pip install matplotlib
pip install shap
pip install captum
pip install datetime
pip install kaggle

# # # Run your Python script
python -u ./medmnist_finetune.py 2> >(tee run.err) | tee run.log
