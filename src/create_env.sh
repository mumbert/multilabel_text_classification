#!/bin/bash

# ARGUMENTS
ENV_NAME="multilabel_text_classification"

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# STEPS
conda remove -n $ENV_NAME --all -y
conda create --name $ENV_NAME python=3.8 -y
conda activate $ENV_NAME
pip install -r src/requirements.txt
conda env list