#!/bin/bash

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate chemprop-env

# Predict CLint endpoints with the corresponding ensemble
chemprop predict \
    --test-path ../../data/CLint_dataset_with_splits.csv \
    --model-paths ../../CLint_model \
    --preds-path ../../CLint_model/predictions.csv \
    --smiles-columns "Structure" \
    --uncertainty-method ensemble

# Predict PPB endpoints with the corresponding ensemble
chemprop predict \
    --test-path ../../data/PPB_dataset_with_splits.csv \
    --model-paths ../../PPB_model \
    --preds-path ../../PPB_model/predictions.csv \
    --smiles-columns "Structure" \
    --uncertainty-method ensemble

# Predict MDR1 endpoint with the corresponding ensemble
chemprop predict \
    --test-path ../../data/MDR1_dataset_with_splits.csv \
    --model-paths ../../MDR1_model \
    --preds-path ../../MDR1_model/predictions.csv \
    --smiles-columns "Structure" \
    --uncertainty-method ensemble
