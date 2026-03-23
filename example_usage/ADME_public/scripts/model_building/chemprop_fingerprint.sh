#!/bin/bash

# Activate the conda environment
conda activate chemprop-env

# Calculate the latent representations (encodings) of CLint model inputs
chemprop fingerprint \
    --test-path ../data/CLint_dataset_with_splits.csv \
    --model-path ../CLint_model \
    --output ../CLint_model/latent_fps.csv \
    --smiles-columns "Structure" \
    --ffn-block-index 0

# Calculate the latent representations (encodings) of PPB model inputs
chemprop fingerprint \
    --test-path ../data/PPB_dataset_with_splits.csv \
    --model-path ../PPB_model \
    --output ../PPB_model/latent_fps.csv \
    --smiles-columns "Structure" \
    --ffn-block-index 0

# Calculate the latent representations (encodings) of MDR1 model inputs
chemprop fingerprint \
    --test-path ../data/MDR1_dataset_with_splits.csv \
    --model-path ../MDR1_model \
    --output ../MDR1_model/latent_fps.csv \
    --smiles-columns "Structure" \
    --ffn-block-index 0
