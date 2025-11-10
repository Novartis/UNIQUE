#!/bin/bash

# Activate the conda environment
conda activate chemprop-env

# Train the CLint model using chemprop
chemprop train \
    --data-path ../data/CLint_dataset_with_splits.csv \
    --task-type regression \
    --output-dir ../CLint_model \
    --epochs 50 \
    --target-columns "rLM LogCLint" "hLM LogCLint" \
    --smiles-columns "Structure" \
    --num-replicates 1 \
    --ensemble-size 10 \
    --splits-column "split"

# Train the PPB model using chemprop
chemprop train \
    --data-path ../data/PPB_dataset_with_splits.csv \
    --task-type regression \
    --output-dir ../PPB_model \
    --epochs 50 \
    --target-columns "LogFu-Rat" "LogFu-Human" \
    --smiles-columns "Structure" \
    --num-replicates 1 \
    --ensemble-size 10 \
    --splits-column "split"

# Train the MDR1 model using chemprop
chemprop train \
    --data-path ../data/MDR1_dataset_with_splits.csv \
    --task-type regression \
    --output-dir ../MDR1_model \
    --epochs 50 \
    --target-columns "MDCK-MDR1_LogER" \
    --smiles-columns "Structure" \
    --num-replicates 1 \
    --ensemble-size 10 \
    --splits-column "split"
