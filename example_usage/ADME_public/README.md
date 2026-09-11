# Uncertainty quantification and error models for molecular machine learning property predictions

---

This repository contains files, scripts, and notebooks to reproduce the models and results for the ADME public dataset presented in "Uncertainty quantification in molecular machine learning for property predictions under data shifts" and "Error Models for Uncertainty Quantification in Molecular Machine Learning" by Parrondo-Pizarro et al. It also includes Jupyter notebooks to generate the figures included in both manuscripts for public data.

The first manuscript introduces the UNIQUE framework and reports the results from the original UQ benchmarking. The second manuscript builds on this work by benchmarking alternative error model configurations and analyzing their explainability.

### Conda environments

Two YAML files to create the Conda environments to run Chemprop for model training and evaluation (`chemprop-env.yml`) and execute UNIQUE pipeline to perform UQ benchmarking (`unique-env.py`) are included. To create the Conda environment from a YAML file, run:

```bash
conda env create -f <FILE NAME>.yml
```

### UNIQUE configuration files

The `unique_config_files/` folder contains YAML configuration files used by the UNIQUE framework to compute and evaluate the selected UQ metrics. Each ADME public endpoint has a corresponding YAML configuration file.

### Scripts

The `scripts/` folder includes:

* `run_unique.py`: Python script to run the UNIQUE pipeline for the five public ADME endpoints.
* `model_building/`: Subfolder with bash scripts to: (i) train Chemprop property models, (ii) make predictions using the trained models, and (iii) calculate latent representations. The three commands (train, predict, and fingerprint) are run per each property model (Clearance, Binding, and MDR1).

### Jupyter notebooks

Multiple Jupyter notebooks tailored to specific procedures are included:

* `Prepare_data.ipynb`: Prepares the ADME public dataset for modeling, generating input datasets for each property model (Clearance, PPB, and MDR1).
* `Split_data.ipynb`: Splits the data using a scaffold-based approach.
* `Prepare_UNIQUE_input_<MODEL NAME>.ipynb`: Creates the input data files required to run the UNIQUE pipeline.
* `Figures_PublicData.ipynb`: Reproduces the figures of the first manuscript for public data.
* `Building_Error_Models.ipynb`: Trains additional error model confirguations to benchmark alternative input feature sets and input dataset sizes and compositions.
* `Figures_ErrorModels_PublicData.ipynb`: Reproduces the figures of the second manuscript for public data.
* `Understanding_Error_Models.ipynb`: Analyzes error models' feature importance and explainability through Gini Importance and SHAP analysis.

---

### Run complete pipeline

1. Download and prepare data: `Prepare_data.ipynb`
2. Split data: `Split_data.ipynb`
2. Train models: `./scripts/model_building/`
3. Prepare UNIQUE input files: `Prepare_UNIQUE_input_<MODEL NAME>.ipynb`
4. Run UNIQUE: `./scripts/run_unique.py`
5. Generate figures: `Figures_PublicData.ipynb`
7. Train alternative error models: `Building_Error_Models.ipynb`
8. Generate error model figures: `Figures_ErrorModels_PublicData.ipynb`
9. Inspect error models' feature importance and explainability: `Understanding_Error_Models.ipynb`
