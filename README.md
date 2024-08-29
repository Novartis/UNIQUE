<figure>
  <img src=./docs/source/_static/unique_logo_blue.png alt="UNIQUE Logo">
  <figcaption align=center><u><b>UN</b></u>certa<u><b>I</b></u>nty <u><b>QU</b></u>antification b<u><b>E</b></u>nchmark: a Python library for benchmarking uncertainty estimation and quantification methods for Machine Learning models predictions.</figcaption>
</figure>

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12.1-blue)
![version](https://img.shields.io/badge/Version-v0.2.2-green)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-red.svg)](https://opensource.org/licenses/BSD-3-Clause)


## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
  - [For Developers](#for-developers)
* [Getting Started](#getting-started)
  - [Prepare Your Dataset](#prepare-your-dataset)
  - [Prepare Your Pipeline](#prepare-your-pipeline)
* [Usage](#usage)
  - [Examples](#examples)
* [Deep Dive](#deep-dive)
* [Contributing](#contributing)
* [License](#license)
* [Contacts & Acknowledgements](#contacts--acknowledgements)


## Introduction

`UNIQUE` provides methods for quantifying and evaluating the uncertainty of Machine Learning (ML) models predictions. The library allows to combine and benchmark multiple uncertainty quantification (UQ) methods simultaneously, generates intuitive visualizations, evaluates the goodness of the UQ methods against established metrics, and in general enables the users to get a comprehensive overview of their ML model's performances from an uncertainty quantification perspective.

`UNIQUE` is a model-agnostic tool, meaning that it does not depend on any specific ML model building platform or provides any  ML model training functionality. It is lightweight, because it only requires the user to input their model's inputs and predictions.

<figure>
  <img src=./docs/source/_static/schema_high_level.png alt="UNIQUE High Level Schema">
  <figcaption align=center>High-level schema of <code>UNIQUE</code>'s components.</figcaption>
</figure>


## Installation

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12.1-blue)

`UNIQUE` is currently compatible with Python 3.8 through 3.12.1. To install the latest release and use the package as is, run the following in a compatible environment of choice:

```bash
pip install git+https://github.com/Novartis/UNIQUE.git
```

> [!TIP]
> To create a dedicated virtual environment for `UNIQUE` using `conda`/`mamba` with all the required and compatible dependencies, check out: [For Developers](#for-developers).

### For Developers

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![codestyle](https://img.shields.io/badge/Code%20Style-black-000000.svg)](https://github.com/psf/black)

If you wish to work on the codebase itself, check first [how to best contribute to `UNIQUE`](./CONTRIBUTING.md).

> [!WARNING]
> The following steps are recommended only for expert/power-users.

First, clone the repository and check into the project folder.

```bash
git clone https://github.com/Novartis/UNIQUE.git ./unique
cd unique
```

The project uses [`mamba`](https://mamba.readthedocs.io/en/latest/index.html) for dependencies management, which is a faster drop-in replacement for [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

> [!TIP]
> If you still wish to use `conda`, you can change the backend solver by adding `--solver=libmamba` to your `conda install` standard command ([see the docs](https://conda.github.io/conda-libmamba-solver/user-guide/#try-it-once)).

To setup the project, run:

```bash
# Install conda environment and jupyter kernel locally
make env && make jupyter-kernel
conda activate .conda/unique
# Setup precommit hooks
make pre-commit
# Install UNIQUE
pip install -e .
# Use `pip install -e .[dev]` to also install optional dependencies
```

In this way, you will have access to the `UNIQUE` codebase and be able to make local modifications to the source code, within the `.conda/unique` environment that contains all the required dependencies.

Additionally, if you use Jupyter Notebooks, the `unique` kernel will be available in the "Select kernel" menu of the JupyterLab/JupyterNotebook UI.

Finally, when using `git` for code versioning, the predefined pre-commit hooks will be run against the commited files for automatic formatting and syntax checks.

You can find out more about custom Jupyter kernels [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) and pre-commit hooks [here](https://pre-commit.com/).


## Getting Started

### Prepare Your Dataset

#### Input Data Preparation

In order to use `UNIQUE` the user only needs to input a dataframe containing at the minimum the following columns:

- **IDs**: a column containing the unique IDs of the datapoints (can be the dataframe's `index` if there are no other identifiers).
- **Labels**: a column containing the target labels/values associated with each datapoint.
- **Predictions**: a column containing the trained ML model's predictions (intended as the final model's, or ensemble of models', single-value output, to be compared with the corresponding label).
- **Subset**: a column containing the specification of which subset each datapoint belongs to. The allowed subsets are: `TRAIN`, `TEST`, and `CALIBRATION`.

> [!CAUTION]
> Make sure to use exactly `TRAIN`, `TEST`, and `CALIBRATION` (all upper-case), as these values are hard-coded.

#### Features Type Specification

Then, depending on the UQ methods one wants to evaluate/use, one can add:

- **Data features**: column(s) containing the feature(s) of each datapoint - e.g., the ones used for training the original ML model. These will be used, for example, to compute the distance between each datapoint in the feature's space (therefore, if you wish to aggregate different features together, you need to provide them as a single column comprising of arrays of length _n_, with _n_ being the number of features, one for each datapoint).
- **Model outputs**: column(s) containing output(s) related to the original ML model. For example, it can be a column containing the individual predictions of an ensemble of models (stored as an array of values for each datapoint), before merging them in a single final predicted value, or it can be directly the variance value of the ensemble associated with each datapoint's predictions.

In `UNIQUE`, data-based and model-based features are represented by the [`FeaturesInputType`](./unique/input_type/base.py) and [`ModelInputType`](./unique/input_type/base.py) classes, respectively. Check out [Prepare Your Pipeline](#prepare-your-pipeline) for more details about how to specify your inputs to `UNIQUE`.

#### Schematic Example

For example, an input dataset to `UNIQUE` could look like this:

|   |ID|Labels|Predictions|Subset|Data Feature|Data Features|Ensemble Predictions|Ensemble Variance|
|--:|-:|:----:|:----------|:-----|:----------:|:-----------:|:------------------:|:---------------:|
|   |1|0.12|0.17|TRAIN|45|[65,12,0.12,True,...]|[0.10,0.12,0.07,0.25,...]|0.02|
|   |2|0.43|0.87|TEST|36|[90,124,15.63,True,...]|[0.43,1.52,0.23,0.45,...]|0.13|
|   |3|4.78|5.62|CALIBRATION|8|[0.9,83,-0.4,False,...]|[1.87,7.92,4.32,5.08,...]|0.81|
|   |...|...|...|...|...|[...]|[...]|...|
|`dtype`|`Any`|`int` or `float`|`int` or `float`|`str`|`int` or `float`|`Iterable` or `np.ndarray`|`Iterable` or `np.ndarray`|`float`|

> [!TIP]
> When storing long arrays/lists in a single `pd.DataFrame` column, saving and reloading the dataframe as a `csv` file will cause issues, due to the fact that each array will be saved as a string when saving in the `csv` format and will be truncated with ellipsis if exceeding a certain limit (typically > 1000 elements per array), thus making it impossible to correctly parse the entire original array when loading the `csv` file again.
>
> To overcome this, consider dumping the input dataframe as a `json` or `pickle` file - e.g., with [`pd.DataFrame.to_json`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html) or [`pd.DataFrame.to_pickle`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html), which will not cause any of the aforementioned issues. `UNIQUE` supports input dataframe in `csv`, `json` and `pickle` formats.

> [!CAUTION]
> Only unpickle files you trust. See the [`pickle` module docs](https://docs.python.org/3/library/pickle.html#module-pickle) for more information, and consider safer serialization formats such as `json`.

Check [Examples](#examples) for some practical, hands-on tutorials on data preparation for `UNIQUE`.


### Prepare Your Pipeline

Once the data has been prepared, the easiest way to run `UNIQUE` is through the [`unique.Pipeline`](./unique/pipeline.py) object. `Pipeline` allows you to run the uncertainty quantification benchmark in an end-to-end fashion.

In order to tell `UNIQUE` which inputs to use and which UQ methods to evaluate, you need to prepare a configuration file. This is a `yaml` file which contains all the specifications needed to retrieve and run the `UNIQUE` pipeline.

#### Available Configuration Arguments

|Argument Name|Type|Description|
|:------------|:---|:----------|
|`data_path`|`str`|Full path to the prepared input dataset. Supported formats: [`csv`, `json`, `pkl`]. Check out [Prepare Your Data](prepare_data.md) for more details.|
|`output_path`|`str`|Full path to the output directory where to save `UNIQUE`'s outputs (figures, tables, models, etc.). Check out [Usage](usage.md) and the [Examples](../examples/index.md) for more details.|
|`id_column_name`|`str`|Name of the column containing the unique data IDs. Use `"index"` if there are no such identifiers or column in your dataset. Check out [Prepare Your Data](prepare_data.md) for more details.|
|`labels_column_name`|`str`|Name of the column containing the target labels/values used to train your predictive model.|
|`predictions_column_name`|`str`|Name of the column containing the point-wise, final predictions from your predictive model.|
|`which_set_column_name`|`str`|Name of the column containing the specification of which subset the datapoint belongs to. Subsets must either be: [`"TRAIN"`, `"TEST"`, `"CALIBRATION"`]. Check out [Prepare Your Data](prepare_data.md) for more details.|
|`model_name`|`str`|Name of your predictive model. Only used for logging purposes.|
|`problem_type`|`str`|Predictive task of the model. Must be either [`"classification"`, `"regression"`].|
|`mode`|`str`|Modality by which to sum variances and distance-based UQ methods to compute {py:class}`~unique.uq_metric_factory.combinations.sum_of_variances.SumOfVariances`. Allowed modalities: [`"compact"`, `"full"`, `"extended"`]. Check out {py:class}`~unique.utils.uncertainty_utils.AnalyticsMode` for more details.|
|`inputs_list`|`list[UniqueInputType]` or `list[dict[str, Any]]`|List of pre-initialized {py:class}`~unique.input_type.base.UniqueInputType` (if configuring `Pipeline` directly) or a list of the `UniqueInputType` names and their corresponding arguments as dictionaries (if configuring `Pipeline` via a `yaml` file). Check out [Features Type Specification](prepare_data.md#features-types-specification) for more details on `UniqueInputType` and [Configuration Template](#configuration-template) for more details on how to populate the inputs list in the `yaml` file.|
|`error_models_list`|`list[UniqueErrorModel]` or `list[dict[str, Any]]`|List of pre-initialized {py:class}`~unique.error_models.base.UniqueErrorModel` (if configuring `Pipeline` directly) or a list of all the `UniqueErrorModel` names and their corresponding arguments as dictionaries (if configuring `Pipeline` via a `yaml` file). Check out [Error Models](../indepth/error_models.md) and [Available UQ Methods & Objects in `UNIQUE`](../indepth/available_uq_methods.md) for more details on available error models.|
|`individual_plots`|`bool`|Whether to plot each computed UQ method's evaluation plots. Note: the plots of the overall best UQ methods are always saved (displaying to screen depends on `display_outputs`).|
|`summary_plots`|`bool`|Whether to plot the summary plots with all UQ methods. Note: the summary plots are always saved (displaying to screen depends on `display_outputs`).|
|`save_plots`|`bool`|Whether to save the individual plots (if enabled via `individual_plots`).|
|`evaluate_test_only`|`bool`|Whether to evaluate the UQ methods against the `"TEST"` set only. If "False", evaluation will be carried out for the `"TRAIN"` and `"CALIBRATION"` subsets as well. Check out [Input Data Preparation](prepare_data.md#input-data-preparation) for more details.|
|`display_outputs`|`bool`|Whether to display the enabled plots and output tables to screen. Only works if running `UNIQUE` in a JupyterNotebook cell.|
|`n_bootstrap`|`int`|Number of bootstrapping samples to use. Default is 500 (even if `n_bootstrap` is not specified explicitly). Note: bootstrapping for selecting the overall best UQ methods is always run (unless the `Pipeline._bootstrap` is set to `False`).|
|`verbose`|`bool`|Whether to enable "DEBUG"-level logging verbosity. "INFO"-level messages are always printed to `stdout`.|

#### Configuration Template

You can find below a commented example of a typical `yaml` configuration file for your `Pipeline`:

```yaml
#######
# I/O #
#######
# Path to the prepared input dataset
data_path: "/path/to/your/input/dataset.[csv,json,pkl]"
# Path to the output folder where to save UNIQUE's outputs
output_path: "/path/to/output/folder"

########
# Data #
########
# Name of the column containing the unique data IDs
id_column_name: "ID"
# Name of the column containing the labels
labels_column_name: "Labels"
# Name of the column containing the original model's predictions
predictions_column_name: "Predictions"
# Name of the column containing the subset specification ("TRAIN", "TEST", "CALIBRATION")
which_set_column_name: "Subset"
# Name of the original model
model_name: "MyModel"
# Specify which task your model solves: either "regression" or "classification"
problem_type: "regression"
# Modality by which to sum variances and distance-based UQ methods. Check `unique.utils.uncertainty_utils.AnalyticsMode` for more details.
mode: "compact"

#############
# UQ Inputs #
#############
# List of UNIQUE InputTypes specifying the column name of the inputs and the UQ methods to compute for each of them (if none are specified, all supported UQ methods for each InputType will be computed)
# Note: it needs to be a list, even if only one input type is specified (note the hyphens)
inputs_list:
  # FeaturesInputType are features that can have `int` or `float` values and can be represented as a single value or grouped as a list/array of features for each datapoint
  - FeaturesInputType:
    # Name of the column containing the features (for example here we assume a single value for each datapoint)
      column_name: "Feature"
    # Only the "manhattan_distance" and "euclidean_distance" UQ methods will be computed for this input (note that they are specified as a list using the hyphen)
      metrics:
      - "manhattan_distance"
      - "euclidean_distance"
  - FeaturesInputType:
    # Name of the column containing the features (for example here we assume an array of features for each datapoint)
      column_name: "FeaturesArray"
    # Only "euclidean_distance" UQ method will be computed for this input (note that you can also specify the methods as a single value - no hyphens here)
      metrics: "euclidean_distance"
  # ModelInputType is the variance of the ensemble's predictions
  - ModelInputType:
    # Name of the column containing the variance
      column_name: "Variance"
    # No methods are specified here, which means that all supported UQ methods for this input type will be computed

###################
# UQ Error Models #
###################
# List of UNIQUE ErrorModels specifying available model's hyperparameters as keyword-arguments
# You can specify as many error models as you want, even the same type but with different hyperparameters (GridSearch is not yet implemented in UNIQUE)
# Note: it needs to a list, even if only one error model is specified (note the hyphens)
error_models_list:
  # UniqueRandomForestRegressor is a RF regressor trained to predict the error between the original model's predictions and data labels
  - UniqueRandomForestRegressor:
    # All available arguments to the model can be specified here. See each model's documentation for the full list of arguments. If no hyperparameters are specified, UNIQUE will use the default ones
      max_depth: 10
      n_estimators: 500
    # List of error types to use as target values (note the hyphen). For each error type, a separate model will be built to predict it
    # Supported errors are:
    # "l1" (=absolute error), "l2" (squared error), "unsigned"
      error_type_list:
        - "l1"

#######################
# Evaluation Settings #
#######################
# Whether to plot each UQ method's evaluation plots. Note: the plots of the best UQ methods are always saved (displaying depends on `display_outputs`)
individual_plots: false
# Whether to plot the summary plots with all UQ methods. Note: the summary plots are always saved (displaying depends on `display_outputs`)
summary_plots: true
# Whether to save the enabled plots in the output folder
save_plots: false
# Whether to evaluate the UQ methods against the TEST set only. If "False", evaluation will be carried out for "TRAIN" and "CALIBRATION" sets as well
evaluate_test_only: true
# Whether to display the plots to screen. Only works if running in a JupyterNotebook cell
display_outputs: true
# Number of bootstrapping samples to run. Note: bootstrapping to determine the best UQ metric is ALWAYS run unless the private attribute `Pipeline._bootstrap` is set to False.
n_bootstrap: 500
# Logging messages levels. If True, logger will output DEBUG level messages.
verbose: false
```

> [!TIP]
> Copy and save the above template as a `yaml` file to use in your project.

> [!NOTE]
> Currently supported UQ methods ("metrics" argument) for `FeaturesInputType` inputs:
> * [`manhattan_distance`](./unique/uncertainty/data_based/distance_to_training.py)
> * [`euclidean_distance`](./unique/uncertainty/data_based/distance_to_training.py)
> * [`tanimoto_distance`](./unique/uncertainty/data_based/distance_to_training.py) (for integer-only inputs)
> * [`gaussian_euclidean_kde`](./unique/uncertainty/data_based/kernel_density.py)
> * [`gaussian_manhattan_kde`](./unique/uncertainty/data_based/kernel_density.py)
> * [`exponential_manhattan_kde`](./unique/uncertainty/data_based/kernel_density.py)
>
> Currently supported UQ methods ("metrics" argument) for `ModelInputType` inputs:
> * [`ensemble_variance`](./unique/uncertainty/model_based/ensemble_variance.py) (for regression tasks only)
> * [`probability`](./unique/uncertainty/model_based/probability.py) (for classification tasks only).

> [!TIP]
> See [Available UQ Methods & Objects in `UNIQUE`](#available-uq-methods--objects-in-unique) for more details.

For more examples of `yaml` configuration files, you can check the [`notebooks`](./notebooks/) folder which contains some examples.


## Usage

Finally, once the data and configuration files have been prepared, you can run `UNIQUE` in the following way:

```python
from unique import Pipeline

# Prepare UNIQUE pipeline
pipeline = Pipeline.from_config("/path/to/config.yaml")

# Run UNIQUE pipeline
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()
# Returns: (Dict[str, np.ndarray], Dict[str, pd.DataFrame])
```

Fitting the `Pipeline` will return two dictionaries:

- `uq_methods_outputs`: contains each UQ method's name (as in "UQ_Method_Name[Input_Name(s)]") and computed UQ values.
- `uq_evaluation_outputs`: contains, for each evaluation type (ranking-based, proper scoring rules, and calibration-based), the evaluation metrics outputs for all the corresponding UQ methods organized in `pd.DataFrame`.

Additionally, `UNIQUE` also generates graphical outputs in the form of tables and evaluation plots (if `display_outputs` is enabled and the code is running in a JupyterNotebook cell).


### Examples

For more hands-on examples and detailed usage, check out some of the examples in [`notebooks`](./notebooks).


## Deep Dive

Check out the docs [INSERT LINKS TO DOCS] for an in-depth overview of `UNIQUE`'s concepts, functionalities, outputs, and references.


## Contributing

Any and all contributions and suggestions from the community are more than welcome and highly appreciated. If you wish to help us out in making `UNIQUE` even better, please check out our [contributing guidelines](./CONTRIBUTING.md).

Please note that we have a [Code of Conduct](./CODE_OF_CONDUCT.md) in place to ensure a positive and inclusive community environment. By participating in this project, you agree to abide by its terms.


## License

`UNIQUE` is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file.


## Contacts & Acknowledgements

For any questions or further details about the project, please get in touch with any of the following contacts:

* **[Jessica Lanini](mailto:jessica.lanini@novartis.com?subject=UNIQUE)**
* **[Minh Tam Davide Huynh](https://github.com/mtdhuynh)**
* **[Gaetano Scebba](mailto:gaetano.scebba@novartis.com?subject=UNIQUE)**
* **[Nadine Schneider](mailto:nadine-1.schneider@novartis.com?subject=UNIQUE)**
* **[Raquel Rodríguez-Pérez](mailto:raquel.rodriguez_perez@novartis.com?subject=UNIQUE)**


![Novartis Logo](./docs/source/_static/novartis_logo.png)
