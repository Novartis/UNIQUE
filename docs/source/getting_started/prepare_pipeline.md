# Prepare Your Pipeline

Once the data has been prepared, the easiest way to run `UNIQUE` is through the {py:class}`~unique.pipeline.Pipeline` object. `Pipeline` allows you to run the uncertainty quantification benchmark in an end-to-end fashion.

In order to tell `UNIQUE` which inputs to use and which UQ methods to evaluate, you need to prepare a configuration file. This is a [`yaml`](https://en.wikipedia.org/wiki/YAML) file which contains all the specifications needed to retrieve and run the `UNIQUE` pipeline.

## Available Configuration Arguments

The `Pipeline` configuration options can be either provided through a `yaml` file - and then loaded using {py:meth}`~unique.pipeline.Pipeline.from_config` - or directly to the `Pipeline` object at initialization.

:::{table} Full list of available `Pipeline` configuration arguments.
:widths: auto
:align: center

|Argument Name|Type|Description|Additional Information|
|:------------|:---|:----------|:---------------------|
|`data_path`|`str`|Full path to the prepared input dataset. Supported formats: `csv`, `json`, `pkl`.|Check out [Prepare Your Data](prepare_data.md) for more details.|
|`output_path`|`str`|Full path to the output directory where to save `UNIQUE`'s outputs (figures, tables, models, etc.).|Check out [Usage](usage.md) and the [Examples](../examples/index.md) for more details.|
|`id_column_name`|`str`|Name of the column containing the unique data IDs. Use `"index"` if there are no such identifiers or column in your dataset.|Check out [Prepare Your Data](prepare_data.md) for more details.|
|`labels_column_name`|`str`|Name of the column containing the target labels/values used to train your predictive model.||
|`predictions_column_name`|`str`|Name of the column containing the point-wise, final predictions from your predictive model.||
|`which_set_column_name`|`str`|Name of the column containing the specification of which subset the datapoint belongs to. Subsets must either be: `"TRAIN"`, `"TEST"`, `"CALIBRATION"`.|Check out [Prepare Your Data](prepare_data.md) for more details.|
|`model_name`|`str`|Name of your predictive model. Only used for logging purposes.||
|`problem_type`|`str`|Predictive task of the model. Must be either: `"classification"`, `"regression"`.||
|`mode`|`str`|Modality by which to sum variances and distance-based UQ methods to compute {py:class}`~unique.uq_metric_factory.combinations.sum_of_variances.SumOfVariances`. Allowed modalities: `"compact"`, `"full"`, `"extended"`.|Check out {py:class}`~unique.utils.uncertainty_utils.AnalyticsMode` for more details.|
|`inputs_list`|`list[UniqueInputType]` or `list[dict[str, Any]]`|List of pre-initialized {py:class}`~unique.input_type.base.UniqueInputType` (if configuring `Pipeline` directly) or a list of the `UniqueInputType` names and their corresponding arguments as dictionaries (if configuring `Pipeline` via a `yaml` file).|Check out [Input Types Specification](prepare_data.md#input-types-specification) for more details on `UniqueInputType` and [Configuration Template](#configuration-template) for more details on how to populate the inputs list in the `yaml` file.|
|`error_models_list`|`list[UniqueErrorModel]` or `list[dict[str, Any]]`|List of pre-initialized {py:class}`~unique.error_models.base.UniqueErrorModel` (if configuring `Pipeline` directly) or a list of all the `UniqueErrorModel` names and their corresponding arguments as dictionaries (if configuring `Pipeline` via a `yaml` file).|Check out [Error Models](../indepth/error_models.md) and [Available Inputs & UQ Methods](../indepth/available_inputs_uq_methods.md) for more details on available error models.|
|`individual_plots`|`bool`|Whether to plot each computed UQ method's evaluation plots. Note: the plots of the overall best UQ methods are always saved (displaying to screen depends on `display_outputs`).||
|`summary_plots`|`bool`|Whether to plot the summary plots with all UQ methods. Note: the summary plots are always saved (displaying to screen depends on `display_outputs`).||
|`save_plots`|`bool`|Whether to save the individual plots (if enabled via `individual_plots`).||
|`evaluate_test_only`|`bool`|Whether to evaluate the UQ methods against the `"TEST"` set only. If "False", evaluation will be carried out for the `"TRAIN"` and `"CALIBRATION"` subsets as well.|Check out [Input Data Preparation](prepare_data.md#input-data-preparation) for more details.|
|`display_outputs`|`bool`|Whether to display the enabled plots and output tables to screen. Only works if running `UNIQUE` in a JupyterNotebook cell.||
|`n_bootstrap`|`int`|Number of bootstrapping samples to use. Default is 500 (even if `n_bootstrap` is not specified explicitly). Note: bootstrapping for selecting the overall best UQ methods is always run (unless the `Pipeline._bootstrap` is set to `False`).||
|`verbose`|`bool`|Whether to enable "DEBUG"-level logging verbosity. "INFO"-level messages are always printed to `stdout`.||
:::

You can find below a comprehensive template of a typical `yaml` configuration file for your `Pipeline`.

## Configuration Template

:::{code-block} yaml
:caption: `UNIQUE` configuration template

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
# Modality by which to sum variances and distance-based UQ methods. Check {py:obj}`~unique.utils.uncertainty_utils.AnalyticsMode` for more details.
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
:::

:::{tip}
Copy and save the above template as a `yaml` file to use in your project.
:::

:::{note}
Currently supported UQ methods (`metrics` argument) for {py:class}`~unique.input_type.base.FeaturesInputType` inputs:
* {py:class}`manhattan_distance <unique.uncertainty.data_based.distance_to_training.ManhattanDistance>`
* {py:class}`euclidean_distance <unique.uncertainty.data_based.distance_to_training.EuclideanDistance>`
* {py:class}`tanimoto_distance <unique.uncertainty.data_based.distance_to_training.TanimotoDistance>` (for integer-only inputs)
* {py:class}`gaussian_euclidean_kde <unique.uncertainty.data_based.kernel_density.GaussianEuclideanKDE>`
* {py:class}`gaussian_manhattan_kde <unique.uncertainty.data_based.kernel_density.GaussianManhattanKDE>`
* {py:class}`exponential_manhattan_kde <unique.uncertainty.data_based.kernel_density.ExponentialManhattanKDE>`

---

Currently supported UQ methods (`metrics` argument) for {py:class}`~unique.input_type.base.ModelInputType` inputs:
* {py:class}`ensemble_variance <unique.uncertainty.model_based.ensemble_variance.EnsembleVariance>` (for regression tasks only)
* {py:class}`probability <unique.uncertainty.model_based.probability.Probability>` (for classification tasks only)

---

Currently supported error types (`error_type_list` argument) for {py:class}`~unique.error_models.base.UniqueErrorModel` error models:
* `"L1"` error (defined as {math}`|y - \hat{y}|`);
* `"L2"` error (defined as {math}`(y - \hat{y})^2`);
* `"unsigned"` error (defined as {math}`y - \hat{y}`);

where {math}`y = label` and {math}`\hat{y} = prediction`.
:::

:::{deprecated} 0.1.0
For {py:class}`~unique.input_type.base.ModelInputType` inputs, it is not necessary to specify the `metrics` argument to compute anymore.

`UNIQUE` automatically detects whether to treat the model-based inputs as regression-based predictions/ensemble variance or classification-based probabilities depending on the specified `problem_type`.
:::

:::{seealso}
Check out [In-Depth Overview of `UNIQUE`](../indepth/overview.md) for more details about `UNIQUE`'s input types, UQ methods, error models, and more.

For other real-world examples of `yaml` configuration files, check out the [Examples](../examples/index.md).
:::
