---
myst:
   substitutions:
      low_level_schema: |
         ```{figure} ../_static/schema_low_level.png
         :target: _images/schema_low_level.png
         :alt: UNIQUE Low Level Schema
         :class: dark-light

         Low-level schema of `UNIQUE`'s components.
         ```
---

# In-Depth Overview of `UNIQUE`

`UNIQUE` implements various input types, UQ methods, and evaluation metrics, and allows for an end-to-end uncertainty quantification benchmarking.

Each input type object is associated with a certain real-world input from the user's data; each UQ method directly consists of or can be derived from the input type values; each UQ method is associated with one or multiple evaluation benchmarks and corresponding metrics.

{{low_level_schema}}


## Input Types

The above schema shows a detailed map of `UNIQUE`'s workflow and objects. From a user's input dataset, `UNIQUE` abstracts two different input type objects: _data_- (or _features_-) and _model_-based input types, that represent the input values necessary to estimate and quantify the uncertainty in model's predictions.

:::{seealso}
Check out [Input Type Specification](../getting_started/prepare_data.md#input-types-specification) for more details about _data_- vs _model_-based input types.
:::

## UQ Methods

Each input type object can be used either directly as a representation of model's uncertainty or to compute a UQ proxy using associated UQ methods.

These methods can either directly derive the UQ estimates from the input data (_base_ UQ methods), or combine several _base_ UQ methods to generate more complex and holistic measures of uncertainty (_transformed_ UQ methods).

`UNIQUE` distinguishes between "_base_" UQ methods and "_transformed_" UQ methods:

|_Base_ UQ Methods|_Transformed_ UQ Methods|
|:----------------|:-----------------------|
|UQ methods directly computable from the input data and/or original model (e.g., {py:class}`~unique.uncertainty.model_based.ensemble_variance.EnsembleVariance` corresponds to the variance of the predictions for a given ensemble of models).<br><br>Similarly to input types, these are usually either data-based or model-based.|Combinations of base UQ methods (e.g., {py:class}`~unique.uq_metric_factory.combinations.sum_of_variances.SumOfVariances` is the sum of all the input variances - and distances converted to variances, if enabled/any).<br><br>More generally, any UQ method derived from transformation/further processing of other UQ methods (e.g., error models use base UQ methods as input training data, and their outputs/predictions are UQ methods themselves).|

:::{seealso}
Check out [Available Inputs & UQ Methods](available_inputs_uq_methods.md) for more details about available UQ methods.
:::

## Error Models

Error models are a novel way to measure uncertainty, and are an example of _transformed_ UQ method, as they combine several input features and _base_ UQ methods to try predicting the error of the model's predictions, as a UQ proxy itself.

:::{seealso}
Check out [Error Models](error_models.md) and [Available Inputs & UQ Methods](available_inputs_uq_methods.md) for more details on error models.
:::

## Evaluation Benchmarks

Lastly, each UQ method can be evaluated by three different evaluation benchmarks: ranking-based, calibration-based, and proper scoring rules evaluations.

Each of these encompasses multiple evaluation metrics, which are established scores, concepts, and functions that are tasked with assessing the quality of the UQ methods with respect to the original data and model.

`UNIQUE` then generates a comprehensive report of all the UQ methods (base and transformed) across the different evaluation benchmarks, highlighting the [best-performing UQ method](best_uq.md) for each one of them (according to a selected scoring function).

:::{seealso}
Check out [Evaluation Benchmarks](evaluation_benchmarks.md) for more details about evaluation benchmarks and corresponding metrics.
:::
