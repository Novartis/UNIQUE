# Prepare Your Dataset

## Input Data Preparation

In order to use `UNIQUE` the user only needs to input a dataframe containing at the minimum the following columns:

|Column|Description|
|-----:|:----------|
|**IDs**|Column containing the unique IDs of the datapoints (can be the dataframe's `index` if there are no other identifiers).|
|**Labels**|Column containing the target labels/values associated with each datapoint.|
|**Predictions**|Column containing the trained ML model's predictions (intended as the final model's, or ensemble of models', single-value output, to be compared with the corresponding label).|
|**Subset**|Column containing the specification of which subset each datapoint belongs to. The allowed subsets are: `"TRAIN"`, `"TEST"`, and `"CALIBRATION"`.|

:::{caution}
Make sure to use exactly `TRAIN`, `TEST`, and `CALIBRATION` (all upper-case), as these values are hard-coded.
:::

## Input Types Specification

Depending on the UQ methods one wants to evaluate/use, one can add the following inputs:

|Data Features|Model Outputs|
|:------------|:------------|
|Column(s) containing the feature(s) of each datapoint - e.g., the ones used for training the original ML model.<br><br>These will be used, for example, to compute the distance between each datapoint in the feature's space (therefore, if you wish to aggregate different features together, you need to provide them as a single column comprising of arrays of length _n_, with _n_ being the number of features, one for each datapoint).|Column(s) containing output(s) related to the original predictive model.<br><br>For example, it can be a column containing the individual predictions of an ensemble of models (stored as an array of values for each datapoint), before merging them in a single final predicted value, or it can be directly the variance value of the ensemble associated with each datapoint's predictions.|

In `UNIQUE`, data-based and model-based features are represented by the {py:class}`~unique.input_type.base.FeaturesInputType` and {py:class}`~unique.input_type.base.ModelInputType` classes, respectively.

:::{seealso}
Check out [Available Configuration Arguments](prepare_pipeline.md#available-configuration-arguments) for more details about how to specify your inputs to `UNIQUE`.
:::

## Schematic Example

For example, an input dataset to `UNIQUE` could look like this:

|   |ID|Labels|Predictions|Subset|Data Feature #1|Data Feature #2|Ensemble Predictions|Ensemble Variance|
|--:|-:|:----:|:----------|:-----|:-------------:|:-------------:|:------------------:|:---------------:|
|   |1|0.12|0.17|TRAIN|45|[65,12,0.12,True,...]|[0.10,0.12,0.07,0.25,...]|0.02|
|   |2|0.43|0.87|TEST|36|[90,124,15.63,True,...]|[0.43,1.52,0.23,0.45,...]|0.13|
|   |3|4.78|5.62|CALIBRATION|8|[0.9,83,-0.4,False,...]|[1.87,7.92,4.32,5.08,...]|0.81|
|   |...|...|...|...|...|[...]|[...]|...|
|`dtype`|`Any`|`int` or `float`|`int` or `float`|`str`|`int` or `float`|`Iterable` or `np.ndarray`|`Iterable` or `np.ndarray`|`float`|

:::{tip}
When storing long arrays/lists in a single `pd.DataFrame` column, saving and reloading the dataframe as a `csv` file will cause issues, due to the fact that each array will be saved as a string when saving in the `csv` format and will be truncated with ellipses if exceeding a certain limit (typically > 1000 elements per array), thus making it impossible to correctly parse the entire original array when loading the `csv` file again.

To overcome this, consider dumping the input dataframe as a `json` or `pickle` file - e.g., with [`pd.DataFrame.to_json`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html) or [`pd.DataFrame.to_pickle`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html), which will not cause any of the aforementioned issues.

`UNIQUE` supports input dataframe in `csv`, `json` and `pickle` formats.
:::

:::{caution}
Only unpickle files you trust.

Check out the [`pickle` module docs](https://docs.python.org/3/library/pickle.html#module-pickle) for more information, and consider safer serialization formats such as `json`.
:::

:::{seealso}
Check [Examples](../examples/index.md) for some practical, hands-on tutorials on data preparation for `UNIQUE`.
:::
