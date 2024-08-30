# Available Inputs & UQ Methods

Below, you can find an overview of the input types, UQ methods, and error models implemented in `UNIQUE`.

:::{table} List of available UQ methods & objects in `UNIQUE`.
:widths: auto
:align: center

|Type|Name|Description|Reference(s)|
|---:|---:|:----------|:-----------|
|Input Type|{py:class}`~unique.input_type.base.FeaturesInputType`|Data-based inputs - i.e., features that can be directly computed from/linked to the data. Features can be provided as a single value or as an array of values/features for each datapoint. Numerical features can contain integer-only (binary included) or real-valued (floats) values. Check out [Input Types Specification](../getting_started/prepare_data.md#input-types-specification).||
|Input Type|{py:class}`~unique.input_type.base.ModelInputType`|Model-based inputs - i.e., outputs associated with the original predictive model. Depending on the task (`problem_type`, either "classification" or "regression"), two types of model-based inputs can be provided: for classification tasks, the predicted main class probabilities (as a single value per datapoint); for regression tasks, either the individual ensemble member's predictions (as an array) or the pre-computed ensemble variance (as a single value) for each datapoint. Check out [Input Types Specification](../getting_started/prepare_data.md#input-types-specification).||
|UQ Method|{py:class}`~unique.uncertainty.model_based.ensemble_variance.EnsembleVariance`|Computes the variance of the ensemble's predictions. Either the individual ensemble member's predictions (as an array) or the pre-computed variance (as a single value) for each datapoint can be provided.||
|UQ Method|{py:class}`~unique.uncertainty.model_based.probability.Probability`|Returns the predicted primary class probability. Expects the predicted main class probability value as input, not the ensemble's (class) predictions.||
|UQ Method|{py:class}`~unique.uncertainty.data_based.distance_to_training.ManhattanDistance`|Returns the k-nearest neighbors from the training set in the corresponding feature(s) space using the Manhattan distance metric.||
|UQ Method|{py:class}`~unique.uncertainty.data_based.distance_to_training.EuclideanDistance`|Returns the k-nearest neighbors from the training set in the corresponding feature(s) space using the Euclidean distance metric.||
|UQ Method|{py:class}`~unique.uncertainty.data_based.distance_to_training.TanimotoDistance`|Returns the k-nearest neighbors from the training set in the corresponding feature(s) space using the Tanimoto distance metric.||
|UQ Method|{py:class}`~unique.uncertainty.data_based.kernel_density.GaussianEuclideanKDE`|Returns the kernel density estimation from the training set in the corresponding feature(s) space using the gaussian kernel and Euclidean distance metric.||
|UQ Method|{py:class}`~unique.uncertainty.data_based.kernel_density.GaussianManhattanKDE`|Returns the kernel density estimation from the training set in the corresponding feature(s) space using the gaussian kernel and Manhattan distance metric.||
|UQ Method|{py:class}`~unique.uncertainty.data_based.kernel_density.ExponentialManhattanKDE`|Returns the kernel density estimation from the training set in the corresponding feature(s) space using the exponential kernel and Manhattan distance metric.||
|"Transformed" UQ Method|{py:class}`~unique.uq_metric_factory.combinations.sum_of_variances.SumOfVariances`|Computes the sum of (computed) variances and distances converted to variances using the {py:func}`Calibrated Negative Log-Likelihood (CNLL) <unique.utils.uncertainty_utils.cnll>` method.|Hirschfeld _et al._ (2020) - Eq. 11 & 12[^hirschfeld2020]|
|"Transformed" UQ Method|{py:class}`~unique.uq_metric_factory.combinations.diffknn.DiffkNN`|Computes the absolute mean difference in predicted vs. target value for the k-nearest neighbors from the training set in the corresponding feature(s) space.|Sheridan _et al._ (2022)[^sheridan2022]|
|Error Model/"Transformed" UQ Method|{py:class}`~unique.error_models.models.random_forest_regressor.UniqueRandomForestRegressor`|Builds and trains a Random Forest regressor that predicts the pointwise prediction error.|Adapted from Lahlou _et al._ (2021)[^lahlou2021]|
|Error Model/"Transformed" UQ Method|{py:class}`~unique.error_models.models.LASSO.UniqueLASSO`|Builds and trains a LASSO regressor that predicts the pointwise prediction error.|Adapted from Lahlou _et al._ (2021)[^lahlou2021]|
:::

:::{seealso}
Check out [UQ Methods](overview.md#uq-methods) for more details about the difference between _base_ and _transformed_ UQ methods.
:::

## References

[^hirschfeld2020]: Hirschfeld L., _et al._ (2020). Uncertainty Quantification Using Neural Networks for Molecular Property Prediction. _Journal of Chemical Information and Modeling_, 60, 3770-3780. https://doi.org/10.1021/acs.jcim.0c00502
[^sheridan2022]: Sheridan R.P., _et al._ (2022). Prediction Accuracy of Production ADMET Models as a Function of Version: Activity Cliffs Rule. _Journal of Chemical Information and Modeling_, 62(14), 3275-3280. https://doi.org/10.1021/acs.jcim.2c00699
[^lahlou2021]: Lahlou, S., _et al._ (2021). DEUP: Direct Epistemic Uncertainty Prediction. _arXiv:2102.08501_. https://arxiv.org/abs/2102.08501v4
