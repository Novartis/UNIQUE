---
myst:
   substitutions:
      error_model_schema: |
         :::{figure} ../_static/error_model_schema.png
         :target: _images/error_model_schema.png
         :alt: UNIQUE Error Model Schema
         :align: center
         :class: dark-light

         Detailed workflow schema for `UNIQUE`'s error models.
         :::
---

# Error Models

`UNIQUE` also supports the generation of ML models to predict the error of the original model.

These so-called error models have been proposed in previous works[^lahlou2021] to predict both the absolute and squared differences between predicted and observed values.

`UNIQUE`'s error models are predictive models that are trained to predict up to three different types of prediction errors:
* `"L1"` error (defined as {math}`|y - \hat{y}|`);
* `"L2"` error (defined as {math}`(y - \hat{y})^2`);
* `"unsigned"` error (defined as {math}`y - \hat{y}`);

where {math}`y = label` and {math}`\hat{y} = prediction`.

{{error_model_schema}}

Given the original input features, along with computed UQ values, `UNIQUE` generates three different sets of input features to train the error predictors:
1. Original model’s prediction, UQ methods, and original input features provided by the user;
2. Original model’s prediction and UQ methods only;
3. Original model’s prediction and non-_transformed_ UQ methods only.

The samples are taken from the `"TRAIN"` and `"CALIBRATION"` subsets only; the `"TEST"` subset is left aside to ensure proper validation.

`UNIQUE` currently supports two predictive algorithms: Least Absolute Shrinkage and Selection Operator (LASSO)[^tibshirani1996], and Random Forest (RF)[^breiman2001]. Both rely on their respective [`scikit-learn`](https://scikit-learn.org/stable/index.html) implementations for the hyperparameters definition (i.e., it is possible to specify any supported `kwargs` in the [configuration file](../getting_started/prepare_pipeline.md#configuration-template)).

Error models are a special case of UQ methods themselves, and their predicted errors can be regarded as UQ values and evaluated as such in `UNIQUE`'s benchmark. In `UNIQUE` they are considered as _transformed_  UQ methods, as they are generated from a further transformation/processing of other UQ methods.

:::{seealso}
Check out [Available Inputs & UQ Methods](available_inputs_uq_methods.md) for more details about error models.
:::

## References

[^lahlou2021]: Lahlou, S., _et al._ (2021). DEUP: Direct Epistemic Uncertainty Prediction. _arXiv:2102.08501_. https://arxiv.org/abs/2102.08501v4
[^tibshirani1996]: Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. _Journal of the Royal Statistical Society. Series B (Methodological)_, 58(1), 267–288. http://www.jstor.org/stable/2346178
[^breiman2001]: Breiman, L. (2001). Random Forests. _Machine Learning_, 45, 5–32. https://doi.org/10.1023/A:1010933404324
