# Best UQ Method Selection

`UNIQUE` also provides a way to easily identify the best-performing UQ method - i.e., the one whose values most accurately reflect the true errors from the original ML model, depending on the evaluation benchmark being used.

In particular, for `UNIQUE` the best UQ method is the one that best optimizes the following evaluation metrics in each benchmark:

|Benchmark|Metric|Objective|
|:--------|:-----|:--------|
|Ranking-based Evaluation|Spearman Rank Correlation Coefficient (SRCC)|The **higher**, the better.|
|Proper Scoring Rules Evaluation|Negative Log-Likelihood (NLL)|The **lower**, the better.|
|Calibration-based Evaluation|Mean Absolute Calibration Error (MACE)|The **lower**, the better.|

:::{seealso}
Check out [Evaluation Benchmarks](./evaluation_benchmarks.md) for more details about each evaluation score.
:::

## Bootstrapping

Furthermore, if enabled (`True` by default), `UNIQUE` performs bootstrapping to ensure the selection is as robust as possible. That is, for each UQ method and each evaluation metrics benchmark, the evaluation score to be optimized is computed on 500 (default value) bootstrap samples of the original dataset and corresponding UQ values.

This leads to a distribution of evaluation scores for each UQ method, which is then compared pairwise with the corresponding score distributions from each and every other UQ meethod via a Wilcoxon ranked sum test, to verify whether there are statistically significant differences.

:::{note}
The best UQ method is then the one that achieves the highest number of occurrences in which its evaluation scores distribution is significantly higher/lower (depending on the evaluation method) than the score distribution of another UQ method.
:::

The best UQ method is highlighted in green in the summary tables output by `UNIQUE` (if running in a JupyterNotebook cell) and additional individual evaluation plots are generated for the best UQ method(s).

:::{seealso}
Check out [Available Configuration Arguments](../getting_started/prepare_pipeline.md#available-configuration-arguments) for all the possible `Pipeline` configuration options.
:::
