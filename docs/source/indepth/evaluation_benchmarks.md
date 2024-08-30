# Evaluation Benchmarks

Three different evaluation benchmarks have been identified and implemented in `UNIQUE`: ranking-based, proper scoring rules, and calibration-based evaluations.

These benchmarks differ in the particular properties of uncertainty that they evaluate and in how they rank the "goodness" of the UQ method to represent the actual uncertainty of the model - e.g., how well the UQ values correlate with the prediction error.

:::{table} Evaluation benchmarks in `UNIQUE`.
:widths: auto
:align: center

|Benchmark|Description|Reference(s)|
|--------:|:----------|:-----------|
|{py:class}`~unique.evaluation.base.RankingBasedEvaluation`|Computes evaluation metrics that rank predictions based on their actual prediction error vs. the computed UQ values. Generally speaking, the higher the (positive) correlation between prediction error and computed UQ values, the better/more confident the model that produced the predictions can be considered. Currently implemented metrics are: AUC Difference, Spearman Rank Correlation Coefficient, Increasing/Decreasing Coefficient, and Performance Drop. For more information about the methods, check out [Available Evaluation Metrics](#available-evaluation-metrics).|Inspired by Scalia _et al._ (2020)[^scalia2020], Hirschfeld _et al._ (2020)[^hirschfeld2020]|
|{py:class}`~unique.evaluation.base.ProperScoringRulesEvaluation`|Computes proper scoring rules to evaluate the quality of predictions. Proper scoring rules are functions that assign a scalar summary measure to the performance of distributional predictions, where the maximum score obtainable is reached when the predicted distribution exactly matches the target one (also known as _minimum contrast estimation_). Currently implemented metrics are: Negative Log-Likelihood, Interval Score, Check Score, Continuous Ranked Probability Score, and Brier Score. For more information about the metrics, check out [Available Evaluation Metrics](#available-evaluation-metrics).|Gneiting _et al._ (2007)[^gneiting2007]|
|{py:class}`~unique.evaluation.base.CalibrationBasedEvaluation`|Computes model's calibration - i.e., whether the model's predictions are consistent with the underlying target distribution. Currently implemented metrics are: Mean Absolute Calibration Error, and Root Mean Squared Calibration Error. For more information about the metrics, check out [Available Evaluation Metrics](#available-evaluation-metrics).|Kuleshov _et al._ (2018)[^kuleshov2018]|
:::

## Available Evaluation Metrics

Below you can find an in-depth guide to the evaluation scoring metrics implemented in `UNIQUE` for each benchmark.

:::{table} List of available evaluation metrics in `UNIQUE`.
:widths: auto
:align: center

|Evaluation Benchmark|Metric Name|Description|Reference(s)|
|-------------------:|----------:|:----------|:-----------|
|Ranking-based|{py:func}`AUC Difference <unique.evaluation.evaluation_metrics.auc_difference_bestrank>`|The AUC measures the ranking capabilities of a model. The difference between the AUC computed using predictions ranked by the original model’s performance metric (e.g., true prediction error) and by the computed UQ method measures the ranking goodness of the UQ method. Lower values are better.|Yousef _et al._ (2004)[^yousef2004]|
|Ranking-based|{py:func}`Spearman Rank Correlation Coefficient (SRCC) <unique.evaluation.evaluation_metrics.spearman_correlation>`|The SRCC indicates how well the computed UQ method is able to rank the predictions with respect to the original model’s performance metric (e.g., true prediction error). Higher values are better.|Marino _et al._ (2008)[^marino2008]|
|Ranking-based|{py:func}`Increasing <unique.evaluation.evaluation_metrics.increasing_coefficient>`/{py:func}`Decreasing Coefficient <unique.evaluation.evaluation_metrics.decreasing_coefficient>`|A coefficient that measures how “correct” the UQ-based ranking is with respect to the performance metric-based one when binning the ranked predictions (either in increasing or decreasing order) – i.e., the predictions are ranked and binned according to the computed UQ values; the coefficient is then the number of consecutive bins with decreasing performance metric values divided by the number of bins. Higher values are better.||
|Ranking-based|{py:func}`Performance Drop <unique.evaluation.evaluation_metrics.performance_drop_rank>`|The drop in performance metric’s value between either the highest and lowest UQ-binned predictions or between the original model’s performance metric on all the predictions and the lowest UQ-binned predictions – i.e., the predictions are ranked and binned according to the computed UQ method; the performance metric is computed for the bins associated with the highest and lowest UQ, and for all the predictions being considered; the score corresponds to the difference in computed performance metrics for highest and lowest UQ-based bins, and for all data and lowest UQ-based bin. Higher values are better.||
|Proper Scoring Rules|{py:func}`Negative Log-Likelihood (NLL) <unique.evaluation.evaluation_metrics.nll_gaussian>`|The NLL assesses how well the predicted probability distribution – i.e., predictions and corresponding computed UQ values, fits the observed data or the error distribution. Lower values are better.|Maddox _et al._ (2019)[^maddox2019], Lakshminarayanan _et al._ (2016)[^lakshminarayanan2016], Detlefsen _et al._ (2019)[^detlefsen2019], Pearce _et al._ (2018)[^pearce2018]|
|Proper Scoring Rules|{py:func}`Interval Score <unique.evaluation.evaluation_metrics.interval_score>`|The interval score evaluates the sharpness and calibration of a specific prediction interval, rewarding narrow and accurate prediction intervals whilst penalizing wider prediction intervals that do not cover the observation. Lower values are better.|Gneiting _et al._ (2007)[^gneiting2007]|
|Proper Scoring Rules|{py:func}`Check Score <unique.evaluation.evaluation_metrics.check_score>` (or Pinball Loss)|The check score measures the distance between the computed UQ values (and associated predictions), intended as prediction quantiles, and the true target values. Lower values are better.|Koenker _et al._ (1978)[^koenker1978], Chung _et al._ (2020)[^chung2020]|
|Proper Scoring Rules|{py:func}`Continuous Ranked Probability Score (CRPS) <unique.evaluation.evaluation_metrics.crps_gaussian>`|The CRPS quantifies the difference between the predicted probability distribution – i.e., predictions and computed UQ values, and the observed distribution. Lower values are better.|Matheson _et al._ (1976)[^matheson1976]|
|Proper Scoring Rules|{py:func}`Brier Score <unique.evaluation.evaluation_metrics.brier_score>`|The Brier Score estimates the accuracy of probabilistic predictions, computed as the mean squared difference between predicted probabilities and the actual outcomes. Lower values are better.|Brier _et al._ (1950)[^brier1950]|
|Calibration-based|{py:func}`Mean Absolute Calibration Error (MACE) <unique.evaluation.evaluation_metrics.mean_absolute_calibration_error>`|The MACE assesses the calibration of the predicted probabilities or intervals, by comparing the bin-wise absolute calibration errors between predicted and observed distributions. Lower values are better.||
|Calibration-based|{py:func}`Root Mean Squared Calibration Error (RMSCE) <unique.evaluation.evaluation_metrics.root_mean_squared_calibration_error>`|The RMSCE assesses the calibration of the predicted probabilities or intervals, by comparing the bin-wise root mean squared calibration errors between predicted and observed distributions. Lower values are better.||
:::


## References

[^scalia2020]: Scalia G., _et al._ (2020). Evaluating Scalable Uncertainty Estimation Methods for Deep Learning-Based Molecular Property Prediction. _Journal of Chemical Information and Modeling_, 60(6), 2697-2717. https://doi.org/10.1021/acs.jcim.9b00975
[^hirschfeld2020]: Hirschfeld L., _et al._ (2020). Uncertainty Quantification Using Neural Networks for Molecular Property Prediction. _Journal of Chemical Information and Modeling_, 60, 3770-3780. https://doi.org/10.1021/acs.jcim.0c00502
[^gneiting2007]: Gneiting T., _et al._ (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. _Journal of the American Statistical Association_, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
[^kuleshov2018]: Kuleshov V. _et al._ (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression. _arXiv:1807.00263_. https://doi.org/10.48550/arXiv.1807.00263
[^yousef2004]: Yousef W.A., _et al._ (2004). Comparison of non-parametric methods for assessing classifier performance in terms of ROC parameters. _33rd Applied Imagery Pattern Recognition Workshop (AIPR'04)_, 190-195. https://doi.org/10.1109/AIPR.2004.18
[^marino2008]: Marino S., _et al._ (2008). A methodology for performing global uncertainty and sensitivity analysis in systems biology. _Journal of Theoretical Biology_, 254(1), 178-196. https://doi.org/10.1016/j.jtbi.2008.04.011
[^maddox2019]: Maddox W., _et al._ (2019). A Simple Baseline for Bayesian Uncertainty in Deep Learning. _arXiv:1902.02476_. https://doi.org/10.48550/arXiv.1902.02476
[^lakshminarayanan2016]: Lakshminarayanan B., _et al._ (2016). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. _arXiv:1612.01474_. https://doi.org/10.48550/arXiv.1612.01474
[^detlefsen2019]: Detlefsen N., _et al._ (2019). Reliable training and estimation of variance networks. _arXiv:190603260_. https://doi.org/10.48550/arXiv.1906.03260
[^pearce2018]: Pearce T., _et al._ (2018). Uncertainty in Neural Networks: Approximately Bayesian Ensembling. _arXiv:1810.05546_. https://doi.org/10.48550/arXiv.1810.05546
[^koenker1978]: Koenker R., _et al._ (1978). Regression Quantiles. _Econometrica_. 46(1), 33-50. https://doi.org/10.2307/1913643
[^chung2020]: Chung Y., _et al._ (2020). Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification. _arXiv:2011.09588_. https://doi.org/10.48550/arXiv.2011.09588
[^matheson1976]: Matheson J.E., _et al._ (1976). Scoring Rules for Continuous Probability Distributions. _Management Science_, 22(10), 1051-1173. https://doi.org/10.1287/mnsc.22.10.1087
[^brier1950]: Brier G.W. (1950). Verification of Forecasts Expressed in Terms of Probability. _Monthly Weather Review_, 78(1), 1-3. https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
