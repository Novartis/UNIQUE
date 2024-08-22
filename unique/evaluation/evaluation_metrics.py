"""
Copyright (c) 2024. Novartis Biomedical Research
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

####################################
# Ranking-based Evaluation Metrics #
####################################


def calculate_regression_performance(
    preds: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Calculate predictive performances for regression models."""
    performance = {}
    # MAE/MSE
    performance["MAE"] = mean_absolute_error(labels, preds)
    performance["MSE"] = mean_squared_error(labels, preds)
    # Correlation coefficients
    performance["R2"] = r2_score(labels, preds)
    performance["Pearson"] = stats.pearsonr(labels, preds)[0]
    performance["Spearman"] = stats.spearmanr(labels, preds)[0]
    # Fold errors
    difference = np.abs(preds - labels)
    performance["MAE_sd"] = np.std(difference)
    performance["GMFE"] = 10 ** np.mean(difference)
    performance["%2-Fold"] = (
        len(np.where(difference <= np.log10(2))[0]) / len(difference) * 100
    )
    performance["%3-Fold"] = (
        len(np.where(difference <= np.log10(3))[0]) / len(difference) * 100
    )
    performance["%5-Fold"] = (
        len(np.where(difference <= np.log10(5))[0]) / len(difference) * 100
    )

    performance["Range"] = np.max(labels) - np.min(labels)
    performance["#"] = len(labels)

    return performance


def calculate_classification_performance(
    preds: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Calculate predictive performance for binary classification models."""
    performance = {}
    # Matthew's coefficient
    performance["MCC"] = matthews_corrcoef(labels, preds)
    # Balanced accuracy
    performance["BA"] = balanced_accuracy_score(labels, preds)
    # Cohen K
    performance["Kappa"] = cohen_kappa_score(labels, preds)
    # F1 score
    performance["F1"] = f1_score(labels, preds)

    performance["#"] = len(labels)

    return performance


def get_indices_bin(
    input_array: np.ndarray,
    nbins: int,
    order: str,
) -> List[np.ndarray]:
    """Return corresponding bins according to ordering criteria.

    Args:
        input_array (np.ndarray):
            Input array with values to be binned.
        nbins (int):
            Number of bins into which to split the input array.
        order (str):
            Ordering and binning criteria. Supported values: "increasing", "decreasing", "within".

    Returns:
        A list of arrays containing each bin's indices.
    """
    ORDERS = ("increasing", "decreasing", "within")
    assert (
        order in ORDERS
    ), f"""
        Unsupported ordering criteria. Supported values: {ORDERS}.
        Got: order={order}.
    """

    # Get the binning indices
    thresholds = np.linspace(0, 1, nbins + 1, endpoint=True)[1:]
    bin_ids = (thresholds * len(input_array)).astype(np.uint64)

    # Get sorted indices
    sorted_ids = np.argsort(input_array)

    # Get indices for each bin
    if order == "increasing":
        bins_ids = [sorted_ids[:bi] for bi in bin_ids]
    elif order == "decreasing":
        bins_ids = [sorted_ids[::-1][:bi] for bi in bin_ids]
    elif order == "within":
        bin_ids = np.insert(bin_ids, 0, 0)
        bins_ids = [sorted_ids[bin_ids[i] : bin_ids[i + 1]] for i in range(nbins)]

    return bins_ids


def get_performance_per_bin(
    preds: np.ndarray,
    labels: np.ndarray,
    uq_values: np.ndarray,
    problem_type: str,
    order: str,
    nbins: int,
    average: bool = True,
    bins_labels: str = "Bins",
    uq_thresholds_column_name: str = "Thresholds",
    errors_column_name: str = "Errors",
) -> pd.DataFrame:
    """Compute relevant performance metrics for data ordered and binned according to UQ values.

    Calculate model's performance for different UQ thresholds - e.g., different
    number of % predicted datapoints.

    Uniform binning is considered (same number of datapoints per bin).

    Three data ordering and binning (``order``) modalities are supported:
        1. "increasing": datapoints are ranked from low to high UQ value and
        cumulative model performance is calculated for different data %s.
        2. "decreasing": datapoints are ranked from high to low UQ value and
        cumulative model performance is calculated for different data %s.
        3. "within": datapoints are ranked according to the computed UQ values,
        and model performance is calculated for different UQ-based bins.

    Args:
        preds (np.ndarray):
            Original model's predictions.
        labels (np.ndarray):
            Target values.
        uq_values (np.ndarray):
            Computed UQ values associated with the predictions.
        problem_type (str):
            Task the model is solving (either "classification" or "regression").
        order (str):
            How to order and bin the datapoints using the UQ distribution.
            Supported values: "increasing", "decreasing", "within".
        nbins (int):
            Number of bins into which to split the datapoints.
        average (bool):
            Whether to return the average of the performance metrics or keep
            the performance metric's distribution. Only supported for regression
            problem types. Default: True.
        bins_labels (str):
            Name of the colum into which to save the bins' labels. Default: "Bins".
        uq_thresholds_column_name (str):
            Name of the column into which to save the UQ-based bins' thresholds.
            Default: "Thresholds".
        errors_column_name (str):
            Name of the column into which to save the individual datapoint's errors.
            Needed if ``average=False``. Default: "Errors".

    Returns:
        A pd.DataFrame containing the model's performances for the each of the
        defined UQ-based bins.
    """

    if average:
        assert (
            problem_type == "regression"
        ), f"""
            Returning the average value for the performance metrics is only supported
            for regression problems (``problem_type="regression"``).
            Got problem_type={problem_type}.
        """
    # Get the indices for each bin
    bins_indices = get_indices_bin(input_array=uq_values, nbins=nbins, order=order)

    perfs_bins = []
    for bids in bins_indices:
        if problem_type == "classification":
            perfs = calculate_classification_performance(
                preds=preds[bids], labels=labels[bids]
            )
        else:  # regression
            if average:
                # Return average performance values
                perfs = calculate_regression_performance(
                    preds=preds[bids], labels=labels[bids]
                )
            else:
                # Return individual errors
                perfs = {}
                perfs[errors_column_name] = np.abs(preds[bids] - labels[bids])

        perfs[uq_thresholds_column_name] = np.round(uq_values[bids[-1]], 3)
        perfs[bins_labels] = np.round(len(bids) / len(uq_values) * 100)

        perfs_bins.append(pd.DataFrame(perfs, index=perfs.keys() if average else None))

    # Return as pd.DataFrame
    return pd.concat(perfs_bins)


def get_all_performance_per_bin(
    preds: np.ndarray,
    labels: np.ndarray,
    uq_values: np.ndarray,
    nbins: int,
    problem_type: str,
    bins_labels: str = "Bins",
    uq_thresholds_column_name: str = "Thresholds",
    errors_column_name: str = "Errors",
) -> Dict[str, pd.DataFrame]:
    """Compute the relevant performance metrics for each UQ-based bin and for different data ordering.

    Data is binned according to three criteria based on UQ-values:
        1. Ordered by ascending UQ values and then binned into `nbins`.
        2. Ordered by decreasing UQ values and then binned into `nbins`.
        3. Binned into 3 UQ-based groups: High, Medium, Low UQ values. The bins
        in this case are simply calculated by equally splitting into 3 parts
        the UQ values distribution.

    For 1. and 2., subsequent bins contain cumulative subsets of data (e.g.,
    if `nbins=10`, the first bin contains the first 10% of the UQ-ordered data,
    the second contains the first 20% of the data, including the first 10% of
    the first bin, etc.).

    Once the bins have been computed, relevant performance metrics are computed
    for each bin. Ideally, the performance metrics for each bin should reflect
    the magnitude of the UQ values - i.e., better performance metric's values
    should be associated with lower UQ values (because the model is less uncertain
    about good predictions).

    Args:
        preds (np.ndarray):
            Predictions from the original model.
        labels (np.ndarray):
            Target values.
        uq_values (np.ndarray):
            Computed UQ values associated with the predictions.
        nbins (int):
            Number of bins into which to split the data.
        problem_type (str):
            Problem the original model is trying to solve ("classification" or "regression").
        bins_labels (str):
            Name of the column into which to save the bins' labels. Default: "Bins".
        uq_thresholds_column_name (str):
            Name of the column into which to save the UQ-based thresholds used
            to bin data. Default: "Thresholds".
        errors_column_name (str):
            Name of the column into which to save the individual datapoint's errors.
            Only valid for regression performances with ``average=False`` (in
            ``calculate_regression_performance``. Default: "Errors".

    Returns:
        A dictionary containing the performance metrics computed for different
        types of binning (each is a pd.DataFrame).
    """
    # Define ordering and binning criteria
    criteria = ["increasing", "decreasing", "within"]
    bins_perfs = {}

    # Define common kwargs
    kwargs = {
        "preds": preds,
        "labels": labels,
        "problem_type": problem_type,
        "nbins": nbins,
        "bins_labels": bins_labels,
        "uq_thresholds_column_name": uq_thresholds_column_name,
        "errors_column_name": errors_column_name,
    }

    # Compute the UQ-based bin performances
    for binning in criteria:
        bins_perfs[f"{binning}_bins"] = get_performance_per_bin(
            uq_values=uq_values, order=binning, average=True, **kwargs
        )

    # Compute the performances for bins ranked according to the true errors (and not UQ-based)
    for binning in ["increasing", "decreasing"]:
        bins_perfs[f"{binning}_bins_best"] = get_performance_per_bin(
            uq_values=np.abs(labels - preds), order=binning, average=True, **kwargs
        )
    # Compute the "raw" errors distributions for the "within"-binned bins
    bins_perfs["within_bins_errordistrib"] = get_performance_per_bin(
        uq_values=uq_values, order="within", average=False, **kwargs
    )

    return bins_perfs


def auc_difference_bestrank(
    increasing_bins: pd.DataFrame,
    increasing_bins_best: pd.DataFrame,
    bins_labels: str,
    perf_metric: str,
) -> float:
    """Compute the difference in AUC between UQ-based performances vs. True Error performances."""
    # Compute AUC for UQ-based ranking
    auc_uq_metric = auc(
        x=increasing_bins[bins_labels] / 100, y=increasing_bins[perf_metric]
    )
    # Compute AUC for best possible ranking
    auc_bestrank = auc(
        x=increasing_bins_best[bins_labels] / 100, y=increasing_bins_best[perf_metric]
    )

    return auc_uq_metric - auc_bestrank


def performance_drop_rank(
    decreasing_bins: pd.DataFrame,
    increasing_bins: pd.DataFrame,
    perf_metric: str,
) -> Tuple[float, float]:
    """Compute performance metric's drop (as ratio) between different bins of data."""
    # 1. Performance drop between computing the metric with all the data (last bin) vs.
    # bin associated with lowest UQ
    all_vs_low = (
        increasing_bins.iloc[-1][perf_metric] / increasing_bins.iloc[0][perf_metric]
    )

    # 2. Performance drop between computing the metric with the bin associated with the
    # highest UQ vs. bin associated with lowest UQ
    high_vs_low = (
        decreasing_bins.iloc[0][perf_metric] / increasing_bins.iloc[0][perf_metric]
    )

    return all_vs_low, high_vs_low


def decreasing_coefficient(decreasing_bins: pd.DataFrame, perf_metric: str) -> float:
    """Return number of consecutive bins showing decreasing performance metric's values."""
    # Extract performance metric's values
    perf_values = decreasing_bins[perf_metric].to_numpy()
    # Count how many bins have lower values than the preceding one
    count = np.where(perf_values[1:] < perf_values[:-1])[0].sum()
    # Return decreasing "coefficient"
    return count / (len(perf_values) - 1)


def increasing_coefficient(increasing_bins: pd.DataFrame, perf_metric: str) -> float:
    """Return number of consecutive bins showing increasing performance metric's values."""
    # Extract performance metric's values
    perf_values = increasing_bins[perf_metric].to_numpy()
    # Count how many bins have higher values than the preceding one
    count = np.where(perf_values[1:] < perf_values[:-1])[0].sum()
    # Return increasing "coefficient"
    return count / (len(perf_values) - 1)


def spearman_correlation(
    preds: np.ndarray, labels: np.ndarray, uq_values: np.ndarray
) -> float:
    """Spearman correlation coefficient between errors and UQ values."""
    return stats.spearmanr(np.abs(preds - labels), uq_values)[0]


###########################################
# Proper Scoring Rules Evaluation Metrics #
###########################################

# For regression


def sharpness_regression(y_variance: np.ndarray) -> float:
    """Return sharpness (a single measure of the overall confidence).

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).

    Returns:
        The computed sharpness value.
    """
    # STD
    y_std = np.sqrt(y_variance)

    # Compute sharpness
    sharp_metric = np.sqrt(np.mean(y_std**2))

    return sharp_metric


def nll_gaussian(
    y_pred: np.ndarray, y_true: np.ndarray, y_variance: np.ndarray, scaled: bool = True
) -> float:
    """Negative log-likelihood for a Gaussian distribution.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        y_pred (np.ndarray):
            Original model's prediction (considered as the mean of the distribution).
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        scaled (bool):
            Whether to scale the negative log likelihood by the size of the dataset.
            Default: True.

    Returns:
        The negative log-likelihood value of the Gaussian distribution thus defined.
    """
    # Std
    y_std = np.sqrt(y_variance)

    # Set residuals
    residuals = y_pred - y_true

    # Compute nll
    nll_list = stats.norm.logpdf(residuals, scale=y_std)
    nll = -1 * np.sum(nll_list)

    # Potentially scale so that sum becomes mean
    if scaled:
        nll = nll / len(nll_list)

    return nll


def crps_gaussian(
    y_pred: np.ndarray, y_true: np.ndarray, y_variance: np.ndarray, scaled: bool = True
) -> float:
    """Compute the negatively oriented Continous Ranked Probability Score (CRPS) for Gaussian distributions.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Each test point is given equal weight in the overall score over the test set.

    "Negatively oriented" means a smaller value is more desirable.

    Args:
        y_pred (np.ndarray):
            Original model's prediction (considered as the mean of the distribution).
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        scaled (bool):
            Whether to scale CRPS by the size of the dataset. Default: True.
    Returns:
        The CRP score as a float value.
    """
    # Std
    y_std = np.sqrt(y_variance)

    # Compute CRPS
    y_standardized = (y_true - y_pred) / y_std
    term_1 = 1 / np.sqrt(np.pi)
    term_2 = 2 * stats.norm.pdf(y_standardized, loc=0, scale=1)
    term_3 = y_standardized * (2 * stats.norm.cdf(y_standardized, loc=0, scale=1) - 1)

    crps_list = -1 * y_std * (term_1 - term_2 - term_3)
    crps = np.sum(crps_list)

    # Potentially scale so that sum becomes mean
    if scaled:
        crps = crps / len(crps_list)

    return crps


def check_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_variance: np.ndarray,
    scaled: bool = True,
    start_q: float = 0.01,
    end_q: float = 0.99,
    resolution: int = 99,
) -> float:
    """Compute the negatively oriented check score.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    The score is computed by scanning over a sequence of quantiles of the predicted
    distributions, starting at (start_q) and ending at (end_q).

    Each test point and each quantile is given equal weight in the overall score
    over the test set and list of quantiles.

    "Negatively oriented" means a smaller value is more desirable.

    Args:
        y_pred (np.ndarray):
            Original model's prediction (considered as the mean of the distribution).
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        scaled (bool):
            Whether to scale CRPS by the size of the dataset. Default: True.
        start_q (float):
            The lower bound of the quantiles to use for computing the score. Default: 0.01.
        end_q (float):
            The upper bound of the quantiles to use for computing the score. Default: 0.99.
        resolution (int):
            The number of quantiles to use for computing the score. Default: 99.

    Returns:
        The computed check score as a float value.
    """

    y_std = np.sqrt(y_variance)

    test_qs = np.linspace(start_q, end_q, resolution)

    check_list = []
    for q in test_qs:
        q_level = stats.norm.ppf(q, loc=y_pred, scale=y_std)  # pred quantile
        diff = q_level - y_true
        mask = (diff >= 0).astype(float) - q
        score_per_q = np.mean(mask * diff)
        check_list.append(score_per_q)
    check_score = np.sum(check_list)

    if scaled:
        check_score = check_score / len(check_list)

    return check_score


def interval_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_variance: np.ndarray,
    scaled: bool = True,
    start_p: float = 0.01,
    end_p: float = 0.99,
    resolution: int = 99,
) -> float:
    """Compute the negatively oriented interval score.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    This metric is computed by scanning over a sequence of prediction intervals,
    where p is the amount of probability captured from a centered prediction
    interval, and intervals are formed starting at p=(start_p) and ending at p=(end_p).

    Each test point and each percentile is given equal weight in the
    overall score over the test set and list of quantiles.

    "Negatively oriented" means a smaller value is more desirable.

    Args:
        y_pred (np.ndarray):
            Original model's prediction (considered as the mean of the distribution).
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        scaled (bool):
            Whether to scale the interval score by the size of the dataset. Default: True.
        start_p (float):
            The lower bound of the quantiles to use for computing the score. Default: 0.01.
        end_p (float):
            The upper bound of the quantiles to use for computing the score. Default: 0.99.
        resolution (int):
            The number of quantiles to use for computing the score. Default: 99.

    Returns:
        The computed interval score as a float value.
    """

    y_std = np.sqrt(y_variance)

    test_ps = np.linspace(start_p, end_p, resolution)

    int_list = []
    for p in test_ps:
        low_p, high_p = 0.5 - (p / 2.0), 0.5 + (p / 2.0)  # p% PI
        pred_l = stats.norm.ppf(low_p, loc=y_pred, scale=y_std)
        pred_u = stats.norm.ppf(high_p, loc=y_pred, scale=y_std)

        below_l = ((pred_l - y_true) > 0).astype(float)
        above_u = ((y_true - pred_u) > 0).astype(float)

        score_per_p = (
            (pred_u - pred_l)
            + (2.0 / (1 - p)) * (pred_l - y_true) * below_l
            + (2.0 / (1 - p)) * (y_true - pred_u) * above_u
        )
        mean_score_per_p = np.mean(score_per_p)
        int_list.append(mean_score_per_p)
    int_score = np.sum(int_list)

    if scaled:
        int_score = int_score / len(int_list)

    return int_score


# For classification


def brier_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Brier score loss.

    Args:
        y_pred (np.ndarray):
            Predicted probabilities.
        y_true (np.ndarray):
            Target values.

    Returns:
        The Brier score loss as a float value.
    """
    return brier_score_loss(y_true, y_pred)


def log_loss_prob(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the logarithmic loss between predicted probabilities and target values.

    Args:
        y_pred (np.ndarray):
            Predicted probabilities.
        y_true (np.ndarray):
            Target values.

    Returns:
        The Log loss as a float value.
    """
    return log_loss(y_true, y_pred)


def crps_probability(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute the Continuous Ranked Probability Score (CRPS).

    Adapted from:
    Source: https://github.com/kashif/uncertainty-metrics/
    Author(s): Uncertainty Metrics' authors

    The Continuous Ranked Probability Score is a [proper scoring rule][1] for
    assessing the probabilistic predictions of a model against a realized value.
    The CRPS is

    \\(\textrm{CRPS}(F,y) = \\int_{-\\inf}^{\\inf} (F(z) - 1_{z \\geq y})^2 dz.\\)

    Here \\(F\\) is the cumulative distribution function of the model predictive
    distribution and \\(y)\\ is the realized ground truth value.

    The CRPS can be used as a loss function for training an implicit model for
    probabilistic regression. It can also be used to assess the predictive
    performance of a probabilistic regression model.

    In this implementation we use an equivalent representation of the CRPS,

    \\(\textrm{CRPS}(F,y) = E_{z~F}[|z-y|] - (1/2) E_{z,z'~F}[|z-z'|].\\)

    This equivalent representation has an unbiased sample estimate and our
    implementation of the CRPS has a complexity is O(n m).

    #### References
    [1]: Tilmann Gneiting, Adrian E. Raftery.
         Strictly Proper Scoring Rules, Prediction, and Estimation.
         Journal of the American Statistical Association, Vol. 102, 2007.
         https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    Args:
        y_pred (np.ndarray):
            Predicted probabilities.
        y_true (np.ndarray):
            Target values.

    Returns:
        The CRPS score as a float value.
    """

    num_samples = y_pred.shape[0]
    mean_abs_errors = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(mean_abs_errors)

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs_crps = mean_abs_errors - np.sum(diff * weight, axis=0) / num_samples**2

    return np.average(per_obs_crps, weights=weight)


########################################
# Calibration-based Evaluation Metrics #
########################################


def mean_absolute_calibration_error(
    obs_proportions: np.ndarray, exp_proportions: np.ndarray
) -> float:
    """Compute the Mean absolute Calibration Error (MACE).

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Also known as Expected Calibration Error (ECE).

    Args:
        obs_proportions (np.ndarray):
            Observed proportions.
        exp_proportions (np.ndarray)
            Expected proportions.

    Returns:
        The MACE as a float value.
    """

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)

    return np.mean(abs_diff_proportions)


def root_mean_squared_calibration_error(
    obs_proportions: np.ndarray, exp_proportions: np.ndarray
) -> float:
    """Compute the Root Mean Squared Calibration Error (RMSCE).

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        obs_proportions (np.ndarray):
            Observed proportions.
        exp_proportions (np.ndarray)
            Expected proportions.

    Returns:
        The RMSCE as a float value.
    """

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)

    return np.sqrt(np.mean(squared_diff_proportions))
