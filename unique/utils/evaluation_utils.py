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

from typing import List, Tuple

import numpy as np
from scipy import stats


def calculate_proportions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_variance: np.ndarray,
    strategy: str = "uniform",
    nbins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate observed and expected proportions.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        y_pred (np.ndarray):
            Original model's predictions.
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        strategy (str):
            Strategy to use to compute the proportions. Either "uniform" or "quantile".
            Default: "uniform".
        nbins (int):
            Number of bins in the probability space. Default: 100.

    Returns:
        Two arrays with observed and expected proportions.
    """

    y_std = np.sqrt(y_variance)

    strategies = ["uniform", "quantile"]
    assert (
        strategy in strategies
    ), f"""
        Unsupported strategy. Supported strategies: {strategies}.
        Got: strategy={strategy}.
    """

    exp_proportions = np.linspace(0, 1, nbins)

    if strategy == "uniform":
        obs_proportions = np.array(
            [
                get_proportion_in_interval(y_pred, y_std, y_true, quantile)
                for quantile in exp_proportions
            ]
        )
    elif strategy == "quantile":
        obs_proportions = np.array(
            [
                get_proportion_under_quantile(y_pred, y_std, y_true, quantile)
                for quantile in exp_proportions
            ]
        )

    return obs_proportions, exp_proportions


def filter_subset(input_list: List[np.ndarray], n_subset: int) -> List[np.ndarray]:
    """Return only n_subset random indices from all lists given in input_list.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        input_list (List[np.ndarray]):
            List containing arrays of values.
        n_subset (int):
            Number of points to plot after filtering.

    Returns:
        List of all input arrays with sizes reduced to n_subset.
    """
    n_total = len(input_list[0])

    assert isinstance(
        n_subset, int
    ), f"Please, provide a valid integer of points to filter. Got: {n_subset} ({type(n_subset)})."
    assert (
        n_subset <= n_total
    ), f"Number of datapoints in the subset must be less than or equal to the total number of datapoints. Got: {n_subset}."

    # Randomly select ``n_subset`` indices
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)

    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)

    return output_list


def get_proportion_in_interval(
    y_pred: np.ndarray, y_true: np.ndarray, y_variance: np.ndarray, quantile: float
) -> float:
    """Return proportion of points falling into an interval corresponding to that quantile.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        y_pred (np.ndarray):
            Original model's predictions.
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        quantile (float):
            A specified quantile level.

    Returns:
        A single scalar which is the proportion of the true labels falling into the
        prediction interval for the specified quantile.
    """
    y_std = np.sqrt(y_variance)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true

    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)

    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid and resid <= upper_bound:
            num_within_quantile += 1.0
    proportion = num_within_quantile / len(residuals)

    return proportion


def get_proportion_under_quantile(
    y_pred: np.ndarray,
    y_variance: np.ndarray,
    y_true: np.ndarray,
    quantile: float,
) -> float:
    """Return proportion of data below the predicted quantile.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        y_pred (np.ndarray):
            Original model's predictions.
        y_true (np.ndarray):
            Target values.
        y_variance (np.ndarray):
            Computed UQ values (considered as the variance of the distribution).
        quantile (float):
            A specified quantile level.

    Returns:
        The proportion of data below the quantile level.
    """
    y_std = np.sqrt(y_variance)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    quantile_bound = norm.ppf(quantile)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true

    normalized_residuals = residuals / y_std

    num_below_quantile = 0
    for resid in normalized_residuals:
        if resid <= quantile_bound:
            num_below_quantile += 1.0
    proportion = num_below_quantile / len(residuals)

    return proportion
