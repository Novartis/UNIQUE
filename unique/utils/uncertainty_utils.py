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

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize


class AnalyticsMode:
    """Analysis mode for distance conversions.

    - "compact": only Euclidean and/or Tanimoto distances are converted and summed
    to variances.
    - "extended": all available distances are converted and summed to variances.
    - "full": same as 'extended', but additionally all combinations of distances
    are converted and summed to variances as well.
    """

    COMPACT: str = "compact"
    EXTENDED: str = "extended"
    FULL: str = "full"
    EXPLAINATION: Dict[str, str] = {
        "compact": "only Euclidean and/or Tanimoto distances are converted and summed to variances",
        "extended": "all available distances are converted and summed to variances",
        "full": "same as 'extended', but additionally all combinations of distances are converted and summed to variances as well",
    }

    def __init__(self, mode: str):
        self.mode = mode.lower()
        if self.mode not in [self.COMPACT, self.EXTENDED, self.FULL]:
            raise ValueError(
                f"""
                    Allowed analysis modalities: [{self.COMPACT}, {self.EXTENDED}, {self.FULL}].
                    Got: {self.mode}.
                """
            )

        self.explaination = self.EXPLAINATION[self.mode]

    def __str__(self):
        return f"{self.__class__.__name__}(mode={self.mode}) - {self.explaination}"


class CustomFormatter(logging.Formatter):
    """Logger's formatter."""

    def format(self, record) -> str:
        record.name = record.name.upper()
        formatted_record = super().format(record)
        parts = formatted_record.split(" - ", 1)
        if len(parts) == 2:
            static_padding = len(" | [ - ]: ")
            dynamic_padding = (
                len(record.asctime) + len(record.name) + len(record.levelname)
            )
            padding = static_padding + dynamic_padding + 4

            parts[1] = parts[1].replace("\n", "\n" + " " * padding)
            return " - ".join(parts)
        else:
            return formatted_record


def cnll(
    params: Tuple[float, float],
    distances: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Computes the Calibrated NLL.

    The formula to compute the Calibrated NLL comes from Eq. 11/12 of
    https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.0c00502, which is
    basically a minimization problem to find the optimal set of scalars
    that approximate a distance-based metric as variance.

    Note that the CNLL algorithm only uses the calibration set to
    convert distances to variances.

    Args:
        params (Tuple[float, float]):
            A tuple of 2 floats used to minimize the NLL of errors of the dataset.
            They correspond to ``a`` and ``b`` in the original paper and are the
            minimization objective in the CNLL optimization problem.
        distances (np.ndarray):
            The distance values to convert to variances.
        predictions (np.ndarray):
            The prediction values to use to compute the CNLL method.
        labels (np.ndarray):
            The ground-truth labels to use to compute the CNLL method.

    Returns:
        The CNLL value (float).
    """
    # Eq. 10: Linear relationship between distance and variance
    a, b = params
    d2v = lambda x: (a * x) + b
    conv_vars = d2v(distances)

    # Eq. 11/12: CNLL
    cnll = (
        np.log(2 * np.pi)
        + np.log(conv_vars)
        + ((predictions - labels) ** 2 / conv_vars)
    )

    return 0.5 * np.sum(cnll)  # Eq. 11/12


def convert_distances_to_variances(
    distances: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    which_set: np.ndarray,
    initial_params: Tuple[float, float] = (0.1, 0.1),
    optimization_method: str = "Nelder-Mead",
) -> Union[np.ndarray, str]:
    """Converts distances to variances using the CalibratedNLL method.

    The method is based on https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.0c00502
    to convert distances into variance-like variables, which can then be
    combined with the other variances.

    Args:
        distances (np.ndarray):
            The distance values to convert to variances.
        predictions (np.ndarray):
            The prediction values to use to compute the CNLL method.
        labels (np.ndarray):
            The ground-truth labels to use to compute the CNLL method.
        which_set (np.ndarray):
            The indication to which subset of data each datapoint belongs to.
            Note that there must be a "CALIBRATION" subset specified.
        initial_params (Tuple[float, float]):
            A tuple of 2 floats used to minimize the NLL of errors of the dataset.
            They correspond to ``a`` and ``b`` in the original paper and are the
            minimization objective in the CNLL optimization problem. Default: (0.1, 0.1).
        optimization_method (str):
            The optimization method to use as supported by ``scipy.optimize.minimize``.
            Default: "Nelder-Mead".

    Returns:
        A np.ndarray object containing the corresponding distances converted to variances.
    """
    # Check that calibration set exists
    assert "CALIBRATION" in np.unique(
        which_set
    ), f""""
        Please, provide a calibration set on which to compute the Calibrated NLL.
        You can specify it in the ``which_set`` parameter.
        Got: {np.unique(which_set)}.
    """
    # Extract calibration set only
    data = pd.DataFrame([distances, predictions, labels, which_set]).T
    data.columns = ["distances", "predictions", "labels", "which_set"]

    calibration_set = data.loc[data["which_set"] == "CALIBRATION"]

    # Fit the CNLL
    result = optimize.minimize(
        fun=cnll,
        x0=initial_params,
        args=(
            calibration_set["distances"].to_numpy().astype(np.float64),
            calibration_set["predictions"].to_numpy().astype(np.float64),
            calibration_set["labels"].to_numpy().astype(np.float64),
        ),
        method=optimization_method,
    )

    if not result.success:
        return (
            f"There was an issue in the NLL minimization process while converting "
            f"the distances to variances: {result.message}."
        )
    # assert result.success, f"""
    #     There was an issue in the NLL minimization process while converting
    #     the distances to variances: {result.message}.
    # """
    # If optimization reached a minimum, return the converted distances
    optimized_params = result.x
    # Eq. 10: Linear relationship between distance and variance
    return (optimized_params[0] * distances) + optimized_params[1]


def is_running_in_jupyter_notebook() -> bool:
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except Exception:
        return False
    return True


def tanimoto_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Computes the Tanimoto similarity coefficient."""
    sum_num = np.abs(np.array(x) - np.array(y)).sum()
    sum_den = np.maximum(np.array(x), np.array(y)).sum()

    return sum_num / sum_den
