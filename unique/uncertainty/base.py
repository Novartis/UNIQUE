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

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np


@dataclass
class UniqueUncertaintyMetric:
    """Base UNIQUE UQ metric class.

    Args:
        input_values (np.ndarray):
            The input values from which to compute the corresponding UQ metric.
        which_set (np.ndarray):
            An array containing information on the subset each datapoint belongs to.
            Values must be one of ["TRAIN", "CALIBRATION", "TEST"].
        problem_type ("classification", "regression"):
            The type of problem the original model solves.
        supported_evaluation_types (Sequence[str]):
            A list of supported UQ evaluation proxies.
        name (str, None):
            The name of the UQ metric. If None, defaults to the class name. Default: None.
        is_variance (bool):
            Whether the UQ metric is variance-based. Default: False.
        is_distance (bool):
            Whether the UQ metric is distance-based. Default: False.
    """

    input_values: np.ndarray
    which_set: np.ndarray
    problem_type: Literal["classification", "regression"]
    supported_evaluation_types: Sequence[str]
    name: Optional[str] = None
    is_variance: bool = False
    is_distance: bool = False
    is_error_model: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    def fit(self):
        """Compute metric using input data."""
        raise NotImplementedError

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


@dataclass
class DataBasedUncertaintyMetric(UniqueUncertaintyMetric):
    """UQ metric associated with the features of the input data."""


@dataclass
class ModelBasedUncertaintyMetric(UniqueUncertaintyMetric):
    """UQ metric associated with the output(s) from a generic (ensemble of) machine learning model(s)."""


@dataclass
class ErrorBasedUncertaintyMetric(UniqueUncertaintyMetric):
    """UQ metric associated with the output from an error model.

    This class 'wraps' an error model as an uncertainty metric itself with its
    own evaluation measures. In this case, the input values to this metric are
    the error predictions from the trained error model and are considered
    directly as the computed metric - i.e., the ``fit()`` method will return the predictions
    themselves.
    """

    supported_evaluation_types: Sequence[str] = (
        "RankingBasedEvaluation",
        "ProperScoringRulesEvaluation",
        "CalibrationBasedEvaluation",
    )
    is_error_model = True

    def fit(self):
        return self.input_values


@dataclass
class UQFactoryBasedUncertaintyMetric(UniqueUncertaintyMetric):
    """UQ metric associated with a combination of UQ metrics.

    This class 'wraps' a combination of UQ metrics as an uncertainty metric with
    its own evaluation measures. In this case, the input values to this metric are
    the output values of the corresponding transformed UQ metrics and are considered
    directly as the computed metric - i.e., the ``fit()`` method will return the
    output values of the combined UQ metrics.
    """

    supported_evaluation_types: Sequence[str] = ("RankingBasedEvaluation",)

    def fit(self):
        return self.input_values
