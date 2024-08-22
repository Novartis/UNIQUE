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
from typing import Optional, Sequence

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from unique.uncertainty import DataBasedUncertaintyMetric

SUPPORTED_EVALUATION_TYPES = ("RankingBasedEvaluation",)


@dataclass
class KernelDensityEstimation(DataBasedUncertaintyMetric):
    """Computes the Kernel Density Estimation (KDE) for the input data.

    Args:
        kernel (str, None):
            Kernel name for KDE.
        distance (str, Callable, None):
            Distance metric to use for KDE.
    """

    kernel: Optional[str] = None
    distance: Optional[str] = None
    supported_evaluation_types: Sequence[str] = SUPPORTED_EVALUATION_TYPES

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.kernel is not None
        ), """
            Please, specify a kernel to use for the KDE computation.
        """
        assert (
            self.distance is not None
        ), """
            Please, specify a distance metric to use for the KDE computation.
        """

        # Initialize and fit KDE during instantiation to speed up .fit()
        train_ids = np.where(self.which_set == "TRAIN")[0]
        train_data = self.input_values[train_ids]

        # Define GS
        kde = KernelDensity(kernel=self.kernel, metric=self.distance)
        bandwidth = [0.1, 0.5, 1]
        self.gs = GridSearchCV(kde, {"bandwidth": bandwidth}, cv=5)
        self.gs.fit(train_data)

        kde = self.gs.best_estimator_
        self.bandwidth = self.gs.best_params_["bandwidth"]

        self.log_density = kde.score_samples(self.input_values)

    def fit(self) -> np.ndarray:
        return self.log_density


@dataclass
class GaussianEuclideanKDE(KernelDensityEstimation):
    """Computes the Gaussian Kernel Density Estimation using the Euclidean distance."""

    kernel: str = "gaussian"
    distance: str = "euclidean"


@dataclass
class GaussianManhattanKDE(KernelDensityEstimation):
    """Computes the Gaussian Kernel Density Estimation using the Manhattan distance."""

    kernel: str = "gaussian"
    distance: str = "manhattan"


@dataclass
class ExponentialManhattanKDE(KernelDensityEstimation):
    """Computes the Exponential Kernel Density Estimation using the Manhattan distance."""

    kernel: str = "exponential"
    distance: str = "manhattan"
