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
from typing import Callable, Optional, Sequence, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

from unique.uncertainty import DataBasedUncertaintyMetric
from unique.utils import tanimoto_distance

SUPPORTED_EVALUATION_TYPES = ("RankingBasedEvaluation",)


@dataclass
class DistanceToTraining(DataBasedUncertaintyMetric):
    """Computes the mean distance of each sample to the k-NearestNeighbors from the training set.

    Args:
        k (int): Number of k-nearest neighbors to consider.
        distance (str, Callable, None): Distance metric to use for the computation.
    """

    k: int = 5
    distance: Optional[Union[str, Callable]] = None
    is_distance: bool = True
    supported_evaluation_types: Sequence[str] = SUPPORTED_EVALUATION_TYPES

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.distance is not None
        ), """
            Please, specify a distance metric to compute the k-nearest
            neighbors to each sample.
        """
        # logging.error

        # Initialize and fit kNN at instantiation to save time during .fit()
        train_ids = np.where(self.which_set == "TRAIN")[0]
        train_data = self.input_values[train_ids]

        # Fit kNN
        self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.distance)
        self.knn.fit(train_data)
        # logging.info/debug?

        self.distances, self.neighbors = self.knn.kneighbors(self.input_values)

    def fit(self) -> np.ndarray:
        return np.mean(self.distances, axis=1)


@dataclass
class ManhattanDistance(DistanceToTraining):
    """Computes the Manhattan distance for a set of features."""

    distance: Union[str, Callable] = "manhattan"


@dataclass
class EuclideanDistance(DistanceToTraining):
    """Computes the Euclidean distance for a set of features."""

    distance: Union[str, Callable] = "euclidean"


@dataclass
class TanimotoDistance(DistanceToTraining):
    """Computes the Tanimoto distance for a set of features."""

    distance: Union[str, Callable] = tanimoto_distance
