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
from typing import Tuple

import numpy as np

from unique.uq_metric_factory import UQMetricFactory
from unique.utils import convert_distances_to_variances


@dataclass
class DistanceToVariance(UQMetricFactory):
    """Converts distance-based UQ metrics to variance-based UQ metrics.

    The method to convert distances to variances uses the so-called ``Calibrated NLL``
    (Negative Log-Likelihood), as explained in Eq. 12 of the paper:
    https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.0c00502.
    """

    predictions: np.ndarray
    labels: np.ndarray
    which_set: np.ndarray
    initial_params: Tuple[float, float] = (0.1, 0.1)
    optimization_method: str = "Nelder-Mead"

    def fit(self) -> np.ndarray:
        variances_from_distances = convert_distances_to_variances(
            distances=self.uq_values,
            predictions=self.predictions,
            labels=self.labels,
            which_set=self.which_set,
            initial_params=self.initial_params,
            optimization_method=self.optimization_method,
        )

        return variances_from_distances
