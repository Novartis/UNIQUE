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
from typing import Sequence

import numpy as np

from unique.uncertainty import ModelBasedUncertaintyMetric

SUPPORTED_EVALUATION_TYPES = (
    "RankingBasedEvaluation",
    "ProperScoringRulesEvaluation",
    "CalibrationBasedEvaluation",
)


@dataclass
class EnsembleVariance(ModelBasedUncertaintyMetric):
    """Computes the variance of the predictions of an ensemble of models.

    ..note:
        The input values to this metric can be:
            1. The individual predictions from each model in the ensemble - e.g.,
            for an ensemble of 5 models, an array of 5 prediction values.
            2. The pre-computed variance of the predictions from all the models
            in the ensemble - i.e., an array with one variance value per datapoint.
    """

    is_variance: bool = True
    supported_evaluation_types: Sequence[str] = SUPPORTED_EVALUATION_TYPES

    def fit(self) -> np.ndarray:
        # Individual predictions from each model in the ensemble
        if len(self.input_values.shape) > 1:
            return np.var(self.input_values, axis=1)
        # Pre-computed variance values for each datapoint
        else:  # len(self.input_values.shape) == 1
            return self.input_values
