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

import numpy as np

from unique.uq_metric_factory import UQMetricFactory


@dataclass
class DiffkNN(UQMetricFactory):
    """Computes DiffkNN score from the kNN neighbors of the test samples.

    The method has been adapted from the Diff5NN implementation described in:
    https://pubs.acs.org/doi/10.1021/acs.jcim.2c00699.

    The DiffkNN score is defined as the absolute difference between the target
    value for the test sample and the mean of the target values of the closest
    k-Nearest Neighbors from the training set.

    The DiffkNN score in ``unique`` is obtained by taking the kNNs computed in
    distance-based UQ metrics and
    """

    neighbors: np.ndarray

    def fit(self) -> np.ndarray:
        return np.abs(self.uq_values - self.uq_values[self.neighbors].mean(axis=1))
