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

from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from unique.error_models import UniqueErrorModel


class UniqueRandomForestRegressor(UniqueErrorModel):
    """RandomForestRegressor error model."""

    def _initialize(self, **kwargs) -> RandomForestRegressor:
        """Initialize a RandomForestRegressor model.

        Acceptable kwargs: all kwargs accepted by ``sklearn.ensemble.RandomForestRegressor``.
        """
        kwargs.update({"n_jobs": -1})
        return RandomForestRegressor(**kwargs)

    def _fit(
        self, model: RandomForestRegressor, x_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestRegressor:
        """Fit a RandomForestRegressor model."""
        return model.fit(x_train, y_train)

    def _predict(
        self,
        model: RandomForestRegressor,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform predictions with the trained RandomForestRegressor model."""
        return model.predict(x), y
