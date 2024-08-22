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

import itertools
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Optional, Union

import numpy as np

from unique.uncertainty.base import UQFactoryBasedUncertaintyMetric
from unique.uq_metric_factory import UQMetricFactory
from unique.utils import AnalyticsMode


@dataclass
class SumOfVariances(UQMetricFactory):
    """Sum up multiple variance-based UQ metrics together in a single "transformed" UQ metric."""

    name: Optional[str] = None

    def fit(self) -> np.ndarray:
        return self.uq_values.sum(axis=0)


@dataclass
class SumVarDistDispatcher:
    """Dispatch converted distances for SumOfVarianceAndDistance definition.

    Args:
        transformed_uq_metrics (Dict[str, UQFactoryBasedUncertaintyMetric]):
            Transformed UQ including Dist2Var converted distances and DiffKNN distances
        variance_values (np.ndarray):
            Variance or sum of variance values input by the user
        mode (str):
            Analysis mode for distances dispatching.
                - "compact": only Euclidean and/or Tanimoto converted distances are
                summed to variances.
                - "extended": all distances are converted and summed to variances.
                - "full": Same as "extended", and additionally all combinations of
                distances are summed to variances.
    """

    transformed_uq_metrics: Dict[str, UQFactoryBasedUncertaintyMetric]
    variance_values: np.ndarray
    mode: Union[str, AnalyticsMode] = "compact"
    _logger: Optional[Logger] = None
    _description: str = "Note: "  # "Sum of Variances and Distances description:"

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = AnalyticsMode(self.mode)

        self._logger.debug(self.mode)

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        # self._description += "\n"
        self._description += value

    def _single_distances(self, selected: Dict) -> Dict[str, np.ndarray]:
        """Get all single selected distances"""
        results = {}
        for k, v in selected.items():
            if isinstance(self.variance_values, np.ndarray):
                uq_values = np.vstack((self.variance_values, [v]))
            else:
                continue
            sum_var_dist_metric = SumOfVariances(
                uq_values=uq_values,
                name=f"SumOfVariances[{k}]",
            )
            results[sum_var_dist_metric.name] = sum_var_dist_metric.fit()
        return results

    def _combination_distances(
        self, selected: Dict, combinations: list
    ) -> Dict[str, np.ndarray]:
        """Get all the combinations of selected distances"""
        results = {}
        for c in combinations:
            if isinstance(self.variance_values, np.ndarray):
                uq_values = self.variance_values
                for k in c:
                    uq_values = np.vstack((uq_values, selected[k]))
                sum_var_dist_metric = SumOfVariances(
                    uq_values=uq_values,
                    name=f"SumOfVariances[{c}]",
                )
                results[sum_var_dist_metric.name] = sum_var_dist_metric.fit()
            else:
                continue
        return results

    def _sum_all_distances(self, selected: Dict) -> Dict[str, np.ndarray]:
        """Sums all the selected converted-to-variance distances"""
        results = {}
        if len(selected) > 1:
            converted_distances = [v for k, v in selected.items()]
            if isinstance(self.variance_values, np.ndarray):
                uq_values = np.vstack((self.variance_values, converted_distances))
                sum_var_dist_metric = SumOfVariances(
                    uq_values=uq_values,
                    name=f"SumOfVariancesAndDistances",
                )
                results[sum_var_dist_metric.name] = sum_var_dist_metric.fit()

        str_list = f"\n".join([f"{i+1}. {k}" for i, k in enumerate(selected.keys())])
        self.description = f"UQ method 'SumOfVariancesAndDistances' summed the input variance(s) and the following distances (converted to variances):\n {str_list}"

        return results

    def fit(self):
        # Select distances to convert based on the selected analysis modality
        selected = {}
        combinations = {}
        if self.mode.mode == AnalyticsMode.COMPACT:
            # select only Euclidean and/or Tanimoto
            selected = {
                k: d.input_values
                for k, d in self.transformed_uq_metrics.items()
                if k.startswith("Dist2Var")
                and ("EuclideanDistance" in k or "TanimotoDistance" in k)
            }

        if self.mode.mode in [AnalyticsMode.EXTENDED, AnalyticsMode.FULL]:
            # select all converted distances
            selected = {
                k: d.input_values
                for k, d in self.transformed_uq_metrics.items()
                if k.startswith("Dist2Var")
            }

            # select all combinations of distances
            combinations = []
            for r in range(2, len(selected) + 1):
                combinations.extend(list(itertools.combinations(selected, r)))

        results = {}

        # variances only
        if len(selected) == 0 and isinstance(self.variance_values, np.ndarray):
            self._logger.debug(f"{self.__class__.__name__}: only variances included.")
            self._logger.warning(
                f"{self.__class__.__name__}: only variances will be included in "
                f"SumOfVariancesAndDistances. "
                f"Add Euclidean distance or Tanimoto distance or, alternatively, "
                f"select FULL analysis mode to include all selected distances."
            )
            self._logger.debug(f"{self.__class__.__name__}: no distances included.")
            sum_of_variances = SumOfVariances(
                uq_values=self.variance_values, name="SumOfVariances"
            )
            results[sum_of_variances.name] = sum_of_variances.fit()
            self.description = (
                f"UQ method 'SumOfVariances' only includes input variance(s)."
            )

        # include single distances based on analysis mode
        results.update(self._single_distances(selected=selected))

        # Add combinations in case of FULL analysis mode
        if self.mode.mode == AnalyticsMode.FULL:
            # select all combinations
            results.update(
                self._combination_distances(
                    selected=selected, combinations=combinations
                )
            )

        # sum all selected based on analysis mode
        results.update(self._sum_all_distances(selected=selected))

        return results
