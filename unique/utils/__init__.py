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

from .error_model_utils import (  # weighted_mse,
    compute_error,
    min_max_normalization,
    split_error_model_dataset,
)
from .evaluation_utils import (
    calculate_proportions,
    filter_subset,
    get_proportion_in_interval,
    get_proportion_under_quantile,
)
from .uncertainty_utils import (
    AnalyticsMode,
    cnll,
    convert_distances_to_variances,
    is_running_in_jupyter_notebook,
    tanimoto_distance,
)

__all__ = [
    "AnalyticsMode",
    "calculate_proportions",
    "cnll",
    "compute_error",
    "convert_distances_to_variances",
    "filter_subset",
    "get_proportion_in_interval",
    "get_proportion_under_quantile",
    "min_max_normalization",
    "split_error_model_dataset",
    "tanimoto_distance",
    "is_running_in_jupyter_notebook",
    # "weighted_mse",
]
