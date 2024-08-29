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
import pandas as pd

from unique.evaluation.evaluation_metrics import (
    calculate_classification_performance,
    calculate_regression_performance,
)
from unique.utils import is_running_in_jupyter_notebook


def apply_uq_inference(
    input_data: pd.DataFrame,
    bins: pd.DataFrame,
    uq_column_name: str,
    uq_thresholds_column_name: str = "Thresholds",
    which_set: str = "TEST",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run UNIQUE inference."""
    bins_subset = bins.loc[
        (bins["uq_metric"] == uq_column_name) & (bins["which_set"] == which_set)
    ]

    inference_column = f"UQ Level: {uq_column_name}"
    thresholds = bins_subset[uq_thresholds_column_name].sort_values().unique()

    all_data = input_data.copy(deep=True)
    all_data[inference_column] = "Medium"
    all_data.loc[all_data[uq_column_name] <= thresholds[0], inference_column] = "Low"
    all_data.loc[all_data[uq_column_name] >= thresholds[1], inference_column] = "High"

    all_data[inference_column] = pd.Categorical(
        all_data[inference_column], ["High", "Medium", "Low"]
    )
    all_data.sort_values(inference_column).reset_index(drop=True, inplace=True)

    test_data = (
        all_data.loc[all_data["which_set"] == which_set]
        .sort_values(inference_column)
        .reset_index(drop=True)
    )

    return all_data, test_data, thresholds


def performance_inference_uq(df_test, inference_column, problem_type):
    """
    Calculate performance per each UQ level at the inference phase
    """

    if problem_type == "regression":
        calculate_performance = calculate_regression_performance
        low_better_list = ["MAE", "MSE", "GMFE"]
        high_better_list = ["R2", "SpearmanR", "2-Fold", "3-Fold"]
        all_list = low_better_list.copy()
        all_list.extend(high_better_list)
    elif problem_type == "classification":
        calculate_performance = calculate_classification_performance
        high_better_list = ["MCC", "BA", "Kappa", "F1"]
        all_list = high_better_list.copy()
    else:
        raise ValueError("problem_type not supported")

    df_perf_inference = pd.DataFrame()
    for i in np.unique(df_test[inference_column]):
        df_tmp = df_test[df_test[inference_column] == i]
        df_tmp = calculate_performance(y_pred=df_tmp.predictions, y_true=df_tmp.labels)
        df_tmp[inference_column] = i
        df_perf_inference = pd.concat((df_perf_inference, df_tmp))

    all_list.extend(["#", inference_column])
    df_perf_inference = df_perf_inference[all_list].reset_index(drop=True)

    df_perf_inference[inference_column] = pd.Categorical(
        df_perf_inference[inference_column], ["High", "Medium", "Low"]
    )
    df_perf_inference = df_perf_inference.sort_values([inference_column]).reset_index(
        drop=True
    )

    if is_running_in_jupyter_notebook():
        if problem_type == "regression":
            display(
                df_perf_inference.style.highlight_min(
                    color="lightgreen", axis=0, subset=low_better_list
                )
                .highlight_(color="lightgreen", axis=0, subset=high_better_list)
                .format(precision=2)
            )
        elif problem_type == "classification":
            display(
                df_perf_inference.style.highlight_(
                    color="lightgreen", axis=0, subset=high_better_list
                ).format(precision=2)
            )

    return df_perf_inference
