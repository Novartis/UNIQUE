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

import os
from collections import defaultdict
from dataclasses import dataclass
from logging import Logger
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from unique.evaluation.evaluation_metrics import (
    auc_difference_bestrank,
    brier_score,
    check_score,
    crps_gaussian,
    crps_probability,
    decreasing_coefficient,
    get_all_performance_per_bin,
    increasing_coefficient,
    interval_score,
    mean_absolute_calibration_error,
    nll_gaussian,
    performance_drop_rank,
    root_mean_squared_calibration_error,
    spearman_correlation,
)
from unique.evaluation.evaluation_plots import (
    plot_bin_errordistrib,
    plot_bin_within_performance,
    plot_bins_summary,
    plot_calibration_classification,
    plot_calibration_regression,
    plot_error_correlation,
    plot_intervals_ordered,
    plot_rank_error_correlation,
)
from unique.utils import calculate_proportions


class EvalType:
    RankingBased = "RankingBasedEvaluation"
    CalibrationBased = "CalibrationBasedEvaluation"
    ProperScoringRules = "ProperScoringRulesEvaluation"


@dataclass
class UniqueUncertaintyEvaluation:  # base class for evaluation
    """Base class for UNIQUE's UQ metrics evaluation.

    Args:
        uq_values (np.ndarray):
            Array containing the precomputed UQ values.
        predictions (np.ndarray):
            Array containing the original model's predictions.
        labels (np.ndarray):
            Array containing the target values.
        uq_metric_name (str):
            Name of the UQ metric as "metric_name[column_name]".
        which_set_label (str):
            Name of the subset to evaluate the UQ metric on - e.g., "TEST".
        problem_type (str):
            Type of problem the original model is solving (either "classification" or "regression").
        name (str, None):
            Name to use to identify the UQ evaluation object. Default: None (same as the class name).
        perf_metric (str, None):
            Predictive performance metric to evaluate the UQ metric's goodness.
            Default: None ("MAE" if ``problem_type="regression"`` else "BA").
        nbins (int):
            Number of bins into which to bin the data. Internally, it uses
            ``nbins=3`` as well (hence, two binnings are performed). Default: 10.
        individual_plots (bool):
            Whether to visualize each supported evaluation metric's plot.
            Note: if True, plots will be shown for each UQ metric and data subset
            available ("TEST" only if ``evaluate_test_only=True``). Default: False.
        output_dir (Path, str, None):
            Output directory where to save the plots. Default: None.
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """

    uq_values: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    uq_metric_name: str
    is_error_model: bool
    is_variance: bool
    is_distance: bool
    which_set_label: str
    problem_type: str
    name: Optional[str] = None
    perf_metric: Optional[str] = None
    nbins: int = 10
    bootstrap_test_set: bool = False
    n_bootstrap: int = 500
    individual_plots: bool = False
    output_dir: Optional[Union[str, Path]] = None
    display_outputs: bool = True

    _seed: int = 42
    _use_parallel: bool = True  # for debugging
    _reduce_bs_output: bool = True
    logger: Logger = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

        if self.output_dir is not None:
            # os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(
                Path(self.output_dir) / self.which_set_label / self.uq_metric_name,
                exist_ok=True,
            )

        if self.bootstrap_test_set and self.which_set_label == "TEST":
            self.get_bs_replicas()

        if self.n_bootstrap == 10:  # for debugging
            self._use_parallel = False

    def fit(self) -> Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
        """Compute evaluation metrics w/o bootstrapping on the test set"""
        # the selection of the test set only is done
        bs_output = {}
        if self.bootstrap_test_set:
            bs_collection = []

            num_processes = 16

            # Compute evaluation metric for each bootstrapping replica using parallel pooling
            self.logger.debug(
                f'Bootstrapping for {self.uq_metric_name} in {self.__class__.__name__}{f" using parallel computing on {num_processes} processes" if self._use_parallel else ""}...'
            )
            if self._use_parallel:
                bs_collection = []

                # Create a pool of workers
                with Pool(processes=num_processes) as pool:

                    # Prepare the arguments for each process
                    args_list = [
                        (
                            self.replicas_dict["predictions"][n],
                            self.replicas_dict["labels"][n],
                            self.replicas_dict["uq_values"][n],
                        )
                        for n in range(self.n_bootstrap)
                    ]

                    results = pool.starmap(self.run, args_list)
                    bs_collection.extend(results)
            else:
                for n in range(self.n_bootstrap):
                    result = self.run(
                        predictions=self.replicas_dict["predictions"][n],
                        labels=self.replicas_dict["labels"][n],
                        uq_values=self.replicas_dict["uq_values"][n],
                    )
                    bs_collection.append(result)

            df_names = list(bs_collection[0].keys())
            bs_output = dict((zip(df_names, defaultdict(pd.DataFrame))))

            # for name in df_names:
            #     for i, l in enumerate(bs_collection):
            #         l[name]["bs_replica"] = i
            #     bs_output[name] = pd.concat([d[name] for d in bs_collection], axis=0)

            for i, l in enumerate(bs_collection):
                l["df_eval"]["bs_replica"] = i
            bs_output["df_eval"] = pd.concat(
                [d["df_eval"] for d in bs_collection], axis=0
            )

        # reduce bootstrapping results
        if self.bootstrap_test_set and self._reduce_bs_output:
            output = {
                k: self._get_reduced_bs_output(d)
                for k, d in bs_output.items()
                if k == "df_eval"
            }
            if isinstance(self, RankingBasedEvaluation):
                non_bs_output = self.run(
                    predictions=self.predictions,
                    labels=self.labels,
                    uq_values=self.uq_values,
                )
                output.update(
                    {
                        "decreasing_bins": non_bs_output["decreasing_bins"],
                        "increasing_bins": non_bs_output["increasing_bins"],
                        "within_3bins": non_bs_output["within_3bins"],
                    }
                )
        else:
            output = self.run(
                predictions=self.predictions,
                labels=self.labels,
                uq_values=self.uq_values,
            )
        return output, bs_output

    def _get_reduced_bs_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce bootstrapping dataframe output to mean values of all bs replicas"""
        values = df.drop(columns={"uq_metric", "which_set", "bs_replica"})
        values = pd.DataFrame([values.mean()])
        q = pd.DataFrame([df[["uq_metric", "which_set"]].iloc[0, :]])
        summary_df = pd.concat([q, values], axis=1)
        return summary_df

    def run(
        self, predictions: np.ndarray, labels: np.ndarray, uq_values: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def visualize(self):
        """Visualize relevant evaluation metrics results."""
        raise NotImplementedError

    def get_bs_replicas(self):
        """Perform bootstrapping with replacement on the provided data"""
        self.replicas_dict = {"uq_values": [], "predictions": [], "labels": []}

        np.random.seed(self._seed)
        self.logger.debug("Creating bootstrapping replicas on TEST set...")
        for _ in range(self.n_bootstrap):

            # sampling with replacement with # samples = N
            sample_indices = np.random.choice(
                len(self.uq_values), size=len(self.uq_values)
            )

            self.replicas_dict["uq_values"].append(self.uq_values[sample_indices])
            self.replicas_dict["predictions"].append(self.predictions[sample_indices])
            self.replicas_dict["labels"].append(self.labels[sample_indices])

        # BS with M < N to ?


@dataclass
class RankingBasedEvaluation(UniqueUncertaintyEvaluation):
    """Ranking-based evaluation.

    The relationship between the UQ metric and the model performance is quantified.
    For regression models, correlations between model error and UQ metrics are also assessed.

    Additional args:
        bins_labels (str):
            Name to give to the column indicating the bins labels. Default: "Bins".
        uq_thresholds_column_name (str):
            Name to give to the column containing the UQ thresholds used to bin
            the datapoints. Default: "Thresholds".
        errors_column_name (str):
            Name to give to the column containing the errors associated with each
            UQ-based bin. Default: "Errors".
    """

    bins_labels: str = "Bins"
    uq_thresholds_column_name: str = "Thresholds"
    errors_column_name: str = "Errors"

    def __post_init__(self):
        super().__post_init__()

        self.supported_performance_metrics = (
            ["MAE", "MSE", "R2", "2-Fold"]
            if self.problem_type == "regression"
            else ["BA", "MCC", "F1", "Kappa"]
        )

        if self.perf_metric is not None:
            assert (
                self.perf_metric in self.supported_performance_metrics
            ), f"""
                Unrecognized evaluation metric. Supported evaluation metrics:
                {self.supported_performance_metrics}.
                Got: {self.perf_metric}.
            """
        else:
            self.perf_metric = "MAE" if self.problem_type == "regression" else "BA"

        # Metrics optimization priority
        self.uq_eval_metrics = {
            "low_better": ["AUC_difference_bestrank"],
            "high_better": [
                "Performance_drop_AlltoLowUncertainty",
                "Performance_drop_HightoLowUncertainty",
                "Performance_drop_AlltoLowUncertainty_3bins",
                "Performance_drop_HightoLowUncertainty_3bins",
                "Decreasing_coefficient",
                "Increasing_coefficient",
                "Spearman_correlation",
            ],
        }

        # Define number of bins for UQ-based binning
        # Minimum 5 samples per bin or `nbins` (and 3 for "within")
        self.nbins = np.min([self.nbins, int(len(self.labels) / 5)])

    def run(
        self, predictions: np.ndarray, labels: np.ndarray, uq_values: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """Calculate rank-based evaluation metrics for a UQ metric."""
        # Store evaluation outputs
        eval_dict = {}
        eval_dict["uq_metric"] = self.uq_metric_name
        eval_dict["which_set"] = self.which_set_label

        decreasing_perfs, increasing_perfs, within_perfs = {}, {}, {}
        decreasing_perfs_best, increasing_perfs_best, within_perfs_errordistrib = (
            {},
            {},
            {},
        )

        # Compute ranking-based evaluation
        # 1. Binning performance: which UQ binning strategy ranks the predictions the best?
        # Use ``nbins`` and 3 bins
        for num_bins in [self.nbins, 3]:
            dict_bins_performances = get_all_performance_per_bin(
                preds=predictions,
                labels=labels,
                uq_values=uq_values,
                nbins=num_bins,
                problem_type=self.problem_type,
                bins_labels=self.bins_labels,
                uq_thresholds_column_name=self.uq_thresholds_column_name,
                errors_column_name=self.errors_column_name,
            )
            # The output is a dictionary with the following keys:
            # "increasing_bins", "decreasing_bins", "within_bins",
            # "increasing_bins_best", "decreasing_bins_best", "within_bins_errordistrib"

            # Store the UQ metric used for the ranking and the data subset
            for k in dict_bins_performances.keys():
                dict_bins_performances[k]["uq_metric"] = (
                    self.uq_metric_name
                    if not k.endswith("_best")
                    else "True Absolute Error"
                )
                dict_bins_performances[k]["which_set"] = self.which_set_label

            # 2. Performance drop: how are performances affected by using all data vs.
            # a subset of the data (e.g., the least uncertain data)?
            perf_drop_all_vs_low, perf_drop_high_vs_low = performance_drop_rank(
                decreasing_bins=dict_bins_performances["decreasing_bins"],
                increasing_bins=dict_bins_performances["increasing_bins"],
                perf_metric=self.perf_metric,
            )

            eval_dict[f"Performance Drop: All vs. Low UQ ({num_bins}-Bins)"] = (
                perf_drop_all_vs_low
            )
            eval_dict[f"Performance Drop: High UQ vs. Low UQ ({num_bins}-Bins)"] = (
                perf_drop_high_vs_low
            )

            # Store the performances separately
            decreasing_perfs[num_bins] = dict_bins_performances["decreasing_bins"]
            increasing_perfs[num_bins] = dict_bins_performances["increasing_bins"]
            within_perfs[num_bins] = dict_bins_performances["within_bins"]
            decreasing_perfs_best[num_bins] = dict_bins_performances[
                "decreasing_bins_best"
            ]
            increasing_perfs_best[num_bins] = dict_bins_performances[
                "increasing_bins_best"
            ]
            within_perfs_errordistrib[num_bins] = dict_bins_performances[
                "within_bins_errordistrib"
            ]

        self.decreasing_perfs = decreasing_perfs  # dict
        self.increasing_perfs = increasing_perfs  # dict
        self.within_perfs = within_perfs  # dict
        self.decreasing_perfs_best = decreasing_perfs_best  # dict
        self.increasing_perfs_best = increasing_perfs_best  # dict
        self.within_perfs_errordistrib = within_perfs_errordistrib  # dict

        # 3. Decreasing & increasing ranking coefficients: how many consecutive bins
        # show a consistent performance trend? e.g., decreasing for decreasing bins
        eval_dict["Decreasing Coefficient"] = decreasing_coefficient(
            decreasing_bins=self.decreasing_perfs[self.nbins],
            perf_metric=self.perf_metric,
        )
        eval_dict["Increasing Coefficient"] = increasing_coefficient(
            increasing_bins=self.increasing_perfs[self.nbins],
            perf_metric=self.perf_metric,
        )

        # 4. Compute evaluation metrics based on the best possible ranking (true error)
        if self.problem_type == "regression":
            # AUC difference between UQ values and true error (best possible ranking)
            eval_dict["AUC Difference: UQ vs. True Error"] = auc_difference_bestrank(
                increasing_bins=self.increasing_perfs[self.nbins],
                increasing_bins_best=self.increasing_perfs_best[self.nbins],
                bins_labels=self.bins_labels,
                perf_metric=self.perf_metric,
            )

            ## Spearman correlation between UQ values and errors
            eval_dict["Spearman Correlation"] = spearman_correlation(
                preds=predictions,
                labels=labels,
                uq_values=uq_values,
            )

        # Visualize relevant plots
        if self.individual_plots:
            self.visualize()

        return {
            "df_eval": pd.DataFrame([eval_dict]),
            "decreasing_bins": self.decreasing_perfs[self.nbins],
            "increasing_bins": self.increasing_perfs[self.nbins],
            "within_3bins": self.within_perfs[3],
        }

    def visualize(self):
        """Visualize relevant ranking-based evaluation metrics results."""
        # Define common kwargs
        kwargs = {
            "uq_metric_name": self.uq_metric_name,
            "subset": self.which_set_label,
            "output_path": self.output_dir,
            "display_outputs": self.display_outputs,
        }

        if self.problem_type == "regression":
            # Plot evaluation metric trend for data ordered by increasing/decreasing UQ-value
            plot_bins_summary(
                df_decreasing=pd.concat(
                    (
                        self.decreasing_perfs[self.nbins],
                        self.decreasing_perfs_best[self.nbins],
                    )
                ).reset_index(drop=True),
                df_increasing=pd.concat(
                    (
                        self.increasing_perfs[self.nbins],
                        self.increasing_perfs_best[self.nbins],
                    )
                ).reset_index(drop=True),
                bins_labels=self.bins_labels,
                perf_metric=self.perf_metric,
                **kwargs,
            )
            # Plot error distributions boxplots split by UQ-based bins
            plot_bin_errordistrib(
                data=self.within_perfs_errordistrib[3],
                errors_column_name=self.errors_column_name,
                uq_thresholds_column_name=self.uq_thresholds_column_name,
                **kwargs,
            )
            # Plot error-to-UQmetric (ranked) correlation scatterplots
            abs_errors = np.abs(self.predictions - self.labels)

            plot_error_correlation(
                uq_values=self.uq_values, true_errors=abs_errors, **kwargs
            )
            plot_rank_error_correlation(
                uq_values=self.uq_values, true_errors=abs_errors, **kwargs
            )
        else:  # classification
            plot_bins_summary(
                df_decreasing=self.decreasing_perfs[self.nbins],
                df_increasing=self.increasing_perfs[self.nbins],
                bins_labels=self.bins_labels,
                perf_metric=self.perf_metric,
                **kwargs,
            )

        # Plot evaluation metric barplots split by UQ-based bins
        plot_bin_within_performance(
            data=self.within_perfs[3],
            perf_metric=self.perf_metric,
            uq_thresholds_column_name=self.uq_thresholds_column_name,
            **kwargs,
        )


@dataclass
class ProperScoringRulesEvaluation(UniqueUncertaintyEvaluation):
    """Proper scoring rules evaluation.

    Proper scoring rules are a scalar summary measure of the performance of a
    distributional prediction. Examples are:
        - Negative Log-Likelihood (NLL).
        - Continuous Ranked Probability Score (CRPS).
    """

    def __post_init__(self):
        super().__post_init__()

        # Function inputs
        # self.kwargs = {
        #         "y_pred": self.predictions,
        #         "y_true": self.labels,
        #     }

        if self.problem_type == "classification":
            # Specify supported metrics
            self.supported_eval_metrics = {
                "CRPS": crps_probability,
                "BrierScore": brier_score,
            }
            # Metrics optimization priority
            self.uq_eval_metrics = {
                "low_better": ["CRPS", "BrierScore"],
                "high_better": [],
            }
        else:  # regression
            # Specify supported metrics
            self.supported_eval_metrics = {
                "CRPS": crps_gaussian,
                "NLL": nll_gaussian,
                "CheckScore": check_score,
                "IntervalScore": interval_score,
            }
            # Metrics optimization priority
            self.uq_eval_metrics = {
                "low_better": ["CRPS", "NLL", "CheckScore", "IntervalScore"],
                "high_better": [],
            }
            # Add UQ values as variance to inputs
            # self.kwargs["y_variance"] = self.uq_values

    def run(
        self, predictions: np.ndarray, labels: np.ndarray, uq_values: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """Calculate Proper Scoring Rules as evaluation metrics."""

        kwargs = {"y_pred": predictions, "y_true": labels, "y_variance": uq_values}

        # Store evaluation outputs
        eval_dict = {}
        eval_dict["uq_metric"] = self.uq_metric_name
        eval_dict["which_set"] = self.which_set_label

        # Compute evaluation metrics
        for metric, func in self.supported_eval_metrics.items():
            eval_dict[metric] = func(**kwargs)

        return {"df_eval": pd.DataFrame([eval_dict])}


@dataclass
class CalibrationBasedEvaluation(UniqueUncertaintyEvaluation):
    """Calibration-based evaluation.

    Original model's predictions are evaluated for their "calibration" with respect
    to the labels based on the computed UQ metric values.

    Additional args:
        strategy (str):
            Aggregation strategy to use to bin the expected and observed
            proportions. One of "quantile" or "uniform". Default: "uniform".
    """

    strategy: str = "uniform"

    def __post_init__(self):
        super().__post_init__()

        assert (
            self.problem_type != "classification"
        ), """
            Calibration-based evaluation metrics for classification problems have not
            been implemented yet.
        """

        # Specify supported metrics
        self.supported_eval_metrics = {
            "MACE": mean_absolute_calibration_error,
            "RMSCE": root_mean_squared_calibration_error,
        }
        # Metrics optimization priority
        self.uq_eval_metrics = {
            "low_better": ["MACE", "RMSCE"],
            "high_better": [],
        }

    def _compute_proportions(
        self, predictions: np.ndarray, labels: np.ndarray, uq_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Calculate observed and expected proportions to compute calibration
        obs_props, exp_props = calculate_proportions(
            y_pred=predictions,
            y_variance=uq_values,
            y_true=labels,
            strategy=self.strategy,
            nbins=self.nbins,
        )

        return obs_props, exp_props

    def run(
        self, predictions: np.ndarray, labels: np.ndarray, uq_values: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """Calculate calibration-based evaluation metrics for a UQ metric."""

        obs_props, exp_props = self._compute_proportions(predictions, labels, uq_values)

        # Store evaluation outputs
        eval_dict = {}
        eval_dict["uq_metric"] = self.uq_metric_name
        eval_dict["which_set"] = self.which_set_label

        # Compute evaluation metrics
        for metric, func in self.supported_eval_metrics.items():
            eval_dict[metric] = func(
                obs_proportions=obs_props, exp_proportions=exp_props
            )

        if self.individual_plots:
            self.visualize()

        return {"df_eval": pd.DataFrame([eval_dict])}

    def visualize(self):
        """Visualize relevant calibration-based evaluation metrics results."""
        kwargs = {
            "uq_metric_name": self.uq_metric_name,
            "subset": self.which_set_label,
            "output_path": self.output_dir,
            "display_outputs": self.display_outputs,
        }

        if self.problem_type == "classification":
            # Miscalibration area
            plot_calibration_classification(
                y_prob=self.uq_values,
                y_true=self.labels,
                num_bins=self.nbins,
                strategy=self.strategy,
                **kwargs,
            )
        else:  # regression
            obs_props, exp_props = self._compute_proportions(
                self.predictions, self.labels, self.uq_values
            )

            # Miscalibration area
            plot_calibration_regression(
                observed=obs_props, expected=exp_props, **kwargs
            )
            # Predictions+UQ ordered by target values
            plot_intervals_ordered(
                y_pred=self.predictions,
                y_std=self.uq_values,
                y_true=self.labels,
                n_subset=int(np.min([self.predictions.shape[0], 200])),
                **kwargs,
            )
