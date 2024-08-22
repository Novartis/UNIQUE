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
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from sklearn.calibration import CalibrationDisplay

from unique.utils import filter_subset, is_running_in_jupyter_notebook

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

SAVEFIG_KWARGS = {
    "transparent": True,
    "dpi": 300,
    "bbox_inches": "tight",
}


class EvalType:
    RankingBased = "RankingBasedEvaluation"
    CalibrationBased = "CalibrationBasedEvaluation"
    ProperScoringRules = "ProperScoringRulesEvaluation"


######################################
# Calibration-based Evaluation Plots #
######################################


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_std: np.ndarray,
    uq_metric_name: str,
    subset: str,
    n_subset: Optional[int] = None,
    ylims: Optional[Tuple[float, float]] = None,
    num_stds_confidence_bound: int = 2,
    output_path: Optional[Union[Path, str]] = None,
    display_outputs: bool = True,
):
    """Plot predictions and predictive interval ordered by the corresponding true values.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    Args:
        y_pred (np.ndarray):
            Predictions.
        y_true (np.ndarray):
            Corresponding target values.
        y_std (np.ndarray):
            Predicted uncertainty values associated with the predictions
            (to consider as standard deviations).
        uq_metric_name (str):
            Name of the UQ metric associated with the predicted uncertainty.
        subset (str):
            Name of the data subset the data belongs to.
        n_subset (int, None):
            Number of random points to plot. Default: None (plot all datapoints).
        y_lims (Tuple[float, float], None):
            Lower and upper y-axis limits. Default: None (use min and max of predictions +- std).
        num_stds_confidence_bound (int):
            Width of the predictive intervals, in number of standard deviations. Default: 2.
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    # Order target values from smaller to bigger value
    order = np.argsort(y_true.flatten())
    # Reorder predictions
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    # Get x-axis indices
    xs = np.arange(len(order))
    # Error bars as std*num_stds_confidence_bound
    intervals = num_stds_confidence_bound * y_std

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Errorbars are plotted separately
    ax.errorbar(
        x=xs,
        y=y_pred,
        xerr=np.abs(intervals),
        fmt="o",
        ls="none",
        linewidth=1.5,
        c="#1f77b4",
        alpha=0.5,
    )
    # Scatter plot for preds vs targets' ordered indices
    sns.scatterplot(x=xs, y=y_pred, s=15, label="Predicted Values", c="#1f77b4", ax=ax)
    sns.lineplot(
        x=xs,
        y=y_true,
        ls="--",
        linewidth=2,
        c="#ff7f0e",
        label="Observed Values",
        ax=ax,
    )

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # Format plot
    ax.set_ylim(lims_ext)
    ax.set_xlabel("Index (Ordered by Observed Value)", fontsize=14)
    ax.set_ylabel("Predicted Values + Intervals", fontsize=14)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=13)

    fig.suptitle(
        f"[{subset}] Ordered Prediction Intervals: {uq_metric_name}", fontsize=15
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"ordered_predictions_intervals_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_calibration_regression(
    observed: np.ndarray,
    expected: np.ndarray,
    uq_metric_name: str,
    subset: str,
    n_subset: Optional[int] = None,
    output_path: Optional[Union[Path, str]] = None,
    display_outputs: bool = True,
):
    """Plot calibration plot for regression problems.

    Adapted from:
    Source: https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    Author(s): Uncertainty Toolbox

    The observed vs expected proportion of outputs falling into a range of intervals,
    are plotted against each other along with the miscalibration area.

    Args:
        observed (np.ndarray):
            Observed proportions of predicted values.
        expected (np.ndarray):
            Expected proportions of target values.
        uq_metric_name (str):
            Name of the UQ metric to be calibrated.
        subset (str):
            Name of the data subset the data belongs to.
        n_subset (int, None):
            Number of random points to plot. Default: None (plot all datapoints).
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    exp_proportions = np.array(expected).flatten()
    obs_proportions = np.array(observed).flatten()

    assert (
        exp_proportions.shape == obs_proportions.shape
    ), "Expected and observed proportions shapes do not match."

    # Optionally select a random subset
    if n_subset is not None:
        [exp_proportions, obs_proportions] = filter_subset(
            [exp_proportions, obs_proportions], n_subset
        )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Calibration curve
    sns.lineplot(x=exp_proportions, y=obs_proportions, c="#1f77b4", ax=ax)
    # Miscalibration area
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)
    # Ideal curve
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")

    # Format plot
    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_xlabel("Predicted Proportion in Interval", fontsize=14)
    ax.set_ylabel("Observed Proportion in Interval", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.axis("square")

    # Compute miscalibration area
    polygon_points = [point for point in zip(exp_proportions, obs_proportions)]
    polygon_points.extend(
        [point for point in zip(reversed(exp_proportions), reversed(exp_proportions))]
    )
    polygon_points.append((exp_proportions[0], obs_proportions[0]))

    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate plot with the miscalibration area
    ax.text(
        x=0.95,
        y=0.05,
        s=f"Miscalibration area = {miscalibration_area:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )

    fig.suptitle(f"[{subset}] Calibration Curve: {uq_metric_name}", fontsize=15)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"calibration_regression_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_calibration_classification(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    uq_metric_name: str,
    subset: str,
    num_bins: int = 100,
    strategy: str = "uniform",
    output_path: Optional[Union[Path, str]] = None,
    display_outputs: bool = True,
):
    """Plot calibration plot for classification problems.

    Args:
        y_prob (np.ndarray):
            Predicted probabilities.
        y_true (np.ndarray):
            Target distribution.
        uq_metric_name (str):
            Name of the UQ metric to be calibrated.
        subset (str):
            Name of the data subset the data belongs to.
        num_bins (int):
            Number of bins to split the target distribution into.
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # Calibration curve
    display = CalibrationDisplay.from_predictions(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=num_bins,
        strategy=strategy,
        ax=ax,
    )

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.axis("square")

    fig.suptitle(f"[{subset}] Calibration Curve: {uq_metric_name}", fontsize=15)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"calibration_classification_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


##################################
# Ranking-based Evaluation Plots #
##################################


def plot_error_correlation(
    uq_values: np.ndarray,
    true_errors: np.ndarray,
    uq_metric_name: str,
    subset: str,
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot the correlation between true absolute errors and UQ values.

    Args:
        uq_values (np.ndarray):
            Precomputed UQ metric values associated to each prediction.
        true_errors (np.ndarray):
            True absolute prediction errors (|labels-predictions|).
        uq_metric_name (str):
            Name of the computed UQ metric.
        subset (str):
            Name of the data subset the grouped data belongs to.
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    # Scatterplot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    sns.scatterplot(x=uq_values, y=true_errors, alpha=0.5, ax=ax)

    ax.set_xlabel("UQ Values", fontsize=14)
    ax.set_ylabel("Absolute Errors", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13)

    fig.suptitle(f"[{subset}] Error vs. UQ Values: {uq_metric_name}", fontsize=15)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"correlation_errors_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_rank_error_correlation(
    uq_values: np.ndarray,
    true_errors: np.ndarray,
    uq_metric_name: str,
    subset: str,
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot the ranked correlation between true absolute errors and UQ values.

    Args:
        uq_values (np.ndarray):
            Precomputed UQ metric values associated to each prediction.
        true_errors (np.ndarray):
            True absolute prediction errors (|labels-predictions|).
        uq_metric_name (str):
            Name of the computed UQ metric.
        subset (str):
            Name of the data subset the grouped data belongs to.
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    # Reorder in ascending order both UQ values and true errors
    sorted_uq_values = np.argsort(uq_values)
    sorted_true_errors = np.argsort(true_errors)

    # Scatterplot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    sns.scatterplot(x=uq_values, y=true_errors, alpha=0.5, ax=ax)

    ax.set_xlabel("Ranked UQ Values", fontsize=14)
    ax.set_ylabel("Ranked Absolute Errors", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13)

    fig.suptitle(
        f"[{subset}] Ranked Errors vs. Ranked UQ Values: {uq_metric_name}", fontsize=15
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"ranked_correlation_errors_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_bin_within_performance(
    data: pd.DataFrame,
    uq_metric_name: str,
    subset: str,
    perf_metric: str,
    uq_thresholds_column_name: str = "Thresholds",
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot the evaluation metric's performance for each UQ-based binned data groups.

    Args:
        data (pd.DataFrame):
            Input dataframe containing the evaluation metric's values for each
            UQ binned data groups.
        uq_metric_name (str):
            Name of the binned UQ metric.
        subset (str):
            Name of the data subset the grouped data belongs to.
        perf_metric (str):
            Name of the performance metric being considered (e.g., MAE).
        uq_thresholds_column_name (str):
            Name of the column in the input data containing the UQ metric's threshold
            used for binning the datapoints. Default: "Thresholds".
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    # Barplot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    sns.barplot(data, x=uq_thresholds_column_name, y=perf_metric, ax=ax)

    ax.set_xlabel("UQ Value's Bin Thresholds", fontsize=14)
    ax.set_ylabel(perf_metric, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13)

    fig.suptitle(
        f"[{subset}] {perf_metric} vs. Binned UQ Method: {uq_metric_name}", fontsize=15
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"{perf_metric}_binned_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_bin_errordistrib(
    data: pd.DataFrame,
    uq_metric_name: str,
    subset: str,
    errors_column_name: str = "Errors",
    uq_thresholds_column_name: str = "Thresholds",
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot the error distributions as boxplots for data grouped by UQ-based bins.

    Args:
        data (pd.DataFrame):
            Input dataframe containing the prediction errors for each datapoint
            and grouped by UQ-based bins.
        uq_metric_name (str):
            Name of the binned UQ metric.
        subset (str):
            Name of the data subset the grouped data belongs to.
        errors_column_name (str):
            Name of the column containing the errors. Default: "Errors".
        uq_thresholds_column_name (str):
            Name of the column in the input data containing the UQ metric's thresholds
            used for binning the datapoints. Default: "Thresholds".
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    # Boxplot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    sns.boxplot(data, x=uq_thresholds_column_name, y=errors_column_name, ax=ax)

    ax.set_xlabel("UQ Value's Bin Thresholds", fontsize=14)
    ax.set_ylabel(errors_column_name, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13)

    fig.suptitle(
        f"[{subset}] {errors_column_name} vs. Binned UQ Method: {uq_metric_name}",
        fontsize=15,
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / subset
            / uq_metric_name
            / f"{errors_column_name}_distributions_binned_{uq_metric_name}.png",
            **SAVEFIG_KWARGS,
        )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_bins_summary(
    df_decreasing: pd.DataFrame,
    df_increasing: pd.DataFrame,
    uq_metric_name: str,
    subset: str,
    bins_labels: str,
    perf_metric: str,
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot the evaluation metric computed on incremental data subsets ordered by uncertainty metric values.

    The plot shows the evaluation metric values for incrementally-added subsets of
    data, ordered by their uncertainty metric value.

    For example, a UQ method is applied for the dataset and each datapoint is ordered
    according to its computed uncertainty value. Then, the evaluation metric (e.g., MAE)
    is computed for each 10% incremental subset of the data (i.e., first 10%, then 20%,
    30%, etc.) and plotted.

    We consider two ordering criteria:
    1. from higher to lower UQ values.
    2. from lower to higher UQ values.

    Finally, the evaluation metric computed for the data ordered according to its UQ
    metric value, is then evaluated against the evaluation metric ordered by its true
    value (e.g., incremental subsets ordered by magnitude of the real MAE values).

    Args:
        df_decreasing (pd.DataFrame):
            Dataframe containing the evaluation metric computed on incremental
            subsets of data ordered in decreasing order based on the uncertainty value obtained by the UQ method .
        df_increasing (pd.DataFrame):
            Dataframe containing the evaluation metric computed on incremental
            subsets of data ordered in ascending order based on the UQ metric value.
        uq_metric_name (str):
            Name of the UQ metric used to order the data.
        subset (str):
            Which subset the data belongs to.
        bins_labels (str):
            Name of the column containing the bins' labels.
        perf_metric (str):
            Name of the performance metric being used (e.g., MAE).
        output_path (Path, str, None):
            Output directory where to save the plots. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    # Labels
    orders = ["High to Low UQ", "Low to High UQ"]

    # Plot evaluation metric according to incrementally-added data
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)

    for i, (df, order) in enumerate(zip([df_decreasing, df_increasing], orders)):
        kwargs = {
            "alpha": 0.85,
            "palette": sns.color_palette(cc.glasbey[: df.shape[0]]),
            "marker": "o",
            "markersize": 15,
            "linewidth": 3,
        }
        sns.lineplot(
            df, x=bins_labels, y=perf_metric, hue="uq_metric", ax=axs[i], **kwargs
        )

        axs[i].set_title(f"Addition Order: {order}", fontsize=14)
        axs[i].set_xlabel("% Data", fontsize=14)
        axs[i].set_ylabel(perf_metric, fontsize=14)
        axs[i].tick_params(axis="both", which="major", labelsize=13)
        axs[i].get_legend().remove()

    hdls, lbls = axs[0].get_legend_handles_labels()
    if len(hdls) > 2:  # More than 2 lines, legend to the right and outside plot
        fig.legend(hdls, lbls, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:  # 2 or less lines, legend inside the second subplot
        axs[-1].legend()
    fig.suptitle(
        f"[{subset}] {perf_metric} vs. UQ-ordered Data: {uq_metric_name}", fontsize=15
    )
    fig.tight_layout()

    if output_path is not None:
        # Summary
        if uq_metric_name == "All UQ Methods":
            os.makedirs(Path(output_path) / subset, exist_ok=True)
            fig.savefig(
                Path(output_path)
                / subset
                / f"{subset}_{perf_metric}_orderedby_{uq_metric_name.replace(' ', '_')}.png",
                **SAVEFIG_KWARGS,
            )
        # Individual UQ methods
        else:
            fig.savefig(
                Path(output_path)
                / subset
                / uq_metric_name
                / f"{perf_metric}_orderedby_{uq_metric_name}.png",
                **SAVEFIG_KWARGS,
            )

    # Display figure
    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)


def plot_metrics_summary(
    df: pd.DataFrame,
    title: str,
    subset: str,
    bootstrap_test_set: bool,
    output_path: Optional[Union[Path, str]] = None,
    display_outputs: bool = True,
):
    """Plot a summarizing barplot containing all evaluation and UQ metrics.

    Args:
        df (pd.DataFrame):
            Input dataframe containing the evaluation and UQ metrics.
        title (str):
            Title of the plot to use.
        evaluate_test_only (bool):
            Whether to only use the test subset for evaluation.
        subset (str):
            TRAIN, CALIBRATION, or TEST
        output_path (Path, str, None):
            Output directory where to store the plot. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    data = df.copy()
    data = data.drop(columns=["which_set"])
    if bootstrap_test_set:
        data = data.melt(
            id_vars=["uq_metric", "bs_replica"],
            var_name="perf_metric",
            value_name="value",
        )
    else:
        data = data.melt(
            id_vars=["uq_metric"], var_name="perf_metric", value_name="value"
        )
    data.drop_duplicates(inplace=True)

    # Plot UQ metrics as barplots
    if len(data["uq_metric"]) > 0:
        data["uq_metric"] = pd.Categorical(data["uq_metric"])
        data = data.sort_values("uq_metric")
        perf_metrics = data["perf_metric"].unique().tolist()
        perf_metrics.append("")
        perf_metrics.append("")
        ncols = min(2, len(perf_metrics))
        nrows = len(perf_metrics) // ncols
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(9 * ncols, 6 * nrows), squeeze=False
        )
        for k, pm in enumerate(perf_metrics):
            i = k // ncols
            j = k % ncols
            this_data = data.loc[data["perf_metric"] == pm]
            data_sorted = this_data.sort_values(by="value", ascending=False)
            hue_order = data_sorted["uq_metric"].unique()

            sns.barplot(
                data=data_sorted,
                x="uq_metric",
                y="value",
                hue="uq_metric",
                order=hue_order,
                dodge=False,
                legend=True if k == 0 else False,
                palette=sns.color_palette(cc.glasbey, n_colors=data_sorted.shape[0]),
                ax=axs[i, j],
            )
            axs[i, j].set_title(f"{pm}", fontsize=15)
            axs[i, j].set_xlabel("UQ Methods", fontsize=14)
            axs[i, j].set_ylabel("Value", fontsize=14)
            axs[i, j].tick_params(axis="both", which="major", labelsize=13)
            axs[i, j].set_xticklabels([])
            if k == 0:
                hdls, lbls = axs[i, j].get_legend_handles_labels()

            if axs[i, j].get_legend() is not None:
                axs[i, j].get_legend().remove()

            if k == len(perf_metrics) - 1 or k == len(perf_metrics) - 2:
                axs[i, j].axis("off")

        axs[i, j].legend(hdls, lbls)

        fig.suptitle(f"[{subset}] {title}", fontsize=17)
        fig.tight_layout()

        if output_path is not None:
            os.makedirs(Path(output_path) / subset, exist_ok=True)
            fig.savefig(
                Path(output_path)
                / subset
                / f"{subset}_{title}_all_metrics_summary_plot.png",
                **SAVEFIG_KWARGS,
            )

        # Display figure
        if is_running_in_jupyter_notebook() and display_outputs:
            fig.show()
        else:
            plt.close(fig)


def get_summary_plots(
    eval_dict: Dict,
    bins_labels: str,
    perf_metric: str,
    eval_bs_dict: Optional[Dict] = None,
    evaluate_test_only: bool = True,
    bootstrap_test_set: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot summary UQ metrics evaluation plots.

    Args:
        eval_dict (Dict):
            Dictionary containing the output from a ``UniqueUncertaintyEvaluation`` object.
        bins_labels (str):
            Name of the column containing the bins' labels.
        perf_metric (str):
            Name of the performance metric being used to evaluate the goodness of
            the UQ metric (e.g., "MAE" for regression problems).
        eval_bs_dict (Dict):
            Dictionary containing the output from a ``UniqueUncertaintyEvaluation`` object with bootstrapping
        evaluate_test_only (bool):
            Whether to perform the evaluation only on the test subset. Default: True.
        bootstrap_test_set (bool):
            Bootstrapping on the test set. Default: False.
        output_path (str, Path, None):
            Output directory path where to store the plots. Default: None (plots are not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    # Plot the UQ metrics summary plot for each evaluation type

    if not bootstrap_test_set:
        eval_bs_dict = eval_dict

    for eval_type in [
        EvalType.RankingBased,
        EvalType.CalibrationBased,
        EvalType.ProperScoringRules,
    ]:
        if eval_bs_dict[eval_type]["df_eval"] is not None:
            data = eval_bs_dict[eval_type]["df_eval"]
            if evaluate_test_only:
                subset = "TEST"
                data = data.loc[data["which_set"] == subset].reset_index(drop=True)
            for subset in data["which_set"].unique():
                plot_metrics_summary(
                    df=data.loc[data["which_set"] == subset].reset_index(drop=True),
                    title=eval_type,
                    subset=subset,
                    bootstrap_test_set=bootstrap_test_set,
                    output_path=output_path,
                    display_outputs=display_outputs,
                )

    # Only for ranking-based evaluation
    df_decreasing = eval_dict[EvalType.RankingBased]["decreasing_bins"]
    df_increasing = eval_dict[EvalType.RankingBased]["increasing_bins"]

    list_subsets = (
        ["TEST"] if evaluate_test_only else df_decreasing["which_set"].unique()
    )

    for subset in list_subsets:
        # Plot the incremental bins vs error summary plot for each subset
        decreasing = df_decreasing.loc[df_decreasing["which_set"] == subset]
        increasing = df_increasing.loc[df_increasing["which_set"] == subset]

        plot_bins_summary(
            df_decreasing=decreasing,
            df_increasing=increasing,
            uq_metric_name="All UQ Methods",
            subset=subset,
            bins_labels=bins_labels,
            perf_metric=perf_metric,
            output_path=output_path,
            display_outputs=display_outputs,
        )


def get_summary_tables(
    eval_dict: Dict,
    uq_eval_metrics: Dict,
    evaluate_test_only: bool = True,
    output_path: Optional[Union[Path, str]] = None,
    display_outputs: bool = True,
    best_methods_bs: Optional[Dict] = None,
) -> Dict:
    """Generate summary tables containing the evaluation metric results per subset.

    Args:
        eval_dict (Dict):
            Dictionary containing the output from a ``UniqueUncertaintyEvaluation``
            object - i.e., the different evaluation types.
        uq_eval_metrics (Dict):
            Dictionary of dictionaries containing the indication of the ideal value
            for each evaluation metric (e.g., higher values are better - "high_better").
        evaluate_test_only (bool):
            Whether to perform the evaluation only on the test subset. Default: True.
        output_path (str, Path, None):
            Output directory path where to store the evaluation tables. Default: None (tables are not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
        best_methods_bs (Dict):
            Dictionary with the best methods found from statistical analysis on results
            from bootstrap resampling. Default = None.

    Returns:
        A dictionary containing the best UQ methods for each evaluation type.
    """
    format_precision = 3
    best_of_each_metric = {}
    # For each evaluation type and each subset, show the performances of the evaluated UQ metrics
    for eval_type in [
        EvalType.RankingBased,
        EvalType.CalibrationBased,
        EvalType.ProperScoringRules,
    ]:
        eval_df = eval_dict[eval_type]["df_eval"]
        eval_df.drop_duplicates(inplace=True)
        uq_eval_metrics_type = uq_eval_metrics[eval_type]

        # if eval_df is not None and uq_eval_metrics_type is not None:
        if eval_df is not None:

            subsets = ["TEST"] if evaluate_test_only else eval_df["which_set"].unique()

            for subset in subsets:
                # Filter the computed evaluation metrics by subset
                subset_df = eval_df.loc[eval_df["which_set"] == subset].reset_index(
                    drop=True
                )
                columns_to_style = list(
                    set(subset_df.columns) - set(["which_set", "uq_metric"])
                )

                # Display the evaluation score for each evaluation type and subset, highlighting the best values for each metric in green
                if eval_type == EvalType.RankingBased:
                    metric_selection = [
                        "AUC Difference: UQ vs. True Error",
                        "Spearman Correlation",
                    ]

                    reorder_columns = (
                        ["uq_metric", "which_set"]
                        + metric_selection
                        + list(
                            set(subset_df.columns)
                            - set(metric_selection)
                            - set(["uq_metric", "which_set"])
                        )
                    )
                    subset_df = subset_df[reorder_columns]
                    if best_methods_bs is not None:
                        best_of_each_metric.update(
                            {f"{eval_type}_{subset}": best_methods_bs[eval_type]}
                        )
                        max_spearman = subset_df.loc[
                            subset_df["uq_metric"].eq(best_methods_bs[eval_type][0]), :
                        ].index.values[0]
                        min_auc = subset_df.loc[
                            subset_df["uq_metric"].eq(best_methods_bs[eval_type][-1]), :
                        ].index.values[0]
                    else:
                        min_auc = subset_df[metric_selection[0]].idxmin()
                        max_spearman = subset_df[metric_selection[1]].idxmax()
                        best_of_each_metric.update(
                            {
                                f"{eval_type}_{subset}": list(
                                    set(
                                        [
                                            subset_df.loc[idx, ["uq_metric"]].values[0]
                                            for idx in [min_auc, max_spearman]
                                        ]
                                    )
                                )
                            }
                        )

                    subset_df = subset_df.rename(
                        columns={"uq_metric": "UQ Method", "which_set": "Subset"}
                    )
                    # Apply the highlight_max function to the specified columns
                    styled_df = subset_df.style.apply(
                        lambda s: highlight_(
                            s,
                            list(
                                set(columns_to_style)
                                - set(["AUC Difference: UQ vs. True Error"])
                            ),
                            max,
                        ),
                        axis=0,
                    )

                    styled_df = styled_df.apply(
                        lambda s: highlight_(s, metric_selection[0], min), axis=0
                    )

                    styled_df = (
                        styled_df.apply(
                            color_row, color="lightgreen", row_index=min_auc, axis=1
                        )
                        .apply(
                            color_row,
                            color="lightgreen",
                            row_index=max_spearman,
                            axis=1,
                        )
                        .format(precision=format_precision)
                    )

                elif eval_type == EvalType.ProperScoringRules:
                    metric_selection = ["NLL"]

                    if best_methods_bs is not None:
                        best_of_each_metric.update(
                            {f"{eval_type}_{subset}": best_methods_bs[eval_type]}
                        )
                        min_nll = subset_df.loc[
                            subset_df["uq_metric"].eq(best_methods_bs[eval_type][0]), :
                        ].index.values[0]
                    else:
                        min_nll = subset_df[metric_selection[0]].idxmin()
                        best_of_each_metric.update(
                            {
                                f"{eval_type}_{subset}": subset_df.loc[
                                    min_nll, ["uq_metric"]
                                ].values[0]
                            }
                        )

                    reorder_columns = (
                        ["uq_metric", "which_set"]
                        + metric_selection
                        + list(
                            set(subset_df.columns)
                            - set(metric_selection)
                            - set(["uq_metric", "which_set"])
                        )
                    )
                    subset_df = subset_df[reorder_columns]

                    subset_df = subset_df.rename(
                        columns={"uq_metric": "UQ Method", "which_set": "Subset"}
                    )
                    # Apply the highlight_max function to the specified columns
                    styled_df = subset_df.style.apply(
                        lambda s: highlight_(s, columns_to_style, min), axis=0
                    )
                    styled_df = styled_df.apply(
                        color_row, color="lightgreen", row_index=min_nll, axis=1
                    ).format(precision=format_precision)
                elif eval_type == EvalType.CalibrationBased:
                    metric_selection = ["MACE"]

                    if best_methods_bs is not None:
                        best_of_each_metric.update(
                            {f"{eval_type}_{subset}": best_methods_bs[eval_type]}
                        )
                        min_mace = subset_df.loc[
                            subset_df["uq_metric"].eq(best_methods_bs[eval_type][0]), :
                        ].index.values[0]
                    else:
                        min_mace = subset_df[metric_selection[0]].idxmin()
                        best_of_each_metric.update(
                            {
                                f"{eval_type}_{subset}": subset_df.loc[
                                    min_mace, ["uq_metric"]
                                ].values[0]
                            }
                        )

                    reorder_columns = (
                        ["uq_metric", "which_set"]
                        + metric_selection
                        + list(
                            set(subset_df.columns)
                            - set(metric_selection)
                            - set(["uq_metric", "which_set"])
                        )
                    )
                    subset_df = subset_df[reorder_columns]

                    subset_df = subset_df.rename(
                        columns={"uq_metric": "UQ Method", "which_set": "Subset"}
                    )
                    # Apply the highlight_max function to the specified columns
                    styled_df = subset_df.style.apply(
                        lambda s: highlight_(s, columns_to_style, min), axis=0
                    )
                    styled_df = styled_df.apply(
                        color_row, color="lightgreen", row_index=min_mace, axis=1
                    ).format(precision=format_precision)

                if output_path is not None:
                    os.makedirs(Path(output_path) / subset, exist_ok=True)
                    subset_df.to_csv(
                        Path(output_path)
                        / subset
                        / f"{subset}_{eval_type}_summary.csv",
                        index=False,
                    )

                if is_running_in_jupyter_notebook() and display_outputs:
                    display(styled_df)

    return best_of_each_metric


# Function to apply a style to the rows
def color_row(row, color, row_index):
    return [f"background-color: {color}" if row.name == row_index else "" for _ in row]


def highlight_(s, columns_to_style, fun):
    if s.name in columns_to_style:
        if fun == max:
            is_ = s == s.max()
        elif fun == min:
            is_ = s == s.min()
        return ["font-weight: bold" if v else "" for v in is_]
    return ["" for _ in s]


def scatter_plot_inference(
    data: pd.DataFrame,
    inference_column: str,
    output_path: Optional[Union[str, Path]] = None,
    display_outputs: bool = True,
):
    """Plot predictions vs. target values grouped by UQ-based bins.

    It will plot `n_bins` scatterplots depending on the number of UQ-based bins.

    It is assumed that the predictions and target values belong to the held-out
    test dataset.

    Args:
        data (pd.DataFrame):
            Input dataframe containing original model's predictions and target values.
        inference_column (str):
            Name of the column in the input dataframe containing the UQ metric values
            used for inference (i.e., with which predictions have been binned).
        output_path (Path, str, None):
            Output directory path where to store the plot. Default: None (plot is not saved).
        display_outputs (bool):
            Whether to display the plots to screen. Only works if running in a
            JupyterNotebook cell. Default: True.
    """
    uq_bins = data[inference_column].unique()

    fig, axs = plt.subplots(
        1, len(uq_bins), figsize=(4 * len(uq_bins), 6), sharex=True, sharey=True
    )

    for i, uq_level in enumerate(uq_bins):
        # Filter data based on UQ-based bins
        df_plot = data.loc[data[inference_column] == uq_level]
        # Plot predictions vs labels
        sns.scatterplot(df_plot, x="predictions", y="labels", alpha=0.5, ax=axs[i])
        # sns.regplot(data, x="predictions", y="labels", scatter=False, line_kws={"alpha": 0.5}, color=".01", ax=axs[i])

        # Plot identity line
        axs[i].axline(
            (df_plot["labels"].min(), df_plot["labels"].min()),
            slope=1,
            ls="--",
            c="black",
            linewidth=1,
            alpha=0.5,
        )

        axs[i].set_title(f"UQ Level = {uq_level}")

    fig.suptitle(f"Preds vs. Labels by {inference_column}")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            Path(output_path)
            / f"scatter_plot_inference_uq_{inference_column.split(': ')[-1]}",
            transparent=True,
            bbox_inches="tight",
        )

    if is_running_in_jupyter_notebook() and display_outputs:
        fig.show()
    else:
        plt.close(fig)
