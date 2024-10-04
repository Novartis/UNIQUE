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

import ast
import itertools
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

yaml = YAML()

from unique import error_models, evaluation, input_type, uncertainty, uq_metric_factory
from unique.error_models import UniqueErrorModel
from unique.evaluation import EvalType
from unique.input_type import UniqueInputType
from unique.utils.uncertainty_utils import AnalyticsMode, CustomFormatter

CONFIG_FIELDS = (
    "data_path",
    "output_path",
    "id_column_name",
    "labels_column_name",
    "predictions_column_name",
    "which_set_column_name",
    "model_name",
    "problem_type",
    "mode",
    "inputs_list",
    "error_models_list",
    "individual_plots",
    "summary_plots",
    "save_plots",
    "evaluate_test_only",
    "display_outputs",
    "n_bootstrap",
    "verbose",
)

DATA_FORMATS = {
    ".csv": pd.read_csv,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".json": pd.read_json,
}

PROBLEM_TYPES = ("classification", "regression")


@dataclass
class Pipeline:
    """Concatenate individual UNIQUE modules together to be run in a pipeline fashion.

    Args:
        data (Path, str, pd.DataFrame):
            Either the path to the input dataframe to use or the already-loaded
            dataframe directly. The input dataframe should contain all of
            the available datapoints from each subset.
        inputs_list (Sequence[UniqueInputType], Sequence[Dict[str, Any]]):
            Either a list of already-initialized UNIQUE input types or a
            list of dictionaries specifying the parameters for each UNIQUE
            input type to initialize.
        output_path (Path, str):
            Output directory where to save the results from the UNIQUE pipeline.
        mode (str):
            Analysis mode for sum of variances and distance and error models.
        id_column_name (str):
            Name of the column in the input dataframe containing each datapoint's
            identifier (e.g., unique compound name). If the datapoints do not have
            a unique identifier or there are duplicates, you use `index`.
            Default: "index".
        labels_column_name (str):
            Name of the column in the input dataframe containing the target values
            which the original model is supposed to predict. Default: "labels".
        predictions_column_name (str):
            Name of the column in the input dataframe containing the predictions
            from the original model. Default: "predictions".
        which_set_column_name (str):
            Name of the column in the input dataframe containing the indication of
            which subset each datapoint belongs to: "TRAIN", "TEST", "CALIBRATION".
            Default: "which_set".
        model_name (str):
            Name of the original model used to predict the labels.
        problem_type (str):
            Type of problem/task the model solves. Either "classification" or "regression".
            Default: "regression".
        error_models_list (Sequence[UniqueErrorModel], Sequence[Dict[str, Any]], None):
            Either a list of already-initialized UNIQUE error models or a list of
            dictionaries specifying the parameters for each UNIQUE error model to
            initialize and use. Default: None.
        evaluate_test_only (bool):
            Whether to evaluate the UQ metrics on the test set only or using all
            the available data. Default: True.
        n_bootstrap (Tuple[int]):
            Number of bootstrapping replicas for UQ methods evaluation. Default: (500,)
        individual_plots (bool):
            Whether to plot and show all the individual metric plots from the UQ
            metrics evaluation step. Default: True.
        summary_plots (bool):
            Whether to plot and show the summary UQ metric plots from the UQ metrics
            evaluation step. Default: True.
        save_plots (bool):
            Whether to save the plotted evaluation plots in the output directory. Default: True.
        display_outputs (bool):
            Whether to display the summary tables and plots (either individual,
            summary or both, see `summary_plots` and `individual_plots`).
            Display only works if `Pipeline.fit()` is called in a JupyterNotebook. Default: True.
        verbose (bool):
            If True, the logger will output DEBUG level messages. Default: False.
    """

    data: Union[Path, str, pd.DataFrame]
    inputs_list: Union[Sequence[UniqueInputType], Sequence[Dict[str, Any]]]
    output_path: Union[Path, str]
    mode: Literal["compact", "extended", "full"] = "compact"
    id_column_name: str = "index"
    labels_column_name: str = "labels"
    predictions_column_name: str = "predictions"
    which_set_column_name: str = "which_set"
    model_name: str = "model_name"
    problem_type: str = "regression"
    error_models_list: Optional[
        Union[Sequence[UniqueErrorModel], Sequence[Dict[str, Any]]]
    ] = None
    evaluate_test_only: bool = True
    n_bootstrap: Union[int, Tuple[int]] = (500,)
    individual_plots: bool = True
    summary_plots: bool = True
    save_plots: bool = True
    display_outputs: bool = True
    verbose: bool = False

    _config_path: Optional[Union[Path, str]] = None
    _bootstrap: bool = True
    _best_uq: Optional[Dict[str, str]] = None
    _no_calibration: bool = False
    _sum_var_dist_description: Optional[str] = None

    def __post_init__(self):
        # Initialize logging
        self._init_logger()

        self.logger.info(" UNIQUE - INITIALIZING PIPELINE ".center(80, "*"))
        if self._config_path is not None:
            self.logger.info(
                f"Loaded Pipeline configuration from: '{Path(self._config_path).name}'"
            )

        self.logger.debug(f"Problem type: {self.problem_type}")
        self.logger.debug(
            f"Analysis mode: {self.mode.lower()} ({AnalyticsMode.EXPLAINATION[self.mode.lower()]})"
        )

        # Load data if not already loaded
        # if isinstance(self.data, (Path, str)):
        self.data = self.load_data(data_path=self.data)

        # Perform sanity checks
        self._validate_params()

        # Store all UQ metrics
        self.all_metrics = {}

        # Initialize inputs
        self._initialize_inputs()
        self.logger.debug(
            f"Output files stored in .../{self.output_path.relative_to(self.output_path.parent.parent)}"
        )

    @classmethod
    def from_config(cls, config_path: Union[Path, str]):
        """Configure Pipeline from a yaml config file."""
        # Load config file
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"""Provided config file does not exist: "{config_path}"."""
            )
        # logging.error

        with open(Path(config_path), "r") as f:
            config = yaml.load(f)

        # Check that all required fields are present
        for field in config.keys():
            if field not in CONFIG_FIELDS:
                raise ValueError(
                    f'Unrecognized field "{field}" in provided config file.'
                    f"Allowed fields: {CONFIG_FIELDS}."
                )

        return cls(
            data=Path(config["data_path"]),
            output_path=Path(config["output_path"]),
            mode=config.get("mode", "compact"),
            id_column_name=config["id_column_name"],
            labels_column_name=config["labels_column_name"],
            predictions_column_name=config["predictions_column_name"],
            which_set_column_name=config["which_set_column_name"],
            model_name=config.get("model_name", "MyModel"),
            problem_type=config["problem_type"],
            inputs_list=config["inputs_list"],
            error_models_list=config.get("error_models_list", None),
            evaluate_test_only=config.get("evaluate_test_only", True),
            n_bootstrap=config.get("n_bootstrap", 500),
            individual_plots=config.get("individual_plots", True),
            summary_plots=config.get("summary_plots", True),
            save_plots=config.get("save_plots", True),
            display_outputs=config.get("display_outputs", True),
            verbose=config.get("verbose", False),
            _config_path=Path(config_path),
        )

    @property
    def best_uq_methods(self) -> Dict[str, Union[str]]:
        """Best UQ methods evaluated with .fit()"""
        if self._best_uq is None:
            raise ValueError("Run .fit() to find the best UQ method")
        return self._best_uq

    def load_data(self, data_path: Union[Path, str, pd.DataFrame]) -> pd.DataFrame:
        """Load dataset as pd.DataFrame from supported files."""

        if isinstance(data_path, (Path, str)):
            data_path = Path(data_path) if isinstance(data_path, str) else data_path
            # logging.error
            if data_path.suffix not in DATA_FORMATS.keys():
                raise ValueError(
                    f'Unrecognized input dataframe extension: "{data_path.suffix}".'
                    f"Supported extensions: {DATA_FORMATS.keys()}."
                )

            self.logger.info(f"Loading data from '{data_path.name}'...")
            # Load using corresponding pd.read_<format>
            self.original_data = DATA_FORMATS[data_path.suffix](data_path)
        elif isinstance(data_path, pd.DataFrame):
            self.original_data = data_path.copy()
            del self.data

        # Check consistency of input dataframe
        columns = [
            self.id_column_name,
            self.labels_column_name,
            self.predictions_column_name,
            self.which_set_column_name,
        ]
        # logging.error
        for col in columns:
            if col not in set(self.original_data.columns):
                raise ValueError(
                    f'Column "{col}" not found in provided dataset.'
                    f"Available columns: {self.original_data.columns}."
                )

        # Clean up
        dataset = self.original_data.copy()
        dataset.rename(
            columns={
                self.id_column_name: "id",
                self.predictions_column_name: "predictions",
                self.labels_column_name: "labels",
                self.which_set_column_name: "which_set",
            },
            inplace=True,
        )
        self.logger.info(f"Dataset with {len(dataset)} entries correctly loaded.")

        return dataset

    def fit(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, pd.DataFrame]]]:
        self.logger.info(" UNIQUE - COMPUTING UQ METHODS ".center(80, "*"))
        start = datetime.now()

        # 1. Compute UQ methods from inputs
        self.logger.info("Computing UQ methods for provided inputs...")
        output = {}
        for input_data in self.inputs:
            output.update(input_data.fit())

        self.base_uq_values = deepcopy(output)

        # 1.1 Compute combination of UQ methods ("transformed")
        self.logger.info(
            "Combining 'base' UQ methods and computing 'transformed' UQ methods..."
        )
        self.transformed_uq_values = self.fit_transformed_uq_metrics(output)
        output.update(self.transformed_uq_values)

        # 2. Train error models
        # 2.1. Generate features sets
        if self.error_models_list is not None:
            self.logger.info("Initializing error models...")
            self.error_model_features = self._generate_error_models_features()

            # Initialize error models
            self._initialize_error_models()

            self.error_model_output_values = {}
            self.error_model_uq_metrics = {}

            # 2.2 Train error models for each feature set and for each error type
            self.logger.info("Training error models...")
            for em in self.error_models:
                self.logger.debug(f"Training {em} model...")
                em_outputs = em.fit()
                output.update(em_outputs)
                self.error_model_output_values.update(em_outputs)
                # Save as UQ metric itself
                self.error_model_uq_metrics.update(
                    {
                        k: uncertainty.ErrorBasedUncertaintyMetric(
                            input_values=v,
                            which_set=self.data["which_set"].to_numpy(),
                            problem_type=self.problem_type,
                            is_error_model=True,
                            name=k,
                        )
                        for k, v in em_outputs.items()
                    }
                )

            self.all_metrics.update(self.error_model_uq_metrics)

        self.logger.info(f"Collected and computed {len(self.all_metrics)} UQ methods.")

        if self._sum_var_dist_description is not None:
            self.logger.info(f"{self._sum_var_dist_description}")

        # 3. UQ metrics evaluation
        self.logger.info(" UNIQUE - EVALUATING UQ METHODS ".center(80, "*"))
        self.logger.info(
            f"Evaluating and benchmarking {len(self.all_metrics)} UQ methods"
            f'{f" by bootstrapping (n={self.n_bootstrap}) on the test set..." if self._bootstrap else "..."}'
        )
        _list_all_metrics = "\n".join([m for m, _ in self.all_metrics.items()])
        self.logger.debug(
            f"Evaluating {len(self.all_metrics)} methods: \n{_list_all_metrics}"
        )
        self.evaluation_outputs = self.evaluate_uq_metrics(
            metric_values=deepcopy(output),
            individual_plots=self.individual_plots,
            summary_plots=self.summary_plots,
            save_plots=self.save_plots,
            display_outputs=self.display_outputs,
            evaluate_test_only=self.evaluate_test_only,
            output_dir=self.output_path,
        )

        # Log time elapsed to run pipeline
        elapsed = datetime.now() - start
        hrs, remainder = divmod(elapsed.seconds, 3600)
        mins, secs = divmod(remainder, 60)

        self.logger.info(" UNIQUE - END ".center(80, "*"))
        self.logger.info(
            f"Time elapsed: {int(hrs):02}h:{int(mins):02}m:{int(secs):02}s"
        )

        return output, self.evaluation_outputs

    def fit_transformed_uq_metrics(
        self, metrics_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Computes the combinations of different base UQ metrics.

        These so-called "transformed" UQ metrics are combinations of compatible
        "base" UQ metrics, and are themselves considered UQ metrics.

        Args:
            metrics_outputs (Dict[str, np.ndarray]):
                The output of different metrics

        Returns:
            A dict containing the computed transformed UQ metrics along with the
            corresponding column name (as key).
        """

        predictions = self.data["predictions"].to_numpy().astype(np.float64)
        labels = self.data["labels"].to_numpy().astype(np.float64)
        which_set = self.data["which_set"].to_numpy()

        self.transformed_uq_metrics = {}
        transformed_uq_metrics_outputs = {}
        var_values, var_columns = [], []
        converted_variances, uq_columns = [], []

        # TRANSFORMED UQ METRICS
        # 1. Sum of variances
        self.logger.debug("Transformed UQ: Sum of variances... ")
        # Get variance-based metrics' outputs
        if len(self.variance_metrics.keys()) >= 1:
            var_values = np.stack(
                [
                    base_output
                    for base_name, base_output in metrics_outputs.items()
                    if base_name in self.variance_metrics.keys()
                ],
                axis=0,
            )

            var_columns = [
                base_name
                for base_name in metrics_outputs.keys()
                if base_name in self.variance_metrics.keys()
            ]

            # Compute if more than one variance-based metric
            if len(self.variance_metrics.keys()) > 1:
                # Initialize transformed UQ metrics object: SumOfVariances
                sum_var_metric = uq_metric_factory.SumOfVariances(
                    uq_values=var_values,
                )
                transformed_uq_metrics_outputs[sum_var_metric.name] = (
                    sum_var_metric.fit()
                )
                self.transformed_uq_metrics[sum_var_metric.name] = (
                    uncertainty.UQFactoryBasedUncertaintyMetric(
                        input_values=transformed_uq_metrics_outputs[
                            sum_var_metric.name
                        ],
                        which_set=which_set,
                        problem_type=self.problem_type,
                        name=sum_var_metric.name,
                    )
                )

        # Append predictions to the variance values for DiffkNN
        uq_values = (
            np.vstack((var_values, predictions))
            if isinstance(var_values, np.ndarray)
            else np.expand_dims(predictions, 0)
        )
        uq_columns.extend(var_columns)
        uq_columns.append("predictions")

        # Compute combinations of distance-based UQ metrics
        for base_name in self.distance_metrics.keys():
            # 2. Compute DiffkNN
            self.logger.debug(f"Transformed UQ: DiffkNN [{base_name}]...")
            neighbors = self.distance_metrics[base_name].neighbors
            k = self.distance_metrics[base_name].k
            for uq_val, uq_col in zip(uq_values, uq_columns):
                diffknn = uq_metric_factory.DiffkNN(
                    uq_values=uq_val,
                    name=f"Diff{k}NN[{base_name}, {uq_col}]",
                    neighbors=neighbors,
                )
                # Store transformed UQ metric
                transformed_uq_metrics_outputs[diffknn.name] = diffknn.fit()
                self.transformed_uq_metrics[diffknn.name] = (
                    uncertainty.UQFactoryBasedUncertaintyMetric(
                        input_values=transformed_uq_metrics_outputs[diffknn.name],
                        which_set=which_set,
                        is_distance=False,
                        problem_type=self.problem_type,
                        name=diffknn.name,
                    )
                )

        # 3. Convert all distances to variances using CalibratedNLL
        # and use as transformed UQ metrics themselves
        if self.evaluate_test_only and not self._no_calibration:
            comb = {
                k: d.fit()
                for k, d in self.all_metrics.items()
                if not isinstance(d, uncertainty.ModelBasedUncertaintyMetric)
            }
            for base_name, distance_output in comb.items():
                if not isinstance(distance_output, np.ndarray):
                    distance_output = distance_output.input_values

                self.logger.debug(f"Transformed UQ: Distance2Var [{base_name}]...")
                dist2var = uq_metric_factory.DistanceToVariance(
                    uq_values=distance_output,  # distances
                    predictions=predictions,
                    labels=labels,
                    which_set=which_set,
                    name=f"Dist2Var[{base_name}]",
                )
                # Store transformed UQ metric
                dist2var_values = dist2var.fit()
                if isinstance(dist2var_values, str):
                    self.logger.warning(
                        f"{base_name} not converted to variance because non-convergent"
                        f" CNLL algorithm."
                    )
                    self.logger.debug(f"{dist2var_values}")
                    continue
                transformed_uq_metrics_outputs[dist2var.name] = dist2var_values
                self.transformed_uq_metrics[dist2var.name] = (
                    uncertainty.UQFactoryBasedUncertaintyMetric(
                        input_values=dist2var_values,
                        which_set=which_set,
                        problem_type=self.problem_type,
                        name=dist2var.name,
                        supported_evaluation_types=(
                            EvalType.RankingBased,
                            EvalType.CalibrationBased,
                            EvalType.ProperScoringRules,
                        ),
                    )
                )
                converted_variances.append(dist2var_values)
        else:
            self.logger.warning(
                "Distances will not be converted to variances because of evaluation of full dataset."
            )

        # 4. Sum variances and distances (converted to variances)
        # If no variance- or distance-based UQ metrics were computed,
        # Or if only one of them was computed, exit
        if (
            (not isinstance(var_values, np.ndarray) and not converted_variances)
            or (len(var_values) < 2 and not converted_variances)
            or (not isinstance(var_values, np.ndarray) and len(converted_variances) < 2)
        ):
            # Update computed metrics list
            self.all_metrics.update(self.transformed_uq_metrics)

            return transformed_uq_metrics_outputs
        # Else, sum up variances and converted distances
        elif isinstance(var_values, np.ndarray) or converted_variances:
            self.logger.debug("Transformed UQ: Sum of variances and distances...")
            # if isinstance(var_values, np.ndarray):
            #     uq_values = (
            #         np.vstack((var_values, converted_variances))
            #         if converted_variances
            #         else var_values
            #     )
            # else:
            #     uq_values = np.vstack((converted_variances))
            #
            # sum_var_dist_metric = uq_metric_factory.SumOfVariances(
            #     uq_values=uq_values,
            #     name="SumOfVariancesAndDistances",
            # )
            ################################
            # Sum of variances and distances dispatcher
            svd_dispatcher = uq_metric_factory.SumVarDistDispatcher(
                transformed_uq_metrics=self.transformed_uq_metrics,
                variance_values=var_values,
                mode=self.mode,
                _logger=self.logger,
            )
            sum_var_dist_results = svd_dispatcher.fit()
            transformed_uq_metrics_outputs.update(sum_var_dist_results)

            for svd_name, svd_value in sum_var_dist_results.items():
                self.transformed_uq_metrics[svd_name] = (
                    uncertainty.UQFactoryBasedUncertaintyMetric(
                        input_values=svd_value,
                        which_set=which_set,
                        problem_type=self.problem_type,
                        supported_evaluation_types=(
                            EvalType.RankingBased,
                            EvalType.CalibrationBased,
                            EvalType.ProperScoringRules,
                        ),
                    )
                )

            self._sum_var_dist_description = svd_dispatcher.description
            ##############################
            # transformed_uq_metrics_outputs[sum_var_dist_metric.name] = (
            #     sum_var_dist_metric.fit()
            # )

            # # UM factory for transformed UQ
            # self.transformed_uq_metrics[sum_var_dist_metric.name] = (
            #     uncertainty.UQFactoryBasedUncertaintyMetric(
            #         input_values=transformed_uq_metrics_outputs[
            #             sum_var_dist_metric.name
            #         ],
            #         which_set=which_set,
            #         problem_type=self.problem_type,
            #         name=sum_var_dist_metric.name,
            #         supported_evaluation_types=(
            #             EvalType.RankingBased,
            #             EvalType.CalibrationBased,
            #             EvalType.ProperScoringRules,
            #         ),
            #     )
            # )
        self.all_metrics.update(self.transformed_uq_metrics)

        return transformed_uq_metrics_outputs

    def _initialize_inputs(self):
        """Initialize ``UniqueInputType`` objects from input data."""

        self.logger.info("UQ inputs initialization...")
        # If inputs are provided from config file they will be a list of dicts,
        # with each dict being: key=UniqueInputType names, and values=kwargs (as dict themselves)

        # with evaluation on the entire dataset (evaluate_test_only = False) we combine train and calibration set
        if not self.evaluate_test_only:
            self.logger.debug(
                "Merging TRAIN and CALIBRATION subsets because of evaluation on full dataset."
            )
            self.data["which_set"] = self.data["which_set"].map(
                {"CALIBRATION": "TRAIN", "TEST": "TEST", "TRAIN": "TRAIN"}
            )

        if all(isinstance(ins, Dict) for ins in self.inputs_list):
            self.inputs = []
            for inputs_dict in self.inputs_list:
                for it, kw in inputs_dict.items():
                    self.logger.debug(f'{kw["column_name"]} initialization...')
                    input_obj = input_type.__dict__[it](
                        input_values=self.data[kw.get("column_name")],
                        which_set=self.data["which_set"].to_numpy(),
                        problem_type=self.problem_type,
                        **kw,
                    )
                    self.inputs.append(input_obj)
        # Otherwise, if inputs are provided "manually", they should be a
        # list of already-initialized UniqueInputType objects
        else:
            self.inputs = self.inputs_list

        # Get the list of UQ metrics to be computed from each input
        self.uq_metrics = {}
        for input_data in self.inputs:
            base_uq_metrics = {
                f"{metric.name}[{input_data.column_name}]": metric
                for metric in input_data.metrics_to_compute.values()
            }

            # ``uq_metrics`` holds both the output UQ metric's column names and the metric objects
            self.uq_metrics.update(base_uq_metrics)

        # Get the names of the input columns that are features of the data - i.e., FeaturesInputType
        self.features_column_names = [
            input_data.column_name
            for input_data in self.inputs
            if input_data.type == "data_based"
        ]

        # Get the list of possible UQ metrics combinations given the metrics to compute
        # These are the "transformed" UQ metrics (``SumOfVariance``, ``SumOfVariancesAndDistances``, etc.)
        self.variance_metrics = {
            metric_name: metric
            for metric_name, metric in self.uq_metrics.items()
            if metric.is_variance
        }

        self.distance_metrics = {
            metric_name: metric
            for metric_name, metric in self.uq_metrics.items()
            if metric.is_distance
        }

        self.all_metrics.update(self.uq_metrics)

        # logging input
        log_data = (
            f"\nid_column_name: {self.id_column_name}"
            f"\nlabels_column_name: {self.labels_column_name}"
            f"\npredictions_column_name: {self.predictions_column_name}"
            f"\nwhich_set_column_name: {self.which_set_column_name}"
            f"\nmodel_name: {self.model_name}"
        )
        self.logger.debug(f"Input data summary: {log_data}")

        log_uq_input = "\n".join([f"{i+1}. {inp}" for i, inp in enumerate(self.inputs)])
        self.logger.info(f"UQ inputs summary: \n{log_uq_input}")

        # logging error models
        if self.error_models_list is not None:
            log_error_models = "\n".join(
                [
                    f"{i+1}. {list(k.keys())[0]}"
                    for i, k in enumerate(self.error_models_list)
                ]
            )
            self.logger.info(f"Selected error model(s): \n{log_error_models}")

    def _initialize_error_models(self):
        """Initialize ``UniqueErrorModel``objects from input arguments."""
        # If error_models are provided from config file, they will be a list of dicts,
        # with each dict being: key=UniqueErrorModel name, and value=kwargs (as dict themselves)
        if all(isinstance(em, Dict) for em in self.error_models_list):
            self.error_models = [
                error_models.__dict__[em](
                    output_dir=self.output_path,
                    which_set=self.data["which_set"].to_numpy(),
                    predictions=self.data["predictions"].to_numpy().astype(np.float64),
                    labels=self.data["labels"].to_numpy().astype(np.float64),
                    input_features=self.error_model_features,
                    **kw,
                )
                for error_model in self.error_models_list
                for em, kw in error_model.items()
            ]
        # Otherwise, if error models are provided "manually", they should be a list
        # of already instantiated UniqueErrorModel objects
        else:
            self.error_models = self.error_models_list

    def _generate_error_models_features(self) -> Dict[str, np.ndarray]:
        """Generate different input feature sets for error model training based on computed UQ metrics and input data.

        We generate 3 separate training sets, from the combination of different inputs and UQ metrics:
            1. Data features + UQ metrics + Predictions
            2. UQ metrics + Predictions
            3. Transformed UQ metrics + Predictions

        We then proceed to train 3 different models (one for each training set) for each
        requested error model.
        """

        def convert_to_numpy(x: pd.Series) -> np.ndarray:
            if isinstance(x[0], (list, np.ndarray)):
                return np.stack(x.tolist())
            else:
                return x.to_numpy()

        error_features = {}

        # Prepare error model features
        self.logger.info("Preparing error models inputs...")
        base_features = {}
        for feature_name in self.features_column_names:
            try:
                base_features[feature_name] = self.data[feature_name].apply(
                    ast.literal_eval
                )
                base_features[feature_name] = convert_to_numpy(
                    base_features[feature_name]
                )
            except:
                base_features[feature_name] = convert_to_numpy(self.data[feature_name])

        base_uq_values = np.vstack(list(self.base_uq_values.values())).transpose((1, 0))
        preds = np.expand_dims(
            self.data["predictions"].to_numpy().astype(np.float64), -1
        )
        if self.transformed_uq_values:
            transformed_uq_values = np.vstack(
                list(self.transformed_uq_values.values())
            ).transpose((1, 0))

        # Create features sets
        for feature_name, feature_values in base_features.items():
            error_features[f"{feature_name}+UQmetrics+predictions"] = np.hstack(
                (feature_values, base_uq_values, preds)
            )

        error_features["UQmetrics+predictions"] = np.hstack((base_uq_values, preds))
        if self.transformed_uq_values:
            error_features["transformedUQmetrics+predictions"] = np.hstack(
                (transformed_uq_values, preds)
            )

        return error_features

    def _validate_params(self):
        # logging.error
        """Check whether all required inputs and variables have been correctly provided."""

        ### Mode ###
        if self.mode not in AnalyticsMode.EXPLAINATION.keys():
            spacer = "\n\t\t"
            spacer = spacer.join(
                [f"- '{k}': {v}" for k, v in AnalyticsMode.EXPLAINATION.items()]
            )
            raise ValueError(
                f'Supported analysis modalities: {spacer}. Got: "{self.mode}".'
            )

        ### Data ###
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(
                f"Supported problem types: {PROBLEM_TYPES}. "
                f'Got: "{self.problem_type}".'
            )
        if len(self.data) == 0:
            raise ValueError("Please, provide a non-empty dataframe.")
        # Check for duplicate IDs
        if all(self.data.duplicated("id")):
            raise ValueError(
                f"Found duplicated IDs in the dataset. UNIQUE can only handle "
                f"datasets with unique ID values. Please, check your dataset for "
                f'duplicate datapoints or use "index" as `id_column_name`.'
            )
        if self.data["which_set"].nunique() < 1:
            raise ValueError(
                f"Provide at least 2 subsets of data to enable the correct "
                f'computation of UQ metrics - e.g., ["TRAIN", "TEST", "CALIBRATION"].'
                f'\nGot: "{self.data["which_set"].unique()}".'
            )

        # Evaluation on both train and test set
        if not self.evaluate_test_only:
            self._bootstrap = False
            self.logger.warning(
                "Evaluation set on entire data set. If any, TRAIN and CALIBRATION subsets will be merged."
                "\n-Bootstrapping on TEST subset will NOT be run."
                "\n-Distances will NOT be converted to variances."
            )

        # no calibration set
        if "CALIBRATION" not in self.data["which_set"].unique():
            self.logger.warning(
                "No CALIBRATION subset provided."
                "\n-Distances will not be converted to variances."
                "\n-Error models will be only trained on TRAIN samples."
            )
            self._no_calibration = True

        # disable bootstrapping when test set length < 50
        test_set_len = len(self.data.loc[self.data["which_set"].eq("TEST"), :])
        if test_set_len < 50:
            self.logger.warning(
                f"TEST subset length < 50 samples. Bootstrapping on TEST set disabled"
            )
            self._bootstrap = False

        # Input types
        if all(isinstance(inputs, UniqueInputType) for inputs in self.inputs_list):
            # Provided "manually" as list of UniqueInputType (already initialized)
            for input_data in self.inputs_list:
                if not isinstance(input_data, UniqueInputType):
                    raise ValueError(
                        f"Provide a valid `UniqueInputType` object. "
                        f'Got: "{input_data}".'
                    )
                if len(input_data.input_values) != len(self.data):
                    raise ValueError(
                        f"Make sure number of input values is the same as the "
                        f"number of rows in the dataset. "
                        f"Got {len(input_data.input_values)} input values and "
                        f"{len(self.data)} rows in the dataset."
                    )
            # Provided from config file as dict of UniqueInputType names and keyword-arguments
        elif all(isinstance(ins, Dict) for ins in self.inputs_list):
            for inputs in self.inputs_list:
                for it, kw in inputs.items():
                    if it not in input_type.__dict__["__all__"]:
                        raise ValueError(
                            f"Provide a valid input type name. "
                            f'Available input types: {input_type.__dict__["__all__"]}. '
                            f'Got: "{it}".'
                        )
                    if kw.get("column_name") is None:
                        raise ValueError(
                            f"Please, provide a `column_name` parameter for each "
                            f"input type, specifying the corresponding name of the "
                            f"column in the dataset."
                        )
                    if kw["column_name"] not in self.data.columns:
                        raise ValueError(
                            f'Column "{kw["column_name"]}" not found in dataset. '
                            f"Provide an existing column name under the `column_name` "
                            f'parameter for the input "{it}". '
                            f"Columns in dataset: {list(self.data.columns)}."
                        )

        self.n_bootstrap = (
            self.n_bootstrap[0]
            if isinstance(self.n_bootstrap, tuple)
            else self.n_bootstrap
        )

    def evaluate_individual_uq_metric(
        self,
        metric_values: np.ndarray,
        metric_obj: uncertainty.UniqueUncertaintyMetric,
        metric_name: str,
        perf_metric: str,
        individual_plots: bool = False,
        save_plots: bool = True,
        evaluate_test_only: bool = True,
        bootstrap_test_set: bool = True,
        output_dir: Optional[Union[Path, str]] = None,
        display_outputs: bool = True,
        best_uq: bool = False,
    ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]:
        """Evaluate a single (precomputed) UQ metric with all its supported evaluation types.

        Args:
            metric_values (np.ndarray):
                Array containing the computed UQ values.
            metric_obj (uncertainty.UniqueUncertaintyMetric):
                The UQ metric object. Used to retrieve the supported evaluation types.
            metric_name (str):
                Name of the UQ metric (as "metric_name[column_name]").
            perf_metric (str):
                Predictive performance metric used to evaluate the UQ metric's goodness.
            individual_plots (bool):
                Whether to visualize each supported evaluation metric's plot.
                Note: if True, plots will be shown for each UQ metric and data subset
                available ("TEST" only if ``evaluate_test_only=True``). Default: False.
            save_plots (bool):
                Whether to save the plots. Default: True.
            evaluate_test_only (bool):
                Whether to compute the evaluation metrics on the test set only. Default: True.
            bootstrap_test_set (bool):
                Whether to perform bootstrapping on the test set. Default: True.
            output_dir (Path, str, None):
                Output directory where to save the plots if ``save_plots=True``. Default: None.
            display_outputs (bool):
                Whether to display the plots to screen. Only works if running in a
                JupyterNotebook cell. Default: True.
            best_uq (bool):
                Whether the UQ method being evaluated has been identified as the best performing
                for the specific evaluation type. Default: False.

        Returns:
            A dictionary containing, for each evaluation type (as keys), the evaluation
            outputs for a single UQ metric (as pd.DataFrames).
        """
        # Variables where to store the evaluation results (one per evaluation type)
        eval_outputs = {
            EvalType.RankingBased: defaultdict(pd.DataFrame),
            EvalType.ProperScoringRules: defaultdict(pd.DataFrame),
            EvalType.CalibrationBased: defaultdict(pd.DataFrame),
        }

        eval_bs_outputs = {
            EvalType.RankingBased: defaultdict(pd.DataFrame),
            EvalType.ProperScoringRules: defaultdict(pd.DataFrame),
            EvalType.CalibrationBased: defaultdict(pd.DataFrame),
        }

        self.evaluation_uq_metrics = {
            EvalType.RankingBased: None,
            EvalType.ProperScoringRules: None,
            EvalType.CalibrationBased: None,
        }

        # Prepare data
        predictions = self.data["predictions"].to_numpy().astype(np.float64)
        labels = self.data["labels"].to_numpy().astype(np.float64)

        output_dir = Path(output_dir) / "evaluation_plots" if save_plots else None

        output_dir = output_dir / "best_methods" if best_uq else output_dir

        # Run the supported evaluation types for each data subset
        subsets = ["TEST"] if evaluate_test_only else np.unique(metric_obj.which_set)

        for subset in subsets:

            subset_ids = np.where(metric_obj.which_set == subset)

            for eval in metric_obj.supported_evaluation_types:
                evaluator = evaluation.__dict__[eval](
                    uq_values=metric_values[subset_ids],
                    predictions=predictions[subset_ids],
                    labels=labels[subset_ids],
                    uq_metric_name=metric_name,
                    is_error_model=metric_obj.is_error_model,
                    is_variance=metric_obj.is_variance,
                    is_distance=metric_obj.is_distance,
                    which_set_label=subset,
                    problem_type=self.problem_type,
                    bootstrap_test_set=bootstrap_test_set,
                    perf_metric=perf_metric,
                    individual_plots=individual_plots,
                    output_dir=output_dir,
                    n_bootstrap=self.n_bootstrap,
                    display_outputs=display_outputs,
                    logger=self.logger,
                )

                # self.logger.debug(f"UQ eval: {eval} @ {subset}")
                self.logger.debug(
                    f"Running {eval} for {metric_name} UQ method on '{subset.lower()}' set..."
                )
                if individual_plots and display_outputs:
                    self.logger.debug(
                        f"Saving and displaying individual plots for {metric_name} evaluated on '{subset.lower()}' set..."
                    )
                elif individual_plots and not display_outputs:
                    self.logger.debug(
                        f"Saving individual plots for {metric_name} evaluated on '{subset.lower()}' set..."
                    )

                output, bs_output = evaluator.fit()

                for k in output.keys():
                    eval_outputs[eval][k] = pd.concat(
                        (eval_outputs[eval][k], output[k])
                    )
                    if bootstrap_test_set and k == "df_eval":
                        eval_bs_outputs[eval][k] = pd.concat(
                            (eval_bs_outputs[eval][k], bs_output[k])
                        )

                self.evaluation_uq_metrics[eval] = evaluator.uq_eval_metrics

        return eval_outputs, eval_bs_outputs

    def evaluate_uq_metrics(
        self,
        metric_values: Dict[str, np.ndarray],
        perf_metric: Optional[str] = None,
        individual_plots: bool = False,
        summary_plots: bool = True,
        save_plots: bool = True,
        display_outputs: bool = True,
        evaluate_test_only: bool = True,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Pipeline function to evaluate all UQ metrics available.

        Args:
            metric_values (Dict[str, np.ndarray]):
                Dictionary containing the computed UQ values along with the UQ metric name.
            perf_metric (str, None):
                Predictive performance metric used to evaluate the UQ metric's goodness.
            individual_plots (bool):
                Whether to visualize each supported evaluation metric's plot.
                Note: if True, plots will be shown for each UQ metric and data subset
                available ("TEST" only if ``evaluate_test_only=True``). Default: False.
            summary_plots (bool):
                Whether to visualize the overall UQ metrics' evaluation scores and plots.
                Default: True.
            save_plots (bool):
                Whether to save the plots. Default: True.
            display_outputs (bool):
                Whether to display the outputs to screen. Only works if running in a
                JupyterNotebook cell. Default: True.
            evaluate_test_only (bool):
                Whether to compute the evaluation metrics on the test set only. Default: True.
            output_dir (Path, str, None):
                Output directory where to save the plots if ``save_plots=True``. Default: None.

        Returns:
            A dictionary containing, for each evaluation type (as keys), the evaluation
            outputs for all metrics (concatenated in pd.DataFrames).
        """
        if individual_plots:
            self.logger.info(
                "Plotting all individual plots for each UQ method may take quite some time. Consider disabling `individual_plots` to False in the configuration file for a smoother and faster experience."
            )

        # Dataframe with evaluation metrics
        eval_dfs = {
            EvalType.RankingBased: defaultdict(pd.DataFrame),
            EvalType.ProperScoringRules: defaultdict(pd.DataFrame),
            EvalType.CalibrationBased: defaultdict(pd.DataFrame),
        }

        eval_bs_dfs = {
            EvalType.RankingBased: defaultdict(pd.DataFrame),
            EvalType.ProperScoringRules: defaultdict(pd.DataFrame),
            EvalType.CalibrationBased: defaultdict(pd.DataFrame),
        }

        # Execute evaluation for each uq_metric
        for i, (metric_name, metric_obj) in enumerate(self.all_metrics.items()):
            self.logger.debug(f"UQ evaluation: {metric_name}")
            eval_outputs, eval_bs_outputs = self.evaluate_individual_uq_metric(
                metric_values[metric_name],
                metric_obj,
                metric_name,
                perf_metric=perf_metric,
                individual_plots=individual_plots,
                save_plots=save_plots,
                evaluate_test_only=evaluate_test_only,
                bootstrap_test_set=self._bootstrap,
                output_dir=output_dir,
                display_outputs=display_outputs,
            )

            for eval_name, eval_out in eval_outputs.items():
                for k in eval_out.keys():
                    eval_dfs[eval_name][k] = pd.concat(
                        (eval_dfs[eval_name][k], eval_out[k])
                    ).reset_index(drop=True)

            for eval_bs_name, eval_bs_out in eval_bs_outputs.items():
                for k in eval_bs_out.keys():
                    eval_bs_dfs[eval_bs_name][k] = pd.concat(
                        (eval_bs_dfs[eval_bs_name][k], eval_bs_out[k])
                    ).reset_index(drop=True)
            if (i + 1) % np.floor(len(self.all_metrics) / 3) == 0:
                self.logger.info(
                    f"Evaluated {i + 1} UQ methods out of {len(self.all_metrics)}{'...' if i+1<len(self.all_metrics) else '.'}"
                )
            elif (i + 1) >= len(self.all_metrics):
                self.logger.info(
                    f"Evaluated {i + 1} UQ methods out of {len(self.all_metrics)}."
                )

        if self._bootstrap:
            best_methods = self._stat_analysis_bootstrapping(eval_dfs, eval_bs_dfs)
        else:
            best_methods = None

        self.logger.info("Generating summary tables...")
        best_method_summary = evaluation.get_summary_tables(
            eval_dict=eval_dfs,
            uq_eval_metrics=self.evaluation_uq_metrics,
            evaluate_test_only=evaluate_test_only,
            output_path=Path(output_dir) / "summary",
            display_outputs=display_outputs,
            best_methods_bs=best_methods,
        )

        self.logger.info(
            f"Summary evaluation tables saved to: .../{Path(output_dir).relative_to(Path(output_dir).parent.parent)}/summary."
        )

        self._best_uq = best_method_summary

        if evaluate_test_only:
            c = {
                k.split("_")[0]: [v] if not isinstance(v, list) else v
                for k, v in best_method_summary.items()
                if "TEST" in k
            }
            best_uqs = {
                item: self.all_metrics[item] for _, l in c.items() for item in l
            }

            # Always generate individual plots of the best UQ methods
            for metric_name, metric_obj in best_uqs.items():
                self.evaluate_individual_uq_metric(
                    metric_values[metric_name],
                    metric_obj,
                    metric_name,
                    perf_metric=perf_metric,
                    individual_plots=True,
                    save_plots=True,
                    bootstrap_test_set=False,
                    evaluate_test_only=evaluate_test_only,
                    output_dir=output_dir,
                    display_outputs=display_outputs,
                    best_uq=True,
                )
        else:
            self.logger.warning(
                "Individual plots for best UQ methods are DISABLED. To enable them, set "
                "`evaluate_test_only=True`"
            )

        if summary_plots:
            if perf_metric is None:
                perf_metric = "MAE" if self.problem_type == "regression" else "BA"

            self.logger.info("Generating summary plots...")
            evaluation.get_summary_plots(
                eval_dict=eval_dfs,
                eval_bs_dict=eval_bs_dfs,
                bins_labels="Bins",
                perf_metric=perf_metric,
                evaluate_test_only=evaluate_test_only,
                bootstrap_test_set=self._bootstrap,
                output_path=Path(output_dir) / "summary",
                display_outputs=display_outputs,
            )

            self.logger.info(
                f"Summary plots saved to: .../{Path(output_dir).relative_to(Path(output_dir).parent.parent) / 'summary'}."
            )

        # Print best UQ methods per evaluation type
        best_method_str = [
            f"{i+1}. [{k.split('_')[-1]}] {k.split('_')[0]}: {v if isinstance(v, str) else v[0]}"
            for i, (k, v) in enumerate(best_method_summary.items())
        ]
        formatted_best_method = "\n".join(best_method_str)
        self.logger.info(
            f"Summary of best UQ method for each UQ evaluation type:\n"
            f"{formatted_best_method}"
        )

        return eval_dfs

    def _stat_analysis_bootstrapping(
        self,
        eval_dfs: Dict[str, Dict[str, pd.DataFrame]],
        eval_bs_dfs: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Dict[str, List[str]]:
        """Run Wilcoxon Rank test to determine the best UQ method.

        Bootstrapping is performed for each UQ method of each evaluation type
        and a statistical significance test (Wilcoxon Rank test) is run to
        determine which UQ method is statistically significantly better.

        Args:
            eval_dfs (Dict[str, Dict[str, pd.DataFrame]]):
                The evaluation output from `Pipeline.fit()` containing the
                evaluation scores for each UQ method.
            eval_bs_dfs (Dict[str, Dict[str, pd.DataFrame]]):
                The same evaluation output from `Pipeline.fit()` but in this
                case it contains the bootstrapped evaluation scores for each UQ method.

        Returns:
            A dictionary containing the best UQ method (as in, statistically
            significantly better than the rest) for each evaluation type.
        """
        metric_selection = {
            EvalType.RankingBased: "Spearman Correlation",
            EvalType.CalibrationBased: "MACE",
            EvalType.ProperScoringRules: "NLL",
        }
        results = {}
        self.logger.debug("Stats analysis on BS results")
        for eval_type in metric_selection.keys():
            metric = metric_selection[eval_type]
            df = eval_bs_dfs[eval_type]["df_eval"]
            df_reduced = eval_dfs[eval_type]["df_eval"]
            methods = df["uq_metric"].unique()
            comparisons = list(itertools.combinations(methods, 2))

            if len(methods) == 1:
                return None

            wilcoxon_results = []

            # Perform pairwise comparisons using Wilcoxon signed-rank test
            for method1, method2 in comparisons:
                data1 = df[df["uq_metric"] == method1][metric]
                data2 = df[df["uq_metric"] == method2][metric]
                differences = data1.values - data2.values

                # Check if all differences are zero
                if np.all(differences == 0):
                    p_value = 1.0
                else:
                    _, p_value = wilcoxon(differences)

                mean_diff = data1.mean() - data2.mean()
                wilcoxon_results.append((method1, method2, mean_diff, p_value))

            wilcoxon_df = pd.DataFrame(
                wilcoxon_results, columns=["group1", "group2", "meandiff", "p"]
            )

            # Multiple testing correction
            p_values = wilcoxon_df["p"].values
            reject, p_values_corrected, _, _ = multipletests(
                p_values, alpha=0.05, method="fdr_bh"
            )
            wilcoxon_df["p_adj"] = p_values_corrected
            wilcoxon_df["reject"] = reject

            # select only statistically significant
            wilcoxon_df = wilcoxon_df.loc[wilcoxon_df.reject, :]

            higher_is_better = metric == "Spearman Correlation"

            method_scores = {method: 0 for method in methods}

            # Update scores based on pairwise comparison results
            for _, row in wilcoxon_df.iterrows():
                method1, method2, mean_diff, p_value, p_adj, reject = row
                if reject:  # Only consider if the result is statistically significant
                    if (higher_is_better and mean_diff > 0) or (
                        not higher_is_better and mean_diff < 0
                    ):
                        method_scores[method1] += 1
                    elif (higher_is_better and mean_diff < 0) or (
                        not higher_is_better and mean_diff > 0
                    ):
                        method_scores[method2] += 1

                        # Select the best method based on the highest score
            best_method = [
                method
                for method, score in method_scores.items()
                if score == max(method_scores.values())
            ]
            results[eval_type] = best_method
        return results

    def _init_logger(self, log_file_path: Optional[Union[Path, str]] = None):
        """Initialize the logger for the class.

        Args:
            log_file_path (Path, str, None):
                The path to the log file. If provided, the logger will also save
                log messages to the specified file. Default: None.
        """
        formatter = CustomFormatter(
            "%(asctime)s | [%(name)s - %(levelname)s]: %(message)s",
            "[%Y-%m-%d %H:%M:%S]",
        )

        if log_file_path is not None:
            rotating_handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_file_path, when="d", interval=1, backupCount=5
            )
            rotating_handler.setLevel(logging.DEBUG)
            rotating_handler.setFormatter(formatter)
            logging.getLogger().addHandler(rotating_handler)

        self.logger = logging.getLogger("UNIQUE")
        self.logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        if self.verbose:
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)

        # Create a Formatter to define the log message format
        stream_handler.setFormatter(formatter)

        # Add the StreamHandler to the logger
        self.logger.addHandler(stream_handler)

        # suppress absl logger if available
        try:
            import os

            from absl import logging as absl_logging

            absl_logging.get_absl_handler().python_handler.stream = open(
                os.devnull, "w"
            )

            absl_logging.set_verbosity(logging.WARNING)
        except ImportError:
            pass

        # self.logger.info(f'{"*" * 80}')
