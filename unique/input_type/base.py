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
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from unique.uncertainty import (
    EnsembleVariance,
    EuclideanDistance,
    ExponentialManhattanKDE,
    GaussianEuclideanKDE,
    GaussianManhattanKDE,
    ManhattanDistance,
    Probability,
    TanimotoDistance,
    UniqueUncertaintyMetric,
)

PROBLEM_TYPES = ("classification", "regression")

DATA_FEATURES_SUPPORTED_METRICS = {
    "count_based": {
        "tanimoto_distance": TanimotoDistance,
        "manhattan_distance": ManhattanDistance,
        "euclidean_distance": EuclideanDistance,
        "gaussian_euclidean_kde": GaussianEuclideanKDE,
        "gaussian_manhattan_kde": GaussianManhattanKDE,
        "exponential_manhattan_kde": ExponentialManhattanKDE,
    },
    "real_valued": {
        "manhattan_distance": ManhattanDistance,
        "euclidean_distance": EuclideanDistance,
        "gaussian_euclidean_kde": GaussianEuclideanKDE,
        "gaussian_manhattan_kde": GaussianManhattanKDE,
        "exponential_manhattan_kde": ExponentialManhattanKDE,
    },
}

MODEL_FEATURES_SUPPORTED_METRICS = {
    "regression": {
        "ensemble_variance": EnsembleVariance,
    },
    "classification": {
        "probability": Probability,
    },
}


@dataclass
class UniqueInputType:
    """Base UNIQUE input type class.

    Args:
        column_name (str):
            The name of the column in the original dataframe to use as input.
        input_values (np.ndarray, pd.Series):
            The actual input values to use for the UQ metrics computations, either
            as a ``np.ndarray`` or a ``pd.Series``.
        which_set (np.ndarray):
            An array containing information on the subset each datapoint belongs to.
            Values must be one of ["TRAIN", "CALIBRATION", "TEST"].
        problem_type ("classification", "regression"):
            The type of problem the original model solves.
        supported_metrics (Dict[str, UniqueUncertaintyMetric]):
            A list of all the supported UQ metrics to compute.
        metrics (Sequence[str], str, None):
            The user-requested metrics to compute. If None, computes
            by default all the available UQ metrics. Default: None.
        type ("data_based", "model_based", None):
            The type of input data as in data features or model-dependent. Default: None.
        incompatible ("classification", "regression", None):
            Which problem task the input type is incompatible with.
    """

    column_name: str
    input_values: Union[np.ndarray, pd.Series]
    which_set: np.ndarray
    problem_type: Literal["classification", "regression"]
    supported_metrics: Optional[Dict[str, Type[UniqueUncertaintyMetric]]] = None
    metrics: Optional[Union[Sequence[str], str]] = None
    type: Optional[Literal["data_based", "model_based"]] = None
    incompatible: Optional[Literal["classification", "regression"]] = None

    def __post_init__(self):
        """Implements data initialization workflow."""
        # Steps:
        # 1. self.sanity_check()
        # 2. self.validate_inputs()
        # 3. self.initialize_metrics()
        raise NotImplementedError

    def initialize_metrics(self):
        """Initialize UQ metrics."""
        metrics_to_compute = {}
        if self.metrics:
            # Select subset of supported metrics
            self.metrics = (
                [self.metrics] if isinstance(self.metrics, str) else self.metrics
            )  # List[str]
            for metric in self.metrics:
                if metric not in self.supported_metrics.keys():
                    raise ValueError(
                        f"""Please, provide a valid (list of) metric(s).
                        Supported metrics: {list(self.supported_metrics.keys())}.
                        Got: "{metric}"."""
                    )
                metrics_to_compute[metric] = self.supported_metrics[metric]
        else:
            # If no metrics are specified, compute all supported metrics
            metrics_to_compute = self.supported_metrics

        # Initialize metrics
        self.metrics_to_compute = {
            metric_name: metric(
                input_values=self.input_values,
                which_set=self.which_set,
                problem_type=self.problem_type,
            )
            for metric_name, metric in metrics_to_compute.items()
        }

    def fit(self) -> Dict[str, np.ndarray]:
        """Compute available/supported UQ metrics using input data."""
        output = {}
        for metric in self.metrics_to_compute.values():
            # Save output using (name of metric + original column name)
            output[f"{metric.name}[{self.column_name}]"] = metric.fit()

        return output

    def sanity_check(self):
        """Perform checks on some input arguments."""
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(
                f"""Please, provide a valid problem type. Supported problem types:
                    {PROBLEM_TYPES}. Got: "{self.problem_type}"."""
            )

        if len(np.unique(self.which_set)) <= 1:
            raise ValueError(
                f"""Please, provide at least 2 subsets of data - e.g., ["TRAIN",
                "CALIBRATION", "TEST"], to enable the correct computation of UQ metrics.
                Got: {np.unique(self.which_set)}."""
            )

    def validate_inputs(self):
        """Check if the input values are valid.

        Note that each datapoint can have a single value or an array of values.
        """
        if not all(self.input_values.apply(lambda x: np.isrealobj(x))):
            raise ValueError(
                """Found complex (non-real) numerical values in the input array.
                Please, provide only real numerical or categorical values."""
            )

        def convert_to_numpy(x: pd.Series) -> np.ndarray:
            if isinstance(x[0], (list, np.ndarray)):
                return np.stack(x.tolist())
            else:
                return x.to_numpy()

        try:
            # If each datapoint is a string of values/list in string-representation
            # e.g., ["[1, 2.7, 13.9]", "[0.3, 1, 5.3]", ...]
            self.input_values = self.input_values.apply(ast.literal_eval)
            self.input_values = convert_to_numpy(self.input_values)
        except:
            self.input_values = convert_to_numpy(self.input_values)

    def __str__(self):
        type_name = self.type.title().replace("_", "-")
        metrics = ", ".join(
            [m.title().replace("_", " ") for m in self.metrics_to_compute]
        )
        return f"[{type_name} Feature] Column: '{self.column_name}' | UQ methods to compute: {metrics}"


@dataclass
class FeaturesInputType(UniqueInputType):
    """UNIQUE inputs associated with data-based features.

    Depending on the type of the numerical features - i.e., whether they are
    integer-only or floats, there are different UQ metrics supported. The
    distinction is mainly done to separate workflows for binary or count-based
    features (e.g., molecular fingerprints) and other numerical features (e.g.,
    PCA components, molecular weight, etc.).

    Supported UQ metrics for integer-only input types are:
        - :class:`unique.uncertainty.TanimotoDistance`.
        - :class:`unique.uncertainty.ManhattanDistance`.
        - :class:`unique.uncertainty.EuclideanDistance`.
        - :class:`unique.uncertainty.GaussianEuclideanKDE`.
        - :class:`unique.uncertainty.GaussianManhattanKDE`.
        - :class:`unique.uncertainty.ExponentialManhattanKDE`.

    Supported UQ metrics for floats input types are:
        - :class:`unique.uncertainty.ManhattanDistance`.
        - :class:`unique.uncertainty.EuclideanDistance`.
        - :class:`unique.uncertainty.GaussianEuclideanKDE`.
        - :class:`unique.uncertainty.GaussianManhattanKDE`.
        - :class:`unique.uncertainty.ExponentialManhattanKDE`.
    """

    type: str = "data_based"

    def __post_init__(self):
        # Sanity check
        self.sanity_check()
        # Clean and initialize input values
        self.validate_inputs()  # self.input_values -> np.ndarray
        # Determine supported UQ metrics depending on the data-type
        if self.input_values.dtype == np.int64:
            self.supported_metrics = DATA_FEATURES_SUPPORTED_METRICS["count_based"]
        elif self.input_values.dtype == np.float64:  # floats
            self.supported_metrics = DATA_FEATURES_SUPPORTED_METRICS["real_valued"]
        else:
            raise ValueError(
                f"""Expected either integer-only or real-valued (floats) data-based
                input features. Got: {self.input_values.dtype}."""
            )

        # Initialize UQ metrics to compute
        self.initialize_metrics()


@dataclass
class ModelInputType(UniqueInputType):
    """UNIQUE inputs associated with the model-based features.

    Depending on the type of task at hand - i.e., classification or regression,
    there are different UQ metrics supported. For 'classification' tasks, the
    predicted probability is used as a UQ proxy itself, whereas for 'regression'
    tasks the variance of the ensemble of models' predictions (either supplied as
    a pre-computed, single variance value or as an array of the individual predictions
    from each model in the ensemble).

    Supported UQ metrics for regression-related outputs are:
        - :class:`unique.uncertainty.EnsembleVariance`.

    Supported UQ metrics for classification-related outputs are:
        - :class:`unique.uncertainty.Probability`.
    """

    type: str = "model_based"

    def __post_init__(self):
        # Sanity check
        self.sanity_check()
        # Clean and initialize input values
        self.validate_inputs()
        # Determine supported UQ metrics depending on the problem-type
        self.supported_metrics = MODEL_FEATURES_SUPPORTED_METRICS[self.problem_type]

        # Initialize UQ metrics to compute
        self.initialize_metrics()
