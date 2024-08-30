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
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# from tensorflow.keras.models import Model, Sequential
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from unique.utils import compute_error, split_error_model_dataset


@dataclass
class UniqueErrorModel:
    """Base class for UNIQUE's error models."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        which_set: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        input_features: Dict[str, np.ndarray],
        error_type_list: Union[Sequence[str], str] = ["l1"],
        standardize: bool = True,
        resampling: bool = False,
        save_models: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the error model.

        Args:
            output_dir (str, Path):
                Output directory where to save the error model's checkpoints.
            which_set (np.ndarray):
                An array containing the data subsets specification for the input dataset.
            predictions (np.ndarray):
                An array containing the predictions from the original model.
            labels (np.ndarray):
                An array containing the target values used to train the original model.
            input_features (Dict[str, np.ndarray]):
                The sets of input features to use to train the error model. The key-value
                pair represent the name of the feature set and the feature values.
                The ``np.ndarray`` must be of shape (n_samples, n_features).
            error_type_list (Sequence[str]):
                List of error types to use to score the model's performance. Default: "l1".
            standardize (bool):
                Whether to standardize the inputs using ``sklearn.preprocessing.StandardScaler``. Default: True.
            resampling (bool):
                Whether to resample the input features to have a more uniform error distribution. Default: False.
            save_models (bool):
                Whether to save each trained error model in the provided output directory.
                Models will be pickled and named according to their error type and input
                features - e.g., "{model_name}_{input_features}_{error_type}.pkl". Default: True.
            name (str):
                Name of the error model instance (used to identify its predictions). Default: None (uses the class name).
            **kwargs:
                Keyword arguments specific to the selected model (e.g., hyperparameters).
        """
        # Create output directory for error models
        self.save_models = save_models
        if self.save_models:
            self.output_dir = Path(output_dir) / "error_models"
            os.makedirs(self.output_dir, exist_ok=True)

        # Store necessary data and parameters
        self.which_set = which_set
        self.predictions = predictions
        self.labels = labels
        self.standardize = standardize
        self.resampling = resampling

        self.name = self.__class__.__name__ if name is None else name

        # Initialize target error values
        self.error_types = (
            [error_type_list] if isinstance(error_type_list, str) else error_type_list
        )
        self.errors = {}
        for error_type in self.error_types:
            self.errors[f"{error_type}_error"] = compute_error(
                self.predictions, self.labels, error_type
            )

        # Make sure the shape is (n_samples, n_errors) or (n_samples,) if only 1 error type
        self.error_values = np.vstack(list(self.errors.values()))
        self.error_values = (
            self.error_values.ravel()
            if len(self.error_types) == 1
            else self.error_values.transpose((1, 0))
        )

        # Split data into training and test sets
        self.input_features = input_features

        self.x_train, self.y_train = {}, {}
        self.x_test, self.y_test = {}, {}
        self.x_all, self.y_all = {}, {}

        for inputs_name, inputs in self.input_features.items():
            # Split data into training and test sets
            x_train, y_train, x_test, y_test, x_all, y_all = split_error_model_dataset(
                inputs=inputs,
                targets=self.error_values,
                which_set=self.which_set,
            )
            # Scale features
            if self.standardize:
                scaler = StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
                x_all = scaler.transform(x_all)
            # Resample features
            if self.resampling:
                pass

            # Store by feature set
            self.x_train[inputs_name], self.y_train[inputs_name] = x_train, y_train
            self.x_test[inputs_name], self.y_test[inputs_name] = x_test, y_test
            self.x_all[inputs_name], self.y_all[inputs_name] = x_all, y_all

        # Initialize model instance
        self.model = self._initialize(**kwargs)

    def fit(
        self,
    ) -> Dict[str, np.ndarray]:
        """Fit error model.

        Returns:
            A dict with the error model name and the error predictions for each
            features combination.
        """
        self.outputs = {}

        for inputs_name in self.input_features.keys():
            # For each combination of features, train a different model
            init_model = deepcopy(self.model)

            # Fit the individual model on one combination of features
            trained_model = self._fit(
                init_model, self.x_train[inputs_name], self.y_train[inputs_name]
            )

            # Predict on the whole input dataset using the trained model
            error_predictions, error_targets = self._predict(
                trained_model, self.x_all[inputs_name], self.y_all[inputs_name]
            )

            for i, error_type in enumerate(self.error_types):
                preds = (
                    error_predictions
                    if len(self.error_types) == 1
                    else error_predictions[:, i]
                )
                # Save error predictions
                self.outputs[f"{self.name}[{inputs_name}]({error_type})"] = preds

            if self.save_models:
                self.save(
                    inputs_name,
                    trained_model,
                    self.x_all[inputs_name],
                    self.y_all[inputs_name],
                )

        return self.outputs

    def _initialize(self, **kwargs) -> Any:
        """Instantiate specific error model instance."""
        raise NotImplementedError

    def _fit(
        self,
        model: BaseEstimator,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Fit a specific error model's instance."""
        raise NotImplementedError

    def _predict(
        self,
        model: BaseEstimator,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform predictions with a trained model and input features and targets."""
        raise NotImplementedError

    def save(
        self,
        features_names: str,
        model: BaseEstimator,
        x: np.ndarray,
        y: np.ndarray,
    ):
        """Dump a trained model and its training set as pickle dumps."""
        # Create subfolder for each feature combination and error type
        output_dir = Path(self.output_dir) / features_names.replace("+", "_")
        os.makedirs(output_dir, exist_ok=True)

        x = np.expand_dims(x, -1) if len(x.shape) == 1 else x
        y = np.expand_dims(y, -1) if len(y.shape) == 1 else y

        # Prepare data
        data = np.hstack(
            (
                np.expand_dims(self.which_set, -1),
                np.expand_dims(self.predictions, -1),
                np.expand_dims(self.labels, -1),
                x,
                y,
            )
        )
        cols = (
            ["which_set", "predictions", "labels"]
            + [f"feature_{i}" for i in range(x.shape[1])]
            + [f"{error}_error" for error in self.error_types]
        )
        # Save as pd.DataFrame
        df = pd.DataFrame(data, columns=cols)

        # Dump model and data
        with open(output_dir / f"{self.name}_model.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # if isinstance(model, Model): # Keras
        #     model.save(output_dir / f"{self.name}_model.h5")
        # else: # sklearn
        #     with open(output_dir / f"{self.name}_model.pkl", "wb") as f:
        #         pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        df.to_csv(output_dir / "data.csv", index=False)

    def __str__(self):
        return self.name
