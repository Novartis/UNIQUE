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

from typing import Sequence

import numpy as np

# import tensorflow as tf
# from tensorflow.keras.losses import mean_squared_error

SUPPORTED_ERROR_TYPES = ["l1", "l2", "unsigned"]


# Utilities
def compute_error(
    preds: np.ndarray, labels: np.ndarray, error_type: str = "unsigned"
) -> np.ndarray:
    """Compute the specified error type given the predictions and the corresponding true labels."""

    assert (
        error_type in SUPPORTED_ERROR_TYPES
    ), f"""
        Supported error types: {SUPPORTED_ERROR_TYPES}. Got: {error_type}.
    """
    error = labels - preds
    if error_type == "l1":
        return np.abs(error)
    elif error_type == "l2":
        return error**2
    else:  # "unsigned"
        return error


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """Perform Min-Max normalization."""
    return (x - min(x)) / max(x)


def split_error_model_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    which_set: np.ndarray,
) -> Sequence[np.ndarray]:
    """Split the error model's input features into training and test set.

    It uses user-provided information about the training/calibration/test splits
    contained in the ``which_set`` array.

    Args:
        inputs (np.ndarray):
            Input features to split. Should be of shape (n_samples, n_features),
            and with the same length as ``targets`` and ``which_set``.
        targets (np.ndarray):
            Target values associated with the input features. Should be of shape
            (n_samples, ) or (n_samples, n_targets), and with the same length as
            ``ìnputs`` and ``which_set``.
        which_set (np.ndarray):
            Array containing information regarding the training/calibration/test
            split in the form of strings indicating each sample's membership - i.e.,
            ["TRAIN", "CALIBRATION", "TRAIN", "TEST"].

    Returns:
        A tuple of 6 ``np.ndarray``: training input features and corresponding targets,
        test input features and corresponding targets, the original full input features
        and corresponding full targets.
    """
    x_train = inputs[np.where(which_set != "TEST")]
    y_train = targets[np.where(which_set != "TEST")]

    x_test = inputs[np.where(which_set == "TEST")]
    y_test = targets[np.where(which_set == "TEST")]

    return x_train, y_train, x_test, y_test, inputs, targets


# def sigmoid(x):
#     """Sigmoid activation function."""
#     return tf.math.sigmoid(x, name=None)
#
#
# def weighted_mse(y_true, y_pred):
#     """Weighted Mean Squared Error (MSE)."""
#     weight = sigmoid(y_true)
#     mse = mean_squared_error(y_true, y_pred)
#     return weight * mse
