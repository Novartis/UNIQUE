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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MLPRegressor(nn.Module):
    """MLP model for regression tasks."""

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: Tuple[int, int, int] = (128, 64, 16),
        output_size: int = 1,
        dropout_rate: float = 0.2,
    ):
        """Initialize MLP architecture.

        Args:
            input_size (int):
                Input size to the first MLP layer.
            hidden_layer_sizes (Tuple[int, int, int]):
                Sizes of subsequent hidden layers. Default: (128, 64, 16).
            output_size (int):
                Number of outputs. Default: 1.
            dropout_rate (float):
                Dropout rate. Default: 0.2.
        """
        super(MLPRegressor, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        # Construct hidden layers
        layer_sizes = [input_size] + list(hidden_layer_sizes)
        for i in range(1, len(layer_sizes)):
            self.hidden_layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x: torch.Tensor):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


class SyntheticDataExamplePreparation:
    """Download, prepare, and run analysis on the CaliforniaHousing dataset."""

    def __init__(self, seed: int = 42):
        # Fix seed
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def prepare_data(self):
        """Download and prepare subset split for ML task."""
        # Load the California Housing dataset
        cali = fetch_california_housing()
        X, y = cali.data, cali.target

        # Create an array of indices and shuffle it
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        # Split indices into training+validation and test sets
        train_val_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=self.seed
        )

        # Split the training+validation indices into separate training and validation sets
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.25, random_state=self.seed
        )

        # Indices
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Create DataLoaders for training and validation
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

        return (
            X,
            y,
            train_indices,
            val_indices,
            test_indices,
            train_loader,
            val_loader,
            X_test_tensor,
            y_test_tensor,
        )

    def train_MLP(
        self,
        mlp: nn.Module,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """Train and validate an MLP model on the input data.

        Args:
            mlp (nn.Module):
                MLP model to use for training.
            n_epochs (int):
                Number of epochs for which to train the model.
            train_loader (DataLoader):
                Training subset.
            val_loader (DataLoader):
                Validation subset.

        Returns:
            The trained model (as nn.Module).
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            mlp.train()
            train_losses = []
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = criterion(outputs, targets)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            train_loss_epoch = np.mean(train_losses)

            with torch.no_grad():
                mlp.eval()  # Set the model to evaluation mode
                val_losses = []
                for inputs, targets in val_loader:
                    outputs = mlp(inputs)
                    val_loss = criterion(outputs, targets)
                    val_losses.append(val_loss.item())
                val_loss_epoch = np.mean(val_losses)

            print(
                f"Epoch: {epoch} | train_loss: {train_loss_epoch:.3f} | val_loss: {val_loss_epoch:.3f}"
            )

        return mlp

    def mc_dropout_predict(
        self, model: nn.Module, input_tensor: torch.Tensor, n_samples: int = 100
    ) -> torch.Tensor:
        """Predict values using Monte-Carlo dropout technique.

        Args:
            model (nn.Module):
                The trained model to run MC-Dropout inference on.
            input_tensor (torch.Tensor):
                The input data to run inference on.
            n_samples (int):
                Number of MC-Dropout sample to obtain. Default: 100.

        Returns:
            The predictions for each input datapoint as a torch.Tensor
            of shape (n_samples, n_datapoints, 1).
        """
        model.train()
        predictions = torch.zeros([n_samples, input_tensor.shape[0], 1])
        print("\n")
        for i in tqdm(range(n_samples), desc="Dropout Monte Carlo..."):
            with torch.no_grad():
                predictions[i] = model(input_tensor)
        model.eval()
        return torch.squeeze(predictions)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print MAE, RMSE, R2 regression evaluation metrics."""
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
        print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
        print(f"R2: {r2_score(y_true, y_pred):.4f}")

    def run(self):
        # prepare data
        (
            X,
            y,
            train_indices,
            val_indices,
            test_indices,
            train_loader,
            val_loader,
            X_test_tensor,
            y_test_tensor,
        ) = self.prepare_data()
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        #######################################################################
        # Initialize the Random Forest regressor
        numberTrees = 8
        rf = RandomForestRegressor(n_estimators=numberTrees, random_state=self.seed)

        print("\nRF training...")
        # Train Random Forest regressor
        rf.fit(X_train, y_train)

        # Evaluate the Random Forest model on the test set
        rf_test_predictions = rf.predict(X_test)

        print("Random Forest Test performance:")
        self.evaluate(y_test, rf_test_predictions)

        #######################################################################
        # Initialize the MLP model
        mlp = MLPRegressor(input_size=X_train.shape[1])

        print("\nMLP training...")
        # train MLP
        mlp = self.train_MLP(
            mlp, n_epochs=10, train_loader=train_loader, val_loader=val_loader
        )

        # Evaluate the MLP model on the test set
        mlp.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            mlp_test_predictions = mlp(X_test_tensor)

        mlp_test_mse = root_mean_squared_error(
            y_test_tensor.numpy(), mlp_test_predictions.numpy()
        )
        print("MLP Test performance:")
        self.evaluate(y_test_tensor.numpy(), mlp_test_predictions.numpy())

        #######################################################################
        # RF Variance
        print(f"\nCollecting RF variance...")
        X_all = np.vstack([X_train, X_val, X_test])
        rf_all_predictions = rf.predict(X_all)

        # collect predictions
        collect = []
        for tree in range(numberTrees):
            collect.append(rf.estimators_[tree].predict(X_all))
        # reshape predictions in order to obtain a vector of predictions for each sample
        tree_num = 0
        collect_all = []
        for sample in range(X_all.shape[0]):
            predictions_trees = []
            for tree in collect:
                predictions_trees.append(tree[sample])
            collect_all.append(predictions_trees)

        rf_variances = [np.var(x) for x in collect_all]
        #######################################################################
        mlp_all_predictions = mlp(torch.tensor(X_all, dtype=torch.float32))
        # Dropout Monte Carlo - MLP Variances
        n_mc_samples = 100
        mlp_mc_pred = self.mc_dropout_predict(
            mlp, torch.tensor(X_all, dtype=torch.float32), n_mc_samples
        )
        mlp_variances = torch.var(mlp_mc_pred, dim=0)

        #######################################################################
        return (
            train_indices,
            val_indices,
            test_indices,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            rf_all_predictions,
            rf_variances,
            mlp_all_predictions,
            mlp_variances,
        )
