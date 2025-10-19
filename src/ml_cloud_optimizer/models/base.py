"""ML models for cloud workload prediction."""

import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TabularModelFactory:
    """Factory for creating and training tabular models."""

    @staticmethod
    def create_xgboost(
        params: Dict[str, Any] = None,
    ) -> xgb.XGBRegressor:
        """Create XGBoost regressor.
        
        Args:
            params: XGBoost parameters
            
        Returns:
            XGBoost regressor instance
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42,
            "tree_method": "hist",
            "device": "cpu",
        }
        if params:
            default_params.update(params)

        return xgb.XGBRegressor(**default_params)

    @staticmethod
    def create_lightgbm(
        params: Dict[str, Any] = None,
    ) -> lgb.LGBMRegressor:
        """Create LightGBM regressor.
        
        Args:
            params: LightGBM parameters
            
        Returns:
            LightGBM regressor instance
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        return lgb.LGBMRegressor(**default_params)

    @staticmethod
    def create_random_forest(
        params: Dict[str, Any] = None,
    ) -> RandomForestRegressor:
        """Create Random Forest regressor.
        
        Args:
            params: Random Forest parameters
            
        Returns:
            Random Forest regressor instance
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        return RandomForestRegressor(**default_params)


class LSTMModel(nn.Module):
    """LSTM model for time-series prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Output dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Output predictions
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.linear(last_hidden)
        return output


class GRUModel(nn.Module):
    """GRU model for time-series prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """Initialize GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_size: Output dimension
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Output predictions
        """
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        output = self.linear(last_hidden)
        return output


class TimeSeriesTrainer:
    """Trainer for PyTorch time-series models."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to use (cpu or cuda)
        """
        self.model = model.to(device)
        self.device = device
        self.history = {"loss": []}

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer instance
            criterion: Loss function
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.history["loss"].append(avg_loss)

        return avg_loss

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
    ) -> Dict[str, Any]:
        """Train model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)

        # Split data
        train_size = int(len(X_train) * (1 - validation_split))
        X_tr, X_val = X_train[:train_size], X_train[train_size:]
        y_tr, y_val = y_train[:train_size], y_train[train_size:]

        # Create dataloaders
        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    loss = criterion(outputs, y.unsqueeze(1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epochs_trained": epoch + 1,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        return outputs.cpu().numpy()


class ModelEvaluator:
    """Evaluate model performance."""

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "f1": f1_score(y_true, y_pred),
        }

        if y_pred_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)

        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage would be in notebooks
