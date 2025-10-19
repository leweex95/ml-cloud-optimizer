"""ML Pipeline orchestration."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_cloud_optimizer.features.engineering import FeatureEngineer
from ml_cloud_optimizer.models.base import (
    TabularModelFactory,
    LSTMModel,
    TimeSeriesTrainer,
    ModelEvaluator,
)

logger = logging.getLogger(__name__)


class MLPipeline:
    """End-to-end ML pipeline."""

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
        test_size: float = 0.2,
    ):
        """Initialize pipeline.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'rf', 'lstm', 'gru')
            random_state: Random seed
            test_size: Test set fraction
        """
        self.model_type = model_type
        self.random_state = random_state
        self.test_size = test_size

        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.preprocessor = {}
        self.evaluator = ModelEvaluator()

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "cpu_utilization",
        feature_cols: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train pipeline end-to-end.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            feature_cols: List of feature columns (if None, all numeric used)
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting {self.model_type} pipeline training")

        # 1. Feature engineering
        logger.info("Performing feature engineering")
        df_features = self.feature_engineer.engineer_features(df)

        # 2. Prepare data
        if feature_cols is None:
            feature_cols = [
                col
                for col in df_features.columns
                if col not in [target_col, "timestamp", "service_id", "cluster_id"]
            ]

        X = df_features[feature_cols]
        y = df_features[target_col]

        logger.info(f"Dataset shape: {X.shape}, Target: {y.shape}")

        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 4. Train model
        if self.model_type == "lstm" or self.model_type == "gru":
            results = self._fit_sequence_model(
                df_features, target_col, feature_cols, X_train, X_test, y_train, y_test
            )
        else:
            results = self._fit_tabular_model(
                X_train, X_test, y_train, y_test, feature_cols
            )

        logger.info(f"Training completed. Results: {results}")
        return results

    def _fit_tabular_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_cols: list,
    ) -> Dict[str, Any]:
        """Fit tabular model.
        
        Args:
            X_train, X_test, y_train, y_test: Train-test split
            feature_cols: Feature column names
            
        Returns:
            Training results
        """
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.preprocessor["scaler"] = scaler

        # Create and train model
        if self.model_type == "xgboost":
            self.model = TabularModelFactory.create_xgboost()
        elif self.model_type == "lightgbm":
            self.model = TabularModelFactory.create_lightgbm()
        else:
            self.model = TabularModelFactory.create_random_forest()

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = self.evaluator.evaluate_regression(y_test.values, y_pred)

        logger.info(f"Test Metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

        return {
            "model_type": self.model_type,
            "metrics": metrics,
            "feature_count": len(feature_cols),
        }

    def _fit_sequence_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Fit LSTM/GRU model.
        
        Args:
            df: Full dataframe
            target_col: Target column
            feature_cols: Feature columns
            X_train, X_test, y_train, y_test: Train-test split
            
        Returns:
            Training results
        """
        # Create sequences
        seq_length = 24
        X_train_seq, y_train_seq = self.feature_engineer.create_sequences(
            df[: int(len(df) * 0.8)], seq_length=seq_length, target_col=target_col
        )
        X_test_seq, y_test_seq = self.feature_engineer.create_sequences(
            df[int(len(df) * 0.8) :], seq_length=seq_length, target_col=target_col
        )

        # Create model
        if self.model_type == "lstm":
            self.model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        else:  # gru
            from ml_cloud_optimizer.models.base import GRUModel

            self.model = GRUModel(input_size=1, hidden_size=64, num_layers=2)

        # Train
        trainer = TimeSeriesTrainer(self.model)
        X_train_seq = X_train_seq.reshape(-1, seq_length, 1)
        X_test_seq = X_test_seq.reshape(-1, seq_length, 1)

        results = trainer.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32)

        # Evaluate
        y_pred = trainer.predict(X_test_seq)
        metrics = self.evaluator.evaluate_regression(y_test_seq, y_pred.flatten())

        logger.info(f"Test Metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

        return {
            "model_type": self.model_type,
            "metrics": metrics,
            "training_result": results,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Predictions
        """
        df_features = self.feature_engineer.engineer_features(df)

        feature_cols = [
            col
            for col in df_features.columns
            if col
            not in ["cpu_utilization", "timestamp", "service_id", "cluster_id"]
        ]

        X = df_features[feature_cols]

        if "scaler" in self.preprocessor:
            X = self.preprocessor["scaler"].transform(X)

        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Save pipeline to disk.
        
        Args:
            path: Output path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        with open(path / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)

        logger.info(f"Pipeline saved to {path}")

    def load(self, path: Path) -> None:
        """Load pipeline from disk.
        
        Args:
            path: Input path
        """
        path = Path(path)

        with open(path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(path / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)

        logger.info(f"Pipeline loaded from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    # df = pd.read_csv("data/raw/cloud_workloads.csv")
    # pipeline = MLPipeline(model_type="xgboost")
    # results = pipeline.fit(df)
