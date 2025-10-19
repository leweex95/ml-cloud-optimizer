"""Feature engineering pipeline for time-series data."""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for ML models."""

    def __init__(self, window_sizes: List[int] = None):
        """Initialize feature engineer.
        
        Args:
            window_sizes: List of rolling window sizes for features
        """
        self.window_sizes = window_sizes or [6, 24, 168]  # 6h, 24h, 7d (hourly data)
        self.scaler_dict: Dict[str, StandardScaler] = {}

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features from raw data.
        
        Args:
            df: Raw dataframe with timestamps and metrics
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Sort by service and timestamp
        df = df.sort_values(["service_id", "cluster_id", "timestamp"]).reset_index(drop=True)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Temporal features
        df = self._create_temporal_features(df)

        # Rolling statistics
        df = self._create_rolling_features(df)

        # Lag features
        df = self._create_lag_features(df)

        # Interaction features
        df = self._create_interaction_features(df)

        # Remove rows with NaN (from rolling/lag operations)
        df = df.dropna()

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using multiple strategies.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with missing values handled
        """
        metric_cols = [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
        ]

        # Forward fill within each service/cluster, then backward fill
        for col in metric_cols:
            df[col] = (
                df.groupby(["service_id", "cluster_id"])[col]
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

        # Fill any remaining with mean
        for col in metric_cols:
            df[col] = df[col].fillna(df[col].mean())

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with temporal features
        """
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Peak hours indicator
        df["is_peak_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with rolling features
        """
        metric_cols = [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
        ]

        for window in self.window_sizes:
            for col in metric_cols:
                # Group by service and cluster for proper windowing
                grouped = df.groupby(["service_id", "cluster_id"])[col]

                # Rolling statistics
                df[f"{col}_rolling_mean_{window}"] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rolling_std_{window}"] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f"{col}_rolling_min_{window}"] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f"{col}_rolling_max_{window}"] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )

        return df

    def _create_lag_features(self, df: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
        """Create lagged features.
        
        Args:
            df: Input dataframe
            n_lags: Number of lags to create
            
        Returns:
            Dataframe with lagged features
        """
        metric_cols = [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
        ]

        for col in metric_cols:
            grouped = df.groupby(["service_id", "cluster_id"])[col]
            for lag in range(1, min(n_lags + 1, 25)):  # Max 24 lags
                df[f"{col}_lag_{lag}"] = grouped.transform(lambda x: x.shift(lag))

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with interaction features
        """
        # Resource utilization combined metric
        df["total_utilization"] = (
            df["cpu_utilization"] * 0.5
            + df["memory_utilization"] * 0.3
            + df["network_utilization"] * 0.2
        )

        # CPU-to-memory ratio
        df["cpu_memory_ratio"] = (
            df["cpu_utilization"] / (df["memory_utilization"] + 1e-6)
        )

        # Utilization variance indicator
        df["util_variance"] = np.sqrt(
            (df["cpu_utilization"] - df["memory_utilization"]) ** 2
        )

        # Cost efficiency
        df["cost_per_util"] = df["cost"] / (df["total_utilization"] + 1e-6)

        return df

    def normalize_features(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """Normalize numerical features.
        
        Args:
            df: Input dataframe
            fit: Whether to fit scalers (True for training data)
            
        Returns:
            DataFrame with normalized features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Separate temporal features (already normalized)
        temporal_features = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_peak_hour",
        ]

        cols_to_scale = [col for col in numeric_cols if col not in temporal_features]

        if fit:
            self.scaler_dict["features"] = StandardScaler()
            scaled = self.scaler_dict["features"].fit_transform(df[cols_to_scale])
        else:
            if "features" not in self.scaler_dict:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled = self.scaler_dict["features"].transform(df[cols_to_scale])

        df_normalized = df.copy()
        df_normalized[cols_to_scale] = scaled

        return df_normalized

    def create_sequences(
        self,
        df: pd.DataFrame,
        seq_length: int = 24,
        target_col: str = "cpu_utilization",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time-series models (LSTM/GRU).
        
        Args:
            df: Input dataframe with features
            seq_length: Sequence length for LSTM/GRU
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        values = df[target_col].values

        for i in range(len(values) - seq_length):
            X.append(values[i : i + seq_length])
            y.append(values[i + seq_length])

        return np.array(X), np.array(y)


class FeatureSelector:
    """Select important features for modeling."""

    @staticmethod
    def get_important_features(
        df: pd.DataFrame, target_col: str, threshold: float = 0.05
    ) -> List[str]:
        """Get features with importance above threshold.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            threshold: Importance threshold
            
        Returns:
            List of selected feature names
        """
        # For this basic implementation, return all numeric features except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        return numeric_cols

    @staticmethod
    def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by type.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary of feature groups
        """
        feature_groups = {
            "temporal": [
                col
                for col in df.columns
                if any(x in col for x in ["hour", "day", "month", "weekend"])
            ],
            "rolling": [
                col for col in df.columns if "rolling" in col
            ],
            "lag": [col for col in df.columns if "lag" in col],
            "interaction": [
                col
                for col in df.columns
                if any(
                    x in col
                    for x in [
                        "total_util",
                        "ratio",
                        "variance",
                        "cost_per",
                    ]
                )
            ],
        }
        return feature_groups


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    # df = pd.read_csv("data/raw/cloud_workloads.csv")
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    # engineer = FeatureEngineer()
    # df_features = engineer.engineer_features(df)
    # df_normalized = engineer.normalize_features(df_features, fit=True)
    # print(df_normalized.head())
