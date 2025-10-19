"""Basic tests for ML Cloud Optimizer."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_cloud_optimizer.data.generator import CloudWorkloadGenerator
from ml_cloud_optimizer.features.engineering import FeatureEngineer
from ml_cloud_optimizer.models.base import ModelEvaluator, TabularModelFactory
from ml_cloud_optimizer.pipeline.imbalanced import (
    ClassWeightCalculator,
    SMOTEHandler,
    ImbalancedEvaluator,
)


class TestDataGeneration:
    """Test synthetic data generation."""

    def test_data_generation(self):
        """Test dataset generation."""
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate_workloads(
            n_records=1000, n_services=5, n_clusters=2
        )

        assert df.shape[0] == 1000
        assert df.shape[1] == 8
        assert "timestamp" in df.columns
        assert "service_id" in df.columns
        assert df["cpu_utilization"].min() >= 0
        assert df["cpu_utilization"].max() <= 1

    def test_anomaly_injection(self):
        """Test anomaly injection."""
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate_workloads(n_records=1000)
        df_with_anomalies = generator.add_anomalies(df, anomaly_rate=0.05)

        # Should have same shape
        assert df_with_anomalies.shape == df.shape
        # Should have more high CPU values
        assert (
            (df_with_anomalies["cpu_utilization"] > 0.8).sum()
            > (df["cpu_utilization"] > 0.8).sum()
        )

    def test_missing_values_injection(self):
        """Test missing value injection."""
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate_workloads(n_records=1000)
        df_with_missing = generator.add_missing_values(df, missing_rate=0.05)

        # Check that NaN values were added
        assert df_with_missing.isnull().sum().sum() > 0


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate_workloads(n_records=1000, n_services=5)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def test_feature_engineering(self, sample_data):
        """Test feature engineering."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(sample_data.copy())

        # Check new features exist
        assert "hour" in df_features.columns
        assert "day_of_week" in df_features.columns
        assert any("rolling" in col for col in df_features.columns)
        assert any("lag" in col for col in df_features.columns)

        # Check feature count increased
        assert df_features.shape[1] > sample_data.shape[1]

    def test_normalization(self, sample_data):
        """Test feature normalization."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(sample_data.copy())
        df_normalized = engineer.normalize_features(df_features, fit=True)

        # Check normalization happened
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
                # Check roughly centered around 0 and scaled
                assert abs(df_normalized[col].mean()) < 1
                assert abs(df_normalized[col].std() - 1) < 1


class TestImbalancedLearning:
    """Test imbalanced learning techniques."""

    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset."""
        X = np.random.randn(1000, 10)
        # Create 97% negative, 3% positive (imbalanced)
        y = np.concatenate([np.zeros(970), np.ones(30)])
        return X, y

    def test_smote(self, imbalanced_data):
        """Test SMOTE resampling."""
        X, y = imbalanced_data
        handler = SMOTEHandler(sampling_strategy=0.5)
        X_resampled, y_resampled = handler.fit_resample(X, y)

        # Check resampling increased minority class
        assert y_resampled.sum() > y.sum()
        assert X_resampled.shape[0] > X.shape[0]

    def test_class_weights(self, imbalanced_data):
        """Test class weight calculation."""
        X, y = imbalanced_data
        weights = ClassWeightCalculator.calculate_weights(y)

        # Check minority class has higher weight
        assert weights[1] > weights[0]
        # Check weights are normalized
        assert abs(sum(weights.values()) - len(weights)) < 0.1

    def test_imbalanced_evaluator(self):
        """Test imbalanced evaluation metrics."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])

        metrics = ImbalancedEvaluator.evaluate_imbalanced(y_true, y_pred)

        # Check metrics are computed
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["f1"] <= 1


class TestModelEvaluation:
    """Test model evaluation."""

    def test_regression_evaluation(self):
        """Test regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.0, 5.2])

        metrics = ModelEvaluator.evaluate_regression(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] > metrics["mae"]

    def test_classification_evaluation(self):
        """Test classification metrics."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])

        metrics = ModelEvaluator.evaluate_classification(y_true, y_pred, y_proba)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics


class TestTabularModels:
    """Test tabular model creation."""

    def test_xgboost_creation(self):
        """Test XGBoost model creation."""
        model = TabularModelFactory.create_xgboost()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_lightgbm_creation(self):
        """Test LightGBM model creation."""
        model = TabularModelFactory.create_lightgbm()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_random_forest_creation(self):
        """Test Random Forest model creation."""
        model = TabularModelFactory.create_random_forest()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


class TestIntegration:
    """Integration tests."""

    def test_data_to_features_pipeline(self):
        """Test data generation through feature engineering."""
        # Generate data
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate_workloads(n_records=500)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Add temporal features
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)

        # Check pipeline output
        assert df_features.shape[0] > 0
        assert df_features.shape[1] > df.shape[1]
        assert not df_features.isnull().any().any()

    def test_imbalanced_to_evaluation(self):
        """Test imbalanced handling through evaluation."""
        # Create imbalanced dataset
        X = np.random.randn(500, 5)
        y = np.concatenate([np.zeros(485), np.ones(15)])  # 3% positive

        # Apply SMOTE
        handler = SMOTEHandler()
        X_resampled, y_resampled = handler.fit_resample(X, y)

        # Check balanced
        ratio = y_resampled.sum() / len(y_resampled)
        assert ratio > 0.4  # At least 40% positive now

        # Train simple model
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_resampled, y_resampled)

        # Evaluate
        y_pred = model.predict(X_resampled[:100])
        metrics = ImbalancedEvaluator.evaluate_imbalanced(y_resampled[:100], y_pred)

        assert metrics["f1"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
