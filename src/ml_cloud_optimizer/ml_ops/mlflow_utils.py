"""MLOps utilities - MLflow, experiment tracking, and model registry."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Track experiments with MLflow."""

    def __init__(self, tracking_uri: str = "http://localhost:5000", experiment_name: str = "cloud-workload-optimization"):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                mlflow.set_experiment(experiment_name)
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow server: {e}")

    def start_run(self, run_name: str = None):
        """Start MLflow run.
        
        Args:
            run_name: Name for the run
        """
        mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")

    def end_run(self):
        """End current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step (optional)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics")

    def log_model(self, model: Any, artifact_path: str = "model", flavor: str = "sklearn") -> None:
        """Log model.
        
        Args:
            model: Model object
            artifact_path: Path to save model
            flavor: Model flavor (sklearn, pytorch, etc.)
        """
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            # Generic approach
            mlflow.log_artifact(str(model), artifact_path)

        logger.info(f"Logged model: {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log artifact.
        
        Args:
            local_path: Local file path
            artifact_path: Remote path (optional)
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_dict(self, data: Dict, name: str = "metadata") -> None:
        """Log dictionary as JSON artifact.
        
        Args:
            data: Dictionary to log
            name: Name of the artifact (without .json)
        """
        artifact_file = f"/tmp/{name}.json"
        with open(artifact_file, "w") as f:
            json.dump(data, f, indent=2)

        mlflow.log_artifact(artifact_file)
        logger.info(f"Logged artifact: {name}.json")


class ModelRegistry:
    """Manage model versions and registry."""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """Initialize model registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.client = MlflowClient(tracking_uri)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: str = None,
    ) -> str:
        """Register model to registry.
        
        Args:
            model_uri: URI of model run (e.g., "runs:/run_id/model")
            model_name: Name for registered model
            description: Model description
            
        Returns:
            Registered model version
        """
        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name}, Version: {result.version}")
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
    ) -> None:
        """Transition model to different stage.
        
        Args:
            model_name: Name of registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")

    def get_latest_model_version(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[Dict[str, Any]]:
        """Get latest model version of specific stage.
        
        Args:
            model_name: Name of registered model
            stage: Model stage
            
        Returns:
            Model version info or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0]
            return None
        except Exception as e:
            logger.warning(f"Could not retrieve model: {e}")
            return None

    def list_models(self) -> list:
        """List all registered models."""
        try:
            return self.client.list_registered_models()
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            return []


class ExperimentComparison:
    """Compare experiments and runs."""

    def __init__(self, experiment_name: str):
        """Initialize comparator.
        
        Args:
            experiment_name: Name of experiment
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()

    def get_experiment_runs(self) -> list:
        """Get all runs in experiment.
        
        Returns:
            List of runs
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.warning(f"Experiment not found: {self.experiment_name}")
            return []

        runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])
        return runs

    def compare_runs(self, run_ids: list) -> Dict[str, Any]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs
            
        Returns:
            Comparison data
        """
        comparison = {
            "runs": {}
        }

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                comparison["runs"][run_id] = {
                    "name": run.data.tags.get("mlflow.runName", run_id),
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                    "status": run.info.status,
                }
            except Exception as e:
                logger.warning(f"Could not retrieve run {run_id}: {e}")

        return comparison

    def get_best_run(self, metric_name: str, mode: str = "max") -> Optional[str]:
        """Get best run by metric.
        
        Args:
            metric_name: Metric to optimize
            mode: Optimization direction (max or min)
            
        Returns:
            Best run ID
        """
        runs = self.get_experiment_runs()
        if not runs:
            return None

        best_run = None
        best_value = float('-inf') if mode == 'max' else float('inf')

        for run in runs:
            value = run.data.metrics.get(metric_name)
            if value is not None:
                if mode == 'max' and value > best_value:
                    best_run = run
                    best_value = value
                elif mode == 'min' and value < best_value:
                    best_run = run
                    best_value = value

        if best_run:
            logger.info(
                f"Best run: {best_run.info.run_id} with {metric_name}={best_value:.4f}"
            )

        return best_run.info.run_id if best_run else None


class DataDriftDetector:
    """Detect data drift between training and serving data."""

    @staticmethod
    def detect_drift(
        baseline_data: dict,
        current_data: dict,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Detect data drift using statistical tests.
        
        Args:
            baseline_data: Baseline statistics
            current_data: Current statistics
            threshold: Significance threshold
            
        Returns:
            Drift detection results
        """
        from scipy import stats

        drift_results = {
            "drift_detected": False,
            "drifted_features": [],
            "p_values": {}
        }

        for feature in baseline_data.get('features', {}):
            baseline_stats = baseline_data['features'][feature]
            current_stats = current_data['features'][feature]

            # Kolmogorov-Smirnov test
            baseline_dist = baseline_stats.get('distribution', [])
            current_dist = current_stats.get('distribution', [])

            if baseline_dist and current_dist:
                ks_stat, p_value = stats.ks_2samp(baseline_dist, current_dist)
                drift_results['p_values'][feature] = p_value

                if p_value < threshold:
                    drift_results['drift_detected'] = True
                    drift_results['drifted_features'].append(feature)

        logger.info(f"Drift detection: {drift_results}")
        return drift_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    # tracker = MLflowTracker(tracking_uri="http://localhost:5000")
    # tracker.start_run(run_name="xgboost_baseline")
    # tracker.log_params({"n_estimators": 100, "max_depth": 8})
    # tracker.log_metrics({"mae": 0.082, "rmse": 0.121})
    # tracker.end_run()
