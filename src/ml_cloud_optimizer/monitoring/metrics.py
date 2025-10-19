"""Monitoring and metrics tracking."""

import logging
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track model and system metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, Any] = {}
        self.history: list = []

    def log_model_metrics(
        self,
        model_name: str,
        mae: float,
        rmse: float,
        timestamp: datetime = None,
    ) -> None:
        """Log model performance metrics.
        
        Args:
            model_name: Name of model
            mae: Mean Absolute Error
            rmse: Root Mean Squared Error
            timestamp: Timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        metric = {
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "timestamp": timestamp.isoformat(),
        }

        self.metrics[f"{model_name}_{timestamp.isoformat()}"] = metric
        self.history.append(metric)

        logger.info(
            f"Model {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

    def log_deployment_metrics(
        self,
        service_id: str,
        cpu_util: float,
        memory_util: float,
        cost_savings: float,
    ) -> None:
        """Log deployment metrics.
        
        Args:
            service_id: Service identifier
            cpu_util: Average CPU utilization
            memory_util: Average memory utilization
            cost_savings: Estimated cost savings
        """
        metric = {
            "service_id": service_id,
            "cpu_utilization": cpu_util,
            "memory_utilization": memory_util,
            "cost_savings": cost_savings,
            "timestamp": datetime.now().isoformat(),
        }

        self.history.append(metric)
        logger.info(
            f"Deployment {service_id}: CPU={cpu_util:.2f}%, "
            f"Memory={memory_util:.2f}%, Savings=${cost_savings:.2f}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.history:
            return {}

        return {
            "total_events": len(self.history),
            "latest_timestamp": self.history[-1].get("timestamp"),
            "events": self.history[-10:],  # Last 10 events
        }

    def export_metrics(self, path: str) -> None:
        """Export metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Metrics exported to {path}")


class AlertManager:
    """Manage performance and anomaly alerts."""

    def __init__(self, thresholds: Dict[str, float] = None):
        """Initialize alert manager.
        
        Args:
            thresholds: Alert thresholds
        """
        self.thresholds = thresholds or {
            "rmse_threshold": 0.15,
            "mae_threshold": 0.10,
            "cpu_threshold": 0.85,
            "cost_threshold": 0.30,
        }
        self.alerts: list = []

    def check_model_drift(self, mae: float, rmse: float) -> bool:
        """Check if model has drifted.
        
        Args:
            mae: Mean Absolute Error
            rmse: Root Mean Squared Error
            
        Returns:
            True if drift detected
        """
        if mae > self.thresholds["mae_threshold"] or rmse > self.thresholds["rmse_threshold"]:
            alert = {
                "type": "model_drift",
                "mae": mae,
                "rmse": rmse,
                "timestamp": datetime.now().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"Model drift detected: MAE={mae:.4f}, RMSE={rmse:.4f}")
            return True

        return False

    def check_resource_anomaly(self, cpu: float, memory: float) -> bool:
        """Check for resource anomalies.
        
        Args:
            cpu: CPU utilization
            memory: Memory utilization
            
        Returns:
            True if anomaly detected
        """
        if cpu > self.thresholds["cpu_threshold"]:
            alert = {
                "type": "high_cpu",
                "cpu": cpu,
                "timestamp": datetime.now().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"High CPU utilization detected: {cpu:.2f}%")
            return True

        return False

    def get_alerts(self) -> list:
        """Get all alerts."""
        return self.alerts

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self.alerts = []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = MetricsTracker()
    tracker.log_model_metrics("xgboost", mae=0.08, rmse=0.12)
