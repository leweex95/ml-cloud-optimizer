"""Configuration management for ML Cloud Optimizer."""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseSettings, Field


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="password", env="DB_PASSWORD")
    database: str = Field(default="ml_cloud", env="DB_NAME")

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class MLConfig(BaseSettings):
    """ML model configuration."""

    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    n_splits: int = 5
    n_jobs: int = -1


class MLflowConfig(BaseSettings):
    """MLflow tracking configuration."""

    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    artifact_location: str = Field(default="./mlflow_artifacts", env="MLFLOW_ARTIFACT_LOCATION")
    experiment_name: str = "cloud-workload-optimization"


class KubernetesConfig(BaseSettings):
    """Kubernetes configuration."""

    context: str = Field(default="docker-desktop", env="K8S_CONTEXT")
    namespace: str = Field(default="default", env="K8S_NAMESPACE")
    replica_count: int = 3


class ProjectConfig(BaseSettings):
    """Main project configuration."""

    project_name: str = "ml-cloud-optimizer"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Sub-configs
    database: DatabaseConfig = DatabaseConfig()
    ml: MLConfig = MLConfig()
    mlflow: MLflowConfig = MLflowConfig()
    kubernetes: KubernetesConfig = KubernetesConfig()

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent.parent.parent
    data_path: Path = Field(default_factory=lambda: ProjectConfig().project_root / "data")
    models_path: Path = Field(default_factory=lambda: ProjectConfig().project_root / "models")
    notebooks_path: Path = Field(default_factory=lambda: ProjectConfig().project_root / "notebooks")

    class Config:
        env_file = ".env"
        case_sensitive = False


def load_config(config_path: Optional[Path] = None) -> ProjectConfig:
    """Load project configuration from file or environment."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return ProjectConfig(**config_data)
    return ProjectConfig()


# Global config instance
CONFIG = load_config()
