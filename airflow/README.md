# Apache Airflow Setup Guide

This directory contains the Apache Airflow orchestration configuration for the ML Cloud Workload Optimizer.

## Directory Structure

```
airflow/
├── dags/                           # DAG definitions
│   └── ml_cloud_optimizer_dag.py   # Main ML pipeline DAG
├── plugins/                        # Custom Airflow plugins
├── feature_store.yaml              # Feast feature store configuration
├── feast_repo.py                   # Feast feature definitions
├── __init__.py                     # Airflow initialization
├── README.md                       # This file
└── airflow.db                      # SQLite database (created at runtime)
```

## Installation

### 1. Install Airflow Dependencies

The required packages are already specified in `pyproject.toml`. Install them:

```bash
# From project root
poetry install
```

### 2. Initialize Airflow Database

```bash
# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow

# Initialize the database
poetry run airflow db init
```

### 3. Create an Admin User

```bash
poetry run airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 4. Setup Connections (Optional)

Edit `airflow/__init__.py` and uncomment `setup_connections()` and `setup_variables()` to automatically create connections:

```bash
poetry run python airflow/__init__.py
```

Or manually create connections in the Airflow UI after starting the server.

## Starting Airflow

### Method 1: Standalone Mode (Recommended for Development)

```bash
# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow

# Start Airflow in standalone mode
poetry run airflow standalone
```

The webserver will be available at: **http://localhost:8080**

### Method 2: Separate Scheduler and Webserver

```bash
export AIRFLOW_HOME=$(pwd)/airflow

# Terminal 1: Start scheduler
poetry run airflow scheduler

# Terminal 2: Start webserver
poetry run airflow webserver --port 8080
```

### Method 3: Docker Compose

```bash
# Build and start services
docker-compose -f docker-compose.airflow.yml up

# Create admin user
docker-compose -f docker-compose.airflow.yml exec airflow \
    airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

## DAGs Overview

### ml_cloud_optimizer_pipeline

Main production pipeline that orchestrates the complete ML workflow:

**Schedule**: Every 6 hours (configurable)

**Tasks**:
1. `generate_data` - Generate synthetic cloud workload data
2. `validate_data` - Quality checks on generated data
3. `engineer_features` - Create temporal, rolling, and lag features
4. `train_tabular_model` - Train XGBoost, LightGBM, CatBoost models
5. `train_timeseries_model` - Train LSTM, GRU models
6. `compare_and_register_models` - Compare models and register to MLflow
7. `detect_data_drift` - Monitor for statistical drift
8. `generate_report` - Create training metrics report

**Dependencies**:
```
generate_data → validate_data → engineer_features
                                   ↓
                         ┌─────────┴─────────┐
                         ↓                   ↓
                 train_tabular_model  train_timeseries_model
                         ↓                   ↓
                         └─────────┬─────────┘
                                   ↓
                       compare_and_register_models
                                   ↓
                         ┌─────────┴─────────┐
                         ↓                   ↓
                  detect_data_drift  generate_report
```

## Configuration

### Environment Variables

Set these in your `.env` file or shell:

```bash
# Airflow
export AIRFLOW_HOME=./airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=LocalExecutor

# Database
export AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:///airflow.db

# MLOps
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_REGISTRY_URI=file:./models/mlflow

# Database connections
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=ml_cloud
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=changeme
```

### Custom Variables

Access in DAGs using:

```python
from airflow.models import Variable

ml_env = Variable.get('ml_environment')
data_path = Variable.get('data_path')
```

## Feast Feature Store Integration

The Airflow environment includes Feast feature store configuration:

### Initialize Feast Repository

```bash
# Apply feature definitions
poetry run python airflow/feast_repo.py
```

### Use Features in DAGs

```python
from airflow.decorators import dag, task
from feast import FeatureStore

@task
def get_features(**context):
    fs = FeatureStore(repo_path='./airflow')
    
    # Get training features
    training_fs = fs.get_feature_service('cloud_optimizer_training')
    
    # Get inference features
    inference_fs = fs.get_feature_service('cloud_optimizer_inference')
```

## MLflow Integration

### Start MLflow Server

```bash
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models/mlflow
```

Access MLflow UI at: **http://localhost:5000**

### Tracking Experiments from DAGs

```python
from ml_cloud_optimizer.ml_ops.mlflow_utils import MLflowTracker

with MLflowTracker() as tracker:
    tracker.log_parameters({'model': 'xgboost'})
    tracker.log_metrics({'mae': 0.082, 'rmse': 0.121})
    tracker.log_model(model, 'xgboost_model')
```

## Monitoring and Debugging

### View DAG Status

```bash
# List all DAGs
poetry run airflow dags list

# Get DAG details
poetry run airflow dags show ml_cloud_optimizer_pipeline

# Get task details
poetry run airflow tasks list ml_cloud_optimizer_pipeline
```

### View Logs

```bash
# Task logs
poetry run airflow logs ml_cloud_optimizer_pipeline task_id execution_date

# Or access through web UI at http://localhost:8080
```

### Trigger DAG Manually

```bash
poetry run airflow dags trigger ml_cloud_optimizer_pipeline

# With configuration
poetry run airflow dags trigger ml_cloud_optimizer_pipeline \
    -c '{"data_path": "./data/custom"}'
```

### Clear Task Failures

```bash
# Clear specific task
poetry run airflow tasks clear ml_cloud_optimizer_pipeline task_id

# Clear all tasks for a DAG
poetry run airflow dags test ml_cloud_optimizer_pipeline
```

## Best Practices

1. **Error Handling**: All PythonOperators include try-catch with AirflowException
2. **XCom Communication**: Tasks pass data using XCom (task_instance.xcom_push/pull)
3. **Logging**: Use standard Python logging for task output
4. **Retries**: Configured with 2 retries and 5-minute backoff
5. **Idempotency**: Design tasks to be idempotent for safe retries
6. **Testing**: Run `airflow dags test <dag_id>` before production deployment

## Troubleshooting

### DAG Not Appearing

```bash
# Check DAG syntax
poetry run airflow dags validate

# Check DAG folder permissions
ls -la ./airflow/dags/

# Restart scheduler
# (Scheduler scans dags/ every 300 seconds)
```

### Connection Errors

```bash
# Test connection
poetry run airflow connections test postgres_ml_cloud

# List all connections
poetry run airflow connections list
```

### Task Failures

1. Check logs in Web UI or terminal
2. Review error message and task logs
3. Fix issue and clear task: `poetry run airflow tasks clear <dag_id> <task_id>`
4. Re-trigger DAG

### Database Issues

```bash
# Reset Airflow (WARNING: Deletes all metadata)
rm airflow/airflow.db
poetry run airflow db init
```

## Production Deployment

### Using Kubernetes

See `deploy/kubernetes/deployment.yaml` for K8s manifests including Airflow Scheduler and Webserver.

### Using Docker

```bash
# Build image
docker build -f deploy/docker/Dockerfile -t ml-cloud-optimizer:latest .

# Run with Docker Compose
docker-compose -f docker-compose.airflow.yml up
```

### Using Cloud Providers

- **GCP**: Cloud Composer (managed Airflow)
- **AWS**: Managed Workflows for Apache Airflow (MWAA)
- **Azure**: Azure Data Factory (Airflow-compatible)

## Additional Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Feast Documentation](https://docs.feast.dev/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project README](../README.md)
- [Architecture Guide](../ARCHITECTURE.md)
- [Deployment Guide](../DEPLOYMENT.md)

## Support

For issues or questions:
1. Check logs: `http://localhost:8080/admin/logs/`
2. Review DAG code: `airflow/dags/ml_cloud_optimizer_dag.py`
3. Check Feast config: `airflow/feast_repo.py`
4. See main project documentation in root directory
