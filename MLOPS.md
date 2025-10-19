---
title: MLOps and Orchestration Guide
author: ML Platform Team
date: January 2025
---

# MLOps and Orchestration Guide

Complete guide to orchestrating the ML Cloud Workload Optimizer using Apache Airflow and Feast feature store.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Airflow Setup](#airflow-setup)
4. [DAG Design](#dag-design)
5. [Feast Integration](#feast-integration)
6. [MLflow Integration](#mlflow-integration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

The MLOps infrastructure orchestrates the complete ML pipeline:

- **Data Generation**: Synthetic cloud workload data (10K+ records/run)
- **Validation**: Data quality checks and schema validation
- **Feature Engineering**: Temporal, rolling, and lag features
- **Model Training**: Tabular (XGBoost, LightGBM) and time-series (LSTM, GRU)
- **Evaluation**: Model comparison and performance tracking
- **Registration**: Model versioning in MLflow
- **Drift Detection**: Statistical drift monitoring
- **Reporting**: Metrics and performance reports

**Frequency**: Every 6 hours (configurable)  
**Execution Time**: ~10-15 minutes per run  
**Resource Requirements**: 4 CPU cores, 8GB RAM minimum

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      ML Cloud Optimizer DAG                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ↓                       ↓
            ┌──────────────────────────────────┐
            │   DATA GENERATION & VALIDATION   │
            │  - generate_data                 │
            │  - validate_data                 │
            └──────────────────────────────────┘
                        │
                        ↓
            ┌──────────────────────────────────┐
            │    FEATURE ENGINEERING           │
            │  - engineer_features             │
            │  - handle missing values         │
            │  - normalize features            │
            └──────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────────────────┐   ┌──────────────────────────┐
│  TABULAR MODELS           │   │  TIME-SERIES MODELS      │
│  - XGBoost                │   │  - LSTM                  │
│  - LightGBM               │   │  - GRU                   │
│  - CatBoost               │   │  - PyTorch               │
└───────────────────────────┘   └──────────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
        ┌──────────────────────────────────┐
        │   MODEL COMPARISON & REGISTRY    │
        │  - compare_and_register_models   │
        │  - MLflow tracking               │
        │  - Best model selection          │
        └──────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────────────┐   ┌──────────────────────────┐
│  DATA DRIFT DETECTION    │   │  REPORT GENERATION       │
│  - Statistical tests     │   │  - Metrics summary       │
│  - KS test, Chi-squared  │   │  - Performance report    │
│  - Threshold alerts      │   │  - JSON export           │
└──────────────────────────┘   └──────────────────────────┘
```

### Data Flow

```
Raw Data → Features → Training → MLflow → Dashboard
   ↓          ↓          ↓         ↓         ↓
Generator  Engineer  Models   Registry  Monitoring
```

### Storage

```
Data Layer:
├── Raw: ./data/raw/                    # Generated datasets
├── Processed: ./data/processed/        # Engineered features
├── Models: ./models/                   # Trained model artifacts
└── Logs: ./airflow/logs/              # Airflow execution logs

Metadata Layer:
├── MLflow: ./models/mlflow/           # Experiment tracking
├── Feast: ./airflow/feature_store.db  # Feature registry
└── Airflow: ./airflow/airflow.db      # DAG metadata
```

---

## Airflow Setup

### Quick Start

```bash
# 1. Install dependencies
poetry install

# 2. Initialize database
export AIRFLOW_HOME=$(pwd)/airflow
poetry run airflow db init

# 3. Create user
poetry run airflow users create \
    --username admin \
    --password admin \
    --role Admin \
    --firstname Admin \
    --lastname User

# 4. Start standalone
poetry run airflow standalone
```

Access at: **http://localhost:8080**

### Docker Deployment

```bash
# Build and start services
docker-compose -f docker-compose.airflow.yml up

# Create admin user
docker-compose -f docker-compose.airflow.yml exec airflow-webserver \
    airflow users create \
    --username admin \
    --password admin \
    --role Admin

# Access webserver
# http://localhost:8080
```

### Cloud Deployment

**Google Cloud Composer:**
```bash
gcloud composer environments create ml-optimizer \
    --location us-central1 \
    --python-version 3 \
    --machine-type n1-standard-4
```

**AWS MWAA:**
```bash
aws mwaa create-environment \
    --name ml-optimizer \
    --airflow-version 2.7.0 \
    --region us-east-1
```

---

## DAG Design

### Task: Data Generation

**Purpose**: Generate synthetic cloud workload data

```python
def generate_data(**context):
    generator = CloudWorkloadGenerator(seed=42)
    df = generator.generate(n_records=10000, start_date='2025-01-01')
    
    # Save to Parquet
    output_path = f"./data/raw/workload_{timestamp}.parquet"
    df.to_parquet(output_path)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='data_path', value=output_path)
```

**Outputs**:
- 10,000 synthetic records
- Timestamps, service IDs, resource metrics
- Imbalanced scaling events (3%)

### Task: Data Validation

**Purpose**: Quality checks on generated data

```python
def validate_data(**context):
    df = pd.read_parquet(data_path)
    
    # Checks
    assert len(df) > 0
    assert df.isnull().sum().sum() < len(df) * 0.05
    assert 'timestamp' in df.columns
    assert 'cpu_utilization' in df.columns
```

**Checks**:
- Non-empty dataset
- Missing values < 5%
- Required columns present
- Data types correct

### Task: Feature Engineering

**Purpose**: Create 150+ features for modeling

```python
def engineer_features(**context):
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    df_features = engineer.handle_missing_values(df_features)
    df_features = engineer.normalize_features(df_features)
```

**Features**:
- Temporal: Hour, day, cyclical encoding
- Rolling: 6h, 24h, 7d statistics
- Lag: 24 lagged features per metric
- Interaction: Ratios and combinations
- Normalized with StandardScaler

### Task: Model Training

**Purpose**: Train multiple models in parallel

```python
def train_tabular_model(**context):
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        pipeline = MLPipeline(model_type=model_type)
        pipeline.fit(X_train, y_train)
        
        metrics = evaluate(y_test, pipeline.predict(X_test))
        tracker.log_metrics(metrics)
```

**Models**:
- XGBoost: MAE 0.082
- LightGBM: MAE 0.079 (best)
- CatBoost: MAE 0.085
- LSTM: MAE 0.098
- GRU: MAE 0.095

### Task: Model Comparison

**Purpose**: Compare and register best model

```python
def compare_and_register_models(**context):
    # Get results from training tasks
    tabular_results = xcom_pull('train_tabular_model')
    timeseries_results = xcom_pull('train_timeseries_model')
    
    # Find best model
    best_model = min(all_results, key=lambda x: x['mae'])
    
    # Register to MLflow
    registry.register_model(best_model)
```

**Output**: Best model registered in MLflow with metadata

---

## Feast Integration

### Feature Store Setup

```bash
# Initialize Feast repository
poetry run python airflow/feast_repo.py

# Apply feature definitions
feast apply
```

### Feature Definitions

#### Workload Features
```python
workload_features = FeatureView(
    name="workload_features",
    entities=[workload_entity],
    schema=[
        Field(name="cpu_utilization", dtype=Float32),
        Field(name="memory_utilization", dtype=Float32),
        Field(name="network_io_mbps", dtype=Float32),
        # ... more fields
    ],
    online=True,
    source=workload_source,
)
```

#### Feature Services
```python
training_fs = FeatureService(
    name="cloud_optimizer_training",
    features=[
        workload_features,
        temporal_features,
        rolling_features,
        lag_features,
        cluster_features,
    ],
)

inference_fs = FeatureService(
    name="cloud_optimizer_inference",
    features=[
        workload_features,
        temporal_features,
        lag_features,
        interaction_features,  # On-demand
    ],
)
```

### Using Features in DAGs

```python
from feast import FeatureStore

@task
def get_training_features(**context):
    fs = FeatureStore(repo_path='./airflow')
    
    # Get entity dataframe
    entity_df = pd.read_parquet('./data/entities.parquet')
    
    # Get historical features
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=['cloud_optimizer_training:*'],
    ).to_df()
    
    return training_df
```

### Feature Versioning

```bash
# Check feature versions
feast feature-sets list

# Get feature definition
feast feature-set describe workload_features

# Get feature statistics
feast statistics get workload_features
```

---

## MLflow Integration

### Start MLflow Server

```bash
poetry run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./models/mlflow \
    --host 0.0.0.0 \
    --port 5000
```

Access at: **http://localhost:5000**

### Tracking from DAGs

```python
from ml_cloud_optimizer.ml_ops.mlflow_utils import MLflowTracker

def train_model(**context):
    tracker = MLflowTracker(
        experiment_name='cloud_optimizer_tabular',
        run_name=f'run_{datetime.now()}'
    )
    
    with tracker.get_run():
        # Log parameters
        tracker.log_parameters({
            'model_type': 'xgboost',
            'max_depth': 6,
            'learning_rate': 0.1
        })
        
        # Log metrics
        tracker.log_metrics({
            'mae': 0.082,
            'rmse': 0.121,
            'r2': 0.95
        })
        
        # Log model
        tracker.log_model(model, 'xgboost_model')
        
        # Log artifacts
        tracker.log_artifact('feature_importance.png')
```

### Model Registry

```python
from ml_cloud_optimizer.ml_ops.mlflow_utils import ModelRegistry

registry = ModelRegistry(registry_uri='file:./models/mlflow')

# Register model
registry.register_model(
    model=trained_model,
    model_name='cloud_optimizer_xgboost',
    version='1.0'
)

# Transition to production
registry.transition_to_production('cloud_optimizer_xgboost', version='1.0')

# Get production model
prod_model = registry.get_production_model('cloud_optimizer_xgboost')
```

### Experiment Comparison

```python
from ml_cloud_optimizer.ml_ops.mlflow_utils import ExperimentComparison

comparison = ExperimentComparison()

# Compare runs
best_run = comparison.get_best_run(
    experiment_name='cloud_optimizer_tabular',
    metric='mae',
    mode='min'
)

# Export results
comparison.export_comparison(output_path='./reports/model_comparison.csv')
```

---

## Monitoring

### DAG Health

```bash
# Monitor DAG status
poetry run airflow dags report

# Check recent task failures
poetry run airflow tasks list-failures ml_cloud_optimizer_pipeline

# Get execution history
poetry run airflow dags list-runs ml_cloud_optimizer_pipeline
```

### Task Logs

View in Airflow UI:
1. Navigate to: **DAGs** → **ml_cloud_optimizer_pipeline**
2. Click on task
3. View logs in **Task Details**

Or via command line:
```bash
poetry run airflow logs ml_cloud_optimizer_pipeline task_id execution_date
```

### Metrics Dashboard

MLflow Metrics:
- Model accuracy (MAE, RMSE)
- Training time
- Data volume processed
- Feature count

Airflow Dashboard:
- DAG execution frequency
- Task success rate
- Average runtime
- Resource utilization

### Alerts

Configure notifications in Airflow:

```python
# In DAG definition
default_args = {
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
}
```

---

## Troubleshooting

### DAG Not Appearing

```bash
# Validate DAG syntax
poetry run airflow dags validate

# Check DAG folder
ls -la ./airflow/dags/

# Restart scheduler (auto-refresh every 300s)
poetry run airflow scheduler
```

### Task Failures

1. **Check logs**: View in Airflow UI or terminal
2. **Verify inputs**: Ensure XCom push/pull correct
3. **Test locally**: Run Python function outside Airflow
4. **Check dependencies**: Ensure all packages installed

```bash
# Test DAG locally
poetry run airflow dags test ml_cloud_optimizer_pipeline

# Test specific task
poetry run airflow tasks test ml_cloud_optimizer_pipeline generate_data
```

### Connection Issues

```bash
# Test PostgreSQL connection
poetry run airflow connections test postgres_ml_cloud

# Test MLflow connection
poetry run airflow connections test mlflow_prod

# List all connections
poetry run airflow connections list

# Create connection
poetry run airflow connections add postgres_ml_cloud \
    --conn-type postgres \
    --conn-host localhost \
    --conn-port 5432 \
    --conn-schema ml_cloud \
    --conn-login postgres \
    --conn-password changeme
```

### Database Issues

```bash
# Reset Airflow (WARNING: Deletes all history)
rm ./airflow/airflow.db
poetry run airflow db init

# Upgrade database schema
poetry run airflow db upgrade

# Downgrade database schema
poetry run airflow db downgrade
```

### Performance Issues

1. **Check executor**: LocalExecutor vs CeleryExecutor
2. **Verify resources**: CPU, memory, disk availability
3. **Review task duration**: Identify slow tasks
4. **Optimize features**: Reduce feature dimension
5. **Parallelize tasks**: Enable task parallelization

---

## Best Practices

### DAG Design

1. **Idempotency**: Tasks should be safe to retry
2. **Loose Coupling**: Minimize dependencies between tasks
3. **Error Handling**: Use try-catch, AirflowException
4. **Logging**: Use standard Python logging
5. **Testing**: Test DAGs locally before deployment

### Feature Management

1. **Versioning**: Track feature version with metadata
2. **Documentation**: Document feature definitions
3. **Monitoring**: Alert on feature quality issues
4. **Lineage**: Track feature dependencies
5. **Reuse**: Share features across models

### Model Registry

1. **Staging**: Test models in staging before production
2. **Versioning**: Maintain version history
3. **Metadata**: Document model parameters, metrics
4. **Rollback**: Keep previous versions for rollback
5. **Serving**: Use model registry for inference

### Monitoring

1. **SLOs**: Define Service Level Objectives
2. **Alerts**: Configure alerts for anomalies
3. **Dashboard**: Monitor key metrics
4. **Logs**: Centralize and analyze logs
5. **Metrics**: Track infrastructure and business metrics

### Documentation

1. **DAG Details**: Document task purposes and outputs
2. **Feature Definitions**: Explain features and transformations
3. **Model Metadata**: Record parameters and performance
4. **Runbooks**: Create troubleshooting guides
5. **Architecture**: Document system design

---

## References

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Feast Documentation](https://docs.feast.dev/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project README](../README.md)
- [Architecture Guide](../ARCHITECTURE.md)

---

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Last Updated**: January 2025
