"""
Feast Feature Store Configuration for Cloud Workload Optimizer.

This module defines the feature repository for managing ML features:
- Feature definitions (entities, features, feature views)
- Data sources (PostgreSQL, Parquet, Delta)
- Registry configuration for production deployments
"""

from feast import (
    FeatureStore,
    FeatureView,
    Entity,
    Field,
    FileSource,
    FeatureService,
    PushSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Int64, UnixTimestamp
from datetime import datetime, timedelta
import os


# Define entities
workload_entity = Entity(
    name="service",
    description="Cloud service/workload",
    owner="ml-platform",
    tags={"team": "ml-ops"},
)

cluster_entity = Entity(
    name="cluster",
    description="Kubernetes cluster",
    owner="ml-platform",
    tags={"team": "platform"},
)


# Define data sources - Parquet files (can be replaced with PostgreSQL)
workload_source = FileSource(
    path="./data/processed/workload_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Alternative: PostgreSQL source (commented for reference)
"""
from feast import PostgreSQLSource

workload_source = PostgreSQLSource(
    name="workload_metrics",
    host="localhost",
    port=5432,
    database="ml_cloud",
    table="workload_metrics",
    user="postgres",
    password="changeme",
    event_timestamp_column="timestamp",
    created_timestamp_column="created_at",
)
"""


# Define features
# Core workload features
workload_features = FeatureView(
    name="workload_features",
    entities=[workload_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="cpu_utilization", dtype=Float32),
        Field(name="memory_utilization", dtype=Float32),
        Field(name="network_io_mbps", dtype=Float32),
        Field(name="disk_io_mbps", dtype=Float32),
        Field(name="request_latency_ms", dtype=Float32),
        Field(name="error_rate", dtype=Float32),
    ],
    online=True,
    source=workload_source,
    tags={
        "category": "core",
        "criticality": "high",
    },
    description="Core workload metrics",
    owner="ml-platform",
)


# Temporal features
temporal_features = FeatureView(
    name="temporal_features",
    entities=[workload_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="day_of_month", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_peak_hour", dtype=Int64),
        Field(name="hour_sin", dtype=Float32),
        Field(name="hour_cos", dtype=Float32),
        Field(name="dow_sin", dtype=Float32),
        Field(name="dow_cos", dtype=Float32),
    ],
    online=True,
    source=workload_source,
    tags={
        "category": "temporal",
        "criticality": "medium",
    },
    description="Temporal features",
    owner="ml-platform",
)


# Rolling statistics
rolling_features = FeatureView(
    name="rolling_features",
    entities=[workload_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="cpu_mean_6h", dtype=Float32),
        Field(name="cpu_std_6h", dtype=Float32),
        Field(name="cpu_min_6h", dtype=Float32),
        Field(name="cpu_max_6h", dtype=Float32),
        Field(name="memory_mean_6h", dtype=Float32),
        Field(name="memory_std_6h", dtype=Float32),
        Field(name="cpu_mean_24h", dtype=Float32),
        Field(name="cpu_std_24h", dtype=Float32),
        Field(name="cpu_min_24h", dtype=Float32),
        Field(name="cpu_max_24h", dtype=Float32),
        Field(name="memory_mean_24h", dtype=Float32),
        Field(name="memory_std_24h", dtype=Float32),
        Field(name="cpu_mean_7d", dtype=Float32),
        Field(name="cpu_std_7d", dtype=Float32),
    ],
    online=True,
    source=workload_source,
    tags={
        "category": "rolling",
        "criticality": "high",
    },
    description="Rolling statistics over different time windows",
    owner="ml-platform",
)


# Lag features
lag_features = FeatureView(
    name="lag_features",
    entities=[workload_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="cpu_lag_1h", dtype=Float32),
        Field(name="cpu_lag_2h", dtype=Float32),
        Field(name="cpu_lag_4h", dtype=Float32),
        Field(name="cpu_lag_8h", dtype=Float32),
        Field(name="cpu_lag_12h", dtype=Float32),
        Field(name="cpu_lag_24h", dtype=Float32),
        Field(name="memory_lag_1h", dtype=Float32),
        Field(name="memory_lag_2h", dtype=Float32),
        Field(name="memory_lag_4h", dtype=Float32),
        Field(name="memory_lag_8h", dtype=Float32),
        Field(name="memory_lag_12h", dtype=Float32),
        Field(name="memory_lag_24h", dtype=Float32),
    ],
    online=True,
    source=workload_source,
    tags={
        "category": "lag",
        "criticality": "high",
    },
    description="Lag features for autoregressive modeling",
    owner="ml-platform",
)


# Interaction features
@on_demand_feature_view(
    sources=[workload_features],
    schema=[
        Field(name="total_utilization", dtype=Float32),
        Field(name="cpu_memory_ratio", dtype=Float32),
        Field(name="utilization_variance", dtype=Float32),
        Field(name="efficiency_score", dtype=Float32),
    ],
)
def interaction_features(inputs):
    """Compute interaction features from base workload features."""
    df = inputs["workload_features"]
    
    df["total_utilization"] = (
        df["cpu_utilization"] + df["memory_utilization"]
    ) / 2.0
    
    df["cpu_memory_ratio"] = (
        df["cpu_utilization"] / (df["memory_utilization"] + 0.001)
    )
    
    df["utilization_variance"] = (
        (df["cpu_utilization"] - df["memory_utilization"]) ** 2
    )
    
    df["efficiency_score"] = (
        1.0 - (df["cpu_utilization"] + df["memory_utilization"]) / 2.0
    )
    
    return df


# Cluster-level aggregations
cluster_features = FeatureView(
    name="cluster_features",
    entities=[cluster_entity],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="total_services", dtype=Int64),
        Field(name="avg_cpu_utilization", dtype=Float32),
        Field(name="avg_memory_utilization", dtype=Float32),
        Field(name="max_cpu_utilization", dtype=Float32),
        Field(name="max_memory_utilization", dtype=Float32),
        Field(name="cluster_health_score", dtype=Float32),
    ],
    online=True,
    source=workload_source,
    tags={
        "category": "cluster",
        "criticality": "medium",
    },
    description="Cluster-level aggregated features",
    owner="ml-platform",
)


# Define feature services for easy retrieval
training_fs = FeatureService(
    name="cloud_optimizer_training",
    features=[
        workload_features,
        temporal_features,
        rolling_features,
        lag_features,
        cluster_features,
    ],
    tags={
        "use_case": "training",
        "environment": "dev",
    },
)

inference_fs = FeatureService(
    name="cloud_optimizer_inference",
    features=[
        workload_features,
        temporal_features,
        rolling_features,
        lag_features,
        interaction_features,
        cluster_features,
    ],
    tags={
        "use_case": "inference",
        "environment": "prod",
    },
)

realtime_fs = FeatureService(
    name="cloud_optimizer_realtime",
    features=[
        workload_features,
        temporal_features,
        lag_features,
        cluster_features,
    ],
    tags={
        "use_case": "realtime",
        "latency_sla_ms": 100,
        "environment": "prod",
    },
)


# Feature store initialization function
def init_feature_store() -> FeatureStore:
    """Initialize and return the feature store."""
    fs = FeatureStore(repo_path=os.path.dirname(__file__))
    return fs


if __name__ == "__main__":
    # Initialize and apply feature store
    fs = init_feature_store()
    print("Feature Store initialized successfully!")
    print(f"Repository path: {fs.repo_path}")
    print(f"Registry path: {fs.registry_path}")
    
    # List all feature views
    print("\nFeature Views:")
    print(f"  - {workload_features.name}")
    print(f"  - {temporal_features.name}")
    print(f"  - {rolling_features.name}")
    print(f"  - {lag_features.name}")
    print(f"  - {cluster_features.name}")
    
    # List all feature services
    print("\nFeature Services:")
    print(f"  - {training_fs.name}")
    print(f"  - {inference_fs.name}")
    print(f"  - {realtime_fs.name}")
