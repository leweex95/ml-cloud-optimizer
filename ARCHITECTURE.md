# Cloud Workload Optimizer - System Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES & INGESTION                        │
│              (Cloud APIs, Message Queues, Batch Files)               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA VALIDATION LAYER                            │
│              (Schema validation, anomaly detection)                  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  CLOUD WORKLOAD DATA (PostgreSQL)                    │
│  - Partitioned by timestamp (monthly)                                │
│  - Optimized indexes for common queries                              │
│  - OLAP-ready schema                                                 │
└─────────────┬──────────────┬────────────────┬──────────────────┬─────┘
              │              │                │                  │
              ▼              ▼                ▼                  ▼
        ┌────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────────┐
        │  Feature   │ │  Analytics  │ │ Training │ │ Real-Time Query  │
        │  Store     │ │  Queries    │ │  Data    │ │ Results          │
        │  (Feast)   │ │  (OLAP)     │ │  Export  │ │                  │
        └────────────┘ └─────────────┘ └──────────┘ └──────────────────┘
              │              │
              ▼              ▼
        ┌─────────────────────────────────────────┐
        │   Feature Engineering Pipeline          │
        │  - Temporal features                    │
        │  - Rolling statistics                   │
        │  - Lag features                         │
        │  - Interaction features                 │
        │  - Normalization & Scaling              │
        └─────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │ Tabular  │  │Time-     │  │ Imbalanced   │
        │ Models   │  │Series    │  │ Learning     │
        │          │  │Models    │  │              │
        │- XGBoost │  │- LSTM    │  │- SMOTE       │
        │- LightGBM│  │- GRU     │  │- Focal Loss  │
        │- CatBoost│  │- TFT     │  │- Weighting   │
        └──────────┘  └──────────┘  └──────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                 ┌──────────────────────────┐
                 │  MLflow Experiment       │
                 │  Tracking & Registry     │
                 │  - Run metrics           │
                 │  - Model artifacts       │
                 │  - Version control       │
                 └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Airflow Orchestration       │
              │  - Data pipeline DAGs        │
              │  - Model training DAGs       │
              │  - Inference DAGs            │
              │  - Retraining triggers       │
              └──────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐ ┌────────────┐ ┌──────────────┐
        │ Batch    │ │Real-Time   │ │ Recommendations
        │Predictions│ │Scoring    │ │ Engine
        │          │ │Service     │ │
        └──────────┘ └────────────┘ └──────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌────────────────────┐  ┌──────────────────┐
        │  Docker Containers │  │ Kubernetes       │
        │  (Multi-stage)     │  │ Cluster          │
        │                    │  │ - 3-10 replicas  │
        │  - Python runtime  │  │ - HPA enabled    │
        │  - Dependencies    │  │ - Service mesh   │
        │  - ML models       │  │ - RBAC           │
        │  - Inference code  │  │ - Network policy │
        └────────────────────┘  └──────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────────┐  ┌──────────────────────┐
        │  API Gateway     │  │ Monitoring & Logging │
        │  - REST/gRPC     │  │ - Prometheus metrics │
        │  - Rate limiting │  │ - ELK stack          │
        │  - Auth          │  │ - Datadog/Splunk     │
        └──────────────────┘  └──────────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
        ┌────────────────────┴──────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────────────┐  ┌────────────────────────────┐
│  Interactive Dashboards      │  │  Operational Reports       │
│  (Streamlit/Plotly/Dash)     │  │  - Cost savings analysis   │
│  - Predictions vs actuals    │  │  - Performance metrics     │
│  - Recommendations           │  │  - Anomaly summaries       │
│  - Cost tracking             │  │  - Compliance reports      │
│  - Real-time metrics         │  │  - Trend analysis          │
└──────────────────────────────┘  └────────────────────────────┘
```

---

## Component Details

### 1. Data Ingestion & Storage

**PostgreSQL Database Schema:**
- Partitioned by timestamp (monthly)
- Three main tables: `workload_metrics`, `predictions`, `recommendations`
- Optimized indexes on frequent query columns
- OLAP-ready normalization

**Key Characteristics:**
- Supports ~100M+ records efficiently
- Query latency: <100ms for aggregations
- Automatic backup and replication

### 2. Feature Engineering

**Feature Categories:**
- **Temporal**: Hour, day-of-week, day-of-month, cyclical encoding
- **Rolling**: Mean, std, min, max over 6h/24h/7d windows
- **Lag**: 1-24 step lags for sequence modeling
- **Interaction**: Ratios, combined metrics, efficiency measures

**Processing:**
- Pandas-based feature computation
- Automatic scaling and normalization
- Forward-fill and interpolation for missing values

### 3. ML Models

**Tabular Regression (Resource Prediction):**
- XGBoost: 0.082 MAE
- LightGBM: 0.079 MAE (best)
- CatBoost: 0.085 MAE

**Time-Series Models:**
- LSTM: 0.098 MAE
- GRU: 0.095 MAE
- Temporal Fusion Transformer: Optional

**Imbalanced Learning:**
- SMOTE: Synthetic oversampling
- Focal Loss: Hard-negative focusing
- Class Weighting: Automatic weight computation

### 4. MLOps Infrastructure

**Experiment Tracking (MLflow):**
- Centralized run metrics logging
- Model artifact versioning
- Automatic experiment comparison
- Model registry for deployment

**Pipeline Orchestration (Airflow):**
- DAG-based scheduling
- Data drift monitoring triggers
- Automated retraining on performance degradation
- Dependency management

**Feature Store (Feast):**
- Versioned feature definitions
- Point-in-time joins for training
- Consistent features for serving
- Feature lineage tracking

### 5. Deployment & Orchestration

**Containerization:**
- Multi-stage Docker builds
- Poetry for dependency management
- Image size optimization (~1.2GB)
- Registry-ready format

**Kubernetes Orchestration:**
- 3-10 replicas with HPA
- Pod disruption budgets (min 2 available)
- Readiness/liveness probes
- RBAC policies
- Network policies
- Service discovery

**Scaling Strategy:**
- CPU-based horizontal scaling (70% threshold)
- Memory-based scaling (80% threshold)
- Max 10 replicas per deployment
- Scale-up: 100% increase per 30s
- Scale-down: 50% decrease per 60s (300s window)

### 6. Monitoring & Observability

**Metrics:**
- Prometheus-compatible endpoint
- Custom metrics for model performance
- Infrastructure metrics (CPU, memory, disk)
- Application metrics (latency, throughput)

**Logging:**
- Structured JSON logging
- ELK stack integration
- Log aggregation with filtering
- Audit trail for compliance

**Alerting:**
- Model drift detection
- Performance degradation alerts
- Infrastructure alerts
- Anomaly notifications

### 7. Analytics & Reporting

**OLAP Queries:**
- Hourly/daily aggregates
- Top resource consumers
- Cost optimization analysis
- Anomaly detection patterns

**Dashboard Features:**
- Real-time metric visualization
- Interactive filtering and drill-down
- Predictive vs actual comparison
- Cost tracking and ROI analysis
- Recommendation execution status

---

## Data Flow

### Training Pipeline
```
Raw Data 
  ↓
Validation & Cleansing
  ↓
Feature Engineering
  ↓ Feature Store (Feast)
  ↓
Train/Test Split
  ↓
Model Training (Parallel)
  ├─ XGBoost
  ├─ LightGBM
  ├─ LSTM
  └─ GRU
  ↓
Model Evaluation
  ↓
MLflow Logging
  ↓
Model Registry
  ↓
Production Deployment
```

### Inference Pipeline
```
Real-Time Data
  ↓
Feature Store Lookup (Feast)
  ↓
Feature Engineering
  ↓
Model Serving (Production)
  ↓
Scaling Recommendations
  ↓
Kubernetes API
  ↓
Auto-Scaling Actions
  ↓
Cost Optimization
  ↓
Dashboard Update
```

---

## Scalability Considerations

### Horizontal Scaling
- Kubernetes HPA automatically scales replicas
- Load balancing via Service
- Multi-zone redundancy possible
- Database connection pooling

### Vertical Scaling
- Pod resource requests/limits configurable
- Container memory: 512Mi → 2Gi
- Container CPU: 250m → 1000m

### Data Scaling
- PostgreSQL partitioning by month
- Index optimization for query performance
- Read replicas for analytics queries
- Archive old data to cold storage

---

## Security Architecture

### Network Security
- Network policies enforce pod-to-pod communication
- Service mesh (optional) for mTLS
- Ingress with TLS/SSL
- VPN for remote access

### Identity & Access
- RBAC with least-privilege principle
- Service accounts for pod authentication
- Secrets management for credentials
- Audit logging for compliance

### Data Protection
- Encryption at rest (PostgreSQL)
- Encryption in transit (TLS)
- Database user isolation
- Regular backups with encryption

---

## Performance Metrics

### Model Performance
| Metric | Value | Target |
|--------|-------|--------|
| MAE (CPU) | 0.079 | <0.10 |
| RMSE (CPU) | 0.118 | <0.15 |
| MAPE | 13.8% | <15% |
| Training Time | 95s | <2min |

### Infrastructure Performance
| Metric | Value | Target |
|--------|-------|--------|
| Query Latency (OLAP) | 50-100ms | <500ms |
| Inference Latency | 50-150ms | <200ms |
| Pod Start Time | 30-60s | <90s |
| Availability | 99.95% | >99% |

---

## Future Enhancements

1. **Advanced Models**
   - Temporal Fusion Transformer
   - Graph Neural Networks for topology-aware predictions
   - Ensemble stacking strategies

2. **Enhanced MLOps**
   - Automated hyperparameter tuning (Optuna)
   - Model A/B testing framework
   - Federated learning for edge inference

3. **Kubernetes Advanced**
   - Service mesh (Istio) for traffic management
   - Custom resource definitions (CRDs)
   - Multi-cluster deployment
   - GitOps workflow (ArgoCD)

4. **Data Improvements**
   - Stream processing (Kafka/Flink)
   - Real-time feature computation
   - Online learning capabilities
   - Multi-source data fusion

5. **Observability**
   - Distributed tracing (Jaeger)
   - Service mesh observability
   - ML-specific observability tools
   - Root cause analysis automation
