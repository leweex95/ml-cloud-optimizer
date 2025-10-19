# Deployment Guide - ML Cloud Workload Optimizer

## Table of Contents
1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Provider Setup](#cloud-provider-setup)
5. [Monitoring & Logging](#monitoring--logging)
6. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites
- Python 3.10+
- Poetry
- PostgreSQL 12+ (or Docker)
- Git

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <repo-url>
cd ml-cloud-optimizer

# Install dependencies
poetry install

# Copy environment template
cp config/.env.example config/.env

# Edit configuration
nano config/.env
```

### Step 2: Database Setup

```bash
# Option A: Using Docker
docker run -d \
  --name ml-cloud-postgres \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=ml_cloud \
  -p 5432:5432 \
  postgres:15

# Option B: Local PostgreSQL
# Follow PostgreSQL installation for your OS
createdb ml_cloud

# Create tables
psql -U postgres -d ml_cloud -f sql/01_create_tables.sql
```

### Step 3: Generate Data

```bash
poetry run python -m ml_cloud_optimizer.data.generator
```

### Step 4: Run Jupyter Notebooks

```bash
# Start Jupyter
poetry run jupyter notebook

# Open http://localhost:8888
# Navigate to notebooks/01_EDA.ipynb
```

### Step 5: Launch Dashboard

```bash
# Terminal 1: Start MLflow server
poetry run mlflow ui --backend-store-uri postgresql://postgres:changeme@localhost/ml_cloud

# Terminal 2: Start dashboard
poetry run streamlit run dashboards/main.py

# Access at http://localhost:8501
```

---

## Docker Deployment

### Building the Image

```bash
# Build image
docker build -f deploy/docker/Dockerfile -t ml-cloud-optimizer:latest .

# Verify build
docker images | grep ml-cloud-optimizer

# Tag for registry (if using external registry)
docker tag ml-cloud-optimizer:latest <registry>/ml-cloud-optimizer:v0.1.0
```

### Running Container

```bash
# Basic run
docker run -d \
  --name ml-optimizer \
  -p 8000:8000 \
  -e DB_HOST=postgres \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  ml-cloud-optimizer:latest

# Run with volume mounts
docker run -d \
  --name ml-optimizer \
  -v /path/to/models:/app/models \
  -v /path/to/data:/app/data \
  -e DB_HOST=host.docker.internal \
  ml-cloud-optimizer:latest

# Check logs
docker logs -f ml-optimizer
```

### Docker Compose Setup

```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ml_cloud
      POSTGRES_PASSWORD: changeme
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --backend-store-uri postgresql://postgres:changeme@postgres/ml_cloud --default-artifact-root /mlflow

  ml-optimizer:
    build:
      context: .
      dockerfile: deploy/docker/Dockerfile
    depends_on:
      - postgres
      - mlflow
    environment:
      DB_HOST: postgres
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models

volumes:
  postgres_data:
EOF

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## Kubernetes Deployment

### Prerequisites
- kubectl configured
- Kubernetes cluster running (local minikube, EKS, GKE, AKS)
- Image pushed to registry (or available locally)

### Step 1: Create Namespace and Secrets

```bash
# Create namespace
kubectl create namespace ml-cloud-optimizer

# Create secrets
kubectl create secret generic ml-optimizer-secrets \
  --from-literal=DB_USER=postgres \
  --from-literal=DB_PASSWORD=changeme \
  --from-literal=MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -n ml-cloud-optimizer
```

### Step 2: Deploy Services

```bash
# Deploy PostgreSQL (using Helm or manifests)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --namespace ml-cloud-optimizer \
  --set auth.password=changeme

# Deploy MLflow
kubectl apply -f deploy/kubernetes/mlflow.yaml -n ml-cloud-optimizer

# Deploy ML Pipeline
kubectl apply -f deploy/kubernetes/deployment.yaml -n ml-cloud-optimizer
```

### Step 3: Verify Deployment

```bash
# Check pods
kubectl get pods -n ml-cloud-optimizer

# Check services
kubectl get svc -n ml-cloud-optimizer

# Get pod logs
kubectl logs -f <pod-name> -n ml-cloud-optimizer

# Check HPA status
kubectl get hpa -n ml-cloud-optimizer
```

### Step 4: Access Services

```bash
# Port forward to dashboard
kubectl port-forward svc/ml-optimizer 8000:8000 -n ml-cloud-optimizer

# Access at http://localhost:8000
```

### Scaling Configuration

```bash
# Manual scaling
kubectl scale deployment ml-optimizer-pipeline \
  --replicas=5 \
  -n ml-cloud-optimizer

# Check current replicas
kubectl get deployment -n ml-cloud-optimizer

# Watch HPA in action
kubectl get hpa -n ml-cloud-optimizer --watch
```

---

## Cloud Provider Setup

### Google Cloud (GCP) - BigQuery

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# Create BigQuery dataset
bq mk --dataset --location=US ml_cloud_optimizer

# Load data
bq load \
  --source_format=CSV \
  --skip_leading_rows=1 \
  ml_cloud_optimizer.workload_metrics \
  data/raw/cloud_workloads.csv \
  timestamp:TIMESTAMP,service_id:STRING,cluster_id:STRING,cpu_utilization:FLOAT,memory_utilization:FLOAT,network_utilization:FLOAT,cost:FLOAT,scale_event:INTEGER

# Deploy to Cloud Run
gcloud run deploy ml-optimizer \
  --source . \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars MLFLOW_TRACKING_URI=<your-uri>
```

### Amazon Web Services (AWS) - Redshift/S3

```bash
# Setup AWS CLI
aws configure

# Create S3 bucket for data
aws s3 mb s3://ml-cloud-optimizer-data

# Upload data
aws s3 cp data/raw/cloud_workloads.csv s3://ml-cloud-optimizer-data/

# Create Redshift cluster (via console or CLI)
aws redshift create-cluster \
  --cluster-identifier ml-optimizer \
  --node-type dc2.large \
  --master-username postgres \
  --master-user-password <password>
```

### Azure - Synapse Analytics

```bash
# Login to Azure
az login

# Create resource group
az group create --name ml-optimizer --location eastus

# Create Synapse workspace
az synapse workspace create \
  --name ml-optimizer-ws \
  --resource-group ml-optimizer \
  --storage-account <storage-account>

# Deploy container to ACI
az container create \
  --resource-group ml-optimizer \
  --name ml-optimizer \
  --image ml-cloud-optimizer:latest \
  --cpu 1 --memory 2
```

---

## Monitoring & Logging

### Prometheus Metrics

```bash
# Add to deployment.yaml
ports:
  - name: metrics
    containerPort: 8000

# Prometheus scrape config
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-optimizer'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ml-cloud-optimizer
```

### ELK Stack (Elasticsearch, Logstash, Kibana)

```bash
# Deploy ELK
helm repo add elastic https://helm.elastic.co
helm install elastic elastic/elasticsearch -n ml-cloud-optimizer
helm install kibana elastic/kibana -n ml-cloud-optimizer
```

### Datadog Integration

```bash
# Add Datadog agent
kubectl apply -f deploy/kubernetes/datadog-agent.yaml

# Update deployment with Datadog tags
env:
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
```

---

## Troubleshooting

### Common Issues

#### Database Connection Fails
```bash
# Check if postgres is running
docker ps | grep postgres

# Test connection
psql -h localhost -U postgres -d ml_cloud

# Check env variables
docker exec ml-optimizer env | grep DB
```

#### Out of Memory Errors
```bash
# Increase container memory
docker run -m 4g ml-cloud-optimizer:latest

# Or in k8s:
kubectl set resources deployment ml-optimizer-pipeline \
  --limits=memory=4Gi --requests=memory=2Gi \
  -n ml-cloud-optimizer
```

#### Model Not Found
```bash
# Check MLflow UI
# http://localhost:5000

# Check artifact storage
ls -la mlflow_artifacts/

# Verify model registration
mlflow models list
```

#### Slow Queries
```bash
# Check indexes
psql -U postgres -d ml_cloud << EOF
SELECT * FROM pg_indexes WHERE tablename = 'workload_metrics';
EOF

# Run explain plan
EXPLAIN ANALYZE SELECT * FROM workload_metrics WHERE timestamp > NOW() - INTERVAL '7 days';
```

---

## Performance Tuning

### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_workload_timestamp_service 
ON workload_metrics(timestamp DESC, service_id);

-- Analyze table
ANALYZE workload_metrics;

-- Check query performance
EXPLAIN ANALYZE 
SELECT DATE_TRUNC('hour', timestamp), AVG(cpu_utilization) 
FROM workload_metrics 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('hour', timestamp);
```

### Application Optimization
```python
# Cache feature computations
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_hourly_features(service_id: str, timestamp: str):
    # Cached feature retrieval
    pass
```

---

## Backup & Recovery

```bash
# Backup PostgreSQL database
pg_dump -U postgres ml_cloud > backup.sql

# Backup MLflow artifacts
tar -czf mlflow_artifacts.tar.gz mlflow_artifacts/

# Restore database
psql -U postgres ml_cloud < backup.sql
```

---

## Production Checklist

- [ ] Database backups configured
- [ ] Monitoring and alerts setup
- [ ] SSL/TLS certificates installed
- [ ] RBAC policies configured
- [ ] Network policies enforced
- [ ] Resource quotas set
- [ ] Pod disruption budgets defined
- [ ] Liveness/readiness probes configured
- [ ] Logging aggregation enabled
- [ ] Security scanning enabled
- [ ] Load testing completed
- [ ] Disaster recovery plan documented
