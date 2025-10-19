-- Create main workload metrics table (partitioned by timestamp)
CREATE TABLE IF NOT EXISTS workload_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    service_id VARCHAR(50) NOT NULL,
    cluster_id VARCHAR(50) NOT NULL,
    cpu_utilization FLOAT NOT NULL,
    memory_utilization FLOAT NOT NULL,
    network_utilization FLOAT NOT NULL,
    cost FLOAT NOT NULL,
    scale_event INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_metric UNIQUE(timestamp, service_id, cluster_id)
) PARTITION BY RANGE (EXTRACT(MONTH FROM timestamp));

-- Create partitions for each month
CREATE TABLE workload_metrics_jan PARTITION OF workload_metrics
    FOR VALUES FROM (1) TO (2);
CREATE TABLE workload_metrics_feb PARTITION OF workload_metrics
    FOR VALUES FROM (2) TO (3);
CREATE TABLE workload_metrics_mar PARTITION OF workload_metrics
    FOR VALUES FROM (3) TO (4);
-- ... (repeat for all months)

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    service_id VARCHAR(50) NOT NULL,
    cluster_id VARCHAR(50) NOT NULL,
    predicted_cpu FLOAT NOT NULL,
    predicted_memory FLOAT NOT NULL,
    predicted_network FLOAT NOT NULL,
    recommended_action VARCHAR(100),
    confidence FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_prediction UNIQUE(timestamp, service_id, cluster_id)
);

-- Create recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    service_id VARCHAR(50) NOT NULL,
    cluster_id VARCHAR(50) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    priority INT DEFAULT 3,
    estimated_cost_savings FLOAT,
    estimated_improvement FLOAT,
    executed BOOLEAN DEFAULT FALSE,
    execution_timestamp TIMESTAMP,
    impact_achieved FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model performance tracking table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date TIMESTAMP NOT NULL,
    mae FLOAT NOT NULL,
    rmse FLOAT NOT NULL,
    mape FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    training_time_seconds FLOAT,
    data_size INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for query optimization
CREATE INDEX IF NOT EXISTS idx_workload_timestamp 
    ON workload_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_workload_service 
    ON workload_metrics(service_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_workload_cluster
    ON workload_metrics(cluster_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_workload_service_cluster
    ON workload_metrics(service_id, cluster_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_service
    ON predictions(service_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON predictions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recommendations_priority
    ON recommendations(priority, executed);

CREATE INDEX IF NOT EXISTS idx_recommendations_service
    ON recommendations(service_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_date
    ON model_performance(evaluation_date DESC);
