-- Hourly Aggregated Metrics Query
-- Performance: ~100ms on 100M rows with proper indexes
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    service_id,
    cluster_id,
    COUNT(*) as record_count,
    AVG(cpu_utilization) as avg_cpu,
    MIN(cpu_utilization) as min_cpu,
    MAX(cpu_utilization) as max_cpu,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY cpu_utilization) as q1_cpu,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY cpu_utilization) as median_cpu,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY cpu_utilization) as q3_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) as p95_cpu,
    STDDEV(cpu_utilization) as stddev_cpu,
    
    AVG(memory_utilization) as avg_memory,
    MAX(memory_utilization) as max_memory,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_utilization) as p95_memory,
    
    AVG(network_utilization) as avg_network,
    MAX(network_utilization) as max_network,
    
    SUM(cost) as total_cost,
    AVG(cost) as avg_cost,
    SUM(CASE WHEN scale_event = 1 THEN 1 ELSE 0 END) as scale_events
FROM workload_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('hour', timestamp), service_id, cluster_id
ORDER BY hour DESC, service_id;

-- ============================================================================

-- Top Resource Consumers Query
-- Identify services consuming most resources over time windows
SELECT
    service_id,
    cluster_id,
    AVG(cpu_utilization) as avg_cpu,
    MAX(cpu_utilization) as peak_cpu,
    AVG(memory_utilization) as avg_memory,
    MAX(memory_utilization) as peak_memory,
    SUM(cost) as total_cost,
    COUNT(*) as observation_count,
    COUNT(CASE WHEN scale_event = 1 THEN 1 END) as scaling_events,
    SUM(cost) / NULLIF(COUNT(*), 0) as cost_per_observation
FROM workload_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY service_id, cluster_id
ORDER BY total_cost DESC
LIMIT 50;

-- ============================================================================

-- Prediction vs Actual Comparison Query
-- Evaluate model accuracy over time
SELECT
    DATE_TRUNC('hour', m.timestamp) as hour,
    m.service_id,
    COUNT(*) as samples,
    
    AVG(m.cpu_utilization) as actual_cpu,
    AVG(p.predicted_cpu) as predicted_cpu,
    ROUND(AVG(ABS(m.cpu_utilization - p.predicted_cpu))::numeric, 4) as mae_cpu,
    ROUND(SQRT(AVG(POWER(m.cpu_utilization - p.predicted_cpu, 2)))::numeric, 4) as rmse_cpu,
    
    AVG(m.memory_utilization) as actual_memory,
    AVG(p.predicted_memory) as predicted_memory,
    ROUND(AVG(ABS(m.memory_utilization - p.predicted_memory))::numeric, 4) as mae_memory,
    
    COUNT(CASE WHEN p.predicted_cpu IS NOT NULL THEN 1 END) as predictions_available
FROM workload_metrics m
LEFT JOIN predictions p ON 
    m.service_id = p.service_id 
    AND m.timestamp = p.timestamp
    AND m.cluster_id = p.cluster_id
WHERE m.timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', m.timestamp), m.service_id
ORDER BY hour DESC, mae_cpu DESC;

-- ============================================================================

-- Cost Optimization Opportunity Analysis
-- Find services with underutilized resources
SELECT
    service_id,
    cluster_id,
    AVG(cpu_utilization) as avg_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) as p95_cpu,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY cpu_utilization) as p5_cpu,
    MAX(cpu_utilization) as max_cpu,
    
    ROUND((1.0 - PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) 
        / GREATEST(AVG(cpu_utilization), 0.1)) * 100, 2) as potential_optimization_percent,
    
    SUM(cost) as total_cost,
    ROUND(SUM(cost) * (1.0 - PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) 
        / GREATEST(AVG(cpu_utilization), 0.1)), 2) as potential_savings,
    
    COUNT(*) as observation_count
FROM workload_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY service_id, cluster_id
HAVING AVG(cpu_utilization) > 0.05
ORDER BY potential_savings DESC NULLS LAST
LIMIT 100;

-- ============================================================================

-- Anomaly and Scaling Event Detection
-- Identify periods with high resource demand and scaling events
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    service_id,
    cluster_id,
    COUNT(*) as total_observations,
    COUNT(CASE WHEN scale_event = 1 THEN 1 END) as scale_events,
    ROUND(100.0 * COUNT(CASE WHEN scale_event = 1 THEN 1 END) / COUNT(*)::numeric, 2) as scale_event_rate,
    
    AVG(cpu_utilization) as avg_cpu,
    MAX(cpu_utilization) as max_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) as p95_cpu,
    
    AVG(memory_utilization) as avg_memory,
    MAX(memory_utilization) as max_memory,
    
    COUNT(CASE WHEN cpu_utilization > 0.8 THEN 1 END) as high_cpu_count,
    COUNT(CASE WHEN memory_utilization > 0.8 THEN 1 END) as high_memory_count
FROM workload_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp), service_id, cluster_id
HAVING COUNT(CASE WHEN scale_event = 1 THEN 1 END) > 0
    OR MAX(cpu_utilization) > 0.85
    OR MAX(memory_utilization) > 0.85
ORDER BY hour DESC, scale_events DESC;

-- ============================================================================

-- Daily Summary Report for Operations
-- High-level metrics for daily operational reporting
SELECT
    DATE_TRUNC('day', timestamp)::date as date,
    service_id,
    COUNT(*) as total_records,
    COUNT(DISTINCT cluster_id) as unique_clusters,
    
    ROUND(AVG(cpu_utilization)::numeric, 4) as avg_cpu,
    ROUND(MAX(cpu_utilization)::numeric, 4) as max_cpu,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization)::numeric, 4) as p95_cpu,
    
    ROUND(AVG(memory_utilization)::numeric, 4) as avg_memory,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_utilization)::numeric, 4) as p95_memory,
    
    ROUND(AVG(network_utilization)::numeric, 4) as avg_network,
    
    ROUND(SUM(cost)::numeric, 2) as total_cost,
    ROUND(AVG(cost)::numeric, 4) as avg_cost_per_record,
    
    COUNT(CASE WHEN scale_event = 1 THEN 1 END) as total_scale_events,
    COUNT(CASE WHEN cpu_utilization > 0.85 THEN 1 END) as high_cpu_events,
    COUNT(CASE WHEN memory_utilization > 0.85 THEN 1 END) as high_memory_events
FROM workload_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', timestamp), service_id
ORDER BY date DESC, total_cost DESC;

-- ============================================================================

-- Model Performance Trend Analysis
-- Track model performance metrics over time
SELECT
    DATE_TRUNC('day', evaluation_date)::date as date,
    model_name,
    COUNT(*) as evaluations,
    ROUND(AVG(mae)::numeric, 6) as avg_mae,
    ROUND(MIN(mae)::numeric, 6) as best_mae,
    ROUND(MAX(mae)::numeric, 6) as worst_mae,
    
    ROUND(AVG(rmse)::numeric, 6) as avg_rmse,
    ROUND(AVG(mape)::numeric, 2) as avg_mape,
    
    ROUND(AVG(f1_score)::numeric, 4) as avg_f1,
    ROUND(AVG(auc_roc)::numeric, 4) as avg_auc,
    
    ROUND(AVG(training_time_seconds)::numeric, 2) as avg_training_time,
    ROUND(AVG(data_size)::numeric, 0) as avg_data_size
FROM model_performance
GROUP BY DATE_TRUNC('day', evaluation_date), model_name
ORDER BY date DESC, model_name;

-- ============================================================================

-- Recommendation Impact Assessment
-- Measure impact of executed recommendations
SELECT
    DATE_TRUNC('day', execution_timestamp)::date as execution_date,
    action_type,
    COUNT(*) as total_recommendations,
    COUNT(CASE WHEN executed = TRUE THEN 1 END) as executed,
    
    ROUND(100.0 * COUNT(CASE WHEN executed = TRUE THEN 1 END) / COUNT(*)::numeric, 2) as execution_rate,
    
    ROUND(SUM(estimated_cost_savings)::numeric, 2) as total_estimated_savings,
    ROUND(AVG(estimated_cost_savings)::numeric, 2) as avg_estimated_savings,
    
    ROUND(AVG(estimated_improvement)::numeric, 4) as avg_estimated_improvement,
    ROUND(AVG(impact_achieved)::numeric, 4) as avg_impact_achieved,
    
    COUNT(CASE WHEN impact_achieved > 0 THEN 1 END) as successful_implementations
FROM recommendations
WHERE execution_timestamp IS NOT NULL
GROUP BY DATE_TRUNC('day', execution_timestamp), action_type
ORDER BY execution_date DESC;

-- ============================================================================

-- Cluster-wide Resource Efficiency
-- Overall cluster health and efficiency metrics
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    cluster_id,
    COUNT(DISTINCT service_id) as active_services,
    COUNT(*) as total_observations,
    
    ROUND(AVG(cpu_utilization)::numeric, 4) as cluster_avg_cpu,
    ROUND(STDDEV(cpu_utilization)::numeric, 4) as cpu_stddev,
    
    ROUND(AVG(memory_utilization)::numeric, 4) as cluster_avg_memory,
    ROUND(STDDEV(memory_utilization)::numeric, 4) as memory_stddev,
    
    ROUND(SUM(cost)::numeric, 2) as cluster_cost,
    
    COUNT(CASE WHEN cpu_utilization > 0.8 THEN 1 END) as high_cpu_observations,
    COUNT(CASE WHEN scale_event = 1 THEN 1 END) as scaling_events,
    
    ROUND(AVG(cpu_utilization + memory_utilization) / 2::numeric, 4) as avg_resource_utilization
FROM workload_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp), cluster_id
ORDER BY hour DESC, cluster_id;
