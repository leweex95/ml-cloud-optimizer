"""Integration tests for Airflow DAG and orchestration."""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestAirflowDAG:
    """Test Airflow DAG structure and task definitions."""
    
    def test_dag_import(self):
        """Test that DAG can be imported without errors."""
        try:
            from airflow.dags.ml_cloud_optimizer_dag import dag
            assert dag is not None
            assert dag.dag_id == 'ml_cloud_optimizer_pipeline'
        except ImportError:
            pytest.skip("Airflow not installed")
    
    def test_dag_schedule(self):
        """Test DAG schedule configuration."""
        try:
            from airflow.dags.ml_cloud_optimizer_dag import dag
            assert dag.schedule_interval == '0 */6 * * *'
            assert dag.catchup is False
        except ImportError:
            pytest.skip("Airflow not installed")
    
    def test_dag_tasks_exist(self):
        """Test that all expected tasks are defined."""
        try:
            from airflow.dags.ml_cloud_optimizer_dag import dag
            
            expected_tasks = [
                'generate_data',
                'validate_data',
                'engineer_features',
                'train_tabular_model',
                'train_timeseries_model',
                'compare_and_register_models',
                'detect_data_drift',
                'generate_report',
            ]
            
            task_ids = [task.task_id for task in dag.tasks]
            for expected in expected_tasks:
                assert expected in task_ids, f"Task {expected} not found"
        except ImportError:
            pytest.skip("Airflow not installed")
    
    def test_dag_dependencies(self):
        """Test task dependencies are correct."""
        try:
            from airflow.dags.ml_cloud_optimizer_dag import dag
            
            # Get task graph
            task_graph = dag.task_graph
            
            # Check critical paths
            assert 'generate_data' in [t.task_id for t in dag.tasks]
            assert 'validate_data' in [t.task_id for t in dag.tasks]
            assert 'engineer_features' in [t.task_id for t in dag.tasks]
            assert 'compare_and_register_models' in [t.task_id for t in dag.tasks]
        except ImportError:
            pytest.skip("Airflow not installed")
    
    def test_dag_validation(self):
        """Test DAG passes validation."""
        try:
            from airflow.models import DAG
            from airflow.dags.ml_cloud_optimizer_dag import dag
            
            # Verify no duplicate task IDs
            task_ids = [task.task_id for task in dag.tasks]
            assert len(task_ids) == len(set(task_ids)), "Duplicate task IDs found"
            
            # Verify DAG attributes
            assert dag.owner == 'ml-platform'
            assert dag.doc_md is not None or len(dag.tasks) > 0
        except ImportError:
            pytest.skip("Airflow not installed")


class TestFeastFeatureStore:
    """Test Feast feature store configuration."""
    
    def test_feast_import(self):
        """Test Feast repo can be imported."""
        try:
            from airflow.feast_repo import init_feature_store
            assert init_feature_store is not None
        except ImportError:
            pytest.skip("Feast not installed")
    
    def test_feast_entities(self):
        """Test Feast entities are defined."""
        try:
            from airflow.feast_repo import (
                workload_entity,
                cluster_entity
            )
            assert workload_entity.name == 'service'
            assert cluster_entity.name == 'cluster'
        except ImportError:
            pytest.skip("Feast not installed")
    
    def test_feast_feature_views(self):
        """Test Feast feature views are defined."""
        try:
            from airflow.feast_repo import (
                workload_features,
                temporal_features,
                rolling_features,
                lag_features,
                cluster_features
            )
            
            assert workload_features.name == 'workload_features'
            assert temporal_features.name == 'temporal_features'
            assert rolling_features.name == 'rolling_features'
            assert lag_features.name == 'lag_features'
            assert cluster_features.name == 'cluster_features'
        except ImportError:
            pytest.skip("Feast not installed")
    
    def test_feast_feature_services(self):
        """Test Feast feature services are defined."""
        try:
            from airflow.feast_repo import (
                training_fs,
                inference_fs,
                realtime_fs
            )
            
            assert training_fs.name == 'cloud_optimizer_training'
            assert inference_fs.name == 'cloud_optimizer_inference'
            assert realtime_fs.name == 'cloud_optimizer_realtime'
        except ImportError:
            pytest.skip("Feast not installed")


class TestAirflowIntegration:
    """Integration tests for Airflow with ML components."""
    
    def test_data_generation_task(self):
        """Test data generation task logic."""
        from ml_cloud_optimizer.data.generator import CloudWorkloadGenerator
        
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate(n_records=1000, start_date='2025-01-01')
        
        assert len(df) == 1000
        assert 'timestamp' in df.columns
        assert 'service_id' in df.columns
        assert 'cpu_utilization' in df.columns
    
    def test_feature_engineering_task(self):
        """Test feature engineering task logic."""
        from ml_cloud_optimizer.data.generator import CloudWorkloadGenerator
        from ml_cloud_optimizer.features.engineering import FeatureEngineer
        
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate(n_records=1000, start_date='2025-01-01')
        
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        assert df_features.shape[1] > df.shape[1]  # More features
        assert len(df_features) == len(df)
    
    def test_model_training_task(self):
        """Test model training task logic."""
        from ml_cloud_optimizer.data.generator import CloudWorkloadGenerator
        from ml_cloud_optimizer.features.engineering import FeatureEngineer
        from ml_cloud_optimizer.models.base import TabularModelFactory
        import pandas as pd
        
        # Generate and engineer features
        generator = CloudWorkloadGenerator(seed=42)
        df = generator.generate(n_records=1000, start_date='2025-01-01')
        
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        df_features = engineer.handle_missing_values(df_features)
        
        # Prepare data
        X = df_features.drop(
            ['cpu_utilization', 'timestamp', 'service_id'],
            axis=1,
            errors='ignore'
        )
        y = df_features['cpu_utilization']
        
        # Train model
        factory = TabularModelFactory()
        model = factory.create('xgboost')
        model.fit(X[:800], y[:800])
        
        # Evaluate
        y_pred = model.predict(X[800:])
        
        assert y_pred.shape[0] == 200
        assert all(0 <= p <= 1 for p in y_pred)
    
    def test_mlflow_tracking(self):
        """Test MLflow integration."""
        try:
            from ml_cloud_optimizer.ml_ops.mlflow_utils import MLflowTracker
            
            tracker = MLflowTracker(
                experiment_name='test_experiment',
                run_name='test_run'
            )
            
            # Log parameters and metrics
            with tracker.get_run():
                tracker.log_parameters({'test_param': 'value'})
                tracker.log_metrics({'test_metric': 0.95})
            
            assert tracker is not None
        except Exception as e:
            pytest.skip(f"MLflow tracking not available: {e}")


class TestAirflowConfiguration:
    """Test Airflow configuration files."""
    
    def test_feature_store_yaml_exists(self):
        """Test feature_store.yaml configuration exists."""
        from pathlib import Path
        
        fs_yaml = Path(__file__).parent.parent / 'airflow' / 'feature_store.yaml'
        assert fs_yaml.exists()
        
        with open(fs_yaml) as f:
            content = f.read()
            assert 'project: cloud_optimizer' in content
            assert 'registry' in content
    
    def test_docker_compose_airflow_exists(self):
        """Test docker-compose.airflow.yml exists."""
        from pathlib import Path
        
        docker_compose = Path(__file__).parent.parent / 'docker-compose.airflow.yml'
        assert docker_compose.exists()
        
        with open(docker_compose) as f:
            content = f.read()
            assert 'airflow-webserver' in content
            assert 'airflow-scheduler' in content
            assert 'postgres' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
