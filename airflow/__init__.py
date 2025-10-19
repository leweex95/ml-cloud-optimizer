"""
Airflow initialization and configuration.

Sets up the Airflow environment, connections, and variables for the ML pipeline.
"""

import os
from pathlib import Path

# Airflow configuration
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', str(Path(__file__).parent))

# Set Airflow environment
os.environ['AIRFLOW_HOME'] = AIRFLOW_HOME
os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = os.path.join(AIRFLOW_HOME, 'dags')
os.environ['AIRFLOW__CORE__PLUGINS_FOLDER'] = os.path.join(AIRFLOW_HOME, 'plugins')
os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
os.environ['AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS'] = 'False'
os.environ['AIRFLOW__CORE__UNIT_TEST_MODE'] = 'True'

# Database
os.environ['AIRFLOW__CORE__SQL_ALCHEMY_CONN'] = 'sqlite:///airflow.db'
os.environ['AIRFLOW__CORE__EXECUTOR'] = 'LocalExecutor'

# Logging
os.environ['AIRFLOW__LOGGING__BASE_LOG_FOLDER'] = os.path.join(AIRFLOW_HOME, 'logs')
os.environ['AIRFLOW__LOGGING__DAG_PROCESSOR_MANAGER_LOG_LOCATION'] = os.path.join(
    AIRFLOW_HOME, 'logs', 'dag_processor_manager'
)

# Security
os.environ['AIRFLOW__CORE__FERNET_KEY'] = 'AIRFLOW_FERNET_KEY_PLACEHOLDER'

# Features
os.environ['AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION'] = 'True'
os.environ['AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL'] = '300'

print(f"✓ Airflow initialized with AIRFLOW_HOME={AIRFLOW_HOME}")


def setup_connections():
    """Setup Airflow connections for external services."""
    from airflow.models import Connection
    from airflow import settings
    from sqlalchemy.orm import sessionmaker
    
    # Create session
    Session = sessionmaker(bind=settings.engine)
    session = Session()
    
    try:
        # PostgreSQL connection
        postgres_conn = session.query(Connection).filter(
            Connection.conn_id == 'postgres_ml_cloud'
        ).first()
        
        if not postgres_conn:
            postgres_conn = Connection(
                conn_id='postgres_ml_cloud',
                conn_type='postgres',
                host='localhost',
                port=5432,
                schema='ml_cloud',
                login='postgres',
                password='changeme',
                description='PostgreSQL for ML Cloud Optimizer'
            )
            session.add(postgres_conn)
            print("✓ PostgreSQL connection created")
        
        # MLflow connection
        mlflow_conn = session.query(Connection).filter(
            Connection.conn_id == 'mlflow_prod'
        ).first()
        
        if not mlflow_conn:
            mlflow_conn = Connection(
                conn_id='mlflow_prod',
                conn_type='http',
                host='localhost',
                port=5000,
                description='MLflow tracking server'
            )
            session.add(mlflow_conn)
            print("✓ MLflow connection created")
        
        session.commit()
        print("✓ Connections setup complete")
        
    except Exception as e:
        print(f"✗ Error setting up connections: {e}")
        session.rollback()
    finally:
        session.close()


def setup_variables():
    """Setup Airflow variables for the ML pipeline."""
    from airflow.models import Variable
    
    variables = {
        'ml_environment': 'production',
        'data_path': './data',
        'models_path': './models',
        'mlflow_tracking_uri': 'http://localhost:5000',
        'postgres_host': 'localhost',
        'postgres_port': '5432',
        'postgres_db': 'ml_cloud',
        'feature_store_path': './airflow',
        'notification_email': 'ml-alerts@example.com',
        'max_parallel_tasks': '4',
        'model_retraining_frequency': 'daily',
    }
    
    for key, value in variables.items():
        try:
            var = Variable.get(key)
        except:
            Variable.set(key, value)
            print(f"✓ Variable set: {key} = {value}")


if __name__ == '__main__':
    print("Setting up Airflow...")
    
    # Uncomment the following lines to setup connections and variables
    # setup_connections()
    # setup_variables()
    
    print("✓ Airflow setup complete!")
