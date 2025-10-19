"""Database management and OLAP utilities."""

import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self, connection_string: str):
        """Initialize database manager.
        
        Args:
            connection_string: SQLAlchemy connection string
        """
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None

    def connect(self) -> Engine:
        """Create database connection."""
        if self.engine is None:
            self.engine = create_engine(self.connection_string, pool_pre_ping=True)
            logger.info(f"Connected to database: {self.connection_string.split('@')[1]}")
        return self.engine

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results as DataFrame
        """
        engine = self.connect()
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = False,
    ) -> int:
        """Insert DataFrame into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            index: Whether to write index
            
        Returns:
            Number of rows inserted
        """
        engine = self.connect()
        df.to_sql(table_name, engine, if_exists=if_exists, index=index)
        logger.info(f"Inserted {len(df)} rows into {table_name}")
        return len(df)

    def create_tables(self) -> None:
        """Create required OLAP tables."""
        engine = self.connect()
        with engine.begin() as conn:
            # Main workload metrics table
            conn.execute(text("""
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
                )
            """))

            # Predictions table
            conn.execute(text("""
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_prediction UNIQUE(timestamp, service_id, cluster_id)
                )
            """))

            # Recommendations table
            conn.execute(text("""
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Create indexes for query performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workload_timestamp 
                ON workload_metrics(timestamp)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workload_service 
                ON workload_metrics(service_id, timestamp)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_predictions_service
                ON predictions(service_id, timestamp)
            """))

        logger.info("Created database tables and indexes")


class OLAPQueryBuilder:
    """Build optimized OLAP queries."""

    @staticmethod
    def hourly_aggregates(service_id: str = None, cluster_id: str = None) -> str:
        """Query hourly aggregated metrics."""
        query = """
        SELECT
            DATE_TRUNC('hour', timestamp) as hour,
            service_id,
            cluster_id,
            AVG(cpu_utilization) as avg_cpu,
            MAX(cpu_utilization) as max_cpu,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) as p95_cpu,
            AVG(memory_utilization) as avg_memory,
            MAX(memory_utilization) as max_memory,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_utilization) as p95_memory,
            AVG(network_utilization) as avg_network,
            MAX(network_utilization) as max_network,
            SUM(cost) as total_cost,
            COUNT(*) as record_count,
            SUM(scale_event) as scale_events
        FROM workload_metrics
        WHERE 1=1
        """
        if service_id:
            query += f" AND service_id = '{service_id}'"
        if cluster_id:
            query += f" AND cluster_id = '{cluster_id}'"

        query += """
        GROUP BY DATE_TRUNC('hour', timestamp), service_id, cluster_id
        ORDER BY hour DESC, service_id, cluster_id
        """
        return query

    @staticmethod
    def top_resource_consumers(limit: int = 20) -> str:
        """Identify top resource consumers."""
        return f"""
        SELECT
            service_id,
            cluster_id,
            AVG(cpu_utilization) as avg_cpu,
            AVG(memory_utilization) as avg_memory,
            AVG(network_utilization) as avg_network,
            SUM(cost) as total_cost,
            COUNT(*) as observation_count
        FROM workload_metrics
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY service_id, cluster_id
        ORDER BY total_cost DESC
        LIMIT {limit}
        """

    @staticmethod
    def prediction_accuracy() -> str:
        """Compare predictions vs actual values."""
        return """
        SELECT
            DATE_TRUNC('hour', m.timestamp) as hour,
            m.service_id,
            AVG(m.cpu_utilization) as actual_cpu,
            AVG(p.predicted_cpu) as predicted_cpu,
            AVG(ABS(m.cpu_utilization - p.predicted_cpu)) as mae_cpu,
            AVG(m.memory_utilization) as actual_memory,
            AVG(p.predicted_memory) as predicted_memory,
            AVG(ABS(m.memory_utilization - p.predicted_memory)) as mae_memory,
            COUNT(*) as samples
        FROM workload_metrics m
        LEFT JOIN predictions p ON 
            m.service_id = p.service_id 
            AND m.timestamp = p.timestamp
        WHERE m.timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY hour, m.service_id
        ORDER BY hour DESC
        """

    @staticmethod
    def cost_optimization_potential() -> str:
        """Identify cost optimization opportunities."""
        return """
        SELECT
            service_id,
            cluster_id,
            AVG(cpu_utilization) as avg_cpu,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) as p95_cpu,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY cpu_utilization) as p5_cpu,
            SUM(cost) as total_cost,
            SUM(cost) * (1 - (PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_utilization) 
                / GREATEST(AVG(cpu_utilization), 0.1))) as potential_savings
        FROM workload_metrics
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY service_id, cluster_id
        HAVING AVG(cpu_utilization) > 0.1
        ORDER BY potential_savings DESC
        """

    @staticmethod
    def anomaly_detection() -> str:
        """Identify anomalous periods."""
        return """
        SELECT
            DATE_TRUNC('hour', timestamp) as hour,
            service_id,
            COUNT(*) as scale_events,
            AVG(cpu_utilization) as avg_cpu,
            MAX(cpu_utilization) as max_cpu,
            COUNT(CASE WHEN scale_event = 1 THEN 1 END) as scaling_instances
        FROM workload_metrics
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY hour, service_id
        HAVING COUNT(CASE WHEN scale_event = 1 THEN 1 END) > 0
        ORDER BY hour DESC
        """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage would be:
    # db = DatabaseManager("postgresql://user:password@localhost/ml_cloud")
    # db.create_tables()
