"""Data generation module for synthetic cloud workload data."""

import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from faker import Faker


class CloudWorkloadGenerator:
    """Generate synthetic cloud workload telemetry data."""

    def __init__(self, seed: int = 42):
        """Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)

    def generate_workloads(
        self,
        n_records: int = 100000,
        n_services: int = 20,
        n_clusters: int = 5,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """Generate synthetic cloud workload dataset.
        
        Args:
            n_records: Number of records to generate
            n_services: Number of unique services
            n_clusters: Number of unique clusters
            start_date: Start timestamp
            end_date: End timestamp
            
        Returns:
            DataFrame with cloud workload metrics
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()

        # Generate base data
        timestamps = []
        service_ids = []
        cluster_ids = []
        cpu_util = []
        memory_util = []
        network_util = []
        costs = []
        scale_events = []

        # Pre-generate service and cluster IDs
        services = [f"service-{i:03d}" for i in range(n_services)]
        clusters = [f"cluster-{i:02d}" for i in range(n_clusters)]

        time_delta = (end_date - start_date) / max(1, n_records - 1)

        for i in range(n_records):
            ts = start_date + (time_delta * i)
            service = random.choice(services)
            cluster = random.choice(clusters)

            # Base metrics with temporal patterns
            hour = ts.hour
            day_of_week = ts.weekday()

            # Peak hours (8-18): higher utilization
            peak_factor = 0.7 if 8 <= hour <= 18 else 0.3
            # Weekdays have higher load
            weekday_factor = 0.8 if day_of_week < 5 else 0.5

            # Generate metrics with realistic patterns
            cpu = np.random.beta(2, 5) * peak_factor * weekday_factor + np.random.normal(0, 0.05)
            memory = np.random.beta(2, 3) * peak_factor * weekday_factor + np.random.normal(0, 0.05)
            network = np.random.gamma(2, 2) * peak_factor * 0.1

            # Clip to valid ranges
            cpu = np.clip(cpu, 0, 1)
            memory = np.clip(memory, 0, 1)
            network = np.clip(network, 0, 100)

            # Cost proportional to utilization
            cost = (cpu * 0.5 + memory * 0.3 + network * 0.1) * random.uniform(0.8, 1.2)

            # Imbalanced: ~3% peak load events (scaling needed)
            scale_event = 1 if random.random() < 0.03 and cpu > 0.8 else 0

            timestamps.append(ts)
            service_ids.append(service)
            cluster_ids.append(cluster)
            cpu_util.append(cpu)
            memory_util.append(memory)
            network_util.append(network)
            costs.append(cost)
            scale_events.append(scale_event)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "service_id": service_ids,
            "cluster_id": cluster_ids,
            "cpu_utilization": cpu_util,
            "memory_utilization": memory_util,
            "network_utilization": network_util,
            "cost": costs,
            "scale_event": scale_events,
        })

        return df.sort_values("timestamp").reset_index(drop=True)

    def add_anomalies(
        self, df: pd.DataFrame, anomaly_rate: float = 0.02
    ) -> pd.DataFrame:
        """Add anomalies to dataset.
        
        Args:
            df: Input DataFrame
            anomaly_rate: Fraction of records to add anomalies to
            
        Returns:
            DataFrame with anomalies
        """
        df = df.copy()
        n_anomalies = max(1, int(len(df) * anomaly_rate))
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)

        for idx in anomaly_indices:
            # Spike in CPU/Memory
            df.loc[idx, "cpu_utilization"] = min(1.0, np.random.normal(0.85, 0.1))
            df.loc[idx, "memory_utilization"] = min(1.0, np.random.normal(0.85, 0.1))
            df.loc[idx, "network_utilization"] *= 3
            df.loc[idx, "scale_event"] = 1

        return df

    def add_missing_values(
        self, df: pd.DataFrame, missing_rate: float = 0.01
    ) -> pd.DataFrame:
        """Add missing values to dataset.
        
        Args:
            df: Input DataFrame
            missing_rate: Fraction of values to set as NaN
            
        Returns:
            DataFrame with missing values
        """
        df = df.copy()
        columns_to_mask = [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
        ]
        n_missing = max(1, int(len(df) * missing_rate))

        for col in columns_to_mask:
            missing_indices = np.random.choice(len(df), n_missing, replace=False)
            df.loc[missing_indices, col] = np.nan

        return df


def create_sample_dataset(output_path: str = None) -> pd.DataFrame:
    """Create and optionally save sample dataset."""
    generator = CloudWorkloadGenerator(seed=42)
    df = generator.generate_workloads(n_records=100000, n_services=20, n_clusters=5)
    df = generator.add_anomalies(df, anomaly_rate=0.02)
    df = generator.add_missing_values(df, missing_rate=0.01)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")

    return df


if __name__ == "__main__":
    df = create_sample_dataset("data/raw/cloud_workloads.csv")
    print(f"Generated {len(df)} records")
    print(df.head())
    print(df.info())
