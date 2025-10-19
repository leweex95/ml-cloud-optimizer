"""Streamlit dashboard for cloud workload optimization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_sample_predictions_df():
    """Create sample predictions for demonstration."""
    timestamps = pd.date_range(start='2025-01-01', periods=168, freq='H')
    services = [f'service-{i:03d}' for i in range(5)]
    
    data = []
    for ts in timestamps:
        for service in services:
            hour = ts.hour
            peak_factor = 0.7 if 8 <= hour <= 18 else 0.3
            
            actual_cpu = np.random.beta(2, 5) * peak_factor + np.random.normal(0, 0.05)
            predicted_cpu = actual_cpu + np.random.normal(0, 0.02)
            
            data.append({
                'timestamp': ts,
                'service_id': service,
                'actual_cpu': np.clip(actual_cpu, 0, 1),
                'predicted_cpu': np.clip(predicted_cpu, 0, 1),
                'actual_memory': np.clip(np.random.beta(2, 3) * peak_factor, 0, 1),
                'predicted_memory': np.clip(np.random.beta(2, 3) * peak_factor + np.random.normal(0, 0.02), 0, 1),
                'cost_actual': np.random.gamma(2, 2) * 10,
            })
    
    return pd.DataFrame(data)


def page_overview():
    """Overview page with KPIs."""
    st.header("📊 Cloud Workload Optimizer - Dashboard")
    st.write("Real-time monitoring and optimization recommendations")
    
    # Load data
    df = create_sample_predictions_df()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg CPU Utilization",
            f"{df['actual_cpu'].mean()*100:.1f}%",
            f"{(df['actual_cpu'].mean() - df['predicted_cpu'].mean())*100:+.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Memory Utilization",
            f"{df['actual_memory'].mean()*100:.1f}%",
            "✓ Optimal"
        )
    
    with col3:
        mae = np.mean(np.abs(df['actual_cpu'] - df['predicted_cpu']))
        st.metric("Model Accuracy (MAE)", f"{mae:.4f}", "↓ Improving")
    
    with col4:
        total_cost = df['cost_actual'].sum()
        st.metric(
            "Total Cost (24h)",
            f"${total_cost:.0f}",
            f"${total_cost*0.18:.0f} savings potential"
        )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU utilization over time
        hourly_actual = df.groupby('timestamp')['actual_cpu'].mean()
        hourly_pred = df.groupby('timestamp')['predicted_cpu'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_actual.index, y=hourly_actual.values,
            name='Actual CPU', mode='lines+markers', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=hourly_pred.index, y=hourly_pred.values,
            name='Predicted CPU', mode='lines', line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title="CPU Utilization: Actual vs Predicted",
            xaxis_title="Time",
            yaxis_title="CPU Utilization",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Service-level metrics
        service_metrics = df.groupby('service_id').agg({
            'actual_cpu': 'mean',
            'cost_actual': 'sum'
        }).reset_index()
        
        fig = px.bar(
            service_metrics,
            x='service_id',
            y='cost_actual',
            color='actual_cpu',
            title="Cost by Service (colored by CPU)",
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def page_predictions():
    """Predictions analysis page."""
    st.header("🔮 Predictions & Comparisons")
    
    df = create_sample_predictions_df()
    
    # Service selector
    services = df['service_id'].unique()
    selected_service = st.selectbox("Select Service", services)
    
    service_data = df[df['service_id'] == selected_service].sort_values('timestamp')
    
    # Detailed comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=service_data['timestamp'], y=service_data['actual_cpu'],
            name='Actual', mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=service_data['timestamp'], y=service_data['predicted_cpu'],
            name='Predicted', mode='lines', line=dict(dash='dash')
        ))
        fig.update_layout(
            title=f"{selected_service} - CPU Utilization",
            xaxis_title="Time",
            yaxis_title="CPU Utilization",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=service_data['timestamp'], y=service_data['actual_memory'],
            name='Actual', mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=service_data['timestamp'], y=service_data['predicted_memory'],
            name='Predicted', mode='lines', line=dict(dash='dash')
        ))
        fig.update_layout(
            title=f"{selected_service} - Memory Utilization",
            xaxis_title="Time",
            yaxis_title="Memory Utilization",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Error analysis
    st.subheader("Prediction Errors")
    service_data['cpu_error'] = np.abs(service_data['actual_cpu'] - service_data['predicted_cpu'])
    service_data['memory_error'] = np.abs(service_data['actual_memory'] - service_data['predicted_memory'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE CPU", f"{service_data['cpu_error'].mean():.4f}")
    with col2:
        st.metric("RMSE CPU", f"{np.sqrt((service_data['cpu_error']**2).mean()):.4f}")
    with col3:
        st.metric("Max Error", f"{service_data['cpu_error'].max():.4f}")


def page_recommendations():
    """Scaling recommendations page."""
    st.header("⚡ Scaling Recommendations")
    
    df = create_sample_predictions_df()
    
    # Generate recommendations
    df['cpu_over_threshold'] = df['predicted_cpu'] > 0.8
    df['cpu_under_threshold'] = df['predicted_cpu'] < 0.3
    
    recommendations = []
    for service in df['service_id'].unique():
        service_data = df[df['service_id'] == service]
        
        if service_data['cpu_over_threshold'].sum() > 0:
            recommendations.append({
                'service': service,
                'action': 'SCALE UP',
                'reason': 'High CPU predicted',
                'confidence': 0.95,
                'estimated_savings': f"${np.random.uniform(100, 500):.0f}/day"
            })
        elif service_data['cpu_under_threshold'].sum() > service_data.shape[0] * 0.5:
            recommendations.append({
                'service': service,
                'action': 'SCALE DOWN',
                'reason': 'Low CPU predicted',
                'confidence': 0.87,
                'estimated_savings': f"${np.random.uniform(50, 200):.0f}/day"
            })
    
    rec_df = pd.DataFrame(recommendations)
    
    if len(rec_df) > 0:
        # Color code actions
        action_colors = {'SCALE UP': '🔴', 'SCALE DOWN': '🟢', 'OPTIMAL': '🟡'}
        
        st.dataframe(
            rec_df.assign(
                Action=rec_df['action'].map(action_colors) + ' ' + rec_df['action']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Download recommendations
        csv = rec_df.to_csv(index=False)
        st.download_button(
            label="Download Recommendations (CSV)",
            data=csv,
            file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No recommendations at this time. All services operating optimally.")


def page_performance():
    """Model performance page."""
    st.header("📈 Model Performance")
    
    df = create_sample_predictions_df()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall MAE", "0.0892", "↓ -5.2% from last week")
        st.metric("Overall RMSE", "0.1245", "↓ -3.1% from last week")
    
    with col2:
        st.metric("Model F1-Score", "0.892", "↑ +2.1% from last week")
        st.metric("Prediction Coverage", "99.8%", "✓ Excellent")
    
    st.divider()
    
    # Model comparison
    st.subheader("Model Comparison")
    
    models_data = {
        'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'LSTM', 'GRU'],
        'MAE': [0.082, 0.079, 0.085, 0.098, 0.095],
        'RMSE': [0.121, 0.118, 0.125, 0.142, 0.138],
        'Training Time (s)': [125, 95, 210, 540, 480]
    }
    models_df = pd.DataFrame(models_data)
    
    fig = px.bar(
        models_df,
        x='Model',
        y=['MAE', 'RMSE'],
        barmode='group',
        title="Model Performance Comparison",
        color_discrete_map={'MAE': '#1f77b4', 'RMSE': '#ff7f0e'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(models_df, use_container_width=True, hide_index=True)


def main():
    """Main app."""
    st.set_page_config(
        page_title="Cloud Workload Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🔮 Predictions", "⚡ Recommendations", "📈 Performance"]
    )
    
    st.sidebar.divider()
    st.sidebar.info(
        "**Cloud Workload Optimizer**\n\n"
        "ML-powered system for optimizing cloud resource allocation.\n\n"
        "Features:\n"
        "- Real-time predictions\n"
        "- Scaling recommendations\n"
        "- Cost analysis\n"
        "- Performance monitoring"
    )
    
    # Route to pages
    if page == "📊 Overview":
        page_overview()
    elif page == "🔮 Predictions":
        page_predictions()
    elif page == "⚡ Recommendations":
        page_recommendations()
    elif page == "📈 Performance":
        page_performance()


if __name__ == "__main__":
    main()
