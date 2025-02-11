# pages/1_Data_Generation.py
import streamlit as st
import plotly.graph_objects as go
import torch
import numpy as np
from utils.vi_model import VariationalInference

# Initialize session state if needed
if 'vi_model' not in st.session_state:
    st.session_state.vi_model = VariationalInference()

st.title("Data Generation")

# Sidebar controls
st.sidebar.header("Data Generation Parameters")
true_theta1 = st.sidebar.slider("True θ₁", -3.0, 3.0, 2.0, 0.1)
true_theta2 = st.sidebar.slider("True θ₂", -3.0, 3.0, -1.0, 0.1)
noise_std = st.sidebar.slider("Noise Level", 0.1, 1.0, 0.5, 0.1)

# Get VI model from session state
vi_model = st.session_state.vi_model
vi_model.noise_std = noise_std

# Generate data
true_theta = torch.tensor([true_theta1, true_theta2])

# Add explanation
st.markdown("""
This page allows you to generate synthetic data for the variational inference process.
Adjust the parameters in the sidebar to see how they affect the generated data:

- **θ₁**: Controls the amplitude of the sine component
- **θ₂**: Controls the amplitude of the cosine component
- **Noise Level**: Controls the amount of random noise added to the data
""")

# Create two columns for visualization options
col1, col2 = st.columns([3, 1])

with col2:
    show_true_curve = st.checkbox("Show True Curve", value=True)
    show_grid = st.checkbox("Show Grid", value=True)
    point_size = st.slider("Point Size", 3, 10, 5)

with col1:
    # Generate data and create plot
    y_obs = vi_model.generate_data(true_theta)

    fig = go.Figure()

    if show_true_curve:
        # Plot true curve
        with torch.no_grad():
            y_true = vi_model.simulator(true_theta, vi_model.t)
        
        fig.add_trace(go.Scatter(
            x=vi_model.t.numpy(),
            y=y_true.numpy(),
            mode='lines',
            name='True Function',
            line=dict(color='rgba(0, 100, 80, 0.8)', width=2)
        ))

    # Plot observed data
    fig.add_trace(go.Scatter(
        x=vi_model.t.numpy(),
        y=y_obs.numpy(),
        mode='markers',
        name='Observed Data',
        marker=dict(
            size=point_size,
            color='rgba(100, 0, 80, 0.7)'
        )
    ))

    fig.update_layout(
        title="Generated Data with True Parameters",
        xaxis_title="t",
        yaxis_title="y",
        height=500,
        showlegend=True,
        plot_bgcolor='white' if show_grid else 'rgba(0,0,0,0)',
    )

    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

# Store data in session state
if st.button("Use This Dataset", type="primary"):
    st.session_state.true_theta = true_theta
    st.session_state.y_obs = y_obs
    st.session_state.vi_state = None  # Reset inference state
    st.success("✅ Dataset saved! You can now proceed to the Inference Process page.")
    
    # Display dataset summary
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Points", len(y_obs))
    with col2:
        st.metric("Data Range", f"{y_obs.min():.2f} to {y_obs.max():.2f}")
    with col3:
        st.metric("Noise Level", f"{noise_std:.2f}")