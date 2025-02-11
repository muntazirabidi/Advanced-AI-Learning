import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("Model Comparison")

# Performance Comparison
st.header("Performance Analysis")

tab1, tab2 = st.tabs(["Metrics", "Scaling Analysis"])

with tab1:
    # Create sample metrics
    metrics = pd.DataFrame({
        'Model': ['BNN', 'GP'],
        'RMSE': [0.15, 0.12],
        'NLL': [-0.85, -0.92],
        'Training Time (s)': [2.5, 1.8]
    })
    
    st.dataframe(metrics.style.highlight_min(axis=0, color='lightgreen'))
    
    # Detailed metric plots
    fig = go.Figure()
    for metric in ['RMSE', 'NLL', 'Training Time (s)']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics['Model'],
            y=metrics[metric],
            text=metrics[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        barmode='group'
    )
    st.plotly_chart(fig)

with tab2:
    # Scaling analysis
    st.subheader("Computational Scaling")
    
    # Generate scaling data
    dataset_sizes = np.array([100, 500, 1000, 5000, 10000])
    bnn_time = 0.001 * dataset_sizes + 0.5  # Linear scaling
    gp_time = 0.0001 * dataset_sizes**2 + 0.1  # Quadratic scaling
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataset_sizes, 
        y=bnn_time,
        name="BNN",
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=dataset_sizes,
        y=gp_time,
        name="GP",
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Training Time vs Dataset Size",
        xaxis_title="Number of Points",
        yaxis_title="Training Time (s)",
        yaxis_type="log"
    )
    st.plotly_chart(fig)

# Key Differences
st.header("Key Differences")

st.markdown("""
Let's analyze the fundamental differences between BNNs and GPs:

1. **Scalability**
   - BNNs demonstrate linear scaling with dataset size (O(N))
   - GPs have cubic scaling due to matrix operations (O(N³))
   - This makes BNNs more suitable for larger datasets

2. **Flexibility**
   - BNNs can learn complex feature representations through their hierarchical structure
   - GPs are limited by the expressiveness of their kernel functions
   - BNNs can adapt to varying levels of complexity in different regions of the input space

3. **Uncertainty Estimation**
   - Both methods provide principled uncertainty quantification
   - GPs offer exact posterior inference when using exact inference methods
   - BNNs typically use approximate inference (e.g., variational inference)
   - BNNs can separate epistemic and aleatoric uncertainty

4. **Interpretability**
   - GP kernels provide clear interpretation of assumptions about function properties
   - BNN uncertainty decomposition offers insights into different sources of uncertainty
   - GP hyperparameters have clear statistical interpretations
""")