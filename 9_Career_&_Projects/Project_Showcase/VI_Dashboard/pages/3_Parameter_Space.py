import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import multivariate_normal
import torch
from utils.vi_model import VariationalInference

st.title("Parameter Space Analysis")

# Check prerequisites
if 'vi_state' not in st.session_state or st.session_state.vi_state is None:
    st.warning("⚠️ Please run inference first in the Inference Process page!")
    st.stop()

# Get model and state from session
vi_model = st.session_state.vi_model
vi_state = st.session_state.vi_state

# Educational explanation
st.markdown("""
### Understanding the Parameter Space

The parameter space visualization shows how the variational inference algorithm has 
approximated the posterior distribution of the parameters θ₁ and θ₂. Here's what you're seeing:

- The **contour plot** shows the density of the approximate posterior distribution
- The **red point** marks the true parameter values used to generate the data
- The **green point** shows our estimated parameters
- The **surrounding region** indicates our uncertainty about the parameters

The shape and spread of the distribution tell us about parameter correlations and uncertainties.
""")

# Sidebar controls for visualization
st.sidebar.header("Visualization Controls")
n_grid = st.sidebar.slider("Grid Resolution", 50, 200, 100, 10)
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500, 100)
show_samples = st.sidebar.checkbox("Show Random Samples", value=True)

# Generate grid for contour plot
L = vi_model.get_L(vi_state.final_l_params)
mu = vi_state.final_mu
Sigma = L @ L.t()

# Create grid
margin = 3 * torch.sqrt(torch.diagonal(Sigma))
x = np.linspace(mu[0].item() - margin[0].item(), 
                mu[0].item() + margin[0].item(), n_grid)
y = np.linspace(mu[1].item() - margin[1].item(), 
                mu[1].item() + margin[1].item(), n_grid)
X, Y = np.meshgrid(x, y)

# Compute posterior density
pos = np.dstack((X, Y))
rv = multivariate_normal(mu.detach().numpy(), Sigma.detach().numpy())
Z = rv.pdf(pos)

# Create main figure
fig = go.Figure()

# Add contour plot
fig.add_trace(go.Contour(
    x=x, y=y, z=Z,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='Posterior Density'),
    name='Posterior Distribution'
))

# Add true parameters
fig.add_trace(go.Scatter(
    x=[st.session_state.true_theta[0].item()],
    y=[st.session_state.true_theta[1].item()],
    mode='markers',
    marker=dict(
        size=12,
        color='red',
        symbol='star'
    ),
    name='True Parameters'
))

# Add estimated parameters
fig.add_trace(go.Scatter(
    x=[mu[0].item()],
    y=[mu[1].item()],
    mode='markers',
    marker=dict(
        size=12,
        color='green',
        symbol='circle'
    ),
    name='Estimated Parameters'
))

# Add random samples if requested
if show_samples:
    samples = np.array([
        vi_model.sample_theta(mu, L).detach().numpy()
        for _ in range(n_samples)
    ])
    fig.add_trace(go.Scatter(
        x=samples[:,0],
        y=samples[:,1],
        mode='markers',
        marker=dict(
            size=3,
            color='rgba(100, 100, 100, 0.3)',
            symbol='circle'
        ),
        name='Random Samples'
    ))

# Update layout
fig.update_layout(
    title="Parameter Space: Posterior Distribution",
    xaxis_title="θ₁",
    yaxis_title="θ₂",
    height=700,
    showlegend=True,
    hovermode='closest'
)

# Show plot
st.plotly_chart(fig, use_container_width=True)

# Add parameter analysis
st.subheader("Parameter Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Parameter Estimates")
    st.markdown(f"""
    - θ₁: {mu[0].item():.3f} ± {2*torch.sqrt(Sigma[0,0]).item():.3f}
    - θ₂: {mu[1].item():.3f} ± {2*torch.sqrt(Sigma[1,1]).item():.3f}
    """)

with col2:
    st.markdown("#### Parameter Correlation")
    correlation = Sigma[0,1].item() / torch.sqrt(Sigma[0,0] * Sigma[1,1]).item()
    st.markdown(f"""
    - Correlation coefficient: {correlation:.3f}
    - This indicates {'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'weak'} 
      {'positive' if correlation > 0 else 'negative'} correlation between parameters
    """)

# Add interpretation guidance
st.markdown("""
### Interpreting the Results

The visualization above helps us understand several key aspects of our inference:

1. **Parameter Uncertainty**: The spread of the distribution shows how certain we are about 
   each parameter. A tighter distribution indicates more certainty.

2. **Parameter Correlation**: The shape of the distribution tells us if the parameters are 
   correlated. An angled or tilted distribution suggests correlation between parameters.

3. **Accuracy**: The distance between the true parameters (red star) and our estimates 
   (green circle) shows how well the inference performed.
""")