import streamlit as st
import torch
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.vi_model import VariationalInference, VIState

# Page title and setup
st.title("Inference Process")

# Check if data has been generated
if 'y_obs' not in st.session_state or st.session_state.y_obs is None:
    st.warning("⚠️ Please generate data first in the Data Generation page!")
    st.stop()

# Initialize model if needed
if 'vi_model' not in st.session_state:
    st.session_state.vi_model = VariationalInference()

# Get model from session state
vi_model = st.session_state.vi_model

# Sidebar controls for inference parameters
st.sidebar.header("Inference Parameters")
learning_rate = st.sidebar.slider(
    "Learning Rate", 
    min_value=0.001, 
    max_value=0.1, 
    value=0.01, 
    step=0.001,
    help="Controls how much the parameters update in each optimization step"
)

num_samples = st.sidebar.slider(
    "Monte Carlo Samples", 
    min_value=5, 
    max_value=50, 
    value=10, 
    step=5,
    help="Number of samples used to estimate the ELBO"
)

update_interval = st.sidebar.slider(
    "Update Interval",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    help="How often to update the visualization (iterations)"
)

# Main content area explanation
st.markdown("""
This page runs the variational inference optimization process. The algorithm will:
1. Initialize the variational parameters (mean and covariance)
2. Optimize them to maximize the Evidence Lower BOund (ELBO)
3. Show real-time updates of the optimization progress
""")

# Create placeholders for our visualizations
progress_bar = st.progress(0)
status_text = st.empty()
plot_placeholder = st.empty()

# Initialize the plot with subplots
def create_initial_plot():
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Data Fitting", "ELBO Optimization"),
        vertical_spacing=0.15
    )
    
    # Add initial data points to both subplots
    fig.add_trace(
        go.Scatter(
            x=vi_model.t.numpy(),
            y=st.session_state.y_obs.numpy(),
            mode='markers',
            name='Observed Data',
            marker=dict(color='rgba(100, 0, 80, 0.7)')
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Variational Inference Progress"
    )
    
    return fig

# Function to update plot during optimization
def update_plot(fig, y_current, elbo_history, iteration):
    # Update the fitted curve
    fig.data = [fig.data[0]]  # Keep only the observed data
    
    fig.add_trace(
        go.Scatter(
            x=vi_model.t.numpy(),
            y=y_current,
            mode='lines',
            name=f'Current Fit (Iteration {iteration})',
            line=dict(color='rgba(0, 100, 80, 0.8)')
        ),
        row=1, col=1
    )
    
    # Update ELBO history
    fig.add_trace(
        go.Scatter(
            x=list(range(len(elbo_history))),
            y=elbo_history,
            mode='lines',
            name='ELBO',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    return fig

# Run inference button
if st.button("Start Inference", type="primary"):
    try:
        # Reset the plot
        fig = create_initial_plot()
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Initialize variational parameters
        mu, l_params = vi_model.get_variational_params()
        optimizer = optim.Adam([mu, l_params], lr=learning_rate)
        elbo_history = []
        
        # Run optimization
        iterations = vi_model.num_iterations
        for i in range(iterations):
            # Optimization step
            optimizer.zero_grad()
            loss = -vi_model.elbo(mu, l_params, st.session_state.y_obs)
            loss.backward()
            optimizer.step()
            
            # Update progress and visualization
            if i % update_interval == 0:
                progress = (i + 1) / iterations
                progress_bar.progress(progress)
                status_text.text(f"Running iteration {i+1}/{iterations}")
                
                # Update ELBO history
                current_elbo = -loss.item()
                elbo_history.append(current_elbo)
                
                # Generate current prediction
                with torch.no_grad():
                    y_current = vi_model.simulator(mu, vi_model.t).numpy()
                
                # Update plot
                fig = update_plot(fig, y_current, elbo_history, i+1)
                plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Store final state
        final_state = VIState(
            elbo_history=elbo_history,
            final_mu=mu.detach(),
            final_l_params=l_params.detach(),
            y_obs=st.session_state.y_obs
        )
        st.session_state.vi_state = final_state
        
        # Show final results
        st.success("✅ Inference completed successfully!")
        
        # Display final parameter estimates
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Final θ₁ Estimate",
                f"{mu[0].item():.3f}",
                f"{mu[0].item() - st.session_state.true_theta[0].item():.3f}"
            )
        with col2:
            st.metric(
                "Final θ₂ Estimate",
                f"{mu[1].item():.3f}",
                f"{mu[1].item() - st.session_state.true_theta[1].item():.3f}"
            )
        
    except Exception as e:
        st.error(f"An error occurred during inference: {str(e)}")
        st.error("Please try adjusting the parameters or regenerating the data.")
else:
    if 'vi_state' in st.session_state and st.session_state.vi_state is not None:
        st.info("Inference has already been run. Click the button to run again with current parameters.")