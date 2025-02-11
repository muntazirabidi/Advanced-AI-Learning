import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

st.title("Uncertainty Analysis")

# Check prerequisites
if 'vi_state' not in st.session_state or st.session_state.vi_state is None:
    st.warning("⚠️ Please run inference first in the Inference Process page!")
    st.stop()

# Get model and state from session
vi_model = st.session_state.vi_model
vi_state = st.session_state.vi_state

# Educational explanation
st.markdown("""
### Understanding Prediction Uncertainty

This page helps you explore the uncertainty in our model's predictions. We visualize:

1. **Prediction Intervals**: Shows the range where we expect most observations to fall
2. **Residual Analysis**: Examines how well our model fits the data
3. **Uncertainty Decomposition**: Breaks down sources of uncertainty

These visualizations help us understand how confident we can be in our model's predictions.
""")

# Sidebar controls
st.sidebar.header("Analysis Controls")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500, 100)
confidence_level = st.sidebar.slider("Confidence Level (%)", 50, 99, 95, 5)

# Generate predictions
@st.cache_data
def generate_predictions(_mu, _l_params, n_samples):
    """
    Generate prediction samples and compute statistics.
    Leading underscores in parameter names tell Streamlit not to hash these parameters.
    
    Args:
        _mu: Mean parameter tensor (unhashed)
        _l_params: Cholesky factor parameters (unhashed)
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of predictions and their statistics
    """
    predictions = []
    for _ in range(n_samples):
        theta_sample = vi_model.sample_theta(_mu, vi_model.get_L(_l_params))
        with torch.no_grad():
            pred = vi_model.simulator(theta_sample, vi_model.t)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)
    
    # Calculate prediction intervals
    alpha = (100 - confidence_level) / 100
    pred_lower = np.percentile(predictions, alpha/2 * 100, axis=0)
    pred_upper = np.percentile(predictions, (1-alpha/2) * 100, axis=0)
    
    return predictions, pred_mean, pred_std, pred_lower, pred_upper

# Generate prediction data
predictions, pred_mean, pred_std, pred_lower, pred_upper = generate_predictions(
    vi_state.final_mu,
    vi_state.final_l_params,
    n_samples
)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs([
    "Prediction Intervals", 
    "Residual Analysis",
    "Uncertainty Decomposition"
])

with tab1:
    st.markdown("""
    ### Prediction Intervals
    
    The shaded region shows the range where we expect future observations to fall with the
    specified confidence level. This accounts for both parameter uncertainty and observation noise.
    """)
    
    fig = go.Figure()
    
    # Add prediction interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([vi_model.t.numpy(), vi_model.t.numpy()[::-1]]),
        y=np.concatenate([pred_upper, pred_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{confidence_level}% Prediction Interval'
    ))
    
    # Add mean prediction
    fig.add_trace(go.Scatter(
        x=vi_model.t.numpy(),
        y=pred_mean,
        mode='lines',
        line=dict(color='rgb(0,100,80)'),
        name='Mean Prediction'
    ))
    
    # Add observed data
    fig.add_trace(go.Scatter(
        x=vi_model.t.numpy(),
        y=vi_state.y_obs.numpy(),
        mode='markers',
        marker=dict(color='black', size=5),
        name='Observed Data'
    ))
    
    fig.update_layout(
        title=f"Predictions with {confidence_level}% Confidence Intervals",
        xaxis_title="t",
        yaxis_title="y",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    ### Residual Analysis
    
    Residuals are the differences between observed data and model predictions. Their
    patterns can reveal if our model is capturing all important features of the data.
    """)
    
    # Calculate residuals
    residuals = vi_state.y_obs.numpy() - pred_mean
    
    # Create residual plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residuals vs. Fitted",
            "Residual Distribution",
            "Residual Time Series",
            "Q-Q Plot"
        ]
    )
    
    # 1. Residuals vs. Fitted
    fig.add_trace(
        go.Scatter(
            x=pred_mean,
            y=residuals,
            mode='markers',
            name='Residuals'
        ),
        row=1, col=1
    )
    
    # 2. Residual Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Residual Dist',
            nbinsx=20
        ),
        row=1, col=2
    )
    
    # 3. Residual Time Series
    fig.add_trace(
        go.Scatter(
            x=vi_model.t.numpy(),
            y=residuals,
            mode='lines+markers',
            name='Residual Series'
        ),
        row=2, col=1
    )
    
    # 4. Q-Q Plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.random.normal(
        0, np.std(residuals), len(residuals)
    )
    theoretical_quantiles.sort()
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Q-Q Plot'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add residual statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Residual", f"{np.mean(residuals):.3f}")
    with col2:
        st.metric("Residual Std Dev", f"{np.std(residuals):.3f}")
    with col3:
        st.metric("Max Absolute Residual", f"{np.max(np.abs(residuals)):.3f}")

with tab3:
    st.markdown("""
    ### Uncertainty Decomposition
    
    We can break down prediction uncertainty into different sources:
    1. **Parameter Uncertainty**: Uncertainty in our estimates of θ₁ and θ₂
    2. **Observation Noise**: Random variation in the data
    """)
    
    # Calculate uncertainty components
    param_std = pred_std
    obs_std = vi_model.noise_std * np.ones_like(param_std)
    total_std = np.sqrt(param_std**2 + obs_std**2)
    
    # Create uncertainty decomposition plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vi_model.t.numpy(),
        y=param_std,
        mode='lines',
        name='Parameter Uncertainty',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=vi_model.t.numpy(),
        y=obs_std,
        mode='lines',
        name='Observation Noise',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=vi_model.t.numpy(),
        y=total_std,
        mode='lines',
        name='Total Uncertainty',
        line=dict(color='purple')
    ))
    
    fig.update_layout(
        title="Decomposition of Prediction Uncertainty",
        xaxis_title="t",
        yaxis_title="Standard Deviation",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add uncertainty summary
    st.markdown(f"""
    ### Uncertainty Summary
    
    The average contribution of each uncertainty source across the time range:
    - Parameter Uncertainty: {np.mean(param_std):.3f} (std. dev.)
    - Observation Noise: {vi_model.noise_std:.3f} (std. dev.)
    - Total Uncertainty: {np.mean(total_std):.3f} (std. dev.)
    
    This means that {'parameter uncertainty' if np.mean(param_std) > vi_model.noise_std else 'observation noise'} 
    is the dominant source of uncertainty in our predictions.
    """)

    # Add interpretation guidance
    st.markdown("""
    ### Interpreting the Results
    
    The uncertainty analysis reveals several important aspects of our model:
    
    1. **Prediction Coverage**: The prediction intervals show where we expect future 
    observations to fall. If our model is well-calibrated, approximately {confidence_level}% 
    of new observations should fall within these bands.
    
    2. **Residual Patterns**: The residual analysis helps us identify potential model 
    deficiencies:
        - Systematic patterns in residuals suggest the model might be missing important 
          features
        - The Q-Q plot helps us check if residuals follow a normal distribution
        - The residual distribution should be centered around zero for an unbiased model
    
    3. **Uncertainty Sources**: Understanding the relative contributions of different 
    uncertainty sources helps us know where we could improve:
        - High parameter uncertainty suggests we might need more data or a better 
          inference procedure
        - High observation noise suggests inherent variability in the system that 
          cannot be reduced without improving measurements
    """)

    # Add downloadable results
    if st.button("Download Analysis Results"):
        results = {
            "predictions": {
                "mean": pred_mean.tolist(),
                "lower_bound": pred_lower.tolist(),
                "upper_bound": pred_upper.tolist(),
                "time_points": vi_model.t.numpy().tolist()
            },
            "residuals": {
                "values": residuals.tolist(),
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals))
            },
            "uncertainty": {
                "parameter_std": param_std.tolist(),
                "observation_std": obs_std.tolist(),
                "total_std": total_std.tolist()
            },
            "metadata": {
                "confidence_level": confidence_level,
                "n_samples": n_samples,
                "noise_level": float(vi_model.noise_std)
            }
        }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(results, indent=2),
            file_name="uncertainty_analysis.json",
            mime="application/json"
        )

    # Footer with additional resources
    st.markdown("""
    ---
    ### Additional Resources
    
    To learn more about uncertainty analysis in variational inference:
    
    1. **Residual Analysis**: Understanding patterns in model residuals can help identify 
    areas where the model could be improved or where its assumptions might be violated.
    
    2. **Prediction Intervals**: These intervals combine both parameter uncertainty and 
    observation noise to give a complete picture of prediction uncertainty.
    
    3. **Uncertainty Decomposition**: Breaking down uncertainty into its components helps 
    us understand where we might focus efforts to improve model performance.
    
    The visualizations and analyses on this page are designed to help you make informed 
    decisions about model reliability and potential areas for improvement.
    """)