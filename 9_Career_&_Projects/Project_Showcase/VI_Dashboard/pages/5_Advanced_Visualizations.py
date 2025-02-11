import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import json
import pandas as pd
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Advanced Visualizations - VI Dashboard",
    layout="wide"
)

st.title("Advanced Visualizations")

# Comprehensive error checking for required session state variables
required_vars = ['vi_model', 'vi_state', 'true_theta', 'y_obs']
missing_vars = [var for var in required_vars if var not in st.session_state]

if missing_vars:
    st.warning(f"⚠️ Please complete the previous steps first! Missing: {', '.join(missing_vars)}")
    st.stop()

# Get model and state from session
vi_model = st.session_state.vi_model
vi_state = st.session_state.vi_state

# Sidebar controls for visualization parameters
with st.sidebar:
    st.header("Visualization Settings")
    
    n_samples_correlation = st.slider(
        "Number of Correlation Samples",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="More samples give smoother contours but take longer to compute"
    )
    
    n_samples_elbo = st.slider(
        "Number of ELBO Component Samples",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Samples used to estimate ELBO component distributions"
    )
    
    plot_height = st.slider(
        "Plot Height",
        min_value=400,
        max_value=1000,
        value=600,
        step=50,
        help="Adjust the height of visualization plots"
    )

# Cache function for generating posterior samples
@st.cache_data
def generate_posterior_samples(_mu, _L, n_samples=1000):
    """
    Generate and cache samples from the posterior distribution.
    Leading underscores in parameter names tell Streamlit not to hash these parameters.
    
    Args:
        _mu: Mean parameter tensor (unhashed)
        _L: Cholesky factor matrix (unhashed)
        n_samples: Number of samples to generate
        
    Returns:
        numpy array of samples
    """
    return np.array([
        vi_model.sample_theta(_mu, _L).detach().numpy()
        for _ in range(n_samples)
    ])

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs([
    "Parameter Correlation",
    "ELBO Components",
    "Residual Analysis"
])

with tab1:
    st.header("Parameter Correlation Analysis")
    
    # Generate samples from posterior
    L = vi_model.get_L(vi_state.final_l_params)
    samples = generate_posterior_samples(vi_state.final_mu, L, n_samples_correlation)
    
    # Create correlation plot
    fig = go.Figure()
    
    # Add 2D histogram contour
    fig.add_trace(go.Histogram2dContour(
        x=samples[:,0],
        y=samples[:,1],
        colorscale='Viridis',
        name='Parameter Correlation',
        showscale=True,
        colorbar=dict(title='Density')
    ))
    
    # Add true and estimated parameters
    fig.add_trace(go.Scatter(
        x=[st.session_state.true_theta[0].item()],
        y=[st.session_state.true_theta[1].item()],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='True Parameters'
    ))
    
    fig.add_trace(go.Scatter(
        x=[vi_state.final_mu[0].item()],
        y=[vi_state.final_mu[1].item()],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        name='Estimated Parameters'
    ))
    
    fig.update_layout(
        title="Parameter Correlation Analysis",
        xaxis_title="θ₁",
        yaxis_title="θ₂",
        height=plot_height,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display correlation statistics
    corr_coef = np.corrcoef(samples[:,0], samples[:,1])[0,1]
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
    with col2:
        correlation_strength = (
            "strong" if abs(corr_coef) > 0.7 else
            "moderate" if abs(corr_coef) > 0.3 else
            "weak"
        )
        st.metric(
            "Correlation Strength", 
            f"{correlation_strength} {'positive' if corr_coef > 0 else 'negative'}"
        )
    
    st.markdown("""
    ### Understanding Parameter Correlation
    
    This visualization reveals the relationship between the two parameters (θ₁ and θ₂) in our model:
    
    - The **contour plot** shows the density of samples from the posterior distribution
    - The **red star** marks the true parameter values
    - The **green circle** shows our estimated parameters
    - The **correlation coefficient** quantifies the linear relationship between parameters
    
    Strong correlation between parameters can indicate:
    1. Parameter redundancy in the model
    2. Insufficient data to separately identify parameters
    3. Inherent relationships in the underlying system
    """)

with tab2:
    st.header("ELBO Components Analysis")
    
    # Calculate ELBO components
    components = {
        'log_likelihood': [],
        'log_prior': [],
        'log_q': []
    }
    
    # Progress bar for ELBO component calculation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    L = vi_model.get_L(vi_state.final_l_params)
    for i in range(n_samples_elbo):
        theta_sample = vi_model.sample_theta(vi_state.final_mu, L)
        components['log_likelihood'].append(
            vi_model.log_likelihood(theta_sample, vi_state.y_obs).item()
        )
        components['log_prior'].append(
            vi_model.log_prior(theta_sample).item()
        )
        components['log_q'].append(
            vi_model.log_q(theta_sample, vi_state.final_mu, L).item()
        )
        
        # Update progress
        progress = (i + 1) / n_samples_elbo
        progress_bar.progress(progress)
        status_text.text(f"Calculating ELBO components: {i+1}/{n_samples_elbo}")
    
    status_text.empty()
    
    # Create violin plots
    fig = go.Figure()
    colors = ['rgb(67,147,195)', 'rgb(178,24,43)', 'rgb(77,146,33)']
    
    for (component, values), color in zip(components.items(), colors):
        fig.add_trace(go.Violin(
            y=values,
            name=component,
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.3)')
        ))
    
    fig.update_layout(
        title="Distribution of ELBO Components",
        yaxis_title="Value",
        height=plot_height,
        showlegend=True,
        violingap=0.3,
        violinmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Component statistics
    for component, values in components.items():
        with st.expander(f"{component} Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{np.mean(values):.3f}")
            with col2:
                st.metric("Std Dev", f"{np.std(values):.3f}")
            with col3:
                st.metric("Range", f"{np.max(values) - np.min(values):.3f}")
    
    st.markdown("""
    ### Understanding ELBO Components
    
    The Evidence Lower BOund (ELBO) consists of three main components:
    
    1. **Log Likelihood**: Measures how well our model explains the observed data.
       - Higher values indicate better fit to the data
       - Wider distribution suggests more uncertainty in the fit
    
    2. **Log Prior**: Represents our initial beliefs about parameter values.
       - Acts as a regularizer to prevent extreme parameter values
       - Shape determined by our choice of prior distribution
    
    3. **Log q**: Measures the complexity of our variational approximation.
       - Prevents the approximation from becoming too complex
       - Balances against the likelihood term
    
    The violin plots show the distribution of each component, with embedded box plots
    showing quartiles and means. The width of each violin indicates the density of
    samples at that value.
    """)

with tab3:
    st.header("Residual Analysis")
    
    # Calculate residuals
    with torch.no_grad():
        y_pred = vi_model.simulator(vi_state.final_mu, vi_model.t)
    residuals = vi_state.y_obs.numpy() - y_pred.numpy()
    
    # Create residual analysis plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals vs. Fitted",
            "Residual Distribution",
            "Residual Time Series",
            "Q-Q Plot"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Residuals vs. Fitted
    fig.add_trace(
        go.Scatter(
            x=y_pred.numpy(),
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=8)
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color="red",
        row=1, col=1
    )
    
    # 2. Residual Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Residual Dist',
            nbinsx=20,
            marker_color='blue'
        ),
        row=1, col=2
    )
    
    # Add normal distribution curve
    x_range = np.linspace(min(residuals), max(residuals), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals)) * 
               len(residuals) * (max(residuals) - min(residuals)) / 20,
            name='Normal Fit',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )
    
    # 3. Residual Time Series
    fig.add_trace(
        go.Scatter(
            x=vi_model.t.numpy(),
            y=residuals,
            mode='lines+markers',
            name='Residual Series',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color="red",
        row=2, col=1
    )
    
    # 4. Q-Q Plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.01, 0.99, len(residuals))
    )
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='blue', size=8)
        ),
        row=2, col=2
    )
    
    # Add diagonal line
    qq_min = min(min(theoretical_quantiles), min(sorted_residuals))
    qq_max = max(max(theoretical_quantiles), max(sorted_residuals))
    fig.add_trace(
        go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min, qq_max],
            mode='lines',
            name='Normal Line',
            line=dict(color='red', dash='dash')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    
    fig.update_layout(
        height=plot_height + 200,
        showlegend=True,
        title_text="Residual Analysis Dashboard"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual statistics
    st.subheader("Residual Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Residual", f"{np.mean(residuals):.3f}")
    with col2:
        st.metric("Residual Std Dev", f"{np.std(residuals):.3f}")
    with col3:
        st.metric("Max Absolute Residual", f"{np.max(np.abs(residuals)):.3f}")
    
    # Statistical tests
    st.subheader("Statistical Tests")
    col1, col2 = st.columns(2)
    
    with col1:
        # Shapiro-Wilk test for normality
        _, p_value_sw = stats.shapiro(residuals)
        st.metric(
            "Shapiro-Wilk Test p-value", 
            f"{p_value_sw:.3f}",
            help="Tests if residuals are normally distributed. p > 0.05 suggests normality."
        )
    
    with col2:
        # Durbin-Watson test for autocorrelation
        dw_statistic = stats.durbin_watson(residuals)
        st.metric(
            "Durbin-Watson Statistic", 
            f"{dw_statistic:.3f}",
            help="Tests for autocorrelation in residuals. Values near 2 suggest no autocorrelation, " +
                 "values < 1 or > 3 suggest positive or negative autocorrelation."
        )
        
        # Additional autocorrelation test using lag-1 correlation
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        st.metric(
            "Lag-1 Autocorrelation",
            f"{lag1_corr:.3f}",
            help="Correlation between consecutive residuals. Values near 0 suggest independence."
        )
    
    st.markdown("""
    ### Interpreting the Residual Analysis
    
    The residual analysis dashboard provides a comprehensive view of model fit quality through four complementary plots and statistical tests:

    1. **Residuals vs. Fitted Values**
       This plot helps us identify any systematic patterns in our model's predictions. In an ideal scenario, we should see:
       - Points scattered randomly around the zero line
       - No clear patterns or trends
       - Roughly constant spread across all fitted values
    
    2. **Residual Distribution**
       The histogram shows the distribution of residuals compared to a theoretical normal distribution. We look for:
       - A roughly bell-shaped curve
       - Symmetry around zero
       - No substantial skewness or heavy tails
       The Shapiro-Wilk test provides a formal check of normality.
    
    3. **Residual Time Series**
       This plot reveals any temporal patterns in our residuals. We want to see:
       - Random fluctuations around zero
       - No obvious trends or cycles
       - Consistent variability over time
       The Ljung-Box test helps detect any significant autocorrelation.
    
    4. **Q-Q Plot**
       The Quantile-Quantile plot compares our residuals to theoretical normal quantiles. We look for:
       - Points following the diagonal line
       - Limited deviation at the tails
       - No systematic curves or patterns

    The statistical tests provide formal verification of our visual assessments:
    - A Shapiro-Wilk p-value > 0.05 suggests normally distributed residuals
    - A Ljung-Box p-value > 0.05 suggests no significant autocorrelation
    """)

    # Add download functionality
    st.subheader("Download Analysis Results")
    
    # Prepare results dictionary
    results = {
        'parameters': {
            'true_theta': st.session_state.true_theta.numpy().tolist(),
            'estimated_theta': vi_state.final_mu.numpy().tolist(),
            'noise_std': float(vi_model.noise_std),
            'learning_rate': float(vi_model.learning_rate)
        },
        'correlation_analysis': {
            'correlation_coefficient': float(corr_coef),
            'n_samples': n_samples_correlation
        },
        'elbo_components': {
            component: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for component, values in components.items()
        },
        'residual_analysis': {
            'statistics': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'max_abs': float(np.max(np.abs(residuals)))
            },
            'statistical_tests': {
                'shapiro_wilk_pvalue': float(p_value_sw),
                'ljung_box_pvalue': float(p_value_lb[0])
            }
        },
        'metadata': {
            'timestamp': str(pd.Timestamp.now()),
            'n_samples_correlation': n_samples_correlation,
            'n_samples_elbo': n_samples_elbo
        }
    }

    col1, col2 = st.columns(2)
    with col1:
        # Download full results as JSON
        st.download_button(
            label="Download Full Analysis (JSON)",
            data=json.dumps(results, indent=2),
            file_name="vi_advanced_analysis.json",
            mime="application/json",
            help="Download complete analysis results including all statistics and metadata"
        )
    
    with col2:
        # Option to download just the figures
        st.markdown("""
        💡 **Tip**: To save any plot as an image:
        1. Hover over the plot
        2. Click the camera icon in the toolbar
        3. The image will download automatically
        """)

    # Final notes and recommendations
    st.markdown("""
    ### Summary and Recommendations
    
    Based on the comprehensive analysis presented above, here are the key findings and recommendations:

    1. **Parameter Estimation**
       - The correlation between parameters helps us understand model identifiability
       - Strong correlations might suggest the need for additional data or model simplification
    
    2. **Model Fit**
       - The ELBO components show how well we're balancing data fit with model complexity
       - The balance between likelihood and complexity terms indicates if we're overfitting
    
    3. **Residual Patterns**
       - Any systematic patterns in residuals suggest areas for model improvement
       - The statistical tests help validate our modeling assumptions
    
    Consider the following next steps:
    - If residuals show patterns, you might need to modify the model structure
    - If parameters are highly correlated, consider collecting more data or simplifying the model
    - If ELBO components are imbalanced, adjust the prior distributions or model complexity
    """)

    # Footer with references
    st.markdown("""
    ---
    ### Additional Resources
    
    To learn more about variational inference and diagnostic tools:
    
    1. **Variational Inference**:
       - "Variational Inference: A Review for Statisticians" by Blei et al.
       - "Automatic Differentiation Variational Inference" by Kucukelbir et al.
    
    2. **Residual Analysis**:
       - "Regression Diagnostics: Identifying Influential Data and Sources of Collinearity"
       - "An Introduction to Statistical Learning" Chapter 3
    
    3. **Model Criticism**:
       - "Predictive Model Criticism" by Cook and Gelman
       - "Model Criticism in Latent Space" by Yao et al.
    """)