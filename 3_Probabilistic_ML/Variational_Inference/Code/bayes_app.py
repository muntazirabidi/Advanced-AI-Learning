# requirements.txt
# streamlit
# numpy
# torch
# pyro-ppl
# matplotlib
# plotly
# scipy

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch import nn
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scipy.stats import norm, gaussian_kde

# Set page config
st.set_page_config(
    page_title="Bayesian Inference Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
    }
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Section", [
        "Introduction",
        "1D Bayesian Inference",
        "Multimodal Density Estimation"
    ])

    if app_mode == "Introduction":
        show_introduction()
    elif app_mode == "1D Bayesian Inference":
        run_1d_inference()
    elif app_mode == "Multimodal Density Estimation":
        run_multimodal_estimation()

def show_introduction():
    st.title("Bayesian Inference & Variational Methods")
    st.markdown("""
    ## Learn Bayesian Inference through Interactive Examples
    
    This app helps you understand:
    - **Bayesian inference** fundamentals
    - **Variational Inference** concepts
    - Density estimation techniques
    - Multimodal distribution handling
    
    ### Core Concepts
    """)
    
    with st.expander("Bayesian Inference"):
        st.markdown("""
        $$P(\\theta | X) = \\frac{P(X | \\theta)P(\\theta)}{P(X)}$$
        - Prior $P(\\theta)$: Initial beliefs about parameters
        - Likelihood $P(X|\\theta)$: Data generation process
        - Posterior $P(\\theta|X)$: Updated beliefs after observing data
        """)
    
    with st.expander("Variational Inference"):
        st.markdown("""
        Approximate complex posteriors with simpler distributions by:
        1. Choosing a family of approximate distributions $q(\\theta)$
        2. Minimizing KL divergence between $q(\\theta)$ and true posterior
        3. Optimization objective (ELBO):
        $$\\text{ELBO} = \\mathbb{E}_q[\\log p(x,\\theta)] - \\mathbb{E}_q[\\log q(\\theta)]$$
        """)
    
    with st.expander("Why Density Estimation?"):
        st.markdown("""
        - Model complex data distributions
        - Handle multimodal data
        - Flexible non-parametric modeling
        - Basis for many machine learning tasks
        """)

def run_1d_inference():
    st.title("1D Bayesian Inference")
    st.markdown("""
    ## Interactive 1D Gaussian Inference
    
    Estimate the mean of a normal distribution with known variance using variational inference.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Generation")
        true_mu = st.slider("True mean", -5.0, 5.0, 2.0)
        true_std = st.slider("True std", 0.1, 5.0, 1.0)
        n_samples = st.slider("Number of samples", 10, 1000, 100)
        
        data = np.random.normal(true_mu, true_std, n_samples)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Data', opacity=0.7))
        fig.update_layout(title="Generated Data Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Parameters")
        prior_mu = st.slider("Prior mean", -5.0, 5.0, 0.0)
        prior_std = st.slider("Prior std", 0.1, 5.0, 2.0)
        n_steps = st.slider("Optimization steps", 100, 5000, 1000)
        
        # Define model
        def model(data):
            mu = pyro.sample("mu", dist.Normal(prior_mu, prior_std))
            with pyro.plate("data", len(data)):
                pyro.sample("obs", dist.Normal(mu, true_std), obs=torch.tensor(data))
        
        # Define guide
        def guide(data):
            loc = pyro.param("loc", torch.tensor(0.0))
            scale = pyro.param("scale", torch.tensor(1.0), constraint=dist.constraints.positive)
            pyro.sample("mu", dist.Normal(loc, scale))
        
        # Run inference
        pyro.clear_param_store()
        optimizer = Adam({"lr": 0.01})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        losses = []
        for _ in range(n_steps):
            loss = svi.step(data)
            losses.append(loss)
        
        # Get results
        posterior_mu = pyro.param("loc").item()
        posterior_std = pyro.param("scale").item()
        
        # True posterior calculation (conjugate prior)
        prior_var = prior_std**2
        post_prec = 1/prior_var + n_samples/(true_std**2)
        post_var = 1/post_prec
        post_mu = (prior_mu/prior_var + np.sum(data)/true_std**2) / post_prec
        
        # Visualization
        x = np.linspace(-5, 5, 500)
        prior_pdf = norm.pdf(x, prior_mu, prior_std)
        true_post_pdf = norm.pdf(x, post_mu, np.sqrt(post_var))
        variational_pdf = norm.pdf(x, posterior_mu, posterior_std)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=prior_pdf, name="Prior", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=x, y=true_post_pdf, name="True Posterior"))
        fig.add_trace(go.Scatter(x=x, y=variational_pdf, name="Variational Approx"))
        fig.update_layout(title="Distributions Comparison", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.metric("True Posterior Mean", f"{post_mu:.2f}")
        st.metric("Variational Mean", f"{posterior_mu:.2f}")
        st.metric("KL Divergence", f"{(posterior_std**2 + (posterior_mu - post_mu)**2)/(2*post_var) - 0.5 + np.log(post_var**0.5/posterior_std):.4f}")

def run_multimodal_estimation():
    st.title("Multimodal Density Estimation")
    st.markdown("""
    ## Handling Complex Distributions with Variational Methods
    
    Estimate densities for multimodal distributions using:
    - Variational Gaussian Mixture Models
    - Kernel Density Estimation (KDE)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("True Distribution")
        n_components = st.slider("Number of components", 1, 5, 2)
        component_params = []
        for i in range(n_components):
            st.markdown(f"Component {i+1}")
            c1, c2 = st.columns(2)
            with c1:
                mean = st.slider(f"Mean {i+1}", -5.0, 5.0, float(-3 + i*3))
            with c2:
                std = st.slider(f"Std {i+1}", 0.1, 2.0, 0.5)
            weight = st.slider(f"Weight {i+1}", 0.1, 1.0, 1.0/n_components)
            component_params.append((mean, std, weight))
        
        # Generate data
        all_data = []
        for mean, std, weight in component_params:
            n_samples = int(1000 * weight)
            all_data.extend(np.random.normal(mean, std, n_samples))
        all_data = np.array(all_data)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=all_data, nbinsx=50, opacity=0.7))
        fig.update_layout(title="True Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Estimation Methods")
        method = st.selectbox("Choose method", ["Variational GMM", "KDE"])
        
        if method == "Variational GMM":
            n_components_fit = st.slider("Number of variational components", 1, 5, 2)
            
            # Implement simplified variational GMM
            from sklearn.mixture import BayesianGaussianMixture
            bgmm = BayesianGaussianMixture(n_components=n_components_fit, 
                                         weight_concentration_prior=0.1,
                                         n_init=1)
            bgmm.fit(all_data.reshape(-1, 1))
            
            # Plot results
            x = np.linspace(-8, 8, 500)
            logprob = bgmm.score_samples(x.reshape(-1, 1))
            pdf = np.exp(logprob)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=pdf, name="Variational Fit"))
            fig.update_layout(title="Variational GMM Fit", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Variational Parameters:**
            - Components used: {bgmm.n_components}
            - Convergence: {bgmm.converged_}
            """)
        
        elif method == "KDE":
            bandwidth = st.slider("Bandwidth", 0.05, 1.0, 0.5)
            kde = gaussian_kde(all_data, bw_method=bandwidth)
            x = np.linspace(-8, 8, 500)
            pdf = kde.evaluate(x)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=pdf, name="KDE Estimate"))
            fig.update_layout(title="Kernel Density Estimation", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **KDE Parameters:**
            - Bandwidth: {kde.factor:.4f}
            - Data points: {len(all_data)}
            """)

if __name__ == "__main__":
    main()