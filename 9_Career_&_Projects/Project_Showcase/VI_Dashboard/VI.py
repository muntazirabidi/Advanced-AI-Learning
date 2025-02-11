import streamlit as st
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VariationalInference:
    def __init__(self):
        """
        Initialize the Variational Inference class with default parameters.
        Sets up the simulator, data generation, and optimization components.
        """
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize parameters that will be controlled via the UI
        self.noise_std = 0.5
        self.learning_rate = 0.01
        self.num_iterations = 5000
        self.num_samples = 10
        
        # Create time grid for simulation
        self.t = torch.linspace(0, 2 * np.pi, 100)
        
    def simulator(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Modified simulator function with interaction between parameters.
        
        Args:
            theta: Parameter tensor of shape (2,)
            t: Time points tensor
            
        Returns:
            Simulated values at time points t
        """
        alpha = 0.5  # Interaction strength
        return (theta[0] * torch.sin(t) + 
                theta[1] * torch.cos(t) + 
                alpha * theta[0] * theta[1] * torch.sin(t) * torch.cos(t))

    def generate_data(self, true_theta: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic observed data using true parameters.
        
        Args:
            true_theta: True parameter values
            
        Returns:
            Observed data with added noise
        """
        with torch.no_grad():
            y_obs = (self.simulator(true_theta, self.t) + 
                    self.noise_std * torch.randn(self.t.shape))
        return y_obs

    def get_variational_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize variational parameters (mean and Cholesky factors).
        
        Returns:
            Tuple of (mean vector, L parameters)
        """
        mu = torch.randn(2, requires_grad=True)
        l_params = torch.randn(3, requires_grad=True)  # [l11, l21, l22]
        return mu, l_params

    def get_L(self, l_params: torch.Tensor) -> torch.Tensor:
        """
        Construct lower triangular matrix L from parameters.
        
        Args:
            l_params: Vector of 3 parameters for L matrix
            
        Returns:
            2x2 lower triangular matrix
        """
        L = torch.zeros(2, 2)
        L[0, 0] = torch.exp(l_params[0])
        L[1, 0] = l_params[1]
        L[0, 1] = 0.0
        L[1, 1] = torch.exp(l_params[2])
        return L

    def sample_theta(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Sample parameters using reparameterization trick.
        
        Args:
            mu: Mean vector
            L: Cholesky factor matrix
            
        Returns:
            Sampled parameter vector
        """
        epsilon = torch.randn(2)
        return mu + L @ epsilon

    def log_q(self, theta: torch.Tensor, mu: torch.Tensor, 
              L: torch.Tensor) -> torch.Tensor:
        """
        Compute log density of variational distribution.
        
        Args:
            theta: Parameter vector
            mu: Mean vector
            L: Cholesky factor matrix
            
        Returns:
            Log density value
        """
        d = theta.numel()
        diff = (theta - mu).unsqueeze(1)
        Sigma = L @ L.t()
        Sigma_inv = torch.inverse(Sigma)
        quad = (diff.t() @ Sigma_inv @ diff).squeeze()
        logdet = 2 * (torch.log(L[0, 0]) + torch.log(L[1, 1]))
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log density of prior distribution.
        
        Args:
            theta: Parameter vector
            
        Returns:
            Log prior density
        """
        prior_std = 3.0
        return (-0.5 * torch.sum((theta / prior_std) ** 2) - 
                theta.numel() * torch.log(torch.tensor(prior_std * np.sqrt(2 * np.pi))))

    def log_likelihood(self, theta: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log likelihood of observations.
        
        Args:
            theta: Parameter vector
            y_obs: Observed data
            
        Returns:
            Log likelihood value
        """
        y_sim = self.simulator(theta, self.t)
        n = y_obs.numel()
        return (-0.5 * torch.sum(((y_obs - y_sim) / self.noise_std) ** 2) - 
                n * torch.log(torch.tensor(self.noise_std * np.sqrt(2 * np.pi))))

    def elbo(self, mu: torch.Tensor, l_params: torch.Tensor, 
             y_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute ELBO using reparameterization trick.
        
        Args:
            mu: Mean vector
            l_params: Cholesky factor parameters
            y_obs: Observed data
            
        Returns:
            ELBO estimate
        """
        L = self.get_L(l_params)
        elbo_est = 0.0
        for _ in range(self.num_samples):
            theta_sample = self.sample_theta(mu, L)
            ll = self.log_likelihood(theta_sample, y_obs)
            lp = self.log_prior(theta_sample)
            lq = self.log_q(theta_sample, mu, L)
            elbo_est += (ll + lp - lq)
        return elbo_est / self.num_samples

    def optimize(self, y_obs: torch.Tensor) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
        """
        Run variational optimization.
        
        Args:
            y_obs: Observed data
            
        Returns:
            Tuple of (ELBO history, final mean, final L parameters)
        """
        mu, l_params = self.get_variational_params()
        optimizer = optim.Adam([mu, l_params], lr=self.learning_rate)
        elbo_history = []
        
        for i in range(self.num_iterations):
            optimizer.zero_grad()
            loss = -self.elbo(mu, l_params, y_obs)
            loss.backward()
            optimizer.step()
            
            if i % 500 == 0:
                current_elbo = -loss.item()
                elbo_history.append(current_elbo)
                
        return elbo_history, mu.detach(), l_params.detach()

def main():
    st.set_page_config(page_title="Variational Inference Playground",
                      layout="wide")
    
    st.title("Variational Inference Interactive Dashboard")
    st.markdown("""
    This dashboard demonstrates variational inference on a simple model with two parameters.
    Adjust the parameters and see how they affect the inference process.
    """)
    
    # Initialize VI class
    vi = VariationalInference()
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    true_theta1 = st.sidebar.slider("True θ₁", -3.0, 3.0, 2.0, 0.1)
    true_theta2 = st.sidebar.slider("True θ₂", -3.0, 3.0, -1.0, 0.1)
    vi.noise_std = st.sidebar.slider("Noise Level", 0.1, 1.0, 0.5, 0.1)
    vi.learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    vi.num_samples = st.sidebar.slider("Number of MC Samples", 5, 50, 10, 5)
    
    # Generate data
    true_theta = torch.tensor([true_theta1, true_theta2])
    y_obs = vi.generate_data(true_theta)
    
    # Run optimization when button is clicked
    if st.sidebar.button("Run Inference"):
        with st.spinner("Running variational inference..."):
            elbo_history, final_mu, final_l_params = vi.optimize(y_obs)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            # Plot 1: Data fitting
            with col1:
                st.subheader("Data Fitting")
                fig_data = go.Figure()
                
                # Plot observed data
                fig_data.add_trace(go.Scatter(
                    x=vi.t.numpy(),
                    y=y_obs.numpy(),
                    mode='markers',
                    name='Observed Data'
                ))
                
                # Plot fitted curve
                with torch.no_grad():
                    y_fit = vi.simulator(final_mu, vi.t)
                fig_data.add_trace(go.Scatter(
                    x=vi.t.numpy(),
                    y=y_fit.numpy(),
                    mode='lines',
                    name='Fitted Curve'
                ))
                
                fig_data.update_layout(
                    xaxis_title="t",
                    yaxis_title="y",
                    height=400
                )
                st.plotly_chart(fig_data, use_container_width=True)
            
            # Plot 2: ELBO history
            with col2:
                st.subheader("ELBO Optimization History")
                fig_elbo = px.line(
                    x=range(0, vi.num_iterations, 500),
                    y=elbo_history,
                    labels={'x': 'Iteration', 'y': 'ELBO'}
                )
                fig_elbo.update_layout(height=400)
                st.plotly_chart(fig_elbo, use_container_width=True)
            
            # Display results
            st.subheader("Inference Results")
            col3, col4 = st.columns(2)
            with col3:
                st.metric("True θ₁", f"{true_theta1:.3f}")
                st.metric("Estimated θ₁", f"{final_mu[0].item():.3f}")
            with col4:
                st.metric("True θ₂", f"{true_theta2:.3f}")
                st.metric("Estimated θ₂", f"{final_mu[1].item():.3f}")

if __name__ == "__main__":
    main()