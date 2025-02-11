import torch
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from scipy.stats import multivariate_normal

@dataclass
class VIState:
    """
    A dataclass to store the state of variational inference process.
    This helps keep track of results and intermediate values throughout the optimization.
    """
    elbo_history: List[float]
    final_mu: torch.Tensor
    final_l_params: torch.Tensor
    y_obs: torch.Tensor
    predictions: np.ndarray = None

class VariationalInference:
    """
    A class implementing variational inference for a simple time series model.
    The model assumes data is generated from a combination of sine and cosine
    functions with unknown parameters and additive noise.
    """
    def __init__(self):
        """
        Initialize the Variational Inference model with default parameters
        and set up the simulation environment.
        """
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize parameters that will be controlled via the UI
        self.noise_std = 0.5  # Standard deviation of observation noise
        self.learning_rate = 0.01  # Learning rate for optimization
        self.num_iterations = 5000  # Number of optimization iterations
        self.num_samples = 10  # Number of Monte Carlo samples for ELBO estimation
        
        # Create time grid for simulation
        self.t = torch.linspace(0, 2 * np.pi, 100)
    
    def simulator(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Simulator function that generates data given parameters theta.
        The function combines sine and cosine components with a nonlinear interaction term.
        
        Args:
            theta: Parameter tensor of shape (2,) containing coefficients
            t: Time points tensor
            
        Returns:
            Simulated values at time points t
        """
        alpha = 0.5  # Interaction strength between parameters
        return (theta[0] * torch.sin(t) + 
                theta[1] * torch.cos(t) + 
                alpha * theta[0] * theta[1] * torch.sin(t) * torch.cos(t))

    def generate_data(self, true_theta: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic observed data using true parameters and adding noise.
        
        Args:
            true_theta: True parameter values (tensor of shape (2,))
            
        Returns:
            Observed data with added Gaussian noise
        """
        with torch.no_grad():
            # Generate clean signal using the simulator
            true_signal = self.simulator(true_theta, self.t)
            # Add Gaussian noise
            noise = self.noise_std * torch.randn(self.t.shape)
            y_obs = true_signal + noise
        return y_obs

    def get_variational_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize variational parameters for the posterior approximation.
        These include the mean vector and the Cholesky factors for the covariance matrix.
        
        Returns:
            Tuple of (mean vector, Cholesky factor parameters)
        """
        mu = torch.randn(2, requires_grad=True)  # Mean of q(theta)
        l_params = torch.randn(3, requires_grad=True)  # Parameters for Cholesky factor
        return mu, l_params

    def get_L(self, l_params: torch.Tensor) -> torch.Tensor:
        """
        Construct the lower triangular Cholesky factor matrix from its parameters.
        Uses exponential transform to ensure positive diagonal elements.
        
        Args:
            l_params: Vector of 3 parameters [l11, l21, l22]
            
        Returns:
            2x2 lower triangular matrix
        """
        L = torch.zeros(2, 2)
        L[0, 0] = torch.exp(l_params[0])  # Positive diagonal
        L[1, 0] = l_params[1]  # Unrestricted off-diagonal
        L[0, 1] = 0.0  # Upper triangle is zero
        L[1, 1] = torch.exp(l_params[2])  # Positive diagonal
        return L

    def sample_theta(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Sample parameters using the reparameterization trick: theta = mu + L * epsilon
        where epsilon ~ N(0, I).
        
        Args:
            mu: Mean vector
            L: Cholesky factor matrix
            
        Returns:
            Sampled parameter vector
        """
        epsilon = torch.randn(2)  # Standard normal samples
        return mu + L @ epsilon  # Reparameterization trick

    def log_q(self, theta: torch.Tensor, mu: torch.Tensor, 
              L: torch.Tensor) -> torch.Tensor:
        """
        Compute log density of the variational distribution q(theta).
        
        Args:
            theta: Parameter vector
            mu: Mean vector
            L: Cholesky factor matrix
            
        Returns:
            Log density value
        """
        d = theta.numel()  # Dimension of parameter space
        diff = (theta - mu).unsqueeze(1)  # Difference from mean
        Sigma = L @ L.t()  # Construct covariance matrix
        Sigma_inv = torch.inverse(Sigma)
        quad = (diff.t() @ Sigma_inv @ diff).squeeze()  # Quadratic form
        logdet = 2 * (torch.log(L[0, 0]) + torch.log(L[1, 1]))  # Log determinant
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log density of the prior distribution p(theta).
        Uses a spherical Gaussian prior with standard deviation 3.0.
        
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
        Compute log likelihood of the observations given parameters.
        
        Args:
            theta: Parameter vector
            y_obs: Observed data
            
        Returns:
            Log likelihood value
        """
        y_sim = self.simulator(theta, self.t)  # Generate model predictions
        n = y_obs.numel()
        return (-0.5 * torch.sum(((y_obs - y_sim) / self.noise_std) ** 2) - 
                n * torch.log(torch.tensor(self.noise_std * np.sqrt(2 * np.pi))))

    def elbo(self, mu: torch.Tensor, l_params: torch.Tensor, 
             y_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Evidence Lower BOund (ELBO) using Monte Carlo sampling.
        
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