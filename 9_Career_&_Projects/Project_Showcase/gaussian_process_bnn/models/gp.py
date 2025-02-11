import gpytorch
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPConfig:
    """Configuration for Gaussian Process"""
    kernel_type: str
    learning_rate: float
    n_iterations: int
    noise_prior: float

class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model implementation"""
    def __init__(self, train_x, train_y, likelihood, config: GPConfig):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Configure kernel based on type
        if config.kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif config.kernel_type == 'matern':
            base_kernel = gpytorch.kernels.MaternKernel()
        elif config.kernel_type == 'periodic':
            base_kernel = gpytorch.kernels.PeriodicKernel()
        else:
            raise ValueError(f"Unknown kernel type: {config.kernel_type}")
        
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        """Forward pass through the GP model"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)