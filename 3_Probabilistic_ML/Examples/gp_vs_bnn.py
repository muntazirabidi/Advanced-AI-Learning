import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import gpytorch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# Configuration Classes
@dataclass
class BNNConfig:
    """Configuration for Bayesian Neural Network"""
    hidden_sizes: List[int] = None
    prior_std: float = 0.1        # Standard deviation for weight priors
    learning_rate: float = 0.05
    num_samples: int = 100        # Number of samples for prediction
    epochs: int = 5000
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 64]

@dataclass
class GPConfig:
    """Configuration for Gaussian Process"""
    kernel_type: str = 'rbf'      # Options: 'rbf', 'matern', 'periodic'
    learning_rate: float = 0.1
    n_iterations: int = 100
    noise_prior: float = 0.1      # Prior for observation noise

# Bayesian Neural Network Implementation
class BayesianLinear(nn.Module):
    """Bayesian linear layer with Gaussian prior and posterior"""
    def __init__(self, in_features: int, out_features: int, prior_std: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters for the weight posterior
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Parameters for the bias posterior
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Prior distributions
        self.weight_prior = dist.Normal(0, prior_std)
        self.bias_prior = dist.Normal(0, prior_std)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self) -> torch.Tensor:
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        weight_posterior = dist.Normal(self.weight_mu, weight_std)
        bias_posterior = dist.Normal(self.bias_mu, bias_std)
        
        weight_kl = torch.sum(dist.kl_divergence(weight_posterior, self.weight_prior))
        bias_kl = torch.sum(dist.kl_divergence(bias_posterior, self.bias_prior))
        
        return weight_kl + bias_kl

class BayesianNeuralNetwork(nn.Module):
    """Complete Bayesian Neural Network implementation"""
    def __init__(self, config: BNNConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_size = 1
        
        for hidden_size in config.hidden_sizes:
            layers.append(BayesianLinear(prev_size, hidden_size, config.prior_std))
            prev_size = hidden_size
        
        layers.append(BayesianLinear(prev_size, 1, config.prior_std))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        return self.layers[-1](x)
    
    def kl_loss(self) -> torch.Tensor:
        return sum(layer.kl_loss() for layer in self.layers)

# Gaussian Process Implementation
class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model with configurable kernel"""
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods.GaussianLikelihood, 
                 config: GPConfig):
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
        
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Training Functions
def train_bnn(model: BayesianNeuralNetwork,
              X: np.ndarray,
              y: np.ndarray,
              config: BNNConfig) -> Tuple[BayesianNeuralNetwork, List[float]]:
    """Train Bayesian Neural Network using Variational Inference"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    losses = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # Forward pass and compute loss
        pred = model(X_tensor)
        likelihood = dist.Normal(pred, 0.1)
        nll = -likelihood.log_prob(y_tensor).mean()
        kl = model.kl_loss()
        
        # ELBO loss
        loss = nll + kl / len(X)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f} '
                  f'(NLL: {nll.item():.4f}, KL: {kl.item():.4f})')
    
    return model, losses

def train_gp(model: GPModel,
             likelihood: gpytorch.likelihoods.GaussianLikelihood,
             X_train: torch.Tensor,
             y_train: torch.Tensor,
             config: GPConfig) -> Tuple[GPModel, List[float]]:
    """Train Gaussian Process model"""
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(config.n_iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss.item():.4f}')
    
    return model, losses

# Prediction Functions
def predict_bnn(model: BayesianNeuralNetwork,
                X: np.ndarray,
                num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make predictions with uncertainty estimates using BNN"""
    model.train()
    predictions = []
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(X_tensor).numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    
    # Decompose uncertainty
    aleatoric_uncertainty = np.ones_like(mean) * 0.1
    epistemic_uncertainty = predictions.std(axis=0)
    total_std = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
    
    return mean, total_std, epistemic_uncertainty, aleatoric_uncertainty

def predict_gp(model: GPModel,
               likelihood: gpytorch.likelihoods.GaussianLikelihood,
               X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with Gaussian Process"""
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X))
        mean = observed_pred.mean.numpy()
        std = observed_pred.stddev.numpy()
        
    return mean, std

# Visualization Functions
def plot_comparative_analysis(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_test: np.ndarray,
                            bnn_results: Tuple,
                            gp_results: Tuple,
                            true_function: Optional[callable] = None):
    """Create comprehensive comparison plots"""
    bnn_mean, bnn_total_std, bnn_epistemic, bnn_aleatoric = bnn_results
    gp_mean, gp_std = gp_results
    
    fig = plt.figure(figsize=(15, 10))
    
    # Main prediction plots
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax1.scatter(X_train, y_train, c='black', label='Training Data')
    ax1.plot(X_test, bnn_mean, 'b-', label='BNN Mean')
    ax1.fill_between(X_test.flatten(),
                    bnn_mean.flatten() - 2*bnn_total_std.flatten(),
                    bnn_mean.flatten() + 2*bnn_total_std.flatten(),
                    alpha=0.3, color='blue', label='BNN 95% CI')
    if true_function is not None:
        ax1.plot(X_test, true_function(X_test), 'r--', label='True Function')
    ax1.set_title('BNN Predictions')
    ax1.legend()
    
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax2.scatter(X_train, y_train, c='black', label='Training Data')
    ax2.plot(X_test, gp_mean, 'g-', label='GP Mean')
    ax2.fill_between(X_test.flatten(),
                    gp_mean - 2*gp_std,
                    gp_mean + 2*gp_std,
                    alpha=0.3, color='green', label='GP 95% CI')
    if true_function is not None:
        ax2.plot(X_test, true_function(X_test), 'r--', label='True Function')
    ax2.set_title('Gaussian Process Predictions')
    ax2.legend()
    
    # Uncertainty decomposition
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax3.plot(X_test, bnn_epistemic.flatten(), 'b-', label='Epistemic')
    ax3.plot(X_test, bnn_aleatoric.flatten(), 'r-', label='Aleatoric')
    ax3.set_title('BNN Uncertainty Decomposition')
    ax3.legend()
    
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    ax4.plot(X_test, gp_std, 'g-', label='GP Total Uncertainty')
    ax4.set_title('GP Uncertainty')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# Main Experiment Function
def run_comparison_experiment(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_test: np.ndarray,
                            bnn_config: BNNConfig,
                            gp_config: GPConfig,
                            true_function: Optional[callable] = None) -> Dict:
    """Run complete comparison experiment"""
    # Prepare tensors
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
    
    # Train BNN
    print("Training BNN...")
    bnn_model = BayesianNeuralNetwork(bnn_config)
    bnn_model, bnn_losses = train_bnn(bnn_model, X_train, y_train, bnn_config)
    bnn_results = predict_bnn(bnn_model, X_test, bnn_config.num_samples)
    
    # Train GP
    print("\nTraining GP...")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = GPModel(X_train_tensor, y_train_tensor, likelihood, gp_config)
    gp_model, gp_losses = train_gp(gp_model, likelihood, X_train_tensor, y_train_tensor, gp_config)
    gp_results = predict_gp(gp_model, likelihood, X_test_tensor)
    
    # Plot results
    plot_comparative_analysis(X_train, y_train, X_test, bnn_results, gp_results, true_function)
    
    return {
        'bnn_model': bnn_model,
        'bnn_losses': bnn_losses,
        'bnn_results': bnn_results,
        'gp_model': gp_model,
        'gp_losses': gp_losses,
        'gp_results': gp_results
    }

# Example usage with analysis
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 20)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, 20)
    X_test = np.linspace(-4, 4, 100)
    
    # Configure models with appropriate hyperparameters
    bnn_config = BNNConfig(
        hidden_sizes=[64, 64],
        prior_std=1.0,
        learning_rate=0.01,
        num_samples=100,
        epochs=2000
    )
    
    gp_config = GPConfig(
        kernel_type='rbf',
        learning_rate=0.1,
        n_iterations=100,
        noise_prior=0.1
    )
    
    # Run experiment
    results = run_comparison_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        bnn_config=bnn_config,
        gp_config=gp_config,
        true_function=np.sin
    )
    
    # Analyze convergence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['bnn_losses'])
    plt.title('BNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(results['gp_losses'])
    plt.title('GP Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print metrics
    def calculate_metrics(true_fn, X, mean_pred, std_pred):
        true_y = true_fn(X)
        mse = np.mean((true_y - mean_pred.flatten())**2)
        rmse = np.sqrt(mse)
        
        # Calculate calibration (percentage of true values within 2σ)
        within_bounds = np.logical_and(
            true_y >= mean_pred.flatten() - 2*std_pred.flatten(),
            true_y <= mean_pred.flatten() + 2*std_pred.flatten()
        )
        calibration = np.mean(within_bounds)
        
        return {
            'RMSE': rmse,
            'Calibration': calibration
        }
    
    bnn_metrics = calculate_metrics(
        np.sin,
        X_test,
        results['bnn_results'][0],
        results['bnn_results'][1]
    )
    
    gp_metrics = calculate_metrics(
        np.sin,
        X_test,
        results['gp_results'][0],
        results['gp_results'][1]
    )
    
    print("\nBNN Metrics:")
    for metric, value in bnn_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nGP Metrics:")
    for metric, value in gp_metrics.items():
        print(f"{metric}: {value:.4f}")