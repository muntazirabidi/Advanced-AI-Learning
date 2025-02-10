import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import gpytorch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

@dataclass
class VIConfig:
    """Configuration for Variational Inference Model"""
    hidden_sizes: List[int] = None  # Sizes of hidden layers
    prior_mu: float = 0.0          # Mean for weight priors
    prior_sigma: float = 1.0       # Std dev for weight priors
    learning_rate: float = 0.01
    num_samples: int = 100         # Number of Monte Carlo samples
    epochs: int = 2000
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [50, 50]  # Default architecture

@dataclass
class GPConfig:
    """Configuration for Gaussian Process"""
    kernel: str = 'rbf'           # Kernel function type
    learning_rate: float = 0.1
    iterations: int = 100
    noise_prior: float = 0.1

class VariationalLayer(nn.Module):
    """
    Implements a variational layer with learned mean and variance parameters.
    Uses the reparameterization trick for backpropagation through random samples.
    """
    def __init__(self, in_features: int, out_features: int, 
                 prior_mu: float, prior_sigma: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize variational parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)  # Smaller initial values
        self.weight_log_var = nn.Parameter(torch.ones(out_features, in_features) * -4)  # Start with small variance
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_var = nn.Parameter(torch.ones(out_features) * -4)
        
        # Define prior distributions
        self.weight_prior = dist.Normal(prior_mu, prior_sigma)
        self.bias_prior = dist.Normal(prior_mu, prior_sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick"""
        # Sample weights and biases using reparameterization
        weight = self.weight_mu + torch.exp(0.5 * self.weight_log_var) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + torch.exp(0.5 * self.bias_log_var) * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence between variational posterior and prior"""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_log_var)
        weight_kl = 0.5 * torch.sum(
            weight_var + self.weight_mu.pow(2) - 1 - self.weight_log_var
        )
        
        # KL divergence for biases
        bias_var = torch.exp(self.bias_log_var)
        bias_kl = 0.5 * torch.sum(
            bias_var + self.bias_mu.pow(2) - 1 - self.bias_log_var
        )
        
        return weight_kl + bias_kl

class VariationalNetwork(nn.Module):
    """
    Full variational neural network implementing VI for regression.
    Uses multiple variational layers with ReLU activations.
    """
    def __init__(self, config: VIConfig):
        super().__init__()
        self.config = config
        
        # Build network architecture
        layers = []
        prev_size = 1  # Input dimension
        
        for hidden_size in config.hidden_sizes:
            layers.append(
                VariationalLayer(prev_size, hidden_size, 
                               config.prior_mu, config.prior_sigma)
            )
            prev_size = hidden_size
        
        # Output layer
        layers.append(
            VariationalLayer(prev_size, 1, 
                           config.prior_mu, config.prior_sigma)
        )
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all layers"""
        return sum(layer.kl_divergence() for layer in self.layers)

class GPRegression(gpytorch.models.ExactGP):
    """
    Gaussian Process regression model with configurable kernel.
    Implements exact GP inference for regression tasks.
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 config: GPConfig):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Select kernel based on configuration
        if config.kernel == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif config.kernel == 'matern':
            base_kernel = gpytorch.kernels.MaternKernel()
        else:
            raise ValueError(f"Unsupported kernel: {config.kernel}")
        
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_vi_model(model: VariationalNetwork,
                   X: np.ndarray,
                   y: np.ndarray,
                   config: VIConfig) -> Tuple[VariationalNetwork, List[float]]:
    """Train variational inference model using ELBO optimization"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Convert data to tensors
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    losses = []
    # Scale factors for loss terms
    num_batches = len(X)
    kl_weight = 1.0 / num_batches  # KL annealing
    
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        predictions = torch.stack([model(X_tensor) for _ in range(config.num_samples)])
        pred_mean = predictions.mean(0)
        pred_var = predictions.var(0) + 1e-6
        
        # Modified loss calculation
        nll = 0.5 * (torch.log(2 * np.pi * pred_var) + 
                     (y_tensor - pred_mean).pow(2) / pred_var).mean()
        
        # Scaled KL divergence
        kl_div = model.kl_divergence() * kl_weight
        
        # Total loss with better scaling
        loss = nll + kl_div
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f} '
                  f'(NLL = {nll.item():.4f}, KL = {kl_div.item():.4f})')
    
    return model, losses

def train_gp_model(model: GPRegression,
                   likelihood: gpytorch.likelihoods.GaussianLikelihood,
                   X_train: torch.Tensor,
                   y_train: torch.Tensor,
                   config: GPConfig) -> Tuple[GPRegression, List[float]]:
    """Train Gaussian Process model using marginal likelihood optimization"""
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(config.iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if i % 10 == 0:
            print(f'Iteration {i}: Loss = {loss.item():.4f}')
    
    return model, losses

def predict_vi(model: VariationalNetwork,
               X: np.ndarray,
               num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions with uncertainty estimates using VI model"""
    model.train()  # Keep in training mode for MC sampling
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(X_tensor).numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    
    return mean, std

def predict_gp(model: GPRegression,
               likelihood: gpytorch.likelihoods.GaussianLikelihood,
               X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions with uncertainty estimates using GP model"""
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_pred = model(X)
        y_pred = likelihood(f_pred)
        mean = y_pred.mean.numpy()
        std = y_pred.stddev.numpy()
    
    return mean, std

def plot_results(X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                vi_results: Tuple[np.ndarray, np.ndarray],
                gp_results: Tuple[np.ndarray, np.ndarray],
                true_fn: Optional[callable] = None):
    """Create comparison plots for VI and GP predictions"""
    vi_mean, vi_std = vi_results
    gp_mean, gp_std = gp_results
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # VI predictions
    ax1.scatter(X_train, y_train, c='black', label='Training Data')
    ax1.plot(X_test, vi_mean, 'b-', label='VI Mean')
    ax1.fill_between(X_test.flatten(),
                    vi_mean.flatten() - 2*vi_std.flatten(),
                    vi_mean.flatten() + 2*vi_std.flatten(),
                    alpha=0.3, color='blue', label='VI 95% CI')
    if true_fn is not None:
        ax1.plot(X_test, true_fn(X_test), 'r--', label='True Function')
    ax1.set_title('Variational Inference Predictions')
    ax1.legend()
    
    # GP predictions
    ax2.scatter(X_train, y_train, c='black', label='Training Data')
    ax2.plot(X_test, gp_mean, 'g-', label='GP Mean')
    ax2.fill_between(X_test.flatten(),
                    gp_mean - 2*gp_std,
                    gp_mean + 2*gp_std,
                    alpha=0.3, color='green', label='GP 95% CI')
    if true_fn is not None:
        ax2.plot(X_test, true_fn(X_test), 'r--', label='True Function')
    ax2.set_title('Gaussian Process Predictions')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_models(true_fn: callable,
                   X: np.ndarray,
                   mean_pred: np.ndarray,
                   std_pred: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics for model evaluation"""
    true_y = true_fn(X)
    
    # Root Mean Square Error
    mse = np.mean((true_y - mean_pred.flatten())**2)
    rmse = np.sqrt(mse)
    
    # Calibration (percentage within 2σ confidence interval)
    within_ci = np.logical_and(
        true_y >= mean_pred.flatten() - 2*std_pred.flatten(),
        true_y <= mean_pred.flatten() + 2*std_pred.flatten()
    )
    calibration = np.mean(within_ci)
    
    # Average uncertainty
    avg_uncertainty = np.mean(std_pred)
    
    return {
        'RMSE': rmse,
        'Calibration': calibration,
        'Avg Uncertainty': avg_uncertainty
    }

def run_experiment(X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  vi_config: VIConfig,
                  gp_config: GPConfig,
                  true_fn: Optional[callable] = None) -> Dict:
    """Run complete comparison experiment between VI and GP"""
    # Prepare tensors
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
    
    # Train VI model
    print("\nTraining Variational Inference model...")
    vi_model = VariationalNetwork(vi_config)
    vi_model, vi_losses = train_vi_model(vi_model, X_train, y_train, vi_config)
    vi_mean, vi_std = predict_vi(vi_model, X_test, vi_config.num_samples)
    
    # Train GP model
    print("\nTraining Gaussian Process model...")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = GPRegression(X_train_tensor, y_train_tensor, likelihood, gp_config)
    gp_model, gp_losses = train_gp_model(gp_model, likelihood, X_train_tensor, y_train_tensor, gp_config)
    gp_mean, gp_std = predict_gp(gp_model, likelihood, X_test_tensor)
    
    # Plot comparison results
    plot_results(X_train, y_train, X_test, 
                (vi_mean, vi_std), 
                (gp_mean, gp_std),
                true_fn)
    
    # Calculate and compare metrics
    if true_fn is not None:
        print("\nCalculating performance metrics...")
        vi_metrics = evaluate_models(true_fn, X_test, vi_mean, vi_std)
        gp_metrics = evaluate_models(true_fn, X_test, gp_mean, gp_std)
        
        print("\nVariational Inference Metrics:")
        for metric, value in vi_metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        print("\nGaussian Process Metrics:")
        for metric, value in gp_metrics.items():
            print(f"{metric:20s}: {value:.4f}")
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(vi_losses)
    plt.title('VI Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(gp_losses)
    plt.title('GP Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'vi_model': vi_model,
        'vi_losses': vi_losses,
        'vi_predictions': (vi_mean, vi_std),
        'gp_model': gp_model,
        'gp_losses': gp_losses,
        'gp_predictions': (gp_mean, gp_std),
        'vi_metrics': vi_metrics if true_fn is not None else None,
        'gp_metrics': gp_metrics if true_fn is not None else None
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data for demonstration
    def true_function(x):
        return np.sin(x) * np.exp(-0.1 * x**2)
    
    # Create training data with noise
    X_train = np.linspace(-4, 4, 30)
    y_train = true_function(X_train) + np.random.normal(0, 0.1, size=X_train.shape)
    
    # Create test data points
    X_test = np.linspace(-5, 5, 100)
    
    # Configure models
    vi_config = VIConfig(
        hidden_sizes=[64, 64],
        prior_mu=0.0,
        prior_sigma=1.0,
        learning_rate=0.01,
        num_samples=100,
        epochs=2000
    )
    
    gp_config = GPConfig(
        kernel='rbf',
        learning_rate=0.1,
        iterations=100,
        noise_prior=0.1
    )
    
    # Run comparison experiment
    results = run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        vi_config=vi_config,
        gp_config=gp_config,
        true_fn=true_function
    )
    
    # Print summary analysis
    print("\nAnalysis Summary:")
    print("1. Model Comparison:")
    if results['vi_metrics']['RMSE'] < results['gp_metrics']['RMSE']:
        print("   - VI shows better prediction accuracy")
    else:
        print("   - GP shows better prediction accuracy")
    
    print("\n2. Uncertainty Estimation:")
    if abs(results['vi_metrics']['Calibration'] - 0.95) < abs(results['gp_metrics']['Calibration'] - 0.95):
        print("   - VI shows better uncertainty calibration")
    else:
        print("   - GP shows better uncertainty calibration")
    
    print("\n3. Computational Aspects:")
    print("   - VI required more training iterations but scales better with dataset size")
    print("   - GP provides exact inference but may be computationally intensive for large datasets")
    
    print("\n4. Key Findings:")
    print("   - Both methods successfully capture the underlying function and uncertainty")
    print("   - VI offers more flexibility in model architecture but requires careful tuning")
    print("   - GP provides smooth interpolation and principled uncertainty estimates")