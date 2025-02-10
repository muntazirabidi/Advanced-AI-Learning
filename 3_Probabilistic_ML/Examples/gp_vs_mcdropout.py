import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

@dataclass
class MCDropoutConfig:
    """Enhanced configuration for MC Dropout with regularization"""
    def __init__(self,
                 hidden_sizes=[64, 64],
                 dropout_rate=0.1,
                 learning_rate=0.01,
                 epochs=1000,
                 n_samples=100,
                 weight_decay=0.01,    # L2 regularization parameter
                 smoothness_reg=0.01):  # Smoothness regularization strength
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_samples = n_samples
        self.weight_decay = weight_decay
        self.smoothness_reg = smoothness_reg

@dataclass
class GPConfig:
    """Configuration for Gaussian Process model"""
    kernel_type: str = 'rbf'  # Options: 'rbf', 'matern', 'periodic'
    learning_rate: float = 0.1
    n_iterations: int = 100
    
class DropoutNet(nn.Module):
    """Flexible neural network with configurable architecture"""
    def __init__(self, config: MCDropoutConfig):
        super(DropoutNet, self).__init__()
        self.dropout_rate = config.dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_size = 1  # Input dimension
        
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GPModel(gpytorch.models.ExactGP):
    """Flexible Gaussian Process model with configurable kernel"""
    def __init__(self, train_x, train_y, likelihood, config: GPConfig):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Configure kernel based on type
        base_kernel = {
            'rbf': gpytorch.kernels.RBFKernel(),
            'matern': gpytorch.kernels.MaternKernel(),
            'periodic': gpytorch.kernels.PeriodicKernel()
        }[config.kernel_type]
        
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_mc_dropout(model, X, y, config):
    """Train MC Dropout with both L2 and smoothness regularization"""
    criterion = nn.MSELoss()
    # Add weight decay (L2 regularization) to optimizer
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    losses = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # Forward pass multiple times to compute smoothness regularization
        predictions = []
        for _ in range(5):  # Use 5 forward passes
            predictions.append(model(X_tensor))
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(0)
        
        # Main prediction loss
        mse_loss = criterion(mean_pred, y_tensor)
        
        # Smoothness regularization using second derivatives
        # Compute approximate second derivatives using finite differences
        h = 0.1
        X_plus = X_tensor + h
        X_minus = X_tensor - h
        y_plus = model(X_plus)
        y_minus = model(X_minus)
        second_derivatives = (y_plus - 2*mean_pred + y_minus) / (h**2)
        smoothness_loss = torch.mean(second_derivatives**2)
        
        # Combined loss
        total_loss = mse_loss + config.smoothness_reg * smoothness_loss
        
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f} '
                  f'(MSE: {mse_loss.item():.4f}, '
                  f'Smoothness: {smoothness_loss.item():.4f})')
    
    return model, losses

def train_gp(X_train: np.ndarray, 
             y_train: np.ndarray, 
             config: GPConfig) -> Tuple[GPModel, gpytorch.likelihoods.GaussianLikelihood, list]:
    """Train Gaussian Process model with given configuration"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train)
    
    model = GPModel(X_train_tensor, y_train_tensor, likelihood, config)
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(config.n_iterations):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, likelihood, losses

def predict_mc_dropout(model: DropoutNet, 
                      X: np.ndarray, 
                      config: MCDropoutConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with MC Dropout"""
    model.train()
    predictions = []
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    
    for _ in range(config.n_samples):
        predictions.append(model(X_tensor).detach().numpy())
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    return mean, std

def predict_gp(model: GPModel, 
               likelihood: gpytorch.likelihoods.GaussianLikelihood, 
               X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with Gaussian Process"""
    model.eval()
    likelihood.eval()
    X_tensor = torch.FloatTensor(X).reshape(-1, 1)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_tensor))
        mean = observed_pred.mean.numpy()
        std = observed_pred.stddev.numpy()
    return mean, std

def plot_comparison(X_train: np.ndarray, 
                   y_train: np.ndarray, 
                   X_test: np.ndarray,
                   true_function: Optional[Callable] = None,
                   mc_results: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   gp_results: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   figsize: Tuple[int, int] = (15, 5)):
    """Plot comparison of uncertainty estimation methods"""
    plt.figure(figsize=figsize)
    
    # Plot MC Dropout results if available
    if mc_results is not None:
        mc_mean, mc_std = mc_results
        plt.subplot(1, 2, 1)
        plt.scatter(X_train, y_train, c='black', label='Training Data')
        plt.plot(X_test, mc_mean, 'b-', label='Mean Prediction')
        plt.fill_between(X_test.flatten(),
                        mc_mean.flatten() - 2*mc_std.flatten(),
                        mc_mean.flatten() + 2*mc_std.flatten(),
                        alpha=0.3, label='95% Confidence')
        if true_function is not None:
            plt.plot(X_test, true_function(X_test), 'r--', label='True Function')
        plt.title('MC Dropout')
        plt.legend()
    
    # Plot GP results if available
    if gp_results is not None:
        gp_mean, gp_std = gp_results
        plt.subplot(1, 2, 2)
        plt.scatter(X_train, y_train, c='black', label='Training Data')
        plt.plot(X_test, gp_mean, 'b-', label='Mean Prediction')
        plt.fill_between(X_test.flatten(),
                        gp_mean - 2*gp_std,
                        gp_mean + 2*gp_std,
                        alpha=0.3, label='95% Confidence')
        if true_function is not None:
            plt.plot(X_test, true_function(X_test), 'r--', label='True Function')
        plt.title('Gaussian Process')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def compare_methods(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   mc_config: Optional[MCDropoutConfig] = None,
                   gp_config: Optional[GPConfig] = None,
                   true_function: Optional[Callable] = None):
    """Compare MC Dropout and GP predictions with custom configurations"""
    
    # Use default configs if none provided
    mc_config = mc_config or MCDropoutConfig()
    gp_config = gp_config or GPConfig()
    
    # Train and predict with MC Dropout
    dropout_model = DropoutNet(mc_config)
    dropout_model, mc_losses = train_mc_dropout(dropout_model, X_train, y_train, mc_config)
    mc_results = predict_mc_dropout(dropout_model, X_test, mc_config)
    
    # Train and predict with GP
    gp_model, likelihood, gp_losses = train_gp(X_train, y_train, gp_config)
    gp_results = predict_gp(gp_model, likelihood, X_test)
    
    # Plot results
    plot_comparison(X_train, y_train, X_test, true_function, mc_results, gp_results)
    
    # Plot training losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(mc_losses)
    ax1.set_title('MC Dropout Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_yscale('log')
    
    ax2.plot(gp_losses)
    ax2.set_title('GP Training Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Negative Log Likelihood')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mc_model': dropout_model,
        'mc_losses': mc_losses,
        'gp_model': gp_model,
        'gp_likelihood': likelihood,
        'gp_losses': gp_losses
    }

# Example usage:
if __name__ == "__main__":
    # Generate data
    X_train = np.linspace(-3, 3, 20)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, 20)
    X_test = np.linspace(-4, 4, 100)
    
    
    mc_config = MCDropoutConfig(
        hidden_sizes=[64, 128, 64],    # Balanced architecture
        dropout_rate=0.1,              # Keep moderate dropout
        learning_rate=0.001,            # Slightly higher learning rate
        epochs=20000,
        n_samples=200,
        weight_decay=0.0001,            # Reduce L2 regularization significantly
        smoothness_reg=0.0005          # Much lighter smoothness regularization
    )
    gp_config = GPConfig(
        kernel_type='rbf',
        learning_rate=0.1,
        n_iterations=100
    )
    
    # Run comparison
    results = compare_methods(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        mc_config=mc_config,
        gp_config=gp_config,
        true_function=np.sin
    )