# models/bnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class BNNConfig:
    """
    Configuration class for Bayesian Neural Network hyperparameters.
    This class encapsulates all the hyperparameters needed to define and train a BNN.
    
    Attributes:
        hidden_sizes: List of integers defining the number of neurons in each hidden layer
        prior_std: Standard deviation for the Gaussian prior over weights
        learning_rate: Learning rate for the Adam optimizer
        num_samples: Number of Monte Carlo samples for prediction
        epochs: Number of training epochs
    """
    hidden_sizes: List[int] = None
    prior_std: float = 0.1        # Small prior std for better uncertainty estimates
    learning_rate: float = 0.01
    num_samples: int = 100        # Number of MC samples for prediction
    epochs: int = 2000
    
    def __post_init__(self):
        """Initialize default architecture if none provided"""
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 64]  # Default architecture that works well

class BayesianLinear(nn.Module):
    """
    Bayesian linear layer implementing the local reparameterization trick.
    Instead of sampling weights, we sample the pre-activations directly,
    which reduces gradient variance and improves training stability.
    
    The layer maintains a Gaussian variational posterior over weights and biases,
    parameterized by means (mu) and rhos, where std = softplus(rho).
    """
    def __init__(self, in_features: int, out_features: int, prior_std: float):
        """
        Initialize the Bayesian linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_std: Standard deviation for the Gaussian prior over weights
        """
        super().__init__()
        
        # Initialize variational parameters using Glorot initialization
        std = 2.0 / (in_features + out_features)
        
        # Weight variational parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -3)
        
        # Bias variational parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3)
        
        # Prior distributions
        self.weight_prior = dist.Normal(0, prior_std)
        self.bias_prior = dist.Normal(0, prior_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the local reparameterization trick.
        Instead of sampling weights and computing activations, we directly
        sample the activations from their implied distribution.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Get standard deviations from rho parameters
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)
        
        # Compute activation distribution parameters
        act_mu = F.linear(x, self.weight_mu, self.bias_mu)
        act_var = F.linear(x**2, weight_std**2, bias_std**2)
        act_std = torch.sqrt(act_var + 1e-8)  # Add small constant for numerical stability
        
        # Sample from activation distribution
        eps = torch.randn_like(act_mu)
        return act_mu + act_std * eps
    
    def kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence between the variational posterior and prior.
        Uses the analytical form of KL divergence between two Gaussian distributions.
        
        Returns:
            KL divergence loss term
        """
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)
        
        weight_var = weight_std**2
        bias_var = bias_std**2
        prior_var = self.weight_prior.variance
        
        # KL divergence for weights
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_var) / prior_var
            - 1.0
            - torch.log(weight_var / prior_var)
        )
        
        # KL divergence for biases
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_var) / prior_var
            - 1.0
            - torch.log(bias_var / prior_var)
        )
        
        return weight_kl + bias_kl

class BayesianNeuralNetwork(nn.Module):
    """
    Complete Bayesian Neural Network implementation with proper uncertainty estimation.
    The network uses variational inference for training and provides both
    epistemic (model) and aleatoric (data) uncertainty estimates.
    """
    def __init__(self, config: BNNConfig):
        """
        Initialize the Bayesian Neural Network.
        
        Args:
            config: Configuration object containing hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Build network architecture
        layers = []
        prev_size = 1  # Input dimension
        
        # Add hidden layers
        for hidden_size in config.hidden_sizes:
            layers.append(BayesianLinear(prev_size, hidden_size, config.prior_std))
            prev_size = hidden_size
        
        # Add output layer
        layers.append(BayesianLinear(prev_size, 1, config.prior_std))
        
        self.layers = nn.ModuleList(layers)
        
        # Learnable observation noise (aleatoric uncertainty)
        self.log_noise = nn.Parameter(torch.tensor([-5.0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 1]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        return self.layers[-1](x)
    
    def calculate_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative ELBO (Evidence Lower BOund) loss.
        ELBO = E[log p(y|x,w)] - KL(q(w)||p(w))
        
        Args:
            pred: Predicted values
            target: True values
            
        Returns:
            Negative ELBO loss
        """
        # Compute likelihood term with learnable noise
        noise_var = torch.exp(self.log_noise)
        likelihood = dist.Normal(pred, noise_var.sqrt())
        nll = -likelihood.log_prob(target).mean()
        
        # Compute KL divergence term
        kl = sum(layer.kl_loss() for layer in self.layers)
        
        # Complete ELBO loss
        batch_size = target.size(0)
        return nll + kl / batch_size
    
    def predict(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using multiple forward passes.
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean predictions, total uncertainty)
        """
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self(x)
                predictions.append(pred)
            
            # Stack predictions and compute statistics
            predictions = torch.stack(predictions)
            mean = predictions.mean(dim=0)
            
            # Compute uncertainties
            epistemic_var = predictions.var(dim=0)
            aleatoric_var = torch.exp(self.log_noise)
            total_std = torch.sqrt(epistemic_var + aleatoric_var)
            
            return mean, total_std

def train_model(model: BayesianNeuralNetwork,
                train_loader: torch.utils.data.DataLoader,
                num_epochs: int,
                learning_rate: float = 0.01) -> List[float]:
    """
    Train the Bayesian Neural Network.
    
    Args:
        model: BNN model to train
        train_loader: DataLoader containing training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of training losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = model.calculate_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')
    
    return losses

# Example usage
if __name__ == "__main__":
    # Configure the model
    config = BNNConfig(
        hidden_sizes=[32],
        prior_std=0.1,
        learning_rate=0.01,
        num_samples=100,
        epochs=2000
    )
    
    # Create synthetic data
    X = torch.linspace(-4, 4, 100).reshape(-1, 1)
    y = torch.sin(X) + torch.randn_like(X) * 0.1
    
    # Create dataset and loader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    model = BayesianNeuralNetwork(config)
    losses = train_model(model, loader, config.epochs, config.learning_rate)
    
    # Make predictions
    with torch.no_grad():
        mean, std = model.predict(X)
        print("Prediction shape:", mean.shape)
        print("Uncertainty shape:", std.shape)