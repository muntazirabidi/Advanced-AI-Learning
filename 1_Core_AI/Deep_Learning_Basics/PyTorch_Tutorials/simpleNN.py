import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryClassificationDataset(Dataset):
    """Custom Dataset for binary classification task."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    """Simple neural network for binary classification."""
    
    def __init__(self, input_size: int = 2, hidden_size: int = 8):
        """
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class BinaryClassifier:
    """Binary classifier wrapper with training and evaluation capabilities."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model and return accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)
                
        return correct / total

def create_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic dataset for binary classification."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y

def train_and_visualize(
    batch_size: int = 32,
    epochs: int = 100,
    val_frequency: int = 10
) -> None:
    """Train the model and visualize results."""
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    X, y = create_dataset()
    split_idx = int(0.8 * len(X))
    
    train_dataset = BinaryClassificationDataset(X[:split_idx], y[:split_idx])
    val_dataset = BinaryClassificationDataset(X[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and trainer
    model = SimpleNN()
    classifier = BinaryClassifier(model, device)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        loss = classifier.train_epoch(train_loader)
        train_losses.append(loss)
        
        if (epoch + 1) % val_frequency == 0:
            accuracy = classifier.evaluate(val_loader)
            val_accuracies.append(accuracy)
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Loss: {loss:.4f}, "
                f"Val Accuracy: {accuracy:.4f}"
            )
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs, val_frequency), val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
    
"""
To use this code:

1. Make sure you have PyTorch 2.0+ installed with MPS support:
   pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

2. Run on macOS 12.3+

3. Check device assignment:
   - If MPS is available, it will use the Apple Silicon GPU
   - If not available, it will fallback to CPU

Common operations with device:
- Move tensor to device: tensor.to(device)
- Create tensor on device: torch.rand(3, 4, device=device)
- Move model to device: model.to(device)

Remember:
- Always create/move both the model AND data to the same device
- Use device context in training loops and inference
- Check tensor device location with tensor.device
"""