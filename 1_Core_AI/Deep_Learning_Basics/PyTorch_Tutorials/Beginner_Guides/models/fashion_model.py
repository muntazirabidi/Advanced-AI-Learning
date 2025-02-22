# fashion_model.py
import torch
import torch.nn as nn

class FashionClassifier(nn.Module):
    """
    CNN classifier for Fashion MNIST dataset.
    Architecture:
    - 2 Convolutional layers with batch normalization
    - Max pooling and dropout for regularization
    - 2 Fully connected layers for classification
    """
    def __init__(self):
        super(FashionClassifier, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First conv block: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Second conv block: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Utility function to load a pretrained model
def load_pretrained_model(model_path):
    """
    Loads a pretrained FashionClassifier model from the specified path.
    
    Args:
        model_path (str): Path to the saved model weights
        
    Returns:
        model (FashionClassifier): Loaded model in evaluation mode
    """
    model = FashionClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Optional: Add class names for reference
class_names = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]