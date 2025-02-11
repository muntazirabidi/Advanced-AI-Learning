import torch 
import torch.nn as nn

class MCDropoutModel(nn.Module):
  def __init__(self, input_dim, output_dim, dropout_prob = 0.2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input, 128),
      nn.Dropout(dropout_prob),
      nn.ReLU,
      nn.Linear(128, 128),
      nn.Dropout(dropout_prob), 
      nn.ReLU,
      nn.Linear(128, output_dim)
    )
    
    def forward(self, x):
      return self.net(x)
    
def mc_dropout_predict(model, x, num_samples = 100):
  model.train()
  with torch.no_grad():
    samples = [model(x) for _ in range(num_samples)]
    
  return torch.stack(samples)
