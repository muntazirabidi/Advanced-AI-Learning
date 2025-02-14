import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# -------------------------------
# 1. Data Generation
# -------------------------------
torch.manual_seed(42)  # For reproducibility

# Define true parameters (centers and weights of the mixture)
true_thetas = torch.tensor([[1.5, 2.0], [3.0, 1.0], [2.0, 3.0]])
weights = torch.tensor([0.4, 0.3, 0.3])
n_samples = 100
sigma_true = 0.1

# Sample from the mixture
mixture_indices = torch.multinomial(weights, n_samples, replacement=True)
observations = torch.zeros((n_samples, 2))
for i in range(n_samples):
    center = true_thetas[mixture_indices[i]]
    observations[i] = center + sigma_true * torch.randn(2)

# -------------------------------
# 2. Variational GMM Model
# -------------------------------
class VariationalGMM(nn.Module):
    def __init__(self, n_components, dim, data):
        super().__init__()
        self.n_components = n_components
        self.dim = dim
        
        # Initialize means using KMeans with multiple restarts.
        # Using KMeans is a common heuristic to get a good starting point,
        # especially because VI can be sensitive to initialization.
        kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42)
        kmeans.fit(data.numpy())
        initial_means = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.means = nn.Parameter(initial_means)
        
        # Initialize log variances based on within-cluster variance
        cluster_vars = []
        labels = kmeans.labels_
        for k in range(n_components):
            cluster_data = data[labels == k]
            # If no data in a cluster, fallback to overall variance
            cluster_var = cluster_data.var(dim=0) if len(cluster_data) > 0 else data.var(dim=0)
            cluster_vars.append(cluster_var)
        initial_log_vars = torch.log(torch.stack(cluster_vars) + 1e-6)
        self.log_vars = nn.Parameter(initial_log_vars)
        
        # Initialize weights based on KMeans cluster sizes
        cluster_sizes = np.bincount(labels, minlength=n_components)
        initial_weights = torch.tensor(cluster_sizes / len(data), dtype=torch.float32)
        self.weight_logits = nn.Parameter(torch.log(initial_weights + 1e-6))
        
        # Temperature parameter for softmax (controls the "sharpness" of weight distribution)
        self.temperature = 1.0

    def get_mixture_weights(self):
        # Returns the mixture weights by applying softmax with temperature
        return torch.softmax(self.weight_logits / self.temperature, dim=0)
    
    def get_components(self):
        # Returns the current component means and variances
        return self.means, torch.exp(self.log_vars) + 1e-6
    
    def compute_log_prob(self, x):
        # Compute log probability of data x under the mixture model.
        weights = self.get_mixture_weights()
        means, vars = self.get_components()
        
        n_samples = len(x)
        log_probs = torch.zeros(n_samples, self.n_components)
        
        for k in range(self.n_components):
            diff = x - means[k]
            log_probs[:, k] = (
                -0.5 * torch.sum(torch.pow(diff, 2) / vars[k], dim=1)
                - 0.5 * torch.sum(torch.log(vars[k]))
                + torch.log(weights[k])
                - self.dim * 0.5 * np.log(2 * np.pi)
            )
        
        # Use the log-sum-exp trick for numerical stability
        max_log_probs = torch.max(log_probs, dim=1, keepdim=True)[0]
        log_prob = max_log_probs + torch.log(torch.sum(
            torch.exp(log_probs - max_log_probs), dim=1, keepdim=True
        ))
        
        return log_prob.squeeze()

# -------------------------------
# 3. Training Procedure
# -------------------------------
def train_vi_gmm(model, observations, n_epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    losses = []
    max_grad_norm = 1.0  # For gradient clipping
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute the log likelihood for the current parameters
        log_likelihood = model.compute_log_prob(observations)
        
        # Annealing schedules:
        # - Temperature: Gradually decrease to sharpen the softmax over weights.
        # - Beta: Gradually increase the weight on the KL divergence terms.
        temperature = max(0.5, np.exp(-epoch / 200))
        model.temperature = temperature
        beta = min(1.0, epoch / 200)
        
        # KL divergence penalties (with arbitrary scaling constants)
        weights = model.get_mixture_weights()
        means, vars = model.get_components()
        kl_weights = beta * 0.01 * torch.sum(weights * torch.log(weights * model.n_components + 1e-10))
        kl_gaussian = beta * 0.01 * 0.5 * torch.sum(
            means ** 2 / 10 + vars - torch.log(vars) - 1
        )
        
        # Total loss: negative ELBO (we maximize ELBO, so minimize -ELBO)
        loss = -(log_likelihood.mean() - kl_weights - kl_gaussian)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Temp: {temperature:.3f}')
    
    return losses

# -------------------------------
# 4. Visualization
# -------------------------------
def plot_results(observations, model, losses):
    plt.figure(figsize=(12, 5))
    
    # Plot the data and learned component centers
    plt.subplot(121)
    plt.scatter(observations[:, 0], observations[:, 1], alpha=0.5, label='Observations', color='gray')
    
    means, vars = model.get_components()
    weights = model.get_mixture_weights()
    
    means = means.detach()
    vars = vars.detach()
    weights = weights.detach()
    
    colors = ['red', 'green', 'blue']
    for k in range(model.n_components):
        plt.scatter(means[k, 0], means[k, 1], c=colors[k], s=150, marker='*',
                   label=f'Component {k+1} (w={weights[k]:.2f})')
        # Plot confidence ellipses (approximately 95% confidence)
        angle = np.linspace(0, 2*np.pi, 100)
        std = torch.sqrt(vars[k])
        x = means[k, 0] + 2 * std[0] * np.cos(angle)
        y = means[k, 1] + 2 * std[1] * np.sin(angle)
        plt.plot(x, y, '--', c=colors[k], alpha=0.5)
    
    plt.legend()
    plt.title('Data and Learned Components')
    plt.grid(True, alpha=0.3)
    
    # Plot the training loss
    plt.subplot(122)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# 5. Running the Model
# -------------------------------
model = VariationalGMM(n_components=3, dim=2, data=observations)
losses = train_vi_gmm(model, observations)
plot_results(observations, model, losses)

print("\nLearned Parameters:")
print("Mixture Weights:", model.get_mixture_weights().detach().numpy())
print("Means:\n", model.means.detach().numpy())
print("Variances:\n", torch.exp(model.log_vars).detach().numpy())


means, variances = model.get_components()
print("Learned Means:", means)
print("Learned Variances:", variances)

log_prob = model.compute_log_prob(observations)
print("Log Probability of Observations:", log_prob)

# -------------------------------
# 5. Visualization Probability on a Grid 
# -------------------------------

# Create a grid of points
x = np.linspace(0, 4, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)

# Compute log probability for each grid point and convert to density
with torch.no_grad():
    log_probs = model.compute_log_prob(grid_points)
    density = torch.exp(log_probs).reshape(X.shape)

# Plot the estimated density as a contour plot along with the original observations
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, density.numpy(), levels=50, cmap='viridis')
plt.colorbar(label='Density')
plt.scatter(observations[:, 0], observations[:, 1], color='red', s=20, label='Observations')
plt.title("Estimated Density over the Data Space")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# -------------------------------
# 6. Sampling new Data Points
# -------------------------------

with torch.no_grad():
    # Get learned parameters
    weights = model.get_mixture_weights().detach().numpy()
    means, vars = model.get_components()
    stds = torch.sqrt(vars).detach().numpy()
    
    # Number of new samples to generate
    n_new_samples = 100
    new_samples = []
    
    for i in range(n_new_samples):
        # Choose a component based on the mixture weights
        component = np.random.choice(len(weights), p=weights)
        # Sample from the chosen Gaussian component
        sample = means[component].detach().numpy() + stds[component] * np.random.randn(2)
        new_samples.append(sample)
    
    new_samples = np.array(new_samples)

# Plot the new samples along with the original observations
plt.figure(figsize=(8, 6))
plt.scatter(observations[:, 0], observations[:, 1], alpha=0.5, label='Original Observations', color='gray')
plt.scatter(new_samples[:, 0], new_samples[:, 1], label='New Samples', color='blue')
plt.title("New Samples from the Learned Mixture Model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
