import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------
# 1. Define the Simulator (Implicit Likelihood)
# -------------------------------------------------
def simulator(theta, t):
    """
    Simulator function.
    theta: tensor of shape (2,)
    t: tensor of time points
    Returns: f(t; theta) = theta[0] * sin(t) + theta[1] * cos(t)
    """
    return theta[0] * torch.sin(t) + theta[1] * torch.cos(t)

# -------------------------------------------------
# 2. Generate Observed Data Using True Parameters
# -------------------------------------------------
# True parameters (unknown in practice)
true_theta = torch.tensor([2.0, -1.0])
# Time grid for simulation
t = torch.linspace(0, 2 * np.pi, 100)
# Noise level
noise_std = 0.5

# Generate observed data: simulator output plus Gaussian noise.
with torch.no_grad():
    y_obs = simulator(true_theta, t) + noise_std * torch.randn(t.shape)

# -------------------------------------------------
# 3. Define the Variational Distribution: q(theta) ~ N(mu, diag(sigma^2))
# -------------------------------------------------
# We learn mu (a vector of length 2) and log_sigma (for numerical stability)
mu = torch.randn(2, requires_grad=True)
log_sigma = torch.randn(2, requires_grad=True)  # we'll use sigma = exp(log_sigma)

# -------------------------------------------------
# 4. Define the Prior and Likelihood
# -------------------------------------------------
def log_prior(theta):
    """
    Assume p(theta) is a Gaussian with mean 0 and standard deviation 3.
    """
    prior_std = 3.0
    return -0.5 * torch.sum((theta / prior_std) ** 2) - theta.numel() * torch.log(torch.tensor(prior_std * np.sqrt(2 * np.pi)))

def log_likelihood(theta, t, y_obs):
    """
    Gaussian likelihood:
    log p(y_obs | theta) = -0.5 * sum[((y_obs - f(t; theta)) / noise_std)^2] + constant.
    (Constants are omitted here since they cancel in optimization.)
    """
    y_sim = simulator(theta, t)
    n = y_obs.numel()
    return -0.5 * torch.sum(((y_obs - y_sim) / noise_std) ** 2) - n * torch.log(torch.tensor(noise_std * np.sqrt(2 * np.pi)))

def log_q(theta, mu, sigma):
    """
    Log-density of q(theta) where q is factorized Gaussian.
    """
    return -0.5 * torch.sum(((theta - mu) / sigma) ** 2) - torch.sum(torch.log(sigma)) - theta.numel() * 0.5 * np.log(2 * np.pi)

# -------------------------------------------------
# 5. Define the ELBO (using the Reparameterization Trick)
# -------------------------------------------------
def elbo(mu, log_sigma, t, y_obs, num_samples=10):
    sigma = torch.exp(log_sigma)
    elbo_est = 0.0
    for _ in range(num_samples):
        # Sample epsilon ~ N(0,I)
        epsilon = torch.randn(2)
        # Reparameterize: theta = mu + sigma * epsilon
        theta_sample = mu + sigma * epsilon
        log_like = log_likelihood(theta_sample, t, y_obs)
        log_pr = log_prior(theta_sample)
        log_q_val = log_q(theta_sample, mu, sigma)
        elbo_est += (log_like + log_pr - log_q_val)
    return elbo_est / num_samples

# -------------------------------------------------
# 6. Optimize the ELBO
# -------------------------------------------------
optimizer = optim.Adam([mu, log_sigma], lr=0.01)
num_iterations = 5000
elbo_history = []

for i in range(num_iterations):
    optimizer.zero_grad()
    loss = -elbo(mu, log_sigma, t, y_obs, num_samples=10)  # maximize ELBO => minimize negative ELBO
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        current_elbo = -loss.item()
        elbo_history.append(current_elbo)
        print(f"Iteration {i}: ELBO = {current_elbo:.2f}")

# After optimization, our variational posterior is q(theta) = N(mu, diag(sigma^2)).
final_mu = mu.detach().numpy()
final_sigma = torch.exp(log_sigma).detach().numpy()
print("\nLearned Variational Parameters:")
print("mu =", final_mu)
print("sigma =", final_sigma)

# -------------------------------------------------
# 7. Visualization: Contour and Marginal Plots of q(theta)
# -------------------------------------------------
# Create a grid for theta1 and theta2
theta1_vals = np.linspace(final_mu[0] - 3 * final_sigma[0], final_mu[0] + 3 * final_sigma[0], 100)
theta2_vals = np.linspace(final_mu[1] - 3 * final_sigma[1], final_mu[1] + 3 * final_sigma[1], 100)
Theta1, Theta2 = np.meshgrid(theta1_vals, theta2_vals)

def q_density(theta1, theta2, mu, sigma):
    """Compute the product of two independent Gaussian densities."""
    density1 = 1 / (sigma[0] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((theta1 - mu[0]) / sigma[0]) ** 2)
    density2 = 1 / (sigma[1] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((theta2 - mu[1]) / sigma[1]) ** 2)
    return density1 * density2

Z_q = q_density(Theta1, Theta2, final_mu, final_sigma)

# Contour plot of the variational posterior
plt.figure(figsize=(8, 6))
cp = plt.contourf(Theta1, Theta2, Z_q, levels=30, cmap='viridis')
plt.xlabel("Theta 1")
plt.ylabel("Theta 2")
plt.title("Variational Posterior q(theta) Contour")
plt.colorbar(cp, label="Density")
plt.show()

# Compute marginal distributions via numerical integration
marginal_theta1 = np.trapz(Z_q, theta2_vals, axis=0)
marginal_theta2 = np.trapz(Z_q, theta1_vals, axis=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(theta1_vals, marginal_theta1, 'b-', lw=2)
axs[0].set_title("Marginal Distribution of Theta 1")
axs[0].set_xlabel("Theta 1")
axs[0].set_ylabel("Density")
axs[1].plot(theta2_vals, marginal_theta2, 'r-', lw=2)
axs[1].set_title("Marginal Distribution of Theta 2")
axs[1].set_xlabel("Theta 2")
axs[1].set_ylabel("Density")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 8. Compare Simulator Output Using Learned Parameters
# -------------------------------------------------
with torch.no_grad():
    y_sim_learned = simulator(mu, t)
plt.figure(figsize=(8, 6))
plt.plot(t.numpy(), y_obs.numpy(), 'o', label='Observed data')
plt.plot(t.numpy(), y_sim_learned.numpy(), 'r-', label='Simulator output (learned θ)')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.title("Simulator Output vs. Observed Data")
plt.show()

# -------------------------------------------------
# 9. Plot ELBO Optimization History
# -------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, num_iterations, 500), elbo_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ELBO Optimization History")
plt.show()
