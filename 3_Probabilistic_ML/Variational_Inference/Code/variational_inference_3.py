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
    Modified simulator function with interaction between theta[0] and theta[1].
    
    theta: tensor of shape (2,)
    t: tensor of time points
    
    Returns: 
      f(t; theta) = theta[0]*sin(t) + theta[1]*cos(t) + 0.5*theta[0]*theta[1]*sin(t)*cos(t)
      
    The extra term 0.5*theta[0]*theta[1]*sin(t)*cos(t) introduces a nonlinear interaction,
    thereby coupling theta[0] and theta[1].
    """
    alpha = 0.5  # Interaction strength
    return theta[0] * torch.sin(t) + theta[1] * torch.cos(t) + alpha * theta[0] * theta[1] * torch.sin(t) * torch.cos(t)

# -------------------------------------------------
# 2. Generate Observed Data Using True Parameters
# -------------------------------------------------
# True parameters (unknown in practice)
true_theta = torch.tensor([2.0, -1.0])
# Time grid for simulation
t = torch.linspace(0, 2 * np.pi, 100)
# Noise level
noise_std = 0.5

with torch.no_grad():
    y_obs = simulator(true_theta, t) + noise_std * torch.randn(t.shape)

# -------------------------------------------------
# 3. Define the Variational Distribution with Full Covariance
# -------------------------------------------------
# We now parameterize q(theta) = N(mu, Σ) with Σ = L L^T.
# We learn mu (a vector of length 2) and three parameters for L:
#   L[0,0] = exp(l11),  L[1,0] = l21,  L[1,1] = exp(l22), and L[0,1] = 0.
mu = torch.randn(2, requires_grad=True)
l_params = torch.randn(3, requires_grad=True)  # [l11, l21, l22]

def get_L(l_params):
    """
    Constructs a 2x2 lower-triangular matrix L from l_params.
    Ensures positive diagonal entries by exponentiating.
    """
    L = torch.zeros(2,2)
    L[0,0] = torch.exp(l_params[0])
    L[1,0] = l_params[1]
    L[0,1] = 0.0
    L[1,1] = torch.exp(l_params[2])
    return L

def sample_theta(mu, L):
    """
    Sample theta using the reparameterization: theta = mu + L*epsilon,
    where epsilon ~ N(0,I)
    """
    epsilon = torch.randn(2)
    return mu + L @ epsilon

def log_q(theta, mu, L):
    """
    Log-density of a full multivariate normal:
      q(theta) = N(theta; mu, Σ) with Σ = L L^T.
    We use the formula:
      log q(theta) = -0.5 * (d*log(2π) + log|Σ| + (theta-μ)^T Σ^{-1} (theta-μ))
    """
    d = theta.numel()
    diff = (theta - mu).unsqueeze(1)  # (2,1)
    # Compute covariance matrix Σ = L L^T
    Sigma = L @ L.t()
    # Compute inverse and log determinant of Σ.
    Sigma_inv = torch.inverse(Sigma)
    # Quadratic form:
    quad = (diff.t() @ Sigma_inv @ diff).squeeze()
    # log determinant: |Σ| = (L[0,0]*L[1,1])^2, so log|Σ| = 2*(log L[0,0] + log L[1,1])
    logdet = 2 * (torch.log(L[0,0]) + torch.log(L[1,1]))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

# -------------------------------------------------
# 4. Define the Prior and Likelihood (same as before)
# -------------------------------------------------
def log_prior(theta):
    """
    Prior: p(theta) ~ N(0, 3^2 I)
    """
    prior_std = 3.0
    return -0.5 * torch.sum((theta / prior_std) ** 2) - theta.numel() * torch.log(torch.tensor(prior_std * np.sqrt(2 * np.pi)))

def log_likelihood(theta, t, y_obs):
    """
    Gaussian likelihood:
      log p(y_obs | theta) = -0.5 * sum[((y_obs - f(t;theta))/noise_std)^2] + constant.
    """
    y_sim = simulator(theta, t)
    n = y_obs.numel()
    return -0.5 * torch.sum(((y_obs - y_sim) / noise_std) ** 2) - n * torch.log(torch.tensor(noise_std * np.sqrt(2 * np.pi)))

# -------------------------------------------------
# 5. Define the ELBO (using the Reparameterization Trick)
# -------------------------------------------------
def elbo(mu, l_params, t, y_obs, num_samples=10):
    L = get_L(l_params)
    elbo_est = 0.0
    for _ in range(num_samples):
        theta_sample = sample_theta(mu, L)
        ll = log_likelihood(theta_sample, t, y_obs)
        lp = log_prior(theta_sample)
        lq = log_q(theta_sample, mu, L)
        elbo_est += (ll + lp - lq)
    return elbo_est / num_samples

# -------------------------------------------------
# 6. Optimize the ELBO
# -------------------------------------------------
optimizer = optim.Adam([mu, l_params], lr=0.01)
num_iterations = 5000
elbo_history = []

for i in range(num_iterations):
    optimizer.zero_grad()
    loss = -elbo(mu, l_params, t, y_obs, num_samples=10)  # maximize ELBO
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        current_elbo = -loss.item()
        elbo_history.append(current_elbo)
        print(f"Iteration {i}: ELBO = {current_elbo:.2f}")

# After optimization, our variational posterior is q(theta) = N(mu, Σ) with Σ = L L^T.
final_mu = mu.detach().numpy()
final_L = get_L(l_params).detach().numpy()
final_Sigma = final_L @ final_L.T
print("\nLearned Variational Parameters:")
print("mu =", final_mu)
print("Cholesky factor L =\n", final_L)
print("Covariance Σ =\n", final_Sigma)

# -------------------------------------------------
# 7. Visualization: Contour and Marginal Plots of q(theta)
# -------------------------------------------------
# Create a grid for theta1 and theta2 around the learned mean.
theta1_vals = np.linspace(final_mu[0] - 3*np.sqrt(final_Sigma[0,0]),
                            final_mu[0] + 3*np.sqrt(final_Sigma[0,0]), 100)
theta2_vals = np.linspace(final_mu[1] - 3*np.sqrt(final_Sigma[1,1]),
                            final_mu[1] + 3*np.sqrt(final_Sigma[1,1]), 100)
Theta1, Theta2 = np.meshgrid(theta1_vals, theta2_vals)

def q_density_full(theta1, theta2, mu, Sigma):
    """
    Evaluate the density of a 2D Gaussian with mean mu and covariance Sigma.
    """
    d = 2
    diff = np.stack([theta1 - mu[0], theta2 - mu[1]], axis=-1)
    inv_Sigma = np.linalg.inv(Sigma)
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, inv_Sigma, diff)
    norm_const = 1 / ((2*np.pi)**(d/2) * np.sqrt(np.linalg.det(Sigma)))
    return norm_const * np.exp(exponent)

Z_q_full = q_density_full(Theta1, Theta2, final_mu, final_Sigma)

plt.figure(figsize=(8, 6))
cp = plt.contourf(Theta1, Theta2, Z_q_full, levels=30, cmap='viridis')
plt.xlabel("Theta 1")
plt.ylabel("Theta 2")
plt.title("Variational Posterior q(theta) (Full Covariance) Contour")
plt.colorbar(cp, label="Density")
plt.show()

# Compute marginal distributions via numerical integration.
marginal_theta1 = np.trapz(Z_q_full, theta2_vals, axis=0)
marginal_theta2 = np.trapz(Z_q_full, theta1_vals, axis=1)

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
    # For prediction, use the learned mu as the best estimate for theta.
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

# -------------------------------------------------
# 10. Additional Plots: Uncertainty Visualization
# -------------------------------------------------
# 10a. Posterior Samples Scatter Plot
num_samples_theta = 200
theta_samples = np.array([sample_theta(mu, get_L(l_params)).detach().numpy() for _ in range(num_samples_theta)])
plt.figure(figsize=(8,6))
plt.contourf(Theta1, Theta2, Z_q_full, levels=30, cmap='viridis', alpha=0.7)
plt.scatter(theta_samples[:,0], theta_samples[:,1], c='white', edgecolor='black', label='Posterior samples')
plt.xlabel("Theta 1")
plt.ylabel("Theta 2")
plt.title("Posterior Samples from q(theta)")
plt.legend()
plt.show()

# 10b. Predictive Distribution Uncertainty
# Draw many samples from the variational posterior, compute simulator outputs,
# and then compute predictive mean and 95% credible intervals.
num_predictive_samples = 1000
predictions = np.zeros((num_predictive_samples, len(t)))
for i in range(num_predictive_samples):
    theta_sample = sample_theta(mu, get_L(l_params)).detach()
    predictions[i,:] = simulator(theta_sample, t).detach().numpy()

pred_mean = np.mean(predictions, axis=0)
pred_lower = np.percentile(predictions, 2.5, axis=0)
pred_upper = np.percentile(predictions, 97.5, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(t.numpy(), y_obs.numpy(), 'o', label='Observed data')
plt.plot(t.numpy(), pred_mean, 'r-', label='Predictive mean')
plt.fill_between(t.numpy(), pred_lower, pred_upper, color='r', alpha=0.3, label='95% CI')
plt.xlabel("t")
plt.ylabel("y")
plt.title("Predictive Distribution with Uncertainty Bands")
plt.legend()
plt.show()
