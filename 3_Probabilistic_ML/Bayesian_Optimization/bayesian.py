import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
np.random.seed(42)

# True parameters of the linear model
true_slope = 2.0
true_intercept = 1.0
sigma_noise = 1.0  # known noise standard deviation

# Generate data: y = true_slope*x + true_intercept + noise
n_samples = 100
x = np.linspace(0, 10, n_samples)
y = true_slope * x + true_intercept + np.random.normal(0, sigma_noise, n_samples)

# -----------------------------
# 2. Define the Bayesian Model
# -----------------------------
# We assume:
#   Likelihood: y_i ~ Normal(m*x_i + b, sigma_noise^2)
#   Prior on m (slope): m ~ Normal(0, sigma_m^2)  with sigma_m = 5
#   Prior on b (intercept): b ~ Normal(0, sigma_b^2) with sigma_b = 5

sigma_m = 5.0
sigma_b = 5.0

def log_likelihood(m, b, x, y, sigma):
    """
    Computes the log-likelihood of the data given model parameters.
    """
    # For each data point, the likelihood is:
    #   p(y_i | m, b) = N(y_i; m*x_i + b, sigma^2)
    return np.sum(norm.logpdf(y, loc=m * x + b, scale=sigma))

def log_prior(m, b, sigma_m, sigma_b):
    """
    Computes the log-prior of the parameters.
    """
    # Independent priors for m and b
    return norm.logpdf(m, loc=0, scale=sigma_m) + norm.logpdf(b, loc=0, scale=sigma_b)

def log_posterior(params, x, y, sigma, sigma_m, sigma_b):
    """
    Computes the joint log-posterior of parameters [m, b].
    """
    m, b = params
    return log_likelihood(m, b, x, y, sigma) + log_prior(m, b, sigma_m, sigma_b)

# -----------------------------
# 3. Metropolis–Hastings Sampler
# -----------------------------
def metropolis_sampler(log_posterior, initial_params, iterations, proposal_width, x, y, sigma, sigma_m, sigma_b):
    """
    A simple Metropolis-Hastings sampler.
    
    Parameters:
      log_posterior    : function that computes the log posterior.
      initial_params   : starting guess for [m, b].
      iterations       : number of iterations.
      proposal_width   : standard deviation of the proposal distribution.
      x, y             : the observed data.
      sigma, sigma_m, sigma_b: noise and prior parameters.
      
    Returns:
      samples: an array of shape (iterations, 2) with the sampled [m, b] values.
    """
    samples = []
    current_params = np.array(initial_params)
    current_log_post = log_posterior(current_params, x, y, sigma, sigma_m, sigma_b)
    
    for i in range(iterations):
        # Propose new parameters by adding a small Gaussian perturbation.
        proposed_params = current_params + np.random.normal(0, proposal_width, size=current_params.shape)
        proposed_log_post = log_posterior(proposed_params, x, y, sigma, sigma_m, sigma_b)
        
        # Compute acceptance probability (in log-space for stability)
        acceptance_prob = np.exp(proposed_log_post - current_log_post)
        if np.random.rand() < acceptance_prob:
            current_params = proposed_params
            current_log_post = proposed_log_post
            
        samples.append(current_params.copy())
    return np.array(samples)

# Run the sampler
initial_params = [0.0, 0.0]    # initial guess for [m, b]
iterations = 5000            # total iterations
proposal_width = 0.1         # proposal standard deviation

samples = metropolis_sampler(log_posterior, initial_params, iterations, proposal_width,
                             x, y, sigma_noise, sigma_m, sigma_b)

# Remove burn-in (first 1000 samples)
burn_in = 1000
posterior_samples = samples[burn_in:]

# -----------------------------
# 4. Visualize the Posterior
# -----------------------------
# Create a DataFrame for easier plotting with seaborn
df_samples = pd.DataFrame(posterior_samples, columns=['Slope (m)', 'Intercept (b)'])

# Joint plot: shows scatter and 2D density contours of the posterior samples.
sns.jointplot(x='Slope (m)', y='Intercept (b)', data=df_samples, kind='scatter', alpha=0.5)
plt.suptitle('Joint Posterior Samples for Slope and Intercept', y=1.02)
plt.show()

# Marginal distributions: histograms with kernel density estimates.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_samples['Slope (m)'], kde=True, ax=axes[0])
axes[0].set_title('Posterior of Slope (m)')
sns.histplot(df_samples['Intercept (b)'], kde=True, ax=axes[1])
axes[1].set_title('Posterior of Intercept (b)')
plt.show()

# Optionally: Evaluate the log-posterior on a grid for a contour plot
m_vals = np.linspace(1.5, 2.5, 100)
b_vals = np.linspace(0.5, 1.5, 100)
M, B = np.meshgrid(m_vals, b_vals)

# Compute the log-posterior on the grid (shifted for numerical stability)
Z = np.array([[log_posterior([m, b], x, y, sigma_noise, sigma_m, sigma_b) for m in m_vals] for b in b_vals])
Z = Z - np.max(Z)  # subtract max for numerical stability
density = np.exp(Z)

plt.figure(figsize=(8, 6))
plt.contourf(M, B, density, levels=50, cmap='viridis')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.title('Posterior Density Contours (Grid Evaluation)')
plt.colorbar(label='Density')
plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], color='white', s=10, alpha=0.3)
plt.show()
