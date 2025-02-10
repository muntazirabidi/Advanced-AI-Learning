# Variational Inference: Mathematical Foundation and Implementation

This document explains the mathematical foundations of variational inference as implemented in the provided code. We'll walk through each component of the implementation, from the prior distribution to the evidence lower bound (ELBO) optimization.

## 1. The Prior Distribution

The implementation uses an isotropic Gaussian prior on the parameters θ. This choice provides a reasonable starting point for many problems while maintaining computational tractability.

### Mathematical Formulation

The prior is defined as:

$$p(\theta) = \mathcal{N}(\theta; 0, \tau^2I)$$

where τ = 3.0. This means each component θᵢ is independently distributed as:

$$\theta_i \sim \mathcal{N}(0, \tau^2)$$

### Log Prior Density

The log density of a single component is:

$$\log \mathcal{N}(\theta_i; 0, \tau^2) = -\frac{1}{2}\left(\frac{\theta_i}{\tau}\right)^2 - \log(\tau\sqrt{2\pi})$$

For the full parameter vector, the joint log-prior becomes:

$$\log p(\theta) = \sum_{i=1}^d \log \mathcal{N}(\theta_i; 0, \tau^2) = -\frac{1}{2}\sum_{i=1}^d \left(\frac{\theta_i}{\tau}\right)^2 - d\log(\tau\sqrt{2\pi})$$

## 2. The Likelihood Function

The likelihood function connects our model's predictions to the observed data. It quantifies how well different parameter values explain the observations.

### Model Assumptions

The simulator produces predictions f(t; θ) for input t. We assume the observed data follows:

$$y_{obs} = f(t; \theta) + \epsilon$$

where ε ~ 𝒩(0, σ²) and σ is specified by noise_std.

### Log Likelihood

For each observation:

$$p(y_{obs,i}|\theta) = \mathcal{N}(y_{obs,i}; f(t_i; \theta), \sigma^2)$$

The total log likelihood across all n data points is:

$$\log p(y_{obs}|\theta) = -\frac{1}{2}\sum_{i=1}^n \left(\frac{y_{obs,i} - f(t_i; \theta)}{\sigma}\right)^2 - n\log(\sigma\sqrt{2\pi})$$

## 3. The Variational Distribution

The variational distribution q(θ) approximates the true posterior p(θ|yₒbₛ). We use a factorized (diagonal) Gaussian for computational efficiency.

### Structure

The variational distribution is defined as:

$$q(\theta; \mu, \sigma) = \prod_{i=1}^d \mathcal{N}(\theta_i; \mu_i, \sigma_i^2)$$

where μ ∈ ℝᵈ is the mean vector and σ ∈ ℝ₊ᵈ contains the standard deviations.

### Log Density

The log density for a single component:

$$\log \mathcal{N}(\theta_i; \mu_i, \sigma_i^2) = -\frac{1}{2}\left(\frac{\theta_i - \mu_i}{\sigma_i}\right)^2 - \log(\sigma_i\sqrt{2\pi})$$

For the full vector:

$$\log q(\theta; \mu, \sigma) = -\frac{1}{2}\sum_{i=1}^d \left(\frac{\theta_i - \mu_i}{\sigma_i}\right)^2 - \sum_{i=1}^d \log(\sigma_i) - \frac{d}{2}\log(2\pi)$$

## 4. The Evidence Lower Bound (ELBO)

The ELBO is our optimization objective, providing a lower bound on the marginal likelihood of the data.

### Definition

$$\text{ELBO} = \mathbb{E}_{q(\theta; \mu, \sigma)}[\log p(y_{obs}|\theta) + \log p(\theta) - \log q(\theta; \mu, \sigma)]$$

### Monte Carlo Estimation

We use the reparameterization trick to enable gradient-based optimization:

$$\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The Monte Carlo estimate of the ELBO becomes:

$$\widehat{\text{ELBO}} = \frac{1}{S}\sum_{s=1}^S [\log p(y_{obs}|\theta^{(s)}) + \log p(\theta^{(s)}) - \log q(\theta^{(s)}; \mu, \sigma)]$$

where θ⁽ˢ⁾ = μ + σ ⊙ ε⁽ˢ⁾.

## Implementation Notes

The code implements this mathematical framework using PyTorch for automatic differentiation and optimization. Key components include:

1. A simulator function that defines the relationship between parameters and observations
2. Prior and likelihood functions implementing the log densities
3. The variational distribution implementation using the reparameterization trick
4. ELBO computation and optimization using ADAM

The implementation includes visualization tools to monitor convergence and examine the learned posterior approximation through contour plots and marginal distributions.

## Practical Considerations

- The choice of prior standard deviation (τ = 3.0) affects the regularization strength
- The number of Monte Carlo samples in ELBO estimation (default 10) trades off accuracy vs. computational cost
- The optimization uses ADAM with learning rate 0.01 and runs for 5000 iterations
- The code includes comprehensive visualization tools to assess the quality of the posterior approximation

## Code Implementation

This section provides a detailed walkthrough of how the mathematical concepts are implemented in code. We'll examine each component and explain how it relates to the mathematical foundation.

### Setup and Dependencies

```python
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

The implementation uses PyTorch for automatic differentiation and optimization, NumPy for numerical operations, and Matplotlib/Seaborn for visualization.

### The Simulator (Implicit Likelihood)

```python
def simulator(theta, t):
    """
    Simulator function.
    theta: tensor of shape (2,)
    t: tensor of time points
    Returns: f(t; theta) = theta[0] * sin(t) + theta[1] * cos(t)
    """
    return theta[0] * torch.sin(t) + theta[1] * torch.cos(t)
```

This implements our forward model f(t; θ). In this case, it's a simple harmonic function where:

- θ₁ controls the amplitude of the sine component
- θ₂ controls the amplitude of the cosine component

### Data Generation

```python
# True parameters (unknown in practice)
true_theta = torch.tensor([2.0, -1.0])
# Time grid for simulation
t = torch.linspace(0, 2 * np.pi, 100)
# Noise level
noise_std = 0.5

# Generate observed data: simulator output plus Gaussian noise
with torch.no_grad():
    y_obs = simulator(true_theta, t) + noise_std * torch.randn(t.shape)
```

This section generates synthetic data following our model assumptions:

- Sets true parameters θ = [2.0, -1.0]
- Creates a time grid from 0 to 2π with 100 points
- Adds Gaussian noise with σ = 0.5 to the simulator output

### Prior Implementation

```python
def log_prior(theta):
    """
    Assume p(theta) is a Gaussian with mean 0 and standard deviation 3.
    """
    prior_std = 3.0
    return -0.5 * torch.sum((theta / prior_std) ** 2) - \
           theta.numel() * torch.log(torch.tensor(prior_std * np.sqrt(2 * np.pi)))
```

This implements the log prior density derived in Section 1:

- Uses an isotropic Gaussian with τ = 3.0
- Computes both the quadratic term and the normalization constant
- Returns the sum over all parameters

### Likelihood Implementation

```python
def log_likelihood(theta, t, y_obs):
    """
    Gaussian likelihood:
    log p(y_obs | theta) = -0.5 * sum[((y_obs - f(t; theta)) / noise_std)^2] + constant
    """
    y_sim = simulator(theta, t)
    n = y_obs.numel()
    return -0.5 * torch.sum(((y_obs - y_sim) / noise_std) ** 2) - n * torch.log(torch.tensor(noise_std * np.sqrt(2 * np.pi)))
```

This implements the log likelihood derived in Section 2:

- Computes simulator predictions for given parameters
- Calculates the squared error normalized by noise standard deviation
- Includes the normalization term

### Variational Distribution Implementation

```python
def log_q(theta, mu, sigma):
    """
    Log-density of q(theta) where q is factorized Gaussian.
    """
    return -0.5 * torch.sum(((theta - mu) / sigma) ** 2) - torch.sum(torch.log(sigma)) -  theta.numel() * 0.5 * np.log(2 * np.pi)
```

This implements the log density of the variational distribution from Section 3:

- Uses a diagonal Gaussian parameterized by mean μ and standard deviation σ
- Computes both the quadratic term and normalization constants

### ELBO Implementation

```python
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
```

This implements the Monte Carlo estimation of the ELBO from Section 4:

- Uses the reparameterization trick for sampling
- Averages over multiple samples to reduce variance
- Works with log_sigma for numerical stability

### Optimization

```python
# Initialize variational parameters
mu = torch.randn(2, requires_grad=True)
log_sigma = torch.randn(2, requires_grad=True)

# Setup optimizer
optimizer = optim.Adam([mu, log_sigma], lr=0.01)
num_iterations = 5000
elbo_history = []

# Optimization loop
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = -elbo(mu, log_sigma, t, y_obs, num_samples=10)
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        current_elbo = -loss.item()
        elbo_history.append(current_elbo)
        print(f"Iteration {i}: ELBO = {current_elbo:.2f}")
```

The optimization process:

1. Initializes variational parameters randomly
2. Uses ADAM optimizer with learning rate 0.01
3. Maximizes the ELBO by minimizing its negative
4. Tracks the optimization progress

### Visualization Functions

The implementation includes several visualization functions to examine the results:

1. Contour Plot of Posterior:

```python
def q_density(theta1, theta2, mu, sigma):
    """Compute the product of two independent Gaussian densities."""
    density1 = 1 / (sigma[0] * np.sqrt(2 * np.pi)) * \
              np.exp(-0.5 * ((theta1 - mu[0]) / sigma[0]) ** 2)
    density2 = 1 / (sigma[1] * np.sqrt(2 * np.pi)) * \
              np.exp(-0.5 * ((theta2 - mu[1]) / sigma[1]) ** 2)
    return density1 * density2
```

2. Data Fit Visualization:

```python
with torch.no_grad():
    y_sim_learned = simulator(mu, t)
plt.figure(figsize=(8, 6))
plt.plot(t.numpy(), y_obs.numpy(), 'o', label='Observed data')
plt.plot(t.numpy(), y_sim_learned.numpy(), 'r-',
         label='Simulator output (learned θ)')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.title("Simulator Output vs. Observed Data")
plt.show()
```

## Implementation Tips and Best Practices

1. **Numerical Stability**:

   - Work with log_sigma instead of sigma directly
   - Use log probabilities instead of raw probabilities
   - Implement careful error checking for numerical operations

2. **Optimization**:

   - Monitor the ELBO during training to ensure convergence
   - Use appropriate learning rates (0.01 works well for this problem)
   - Consider implementing early stopping if the ELBO plateaus

3. **Sampling**:

   - Increase num_samples for more accurate ELBO estimates
   - Use the reparameterization trick for lower variance gradients
   - Consider implementing importance sampling for better estimates

4. **Diagnostics**:
   - Plot the ELBO history to check convergence
   - Visualize the learned posterior to assess uncertainty
   - Compare simulator outputs with observed data

## Common Issues and Solutions

1. **Poor Convergence**:

   - Try different learning rates
   - Increase the number of Monte Carlo samples
   - Check for numerical instabilities in the implementation

2. **High Variance**:

   - Increase the number of Monte Carlo samples
   - Implement variance reduction techniques
   - Consider using control variates

3. **Numerical Overflow/Underflow**:
   - Work in log space
   - Use stable implementations of mathematical operations
   - Add small constants to denominators where necessary
