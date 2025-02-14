# Gaussian Processes and Their Variations

## 1. Foundation: Gaussian Processes

### 1.1 Basic Intuition

A Gaussian Process (GP) is a collection of random variables where any finite subset follows a multivariate normal distribution. Think of it as an infinite-dimensional extension of multivariate Gaussian distributions. Instead of dealing with vectors, we're working with functions.

The key insight is that a GP defines a distribution over functions. When we observe data points, we're essentially constraining this distribution to pass through (or near) these points.

### 1.2 Mathematical Definition

Formally, a Gaussian Process is defined by its mean function $m(x)$ and covariance function $k(x, x')$:

$f(x) \sim \mathcal{GP}(m(x), k(x,x'))$

where:

- $m(x) = \mathbb{E}[f(x)]$
- $k(x,x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$

### 1.3 Key Components

#### Mean Function

The mean function $m(x)$ represents our prior belief about the function's expected value at any point. Often set to zero for simplicity:

$m(x) = 0$

#### Covariance Function (Kernel)

The kernel $k(x,x')$ defines the similarity between points. Common choices include:

1. Squared Exponential (RBF):
   $k(x,x') = \sigma^2 \exp(-\frac{||x-x'||^2}{2l^2})$

2. Matérn Kernel:
   $k(x,x') = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}\frac{||x-x'||}{l}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{||x-x'||}{l}\right)$

### 1.4 Inference

For a training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, the posterior distribution is:

$p(f_*|X_*, X, y) = \mathcal{N}(\mu_*, \Sigma_*)$

where:

- $\mu_* = K_{*,X}K_{X,X}^{-1}y$
- $\Sigma_* = K_{*,*} - K_{*,X}K_{X,X}^{-1}K_{X,*}$

## 2. Sparse Gaussian Process Regression

### 2.1 Motivation

Standard GP regression scales cubically $O(N^3)$ with the number of data points, making it impractical for large datasets. Sparse GP methods address this by using a subset of inducing points.

### 2.2 Mathematical Framework

We introduce $M$ inducing points $Z = \{z_m\}_{m=1}^M$ where $M \ll N$. The key idea is to approximate the full GP using these points.

#### Deterministic Training Conditional (DTC)

The approximation takes the form:

$q(f_*) = \mathcal{N}(K_{*u}K_{uu}^{-1}m_u, K_{**} - K_{*u}K_{uu}^{-1}K_{u*})$

where:

- $K_{uu}$ is the kernel matrix between inducing points
- $K_{*u}$ is the kernel matrix between test points and inducing points

### 2.3 Methods

1. **Subset of Data (SoD)**:

   - Simplest approach
   - Uses a random subset of training data
   - Fast but potentially inaccurate

2. **Fully Independent Training Conditional (FITC)**:
   - Maintains diagonal elements of original covariance
   - Better uncertainty estimates
   - $Q_{ff} = K_{fu}K_{uu}^{-1}K_{uf} + \text{diag}(K_{ff} - K_{fu}K_{uu}^{-1}K_{uf})$

## 3. Variational Gaussian Process

### 3.1 Core Concept

Variational GPs use variational inference to approximate the posterior distribution. This approach provides a principled way to handle non-Gaussian likelihoods.

### 3.2 Mathematical Framework

The variational lower bound (ELBO) is:

$\mathcal{L} = \mathbb{E}_{q(f)}[\log p(y|f)] - KL[q(f)||p(f)]$

where:

- $q(f)$ is the variational approximation
- $p(f)$ is the prior
- $p(y|f)$ is the likelihood

### 3.3 Optimization

The ELBO is optimized with respect to:

1. Variational parameters
2. Kernel hyperparameters
3. Likelihood parameters

## 4. Sparse Variational Gaussian Process

### 4.1 Combining Ideas

SVGP combines sparse methods with variational inference, providing a scalable and flexible framework.

### 4.2 Mathematical Framework

The variational distribution takes the form:

$q(f(x)) = \int p(f(x)|u)q(u)du$

where:

- $u$ are the inducing variables
- $q(u)$ is a free-form variational distribution

### 4.3 Modified ELBO

The ELBO becomes:

$\mathcal{L} = \sum_{i=1}^N \mathbb{E}_{q(f(x_i))}[\log p(y_i|f(x_i))] - KL[q(u)||p(u)]$

### 4.4 Advantages

1. Mini-batch training possible
2. Scales to large datasets
3. Handles non-Gaussian likelihoods
4. Memory efficient

## 5. Comparative Analysis

### 5.1 Computational Complexity

| Method    | Training  | Prediction | Memory   |
| --------- | --------- | ---------- | -------- |
| Full GP   | $O(N^3)$  | $O(N^2)$   | $O(N^2)$ |
| Sparse GP | $O(NM^2)$ | $O(M^2)$   | $O(NM)$  |
| VGP       | $O(N^3)$  | $O(N^2)$   | $O(N^2)$ |
| SVGP      | $O(M^3)$  | $O(M^2)$   | $O(M^2)$ |

### 5.2 Pros and Cons

#### Full GP

Pros:

- Exact inference
- Well-calibrated uncertainty
- Theoretically well-understood

Cons:

- Poor scaling
- Limited to Gaussian likelihoods
- High memory requirements

#### Sparse GP

Pros:

- Better scaling
- Maintains some theoretical guarantees
- Simple implementation

Cons:

- Approximate inference
- Inducing point selection crucial
- Still limited to Gaussian likelihoods

#### VGP

Pros:

- Handles non-Gaussian likelihoods
- Principled approximation
- Well-calibrated uncertainties

Cons:

- Still scales poorly
- Optimization can be challenging
- More complex implementation

#### SVGP

Pros:

- Best scaling properties
- Handles non-Gaussian likelihoods
- Mini-batch training possible

Cons:

- Most approximate
- Many parameters to tune
- Complex implementation

## 6. Implementation Example

Here's a simple example using GPflow for SVGP:

```python
import gpflow
import numpy as np

# Generate synthetic data
X = np.random.randn(1000, 1)
Y = np.sin(X) + 0.1*np.random.randn(1000, 1)

# Create inducing points
Z = np.linspace(-2, 2, 20)[:, None]

# Define kernel
kernel = gpflow.kernels.SquaredExponential()

# Create SVGP model
model = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=gpflow.likelihoods.Gaussian(),
    inducing_variable=Z,
    num_data=len(X)
)

# Optimize
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss_closure(data=(X, Y)),
            model.trainable_variables)
```

## 7. Understanding Gaussian and Non-Gaussian Likelihoods

### 7.1 Gaussian Likelihood: When to Use

Gaussian likelihoods are appropriate in several scenarios:

1. **Continuous Data with Homoscedastic Noise**: When your observations can be modeled as the true function plus independent, identically distributed Gaussian noise:

   $y = f(x) + \epsilon, \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2)$

2. **Central Limit Theorem Cases**: When your observations are the result of many small, independent effects adding up, the Central Limit Theorem suggests a Gaussian likelihood might be appropriate.

3. **Well-Behaved Measurement Error**: In scientific measurements where instrument errors are well-understood and approximately normally distributed.

Example applications include:

- Temperature measurements
- Physical sensor readings
- Financial returns (in some cases)
- Height/weight predictions
- Material property estimation

### 7.2 When to Avoid Gaussian Likelihoods

Gaussian likelihoods are inappropriate for:

1. **Count Data**: When your observations are non-negative integers. Use Poisson or Negative Binomial likelihoods instead:

   $p(y|f) = \text{Poisson}(\exp(f))$

2. **Binary Classification**: When your outcomes are binary (0/1). Use Bernoulli likelihood:

   $p(y|f) = \text{Bernoulli}(\sigma(f))$

3. **Bounded Data**: When your observations have natural bounds (e.g., probabilities between 0 and 1). Consider Beta likelihood.

4. **Heavy-tailed Data**: When your data has more extreme values than a Gaussian would predict. Consider Student-t likelihood:

   $p(y|f) = \text{Student-t}(\nu, f, \sigma)$

5. **Heteroscedastic Noise**: When observation noise varies with input. Consider:
   - Heteroscedastic Gaussian likelihood
   - Student-t likelihood
   - A separate GP for the noise variance

### 7.3 Alternative Likelihoods

Here's a mathematical framework for common non-Gaussian likelihoods:

1. **Bernoulli (Classification)**:
   $p(y|f) = \sigma(f)^y(1-\sigma(f))^{1-y}$
   where $\sigma$ is the logistic function

2. **Poisson (Count Data)**:
   $p(y|f) = \frac{\lambda^y e^{-\lambda}}{y!}$
   where $\lambda = \exp(f)$

3. **Beta (Bounded Data)**:
   $p(y|f) = \frac{y^{\alpha-1}(1-y)^{\beta-1}}{B(\alpha,\beta)}$
   where $\alpha, \beta$ are derived from $f$

### 7.4 Implementation Considerations

When using non-Gaussian likelihoods:

1. **Inference Method**: You must use approximate inference methods:

   - Variational Gaussian Process (VGP)
   - Expectation Propagation (EP)
   - Markov Chain Monte Carlo (MCMC)

2. **Link Functions**: Choose appropriate link functions:

   - logit/probit for classification
   - log for count data
   - identity for regression

3. **Computational Cost**: Non-Gaussian likelihoods often require:
   - Numerical optimization
   - Monte Carlo sampling
   - More complex implementation

Example in GPflow for binary classification:

```python
# Binary classification with Bernoulli likelihood
model = gpflow.models.VGP(
    data=(X, Y),
    kernel=gpflow.kernels.Matern52(),
    likelihood=gpflow.likelihoods.Bernoulli(),
    num_latent_gps=1
)
```

## 8. Further Reading and Advanced Topics

1. Multi-output Gaussian Processes
2. Deep Gaussian Processes
3. Convolutional Gaussian Processes
4. Gaussian Process Latent Variable Models
5. Manifold Gaussian Processes

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning
2. Titsias, M. (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes
3. Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). Scalable Variational Gaussian Process Classification
