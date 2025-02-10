# Gaussian Processes, Variational Inference, and Bayesian Neural Networks

## 1. Gaussian Processes (GPs)

### 1.1 Theoretical Foundation

A Gaussian Process is a collection of random variables, any finite subset of which follows a multivariate Gaussian distribution. GPs are completely specified by their mean function $m(x)$ and covariance function (kernel) $k(x, x')$:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

The mean function is often assumed to be zero for simplicity:

$$m(x) = \mathbb{E}[f(x)] = 0$$

### 1.2 Kernel Functions

The kernel function defines the covariance between any two points:

$$k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$$

Common kernel choices include:

1. Radial Basis Function (RBF):
   $$k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2l^2}\right)$$

2. Matérn Kernel:
   $$k(x, x') = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}\frac{\|x-x'\|}{l}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{\|x-x'\|}{l}\right)$$

### 1.3 GP Regression

For regression, given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, the predictive distribution at a new point $x_*$ is:

$$p(f_*|\mathcal{D}, x_*) = \mathcal{N}(\mu_*, \sigma_*^2)$$

where:
$$\mu_* = K_*K^{-1}y$$
$$\sigma_*^2 = K_{**} - K_*K^{-1}K_*^T$$

Here, $K$ is the kernel matrix for training points, $K_*$ is the kernel between test and training points, and $K_{**}$ is the kernel for the test point.

## 2. Variational Inference (VI)

### 2.1 Fundamentals

VI approximates complex posterior distributions $p(\theta|X)$ with a simpler variational distribution $q_\phi(\theta)$ by minimizing the Kullback-Leibler (KL) divergence:

$$\phi^* = \arg\min_\phi KL(q_\phi(\theta)||p(\theta|X))$$

### 2.2 Evidence Lower Bound (ELBO)

The optimization objective is the ELBO:

$$\text{ELBO}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(X|\theta)] - KL(q_\phi(\theta)||p(\theta))$$

This can be rewritten as:

$$\log p(X) = \text{ELBO}(\phi) + KL(q_\phi(\theta)||p(\theta|X))$$

### 2.3 Mean Field Approximation

Under mean field assumption, the variational distribution factorizes:

$$q_\phi(\theta) = \prod_{i=1}^m q_{\phi_i}(\theta_i)$$

### 2.4 Reparameterization Trick

To enable gradient-based optimization, we use the reparameterization trick:

$$\theta = g_\phi(\epsilon), \epsilon \sim p(\epsilon)$$

For Gaussian variational distributions:

$$\theta = \mu_\phi + \sigma_\phi \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$$

## 3. Bayesian Neural Networks (BNNs) and VI

### 3.1 Connection to VI

BNNs place distributions over network weights instead of point estimates. VI provides a practical way to train BNNs by approximating the true posterior over weights.

For a BNN with weights $W$, the variational objective is:

$$\text{ELBO}(q_\phi) = \mathbb{E}_{q_\phi(W)}\left[\sum_{i=1}^N \log p(y_i|x_i, W)\right] - KL(q_\phi(W)||p(W))$$

### 3.2 Practical Implementation

The variational distribution is typically chosen as:

$$q_\phi(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$$

For each weight $w_{ij}$, we learn variational parameters $\mu_{ij}$ and $\sigma_{ij}^2$.

### 3.3 Local Reparameterization

To reduce variance in gradient estimates, we can use the local reparameterization trick:

Instead of sampling weights:
$$w_{ij} = \mu_{ij} + \sigma_{ij}\epsilon_{ij}, \epsilon_{ij} \sim \mathcal{N}(0,1)$$

We sample activations directly:
$$a_j = \sum_i x_i\mu_{ij} + \sqrt{\sum_i x_i^2\sigma_{ij}^2}\epsilon_j, \epsilon_j \sim \mathcal{N}(0,1)$$

### 3.4 Mathematical Connection to GPs

As the width of a BNN approaches infinity, under certain conditions, it converges to a GP:

For a single-layer BNN with infinite width:
$$f(x) \sim \mathcal{GP}(0, k(x,x'))$$

where the kernel is:
$$k(x,x') = \sigma_w^2\mathbb{E}_{w\sim\mathcal{N}(0,1)}[\phi(w^Tx)\phi(w^Tx')]$$

Here, $\phi$ is the activation function and $\sigma_w^2$ is the prior variance of weights.

## 4. Key Differences and Trade-offs

### 4.1 Computational Complexity

- GPs: $O(N^3)$ for exact inference
- VI in BNNs: $O(N)$ per iteration

### 4.2 Modeling Flexibility

- GPs: Limited by kernel choice
- BNNs with VI: Can learn hierarchical features

### 4.3 Uncertainty Quantification

- GPs: Exact uncertainty (within model assumptions)
- VI: Approximate uncertainty, often underestimated

### 4.4 Scalability

- GPs: Challenging for large datasets
- VI: More scalable, supports mini-batch training

## 5. Recent Advances

### 5.1 Sparse Approximations

- Inducing points methods for GPs
- Structured variational approximations for BNNs

### 5.2 Hybrid Approaches

- Deep Kernel Learning
- Neural Processes
- Gaussian Process Layers in Deep Networks

These approaches combine the strengths of both worlds:

- Feature learning capability of neural networks
- Principled uncertainty quantification of GPs
- Scalability of VI
