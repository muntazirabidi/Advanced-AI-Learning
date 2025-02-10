# SWAGModel: Understanding the Code Structure

The code implements a `SWAGModel` class that combines a neural network with SWAG functionality. Let's analyze the key components:

### 1. Basic Network Architecture

```python
def __init__(self, nin, npars):
    # nin: number of input features
    # npars: number of parameters to predict

    hidden = 64
    layers = (
        [torch.nn.LayerNorm(nin)]
        + [torch.nn.Linear(nin, hidden), torch.nn.ReLU()]
        + [[torch.nn.Linear(hidden, hidden), torch.nn.ReLU()][i%2] for i in range(2*2)]
        + [torch.nn.Linear(hidden, nout)]
    )
    self.out = torch.nn.Sequential(*layers)
```

This creates a neural network with:

- Input normalization
- Hidden layers of size 64
- ReLU activations
- Output layer predicting parameters

### 2. SWAG Implementation

Remember how we discussed SWAG maintains running averages and deviations? Here's how the code implements it:

```python
def aggregate_model(self):
    cur_w = self.flatten()  # Get current weights as vector
    cur_w2 = cur_w ** 2    # Square of weights

    with torch.no_grad():
        # Update running averages
        if self.w_avg is None:
            self.w_avg = cur_w
            self.w2_avg = cur_w2
        else:
            self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
            self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)
```

This matches our earlier discussion where:

- We maintain running averages of weights ($\mu$)
- We keep track of squared weights for variance calculation
- We update these averages during training

### 3. Storing Deviations for Low-Rank Component

```python
# In aggregate_model
if self.pre_D is None:
    self.pre_D = cur_w.clone()[:, None]
else:
    # Store last K deviations
    self.pre_D = torch.cat((self.pre_D, cur_w[:, None]), dim=1)
    if self.pre_D.shape[1] > self.K:
        self.pre_D = self.pre_D[:, 1:]
```

This implements the low-rank approximation we discussed:

- Stores K=20 most recent weight vectors
- Used later to compute uncertainty

### 4. Sampling from SWAG Distribution

```python
def sample_weights(self, scale=1):
    with torch.no_grad():
        avg_w = self.w_avg
        avg_w2 = self.w2_avg
        D = self.pre_D - avg_w[:, None]

        # Sample random vectors
        z_1 = torch.randn((1, d))
        z_2 = torch.randn((K, 1))

        # Combine diagonal and low-rank components
        w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 * torch.abs(
            avg_w2 - avg_w ** 2) ** 0.5
        w += scale * (D @ z_2).T / np.sqrt(2 * (K - 1))
```

This implements the SWAG sampling formula we discussed:
$w = \mu + \frac{1}{\sqrt{2}}\Sigma^{1/2}z_1 + \frac{1}{\sqrt{2(K-1)}}Dz_2$

### 5. Training Loop Integration

```python
def train(self, x_train, y_train, delta_y=None, lr=1e-3, batch_size=32,
          num_epochs=10000, pretrain=False, mom_freq=100):
    # ... training loop ...
    if (not pretrain) and (count % mom_freq == 0):
        self.aggregate_model()
```

The key difference from standard training is:

- Every `mom_freq` steps, it calls `aggregate_model()`
- This updates the running averages and stores deviations
- This allows SWAG to track the trajectory of weights during training

### 6. Making Predictions with Uncertainty

```python
def generate_samples(self, x, nsamples, scale=0.5, verbose=True):
    samples = torch.zeros([nsamples, x.shape[0], self.npars])
    for i in range(nsamples):
        samples[i] = self.forward_swag(x, scale=scale)
    return samples
```

This implements what we discussed about getting predictions with uncertainty:

- Samples multiple sets of weights
- Makes predictions with each set
- Returns array of predictions that capture uncertainty

# Understanding DELFI vs cosmoSWAG for Cosmological Parameter Inference

## 1. Introduction to the Problem

In cosmology, we often need to infer parameters $\theta$ (like the Hubble constant $H_0$ or matter density $\Omega_m$) from observational data $x$ (like the Cosmic Microwave Background power spectrum). Traditional methods rely on explicitly calculating likelihood functions $P(x|\theta)$, but this becomes intractable for complex modern datasets.

## 2. Simulation-Based Inference (SBI)

SBI allows us to perform parameter inference without explicit likelihood calculations. Instead, we:

1. Generate simulations with known parameters
2. Train a model to learn the relationship between:
   - Observables $x$
   - Parameters $\theta$

## 3. DELFI (Density Estimation Likelihood-Free Inference)

### 3.1 Core Concept

DELFI directly learns the probability distribution $P(\theta|x)$ using neural density estimators.

### 3.2 Implementation

```python
# Multiple neural density estimators
NDEs = [
    # Masked Autoregressive Flow
    ndes.ConditionalMaskedAutoregressiveFlow(
        n_parameters=5,
        n_data=5,
        n_hiddens=[50,50]
    ),
    # Mixture Density Network
    ndes.MixtureDensityNetwork(
        n_parameters=5,
        n_data=5,
        n_components=10
    )
]
```

### 3.3 Mathematical Framework

The learned distribution can be expressed as:

- For Mixture Density Networks:
  $P(\theta|x) = \sum_{i=1}^K \alpha_i(x) \mathcal{N}(\theta|\mu_i(x), \Sigma_i(x))$
- For Normalizing Flows:
  $P(\theta|x) = P_0(\theta) \left|\det\frac{\partial f(x)}{\partial \theta}\right|^{-1}$

## 4. cosmoSWAG Approach

### 4.1 Core Concept

Uses Bayesian Neural Networks with Stochastic Weight Averaging-Gaussian (SWAG) to capture model uncertainty.

### 4.2 Implementation

```python
class SWAGModel:
    def __init__(self):
        self.w_avg = None    # Running average of weights
        self.w2_avg = None   # Running average of squared weights
        self.pre_D = None    # Deviation matrix
        self.K = 20          # Number of snapshots to maintain

    def sample_weights(self, scale=1):
        # Sample from weight distribution
        z_1 = torch.randn((1, d))
        z_2 = torch.randn((K, 1))
        w = avg_w + scale * σ * z_1 + D @ z_2
```

### 4.3 Mathematical Framework

SWAG approximates the posterior over weights as:
$w \sim \mathcal{N}(\mu, \frac{1}{2}\Sigma + \frac{1}{2(K-1)}DD^T)$

where:

- $\mu$ is the average of weights
- $\Sigma$ is the diagonal covariance
- $D$ is the deviation matrix
- $K$ is the number of collected models

## 5. Key Differences

### 5.1 Uncertainty Handling

- DELFI: Directly learns output distributions
- cosmoSWAG: Uncertainty comes from weight distributions

### 5.2 Computational Approach

- DELFI: Uses ensemble of different network architectures
- cosmoSWAG: Uses single network with weight uncertainty

### 5.3 Out-of-Distribution Behavior

- DELFI: Relies on ensemble diversity
- cosmoSWAG: Natural uncertainty scaling through weight sampling

## 6. Results and Applications

### 6.1 Performance Metrics

For cosmological parameters $\theta$:

- Coverage probability: $P(θ_{true} \in \text{predicted interval})$
- Excess probability (EP): Fraction of posterior samples with lower probability than true parameters

### 6.2 Key Findings

- cosmoSWAG shows better uncertainty estimates when data differs from simulations
- More computationally efficient than full DELFI implementation
- Better handling of systematic effects not present in training data

## 7. Mathematical Details for CMB Analysis

The power spectrum analysis involves:

- Input data: $C_\ell$ (CMB power spectrum)
- Parameters: $\theta = (H_0, \Omega_bh^2, \Omega_ch^2, \log A_s, n_s)$
- Likelihood approximation:
  $\mathcal{L}(\theta) \approx \int P(\theta|x)P(x|C_\ell)dx$
