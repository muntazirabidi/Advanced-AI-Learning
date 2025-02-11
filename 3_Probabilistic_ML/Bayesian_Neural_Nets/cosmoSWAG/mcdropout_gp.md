Think of Gaussian Processes as a fundamentally different way of doing machine learning compared to neural networks. Instead of learning weights between neurons, a GP defines a probability distribution over functions directly. Let me explain with a simple example:

Let's say we want to learn the relationship between CMB power spectrum (x) and cosmological parameters (θ). Here's how each approach works:

Neural Network Approach:

```python
class CosmoNetwork(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(n_input, 100),
            nn.ReLU(),
            nn.Linear(100, n_params)
        )

    def forward(self, x):
        return self.layers(x)  # Gives a single prediction
```

Gaussian Process Approach:

```python
from sklearn.gaussian_process import GaussianProcessRegressor

class CosmoGP:
    def __init__(self):
        self.gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            normalize_y=True
        )

    def fit_predict(self, x, y):
        # Returns both mean and uncertainty
        mean, std = self.gp.fit(x).predict(y, return_std=True)
        return mean, std
```

The key difference is that a GP naturally gives us uncertainty estimates. For every prediction it makes, it tells us both:

1. The expected value (mean)
2. How uncertain it is (variance)

This happens because a GP defines a multivariate Gaussian distribution over function values. The mathematical form is:

```
P(f|X) = N(μ(X), K(X,X))
```

where:

- μ(X) is the mean function
- K(X,X) is the covariance function (kernel)

For estimating P(θ|x), we could use a GP like this:

```python
class CosmoParameterGP:
    def __init__(self):
        # One GP for each cosmological parameter
        self.H0_gp = GaussianProcessRegressor()
        self.omega_m_gp = GaussianProcessRegressor()

    def predict_parameters(self, cmb_spectrum):
        # Get distributions for each parameter
        H0_mean, H0_std = self.H0_gp.predict(cmb_spectrum, return_std=True)
        omega_mean, omega_std = self.omega_m_gp.predict(cmb_spectrum, return_std=True)

        # Combined distribution P(θ|x)
        return MultivariateNormal(
            mean=[H0_mean, omega_mean],
            covariance=[[H0_std**2, 0], [0, omega_std**2]]
        )
```

Now, here's where MC Dropout becomes interesting. When we use dropout in a neural network and keep it on during testing, we can prove mathematically that this approximates a Gaussian Process! This means:

1. We get the flexibility and scalability of neural networks
2. But we also get uncertainty estimates like a GP would give us
3. All by just keeping dropout switched on

The limitations are:

1. It's only an approximation to a GP
2. The uncertainty estimates might not be as well-calibrated
3. The type of GP it approximates depends on the network architecture

This is why having multiple approaches (SWAG, DELFI, MC Dropout) is valuable - they each capture different aspects of uncertainty in our parameter estimates.
