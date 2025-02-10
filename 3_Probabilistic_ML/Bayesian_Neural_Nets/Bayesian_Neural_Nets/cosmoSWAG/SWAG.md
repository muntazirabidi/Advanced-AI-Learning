# Understanding Stochastic Weight Averaging (SWA) and SWAG

Neural networks are powerful function approximators, but their optimization landscape is complex and finding good solutions can be challenging. This document explores two related techniques: Stochastic Weight Averaging (SWA) and its Bayesian extension, SWAG (SWA-Gaussian).

## Traditional Neural Network Training

In traditional neural network training, we seek to minimize a loss function $L(w)$ where $w$ represents the network weights:

$w^* = \arg\min L(w)$

The optimization typically uses stochastic gradient descent (SGD) or variants:

$w_{t+1} = w_t - \eta \nabla L(w_t)$

where $\eta$ is the learning rate.

```python
def train_step(model, optimizer, data, labels):
    # Standard training loop
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    return model.get_weights()
```

## Stochastic Weight Averaging (SWA)

### Motivation

SGD's trajectory often oscillates between different regions of the loss landscape. SWA leverages this behavior by averaging multiple points along the trajectory.

### Algorithm

1. Train the model normally for a warmup period
2. After warmup, collect weights periodically
3. Average the collected weights

Mathematically, given collected weights $\{w_1, ..., w_C\}$, the SWA solution is:

$w_{SWA} = \frac{1}{C}\sum_{c=1}^C w_c$

```python
class SWA:
    def __init__(self, model, warmup=1000):
        self.model = model
        self.warmup = warmup
        self.collected_weights = []

    def update(self, epoch):
        if epoch > self.warmup:
            current_weights = self.model.get_weights()
            self.collected_weights.append(current_weights)

    def get_swa_weights(self):
        return np.mean(self.collected_weights, axis=0)
```

### Example

```python
# Training with SWA
swa = SWA(model, warmup=1000)
for epoch in range(num_epochs):
    for batch in dataloader:
        train_step(model, optimizer, batch)

    # Collect weights for SWA
    if epoch > warmup and epoch % collection_freq == 0:
        swa.update(epoch)

# Set final SWA weights
model.set_weights(swa.get_swa_weights())
```

## SWAG (Stochastic Weight Averaging - Gaussian)

### Motivation

While SWA provides a point estimate, SWAG extends this to capture uncertainty by modeling the weights as coming from a Gaussian distribution.

### Mathematical Formulation

SWAG models the weight distribution as:

$w \sim \mathcal{N}(\mu, \Sigma)$

where:

- $\mu$ is the SWA mean
- $\Sigma$ is approximated using both diagonal and low-rank components

### Components

1. **Mean** ($\mu$): Same as SWA
2. **Diagonal Variance** ($\Sigma_{diag}$):
   $\Sigma_{diag} = \frac{1}{C-1}\sum_{c=1}^C (w_c - \mu)^2$
3. **Low-rank Covariance** ($\Sigma_{low-rank}$):
   Using deviation matrix $D$ where each column is $(w_c - \mu)$
   $\Sigma_{low-rank} = \frac{1}{2(C-1)}DD^T$

### Implementation

```python
class SWAG:
    def __init__(self, model, K=20):
        self.model = model
        self.K = K  # Number of deviation vectors to store
        self.mean = None
        self.sq_mean = None
        self.deviations = []

    def update(self, weights):
        # Update running averages
        if self.mean is None:
            self.mean = weights
            self.sq_mean = weights**2
        else:
            n = len(self.deviations) + 1
            self.mean = (self.mean * (n-1) + weights) / n
            self.sq_mean = (self.sq_mean * (n-1) + weights**2) / n

        # Store deviations for low-rank approximation
        if len(self.deviations) >= self.K:
            self.deviations.pop(0)
        self.deviations.append(weights - self.mean)

    def sample_weights(self, scale=1.0):
        # Diagonal variance
        var = self.sq_mean - self.mean**2

        # Sample from distribution
        z1 = np.random.normal(0, 1, size=self.mean.shape)
        z2 = np.random.normal(0, 1, size=(self.K, 1))

        D = np.stack(self.deviations)
        sample = (self.mean +
                 scale * np.sqrt(var) * z1 +
                 scale * np.dot(D.T, z2).squeeze() / np.sqrt(2*(self.K-1)))

        return sample
```

### Making Predictions with Uncertainty

```python
def predict_with_uncertainty(swag_model, input_data, num_samples=30):
    predictions = []

    for _ in range(num_samples):
        # Sample weights from SWAG distribution
        weights = swag_model.sample_weights()
        model.set_weights(weights)

        # Make prediction
        pred = model(input_data)
        predictions.append(pred)

    # Calculate mean and variance
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0)

    return mean_pred, var_pred
```

## Practical Considerations

1. **Warmup Period**: Allow the model to reach a good region of the loss landscape before starting collection.

2. **Collection Frequency**: Balance between computational cost and capturing trajectory variation.

3. **Low-rank Approximation**: The number of stored deviations (K) trades off between memory usage and covariance approximation quality.

4. **Calibration**: SWAG uncertainty estimates might need scaling for proper calibration.

## Example Output

For a classification task, SWAG might give:

```python
# Mean predictions with uncertainties
predictions = {
    'class_1': 0.81 ± 0.03,  # 81% confidence ± 3%
    'class_2': 0.12 ± 0.02,  # 12% confidence ± 2%
    'class_3': 0.07 ± 0.01   #  7% confidence ± 1%
}
```
