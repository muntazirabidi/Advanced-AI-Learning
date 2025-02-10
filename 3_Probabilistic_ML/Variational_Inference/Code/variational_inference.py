import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
from scipy.optimize import minimize
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# === Modified data generation: Three-Component Mixture ===
def generate_mixture_data(n_samples=200):
    """
    Generate synthetic data from a mixture of three Gaussians.
    Returns both the samples and the true parameters.
    """
    # True parameters for three Gaussians
    # For example, we choose:
    # - Mixture weights (should sum to 1)
    pi = [0.3, 0.4, 0.3]  
    # - Means for each component
    mu = [-2.0, 0.0, 2.0]  
    # - Standard deviations for each component
    sigma = [0.5, 1.0, 1.5]
    
    # Generate samples for each component
    samples = []
    for weight, mean, std in zip(pi, mu, sigma):
        n_comp = int(weight * n_samples)
        samples.append(np.random.normal(mean, std, n_comp))
    
    # In case rounding leads to fewer than n_samples, add extra samples from the last component.
    data = np.concatenate(samples)
    if data.shape[0] < n_samples:
        extra = np.random.normal(mu[-1], sigma[-1], n_samples - data.shape[0])
        data = np.concatenate([data, extra])
    
    np.random.shuffle(data)
    
    true_params = {
        'pi': pi,      # list of weights
        'mu': mu,      # list of means
        'sigma': sigma # list of standard deviations
    }
    
    return data, true_params

# === Variational Distribution (still 2 components) ===
class VariationalDistribution:
    def __init__(self, n_components=3):
        """
        Initialize variational parameters.
        Here the variational distribution is a mixture of Gaussians.
        """
        self.n_components = n_components
        # Random initialization (avoid symmetry)
        self.mu = np.random.randn(n_components)  # means
        self.sigma = np.abs(np.random.randn(n_components)) + 0.1  # positive standard deviations
        weights = np.random.rand(n_components)
        self.phi = weights / np.sum(weights)  # mixture weights (summing to 1)
        
    def get_params(self):
        """Return all parameters as a flat array for optimization."""
        return np.concatenate([self.mu, np.log(self.sigma), self.phi[:-1]])
    
    def set_params(self, params):
        """
        Set parameters from a flat array.
        We transform parameters so that sigma stays positive and phi sums to one.
        """
        n = self.n_components
        self.mu = params[:n]
        self.sigma = np.exp(params[n:2*n])  # exponentiate to enforce positivity
        # For mixture weights, store K-1 numbers and recover the Kth via softmax.
        phi_n_minus_1 = params[2*n:]
        phi_full = np.append(phi_n_minus_1, 0)  # append zero for numerical stability
        self.phi = special.softmax(phi_full)
        
    def log_prob(self, x):
        """
        Compute log probability under the variational distribution.
        For each component k, compute log(phi[k] * N(x; mu[k], sigma[k]^2)),
        and then combine them using the logsumexp trick.
        """
        log_probs = []
        for k in range(self.n_components):
            log_probs.append(
                np.log(self.phi[k]) + stats.norm.logpdf(x, self.mu[k], self.sigma[k])
            )
        return special.logsumexp(np.array(log_probs), axis=0)

# === ELBO (Evidence Lower Bound) Computation ===
def elbo(params, q_dist, data):
    """
    Compute the negative ELBO.
    We minimize the negative ELBO to maximize the actual ELBO.
    
    Parameters:
      params : flat array of variational parameters.
      q_dist : instance of VariationalDistribution.
      data : observed data points.
    
    Returns:
      Negative ELBO value.
    """
    try:
        q_dist.set_params(params)
        
        # Expected log likelihood (data likelihood under q)
        expected_ll = np.sum(q_dist.log_prob(data))
        
        # KL divergence from the prior.
        # For the means and variances, assume a standard normal prior N(0,1).
        kl_divergence = 0
        for k in range(q_dist.n_components):
            kl_divergence += 0.5 * (q_dist.mu[k]**2 + q_dist.sigma[k]**2 - 1 - np.log(q_dist.sigma[k]**2))
            # For the mixture weights, assume a uniform Dirichlet prior (equivalent to Dirichlet(1,...,1)).
            kl_divergence += q_dist.phi[k] * np.log(q_dist.phi[k] + 1e-10)
        
        return -(expected_ll - kl_divergence)
    except Exception as e:
        print(f"Error in ELBO computation: {str(e)}")
        return np.inf

# === Fit the Variational Distribution ===
def fit_variational_distribution(data, n_iter=1000):
    """
    Fit the variational distribution to the data using ELBO optimization.
    
    Parameters:
      data : observed data points.
      n_iter : maximum number of iterations for optimization.
      
    Returns:
      Fitted VariationalDistribution instance.
    """
    q_dist = VariationalDistribution()  # note: still 2 components
    
    # Get initial parameter values in a flat array
    initial_params = q_dist.get_params()
    
    try:
        result = minimize(
            elbo,
            initial_params,
            args=(q_dist, data),
            method='L-BFGS-B',
            options={'maxiter': n_iter}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        # Set the optimal parameters in our variational distribution
        q_dist.set_params(result.x)
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return None
    
    return q_dist

# === Plotting Function ===
def plot_results(data, q_dist, true_params=None):
    """
    Plot the data histogram, the variational approximation, and the true distribution.
    
    Parameters:
      data : observed data points.
      q_dist : fitted variational distribution.
      true_params : dictionary of true parameters (if available).
    """
    plt.figure(figsize=(12, 6))
    
    # Histogram of the data
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    
    # Plot variational approximation (from the VI model with 2 components)
    x = np.linspace(min(data)-2, max(data)+2, 1000)
    q_density = np.exp([q_dist.log_prob(xi) for xi in x])
    plt.plot(x, q_density, 'r-', label='Variational Approximation (2 components)')
    
    # Plot true density if provided (this time with 3 components)
    if true_params is not None:
        # Check if true_params use list keys ('mu', 'pi', 'sigma') for the 3-component version.
        if 'mu' in true_params:
            true_density = np.zeros_like(x)
            for pi_val, mu_val, sigma_val in zip(true_params['pi'], true_params['mu'], true_params['sigma']):
                true_density += pi_val * stats.norm.pdf(x, mu_val, sigma_val)
        else:
            # (Fallback to previous two-component style if necessary)
            true_density = (
                true_params['pi'] * stats.norm.pdf(x, true_params['mu1'], true_params['sigma1']) +
                (1 - true_params['pi']) * stats.norm.pdf(x, true_params['mu2'], true_params['sigma2'])
            )
        plt.plot(x, true_density, 'g--', label='True Distribution (3 components)')
    
    plt.title('Variational Inference: True vs. Approximate Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# === Run the Example ===
if __name__ == "__main__":
    try:
        # Generate data from a mixture of three Gaussians
        data, true_params = generate_mixture_data(n_samples=1000)
        
        # Fit a variational distribution with 2 components (model mismatch)
        q_dist = fit_variational_distribution(data)
        
        if q_dist is not None:
            # Plot the results: data histogram, VI approximation, and true 3-component density
            plot_results(data, q_dist, true_params)
            
            # Print learned variational parameters
            print("\nLearned Variational Parameters (2 components):")
            print(f"Means: {q_dist.mu}")
            print(f"Standard Deviations: {q_dist.sigma}")
            print(f"Mixture Weights: {q_dist.phi}")
            
            print("\nTrue Parameters (3 components):")
            print(f"Means: {true_params['mu']}")
            print(f"Standard Deviations: {true_params['sigma']}")
            print(f"Mixture Weights: {true_params['pi']}")
        
    except Exception as e:
        print(f"Error running the example: {str(e)}")
