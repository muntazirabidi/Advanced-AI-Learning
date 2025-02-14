import numpy as np
from scipy.stats import norm, multivariate_normal

def generate_synthetic_data(dist_type, n_samples, seed=None):
    """
    Generate synthetic data from various distributions.
    
    Parameters:
        dist_type: str
            Type of distribution to generate from
        n_samples: int
            Number of samples to generate
        seed: int, optional
            Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Generated samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    if dist_type == "Normal":
        return np.random.normal(0, 1, n_samples)
    
    elif dist_type == "Mixture of Gaussians":
        # Generate mixture of two Gaussians
        n1 = n_samples // 2
        n2 = n_samples - n1
        data1 = np.random.normal(-2, 0.5, n1)
        data2 = np.random.normal(2, 0.5, n2)
        return np.concatenate([data1, data2])
    
    elif dist_type == "Bimodal":
        # Generate more separated bimodal distribution
        n1 = n_samples // 2
        n2 = n_samples - n1
        data1 = np.random.normal(-3, 0.8, n1)
        data2 = np.random.normal(3, 0.8, n2)
        return np.concatenate([data1, data2])
    
    elif dist_type == "Uniform":
        return np.random.uniform(-3, 3, n_samples)
    
    elif dist_type == "Skewed":
        # Generate log-normal distribution
        return np.random.lognormal(0, 0.5, n_samples)
    
    elif dist_type == "Heavy-tailed":
        # Generate Student's t distribution
        return np.random.standard_t(df=3, size=n_samples)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def generate_2d_data(dist_type, n_samples, seed=None):
    """
    Generate 2D synthetic data for visualization.
    
    Parameters:
        dist_type: str
            Type of distribution to generate from
        n_samples: int
            Number of samples to generate
        seed: int, optional
            Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Generated samples with shape (n_samples, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if dist_type == "Gaussian":
        return np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[1, 0.5], [0.5, 1]],
            size=n_samples
        )
    
    elif dist_type == "Mixture":
        n1 = n_samples // 2
        n2 = n_samples - n1
        
        samples1 = np.random.multivariate_normal(
            mean=[-2, -2],
            cov=[[0.5, 0], [0, 0.5]],
            size=n1
        )
        
        samples2 = np.random.multivariate_normal(
            mean=[2, 2],
            cov=[[0.5, 0], [0, 0.5]],
            size=n2
        )
        
        return np.vstack([samples1, samples2])
    
    elif dist_type == "Ring":
        # Generate points in a circular pattern
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        r = np.random.normal(2, 0.2, n_samples)  # radius with noise
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])
    
    elif dist_type == "Spiral":
        # Generate spiral pattern
        theta = np.random.uniform(0, 4*np.pi, n_samples)
        r = theta / (4*np.pi) * 3
        noise = np.random.normal(0, 0.1, n_samples)
        x = (r + noise) * np.cos(theta)
        y = (r + noise) * np.sin(theta)
        return np.column_stack([x, y])
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def generate_flow_data(flow_type, n_samples, seed=None):
    """
    Generate data for testing and visualizing normalizing flows.
    
    Parameters:
        flow_type: str
            Type of flow transformation to simulate
        n_samples: int
            Number of samples to generate
        seed: int, optional
            Random seed for reproducibility
    
    Returns:
        tuple: (base_samples, transformed_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate base distribution (usually standard normal)
    base_samples = np.random.normal(0, 1, n_samples)
    
    if flow_type == "Affine":
        # Simple scale and shift
        scale = 2.0
        shift = 1.0
        transformed_samples = scale * base_samples + shift
        
    elif flow_type == "Quadratic":
        # Quadratic transformation
        transformed_samples = np.sign(base_samples) * base_samples**2
        
    elif flow_type == "Sine":
        # Periodic transformation
        transformed_samples = np.sin(2 * base_samples)
        
    elif flow_type == "Spline":
        # Simulate a simple monotonic spline transformation
        transformed_samples = np.tanh(base_samples) + 0.5 * base_samples
        
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
    
    return base_samples, transformed_samples

def generate_gmm_data(n_components, n_samples, random_state=None):
    """
    Generate data from a Gaussian Mixture Model with random parameters.
    
    Parameters:
        n_components: int
            Number of mixture components
        n_samples: int
            Number of samples to generate
        random_state: int, optional
            Random seed for reproducibility
            
    Returns:
        tuple: (samples, true_parameters)
            - samples: Generated data points
            - true_parameters: Dictionary containing the true GMM parameters
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random means between -5 and 5
    means = np.random.uniform(-5, 5, n_components)
    
    # Generate random standard deviations between 0.5 and 1.5
    stds = np.random.uniform(0.5, 1.5, n_components)
    
    # Generate random weights that sum to 1
    weights = np.random.dirichlet(np.ones(n_components))
    
    # Generate samples
    samples = []
    for i in range(n_components):
        n_comp_samples = np.random.binomial(n_samples, weights[i])
        comp_samples = np.random.normal(means[i], stds[i], n_comp_samples)
        samples.append(comp_samples)
    
    samples = np.concatenate(samples)
    np.random.shuffle(samples)
    
    # Ensure exactly n_samples
    if len(samples) > n_samples:
        samples = samples[:n_samples]
    elif len(samples) < n_samples:
        extra_samples = np.random.choice(samples, n_samples - len(samples))
        samples = np.concatenate([samples, extra_samples])
    
    true_parameters = {
        'means': means,
        'stds': stds,
        'weights': weights
    }
    
    return samples, true_parameters