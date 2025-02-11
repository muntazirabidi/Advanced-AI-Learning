# utils/distributions.py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .plotting import create_theme_plot

def plot_beta_distribution(prior_alpha, prior_beta, post_alpha=None, post_beta=None):
    """
    Plot Beta distribution with optional comparison between prior and posterior.
    
    Parameters:
    -----------
    prior_alpha : float
        Alpha parameter for the prior Beta distribution
    prior_beta : float
        Beta parameter for the prior Beta distribution
    post_alpha : float, optional
        Alpha parameter for the posterior Beta distribution
    post_beta : float, optional
        Beta parameter for the posterior Beta distribution
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    fig, ax = create_theme_plot()
    
    # Generate x values for plotting
    x = np.linspace(0, 1, 1000)
    
    # Plot prior distribution
    y_prior = stats.beta.pdf(x, prior_alpha, prior_beta)
    if post_alpha is None and post_beta is None:
        # If no posterior, just plot the single distribution
        label = f'Beta({prior_alpha:.1f}, {prior_beta:.1f})'
    else:
        # If there's a posterior, label this as prior
        label = f'Prior: Beta({prior_alpha:.1f}, {prior_beta:.1f})'
    
    ax.plot(x, y_prior, 'b-', lw=2, label=label)
    ax.fill_between(x, y_prior, alpha=0.3)
    
    # Plot posterior if provided
    if post_alpha is not None and post_beta is not None:
        y_post = stats.beta.pdf(x, post_alpha, post_beta)
        ax.plot(x, y_post, 'r-', lw=2, label=f'Posterior: Beta({post_alpha:.1f}, {post_beta:.1f})')
        ax.fill_between(x, y_post, alpha=0.3, color='red')
    
    ax.set_xlabel('θ')
    ax.set_ylabel('Density')
    ax.set_title('Beta Distribution')
    ax.legend()
    
    return fig

def plot_prior_likelihood_posterior(prior_mean, prior_std, likelihood_mean, likelihood_std):
    """
    Plot prior, likelihood, and resulting posterior distributions.
    
    Parameters:
    -----------
    prior_mean : float
        Mean of the prior normal distribution
    prior_std : float
        Standard deviation of the prior normal distribution
    likelihood_mean : float
        Mean of the likelihood normal distribution
    likelihood_std : float
        Standard deviation of the likelihood normal distribution
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot showing prior, likelihood, and posterior
    """
    fig, ax = create_theme_plot()
    
    # Generate x values for plotting
    x = np.linspace(-10, 10, 1000)
    
    # Calculate distributions
    prior = stats.norm.pdf(x, prior_mean, prior_std)
    likelihood = stats.norm.pdf(x, likelihood_mean, likelihood_std)
    
    # Calculate posterior (unnormalized)
    posterior = prior * likelihood
    posterior = posterior / np.trapz(posterior, x)  # Normalize
    
    # Plot all distributions
    ax.plot(x, prior, 'b-', label='Prior')
    ax.plot(x, likelihood, 'r-', label='Likelihood')
    ax.plot(x, posterior, 'g-', label='Posterior')
    
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    ax.set_title('Bayesian Updating')
    ax.legend()
    
    return fig