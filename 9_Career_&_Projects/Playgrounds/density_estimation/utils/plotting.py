# utils/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def configure_plotting_style():
    """Configure the global plotting style for consistency."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_density_comparison(data, estimated_density, title, ax=None):
    """
    Plot the comparison between data histogram and estimated density.
    
    Args:
        data (numpy.ndarray): Original data points
        estimated_density (callable): Function that returns density estimates
        title (str): Plot title
        ax (matplotlib.axes.Axes, optional): Axes to plot on
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of data
    ax.hist(data, bins='auto', density=True, alpha=0.3, 
            color='gray', label='Data')
    
    # Plot estimated density
    x_plot = np.linspace(min(data) - 1, max(data) + 1, 200)
    ax.plot(x_plot, estimated_density(x_plot), 'r-', 
            label='Estimated Density')
    
    ax.set_title(title)
    ax.legend()
    return ax

def plot_flow_transformation(base_samples, transformed_samples, 
                             base_density=None, title=None):
    """
    Plot the flow transformation from base to target distribution.
    
    Args:
        base_samples (numpy.ndarray): Samples from base distribution
        transformed_samples (numpy.ndarray): Transformed samples
        base_density (callable, optional): Base distribution density function
        title (str, optional): Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot base distribution
    ax1.hist(base_samples, bins='auto', density=True, alpha=0.3,
             color='gray', label='Samples')
    if base_density is not None:
        x_plot = np.linspace(min(base_samples), max(base_samples), 200)
        ax1.plot(x_plot, base_density(x_plot), 'r-', 
                 label='Base Density')
    ax1.set_title("Base Distribution")
    ax1.legend()
    
    # Plot transformed distribution
    ax2.hist(transformed_samples, bins='auto', density=True, alpha=0.3,
             color='gray', label='Transformed')
    ax2.set_title("Transformed Distribution")
    ax2.legend()
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_spline_flow(x, y, knots):
    """
    Plot the spline transformation with knot points.
    
    Args:
        x (numpy.ndarray): Input points
        y (numpy.ndarray): Transformed points
        knots (numpy.ndarray): Knot points of the spline, with shape (n_knots, 2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the spline transformation function
    ax1.plot(x, y, 'b-', label='Transformation')
    ax1.scatter(knots[:, 0], knots[:, 1], color='red', 
                label='Knot Points')
    # Optionally, plot the identity line for reference
    ax1.plot([min(x), max(x)], [min(x), max(x)], 'k--', label='Identity')
    ax1.set_title("Spline Transformation")
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")
    ax1.legend()
    
    # Plot the derivative of the transformation (slope)
    dydx = np.gradient(y, x)
    ax2.plot(x, dydx, 'g-', label='Derivative')
    ax2.set_title("Derivative of the Transformation")
    ax2.set_xlabel("Input")
    ax2.set_ylabel("dy/dx")
    ax2.legend()
    
    plt.tight_layout()
    return fig
