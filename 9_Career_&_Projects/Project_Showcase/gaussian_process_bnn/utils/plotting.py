# utils/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable

def plot_comparative_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    bnn_results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    gp_results: Tuple[np.ndarray, np.ndarray],
    true_function: Optional[Callable] = None
) -> plt.Figure:
    """
    Create comparison plots for BNN and GP predictions with uncertainty.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training target data
    X_test : np.ndarray
        Test input data
    bnn_results : tuple
        Contains (mean, total_std) for BNN
    gp_results : tuple
        Contains (mean, std) for GP
    true_function : callable, optional
        True function for comparison
        
    Returns:
    --------
    plt.Figure
        The complete figure with subplots
    """
    # Unpack results
    bnn_mean, bnn_std = bnn_results[0], bnn_results[1]  # We only need mean and total std
    gp_mean, gp_std = gp_results
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot BNN predictions
    ax1.scatter(X_train, y_train, c='black', label='Training Data', alpha=0.6)
    ax1.plot(X_test, bnn_mean, 'b-', label='BNN Mean', linewidth=2)
    ax1.fill_between(
        X_test.flatten(),
        bnn_mean.flatten() - 2*bnn_std.flatten(),
        bnn_mean.flatten() + 2*bnn_std.flatten(),
        alpha=0.3, color='blue', label='BNN 95% CI'
    )
    if true_function is not None:
        ax1.plot(X_test, true_function(X_test), 'r--', label='True Function', linewidth=2)
    ax1.set_title('Bayesian Neural Network Predictions', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot GP predictions
    ax2.scatter(X_train, y_train, c='black', label='Training Data', alpha=0.6)
    ax2.plot(X_test, gp_mean, 'g-', label='GP Mean', linewidth=2)
    ax2.fill_between(
        X_test.flatten(),
        gp_mean - 2*gp_std,
        gp_mean + 2*gp_std,
        alpha=0.3, color='green', label='GP 95% CI'
    )
    if true_function is not None:
        ax2.plot(X_test, true_function(X_test), 'r--', label='True Function', linewidth=2)
    ax2.set_title('Gaussian Process Predictions', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig

def plot_training_loss(losses: list) -> plt.Figure:
    """
    Plot the training loss over epochs.
    
    Parameters:
    -----------
    losses : list
        List of loss values during training
        
    Returns:
    --------
    plt.Figure
        The figure containing the loss plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, 'b-', label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig