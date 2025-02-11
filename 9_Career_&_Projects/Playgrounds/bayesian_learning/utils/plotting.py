import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_theme_plot(figsize=(10, 6)):
    """Create a themed matplotlib figure."""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#4C72B0', '#55A868', '#C44E52'])
    
    return fig, ax