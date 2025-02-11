import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import create_theme_plot
import seaborn as sns

def metropolis_hastings(target_dist, proposal_width, n_samples, initial_state):
    """
    Implement the Metropolis-Hastings algorithm for MCMC sampling.
    
    Parameters:
    - target_dist: Function that returns the unnormalized probability density
    - proposal_width: Standard deviation for the Gaussian proposal distribution
    - n_samples: Number of samples to generate
    - initial_state: Starting point for the chain
    """
    samples = np.zeros(n_samples)
    current_state = initial_state
    
    for i in range(n_samples):
        # Propose new state
        proposal = current_state + np.random.normal(0, proposal_width)
        
        # Calculate acceptance ratio
        ratio = target_dist(proposal) / target_dist(current_state)
        
        # Accept or reject
        if np.random.random() < ratio:
            current_state = proposal
            
        samples[i] = current_state
    
    return samples

def mcmc_sampling_page():
    st.title("MCMC Sampling: When Conjugacy Isn't Enough")
    
    st.markdown("""
    Markov Chain Monte Carlo (MCMC) methods allow us to sample from complex
    posterior distributions that don't have nice analytical forms. We'll explore
    the Metropolis-Hastings algorithm, one of the most fundamental MCMC methods.
    """)
    
    st.subheader("Interactive Metropolis-Hastings Demonstration")
    
    # Define a mixture of Gaussians as our target distribution
    def target_distribution(x):
        return (0.3 * np.exp(-0.2 * (x - 2)**2) +
                0.7 * np.exp(-0.2 * (x + 2)**2))
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of Samples", 100, 10000, 2000)
        proposal_width = st.slider("Proposal Width", 0.1, 5.0, 1.0, 0.1,
                                 help="Controls how far the chain can jump in one step")
        
    # Run MCMC
    samples = metropolis_hastings(
        target_distribution,
        proposal_width,
        n_samples,
        initial_state=0.0
    )
    
    with col2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot the true distribution
        x = np.linspace(-6, 6, 1000)
        ax1.plot(x, target_distribution(x), 'r-', label='Target Distribution')
        sns.histplot(samples, stat='density', ax=ax1, alpha=0.5, label='MCMC Samples')
        ax1.set_title('Target Distribution vs MCMC Samples')
        ax1.legend()
        
        # Plot the trace
        ax2.plot(samples, alpha=0.5)
        ax2.set_title('MCMC Trace Plot')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("""
    ### Understanding MCMC Sampling
    
    The demonstration above shows how MCMC can approximate a complex target distribution:
    
    1. The top plot shows both the true target distribution (red line) and a histogram
       of our MCMC samples. As we collect more samples, the histogram should better
       approximate the target.
       
    2. The bottom plot is a trace plot, showing where our chain has explored over time.
       Good mixing (the chain moving freely across the parameter space) is crucial for
       accurate sampling.
       
    The proposal width parameter controls how far the chain attempts to jump in each step:
    - Too small: Chain moves very slowly, taking long to explore the space
    - Too large: Chain often proposes unlikely values, leading to many rejections
    - Just right: Chain efficiently explores the target distribution
    
    Try adjusting the parameters to see how they affect the sampling efficiency!
    """)

if __name__ == "__main__":
    mcmc_sampling_page()