import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import create_theme_plot
from utils.distributions import plot_beta_distribution
from utils.bayesian_utils import calculate_posterior_beta

def conjugate_priors_page():
    st.title("Conjugate Priors: A Mathematical Convenience")
    
    st.markdown("""
    Conjugate priors are special pairs of prior and likelihood distributions that result
    in a posterior distribution of the same family as the prior. This mathematical
    convenience makes Bayesian updating particularly elegant and computationally efficient.
    
    Let's explore one of the most common conjugate pairs: the Beta-Binomial conjugacy.
    """)
    
    st.subheader("Beta-Binomial Conjugate Prior Example")
    st.markdown("""
    Imagine we're trying to estimate the probability of success for a binary outcome.
    The Beta distribution serves as our prior for the probability parameter, and the
    Binomial distribution models our observations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        prior_alpha = st.slider("Prior Alpha", 0.1, 10.0, 2.0, 0.1,
                              help="Shape parameter for prior Beta distribution")
        prior_beta = st.slider("Prior Beta", 0.1, 10.0, 2.0, 0.1,
                             help="Shape parameter for prior Beta distribution")
        
        successes = st.slider("Number of Successes", 0, 100, 30,
                            help="Number of successful trials observed")
        trials = st.slider("Total Number of Trials", successes, 100, 50,
                         help="Total number of trials conducted")
    
    # Calculate posterior parameters
    post_alpha, post_beta = calculate_posterior_beta(prior_alpha, prior_beta, successes, trials)
    
    with col2:
        fig = plot_beta_distribution(prior_alpha, prior_beta, post_alpha, post_beta)
        st.pyplot(fig)
        plt.close()
    
    st.markdown(f"""
    ### Understanding the Update
    Starting with a Beta({prior_alpha:.1f}, {prior_beta:.1f}) prior and observing
    {successes} successes in {trials} trials, our posterior distribution is
    Beta({post_alpha:.1f}, {post_beta:.1f}).
    
    This demonstrates how conjugate priors allow us to:
    1. Start with our prior beliefs (Beta distribution)
    2. Incorporate new evidence (Binomial observations)
    3. Get an updated posterior (new Beta distribution)
    
    The parameters simply add: α_posterior = α_prior + successes, β_posterior = β_prior + failures
    """)

if __name__ == "__main__":
    conjugate_priors_page()