import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import create_theme_plot
from utils.distributions import plot_prior_likelihood_posterior

def prior_likelihood_page():
    st.title("Prior & Likelihood: The Building Blocks")
    
    st.markdown("""
    In Bayesian inference, we combine our prior beliefs (prior) with observed
    data (likelihood) to form updated beliefs (posterior).
    """)
    
    # Interactive demonstration
    st.subheader("Interactive Prior-Likelihood Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prior_mean = st.slider("Prior Mean", -5.0, 5.0, 0.0, 0.1)
        prior_std = st.slider("Prior Standard Deviation", 0.1, 5.0, 1.0, 0.1)
        likelihood_mean = st.slider("Likelihood Mean", -5.0, 5.0, 2.0, 0.1)
        likelihood_std = st.slider("Likelihood Standard Deviation", 0.1, 5.0, 0.5, 0.1)
    
    with col2:
        fig = plot_prior_likelihood_posterior(
            prior_mean, prior_std,
            likelihood_mean, likelihood_std
        )
        st.pyplot(fig)
        plt.close()

if __name__ == "__main__":
    prior_likelihood_page()