# Home.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import create_theme_plot
from utils.distributions import plot_beta_distribution

def main():
    st.set_page_config(
        page_title="Learn Bayesian Inference",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📚 Interactive Bayesian Inference Learning")
    
    st.markdown("""
    Welcome to your journey into Bayesian Statistics! This interactive application
    will help you understand the fundamental concepts of Bayesian inference through
    visual demonstrations and hands-on examples.
    
    ### What you'll learn:
    1. Prior and Likelihood
    2. Conjugate Priors
    3. MCMC Sampling
    """)

    # Theoretical Foundation Section
    st.header("📖 Theoretical Foundations of Bayesian Inference")
    
    st.markdown("""
    ### The Heart of Bayesian Inference
    
    Bayesian inference represents a powerful framework for updating our beliefs based on evidence. At its core lies Bayes' Theorem:

    $$ P(θ|D) = \\frac{P(D|θ)P(θ)}{P(D)} $$

    Where:
    - $P(θ|D)$ is the **posterior probability** - our updated belief about parameter θ after seeing data D
    - $P(D|θ)$ is the **likelihood** - the probability of observing data D given parameter θ
    - $P(θ)$ is the **prior probability** - our initial belief about parameter θ
    - $P(D)$ is the **evidence** - the total probability of observing data D
    
    ### Understanding Each Component

    #### 1. Prior Probability
    The prior probability represents our initial beliefs before seeing any data. It can be:
    - **Informative**: Based on previous studies or expert knowledge
    - **Uninformative**: When we want to express minimal prior knowledge
    - **Conjugate**: Chosen to make posterior calculations mathematically convenient
    
    For example, a Beta distribution is often used as a prior for probability parameters because:
    $$ P(θ) = \\frac{θ^{α-1}(1-θ)^{β-1}}{B(α,β)} $$
    where B(α,β) is the Beta function.

    #### 2. Likelihood Function
    The likelihood represents how probable our observed data is under different parameter values:
    $$ L(θ;D) = P(D|θ) $$
    
    For example, in a coin-flipping experiment with n trials and k successes:
    $$ P(D|θ) = \\binom{n}{k}θ^k(1-θ)^{n-k} $$

    #### 3. Posterior Distribution
    The posterior combines our prior beliefs with the evidence from our data:
    $$ P(θ|D) ∝ P(D|θ)P(θ) $$
    
    In many cases, we can write this as:
    $$ \text{Posterior} ∝ \text{Likelihood} × \text{Prior} $$
    """)

    # Interactive Visualization Section
    st.header("🔄 Interactive Beta Distribution")
    st.markdown("""
    The Beta distribution plays a central role in Bayesian inference because:
    1. It's defined on the interval [0,1], making it perfect for probability parameters
    2. It's conjugate to the Binomial likelihood
    3. Its parameters have intuitive interpretations:
       - α represents the number of "successes" plus 1
       - β represents the number of "failures" plus 1
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        prior_alpha = st.slider("α (Alpha) parameter", 0.1, 10.0, 2.0, 0.1,
                              help="Controls the concentration of probability mass above 0.5")
        prior_beta = st.slider("β (Beta) parameter", 0.1, 10.0, 2.0, 0.1,
                             help="Controls the concentration of probability mass below 0.5")
        
        st.markdown(f"""
        **Current Distribution Properties:**
        - Mean: {prior_alpha/(prior_alpha + prior_beta):.3f}
        - Mode: {(prior_alpha - 1)/(prior_alpha + prior_beta - 2):.3f} (when α,β > 1)
        - Variance: {(prior_alpha * prior_beta)/((prior_alpha + prior_beta)**2 * (prior_alpha + prior_beta + 1)):.3f}
        """)
    
    with col2:
        fig = plot_beta_distribution(prior_alpha, prior_beta)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("""
    ### Understanding Shape Parameters
    
    The shape of the Beta distribution is determined by its two parameters:
    
    - When α = β = 1: Uniform distribution (completely uncertain)
    - When α, β > 1: Unimodal (single peak)
    - When α, β < 1: U-shaped (peaks at 0 and 1)
    - When α = β: Symmetric around 0.5
    - Large α, β: More concentrated distribution
    
    Try adjusting the parameters to see how they affect the distribution's shape!
    """)

    # Navigation Guide
    st.header("🗺️ Navigation Guide")
    st.markdown("""
    Continue your learning journey through the following pages:
    
    1. **Prior & Likelihood**: Understand how these components interact
    2. **Conjugate Priors**: Learn about mathematical convenience in Bayesian updating
    3. **MCMC Sampling**: Explore methods for complex posterior distributions
    
    Each section builds upon the concepts introduced here, providing deeper insights
    and more advanced applications of Bayesian inference.
    """)

if __name__ == "__main__":
    main()