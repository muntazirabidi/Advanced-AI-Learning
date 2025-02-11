import streamlit as st
from utils.vi_model import VariationalInference
import torch

def initialize_session_state():
    """Initialize all session state variables if they don't exist"""
    if 'vi_model' not in st.session_state:
        st.session_state.vi_model = VariationalInference()
    if 'y_obs' not in st.session_state:
        st.session_state.y_obs = None
    if 'true_theta' not in st.session_state:
        st.session_state.true_theta = torch.tensor([2.0, -1.0])
    if 'vi_state' not in st.session_state:
        st.session_state.vi_state = None

def render_theoretical_foundations():
    """Render the theoretical foundations section with LaTeX equations"""
    st.header("Theoretical Foundations")
    
    st.markdown("""
    Variational Inference (VI) is a powerful method in Bayesian statistics that approximates complex posterior distributions 
    through optimization. Let's break down the key concepts and mathematics behind it.
    """)

    with st.expander("Bayesian Inference Fundamentals", expanded=True):
        st.markdown(r"""
        **Bayes' Theorem** forms the foundation of Bayesian statistics:
        
        $$p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)}$$
        
        Where:
        - $p(\theta)$: Prior distribution (our beliefs before seeing data)
        - $p(x|\theta)$: Likelihood (probability of data given parameters)
        - $p(x)$: Evidence/marginal likelihood ($\int p(x|\theta)p(\theta)d\theta$)
        - $p(\theta|x)$: Posterior distribution (updated beliefs after seeing data)

        The main computational challenge lies in calculating the evidence $p(x)$, which often requires intractable integrals.
        """)

    with st.expander("Variational Inference Core Concepts", expanded=True):
        st.markdown(r"""
        **Key Idea:** Approximate the true posterior $p(\theta|x)$ with a simpler distribution $q_\phi(\theta)$ from a variational family $Q$.

        **KL Divergence Minimization:**
        We minimize the Kullback-Leibler divergence:
        
        $$\min_{q_\phi \in Q} KL(q_\phi(\theta) \| p(\theta|x))$$
        
        **ELBO Derivation:**
        Through algebraic manipulation, we obtain the Evidence Lower Bound (ELBO):
        
        $$
        \begin{aligned}
        KL(q||p) &= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\theta,x)] + \log p(x) \\
        \log p(x) &= KL(q||p) + \underbrace{\mathbb{E}_q[\log p(\theta,x)] - \mathbb{E}_q[\log q(\theta)]}_{\text{ELBO}}
        \end{aligned}
        $$
        
        Since $\log p(x)$ is constant, maximizing ELBO $\mathcal{L}(\phi)$ is equivalent to minimizing KL divergence.
        """)

    with st.expander("ELBO Components and Interpretation"):
        st.markdown(r"""
        The ELBO can be decomposed into two interpretable terms:
        
        $$\mathcal{L}(\phi) = \underbrace{\mathbb{E}_{q_\phi}[\log p(x|\theta)]}_{\text{Expected log-likelihood}} - \underbrace{KL(q_\phi(\theta)||p(\theta))}_{\text{Regularization}}$$
        
        1. **Data Term:** Encourages parameters that explain observed data
        2. **KL Term:** Regularizes towards prior distribution

        This decomposition reveals VI as a balance between data fitting and prior adherence.
        """)

    with st.expander("Variational Family & Mean-Field Approximation"):
        st.markdown(r"""
        **Choice of Variational Family:**
        - **Gaussian Family:** $q_\phi(\theta) = \mathcal{N}(\theta|\mu_\phi, \Sigma_\phi)$
        - **Mean-Field Approximation:** $q(\theta) = \prod_{i=1}^d q_i(\theta_i)$ (factorized)
        
        **Trade-off:** Richer families improve approximation but increase computational complexity
        
        **Gaussian Parameterization:**
        In our implementation:
        $$\phi = \{\mu_\phi, L_\phi\}$$
        where $L_\phi$ is the lower Cholesky factor of the covariance matrix $\Sigma_\phi = L_\phi L_\phi^\top$
        """)

    with st.expander("Optimization & Reparameterization Trick"):
        st.markdown(r"""
        **Gradient-Based Optimization:**
        $$\phi^* = \arg\max_\phi \mathcal{L}(\phi)$$
        
        **Reparameterization Trick:**
        For $\epsilon \sim \mathcal{N}(0,I)$:
        $$\theta = \mu_\phi + L_\phi\epsilon$$
        
        This enables low-variance gradient estimation:
        $$\nabla_\phi\mathbb{E}_{q_\phi}[f(\theta)] = \mathbb{E}_{\epsilon}[\nabla_\phi f(\mu_\phi + L_\phi\epsilon)]$$
        
        **KL Analytic Computation:**
        For Gaussian $q_\phi$ and prior $p(\theta) = \mathcal{N}(0,I)$:
        $$
        KL(q||p) = \frac{1}{2}\left[\text{tr}(\Sigma_\phi) + \mu_\phi^\top\mu_\phi - d - \log\det\Sigma_\phi\right]
        $$
        """)

    with st.expander("Algorithm Summary"):
        st.markdown(r"""
        **Variational Inference Algorithm:**
        1. Initialize variational parameters $\phi^{(0)} = (\mu^{(0)}, L^{(0)})$
        2. For $t=1$ to $T$:
            - Sample $\epsilon \sim \mathcal{N}(0,I)$
            - Compute $\theta = \mu^{(t)} + L^{(t)}\epsilon$
            - Compute gradient estimate:
              $$\nabla_\phi\mathcal{L} \approx \nabla_\phi[\log p(x|\theta) + \log p(\theta) - \log q_\phi(\theta)]$$
            - Update parameters:
              $$\phi^{(t+1)} \leftarrow \phi^{(t)} + \eta \nabla_\phi\mathcal{L}$$
        3. Return optimized $\phi^*$
        
        **Key Advantages:**
        - Converts integration problems to optimization
        - Scales well to large datasets
        - Provides full posterior approximation
        """)

    st.markdown("""
    ### Further Reading:
    - [Blei et al. (2017) Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)
    - [Bishop (2006) Pattern Recognition and Machine Learning, Chapter 10](https://www.springer.com/gp/book/9780387310732)
    """)

def main():
    st.set_page_config(
        page_title="Variational Inference Explorer",
        page_icon="📊",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    st.title("Variational Inference Interactive Explorer")

    # Add tabs for different sections
    tab1, tab2 = st.tabs(["Overview", "Theory"])

    with tab1:
        st.markdown("""
        Welcome to the Variational Inference Explorer! This interactive dashboard helps you understand
        variational inference through visualization and experimentation.

        ### Navigation Guide:
        1. **Data Generation**: Explore how data is generated and set true parameters
        2. **Inference Process**: Watch the optimization process in real-time
        3. **Parameter Space**: Visualize the parameter posterior distribution
        4. **Uncertainty Analysis**: Analyze prediction uncertainties
        5. **Advanced Visualizations**: Explore additional insights

        ### Getting Started:
        1. Begin with the 'Data Generation' page to create your dataset
        2. Move to 'Inference Process' to run the variational inference
        3. Explore the results in the visualization pages

        Use the sidebar to navigate between pages and adjust parameters.
        """)

    with tab2:
        render_theoretical_foundations()

    # Display current state information
    with st.sidebar:
        st.subheader("Session Information")
        if st.session_state.y_obs is not None:
            st.success("✅ Data Generated")
        else:
            st.warning("⚠️ No Data Generated")
        
        if st.session_state.vi_state is not None:
            st.success("✅ Inference Complete")
        else:
            st.warning("⚠️ Inference Not Run")

        if st.button("Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.success("Reset complete!")

if __name__ == "__main__":
    main()