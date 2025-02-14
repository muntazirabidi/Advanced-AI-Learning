# pages/gmm_page.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from utils.plotting import plot_density_comparison
from utils.data_generation import generate_gmm_data

class GMMPage:
    """
    Page component for Gaussian Mixture Models visualization and explanation.
    Provides interactive demonstrations and educational content about GMMs.
    """
    
    def render(self):
        """Render the GMM page content and interactive elements."""
        st.title("Gaussian Mixture Models (GMM)")
        
        # Introduction and core concepts
        self._render_introduction()
        
        # Interactive demonstration
        self._render_interactive_demo()
        
        # Component analysis and technical details
        self._render_component_analysis()
        self._render_advanced_topics()
    
    def _render_introduction(self):
        """Render the introductory section explaining GMM concepts."""
        st.markdown("""
        Gaussian Mixture Models represent complex probability distributions as a weighted 
        sum of simpler Gaussian distributions. They are a powerful and flexible approach 
        to density estimation that can capture multi-modal and asymmetric distributions.
        
        ### Mathematical Formulation
        
        A GMM defines the probability density as:
        
        $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \sigma_k^2)$
        
        where:
        - $K$ is the number of components
        - $\pi_k$ are the mixture weights (summing to 1)
        - $\mu_k$ are the means of each Gaussian
        - $\sigma_k^2$ are the variances
        """)
    
    def _render_interactive_demo(self):
        """Create and render the interactive GMM demonstration."""
        st.subheader("Interactive GMM Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider(
                "Number of Components",
                min_value=1,
                max_value=5,
                value=2
            )
            
            n_points = st.slider(
                "Number of Data Points",
                min_value=100,
                max_value=1000,
                value=300,
                step=100
            )
        
        # Generate data and fit GMM
        data, true_params = generate_gmm_data(
            n_components=n_components,
            n_samples=n_points,
            random_state=42
        )
        
        gmm = self._fit_gmm(data, n_components)
        self._plot_gmm_demonstration(data, gmm, true_params)
    
    def _fit_gmm(self, data, n_components):
        """Fit a Gaussian Mixture Model to the data."""
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=42
        )
        gmm.fit(data.reshape(-1, 1))
        return gmm
    
    def _plot_gmm_demonstration(self, data, gmm, true_params):
        """Create visualizations demonstrating GMM behavior."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Setup plot range
        x_plot = np.linspace(min(data) - 1, max(data) + 1, 200)
        
        # First plot: overall density
        plot_density_comparison(
            data,
            lambda x: np.exp(gmm.score_samples(x.reshape(-1, 1))),
            "GMM Density Estimate",
            ax=ax1
        )
        
        # Second plot: individual components
        ax2.hist(data, bins='auto', density=True, alpha=0.3, 
                color='gray', label='Data')
        
        # Plot individual components
        for i in range(gmm.n_components):
            component_density = gmm.weights_[i] * norm.pdf(
                x_plot,
                gmm.means_[i][0],
                np.sqrt(gmm.covariances_[i][0])
            )
            ax2.plot(x_plot, component_density, '--', 
                    label=f'Component {i+1}')
        
        # Plot overall density
        gmm_density = np.exp(gmm.score_samples(x_plot.reshape(-1, 1)))
        ax2.plot(x_plot, gmm_density, 'r-', label='Full GMM')
        ax2.set_title("Individual Components")
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        self._create_parameter_table(gmm, true_params)
    
    def _create_parameter_table(self, gmm, true_params):
        """Create and display a table comparing true and estimated parameters."""
        st.subheader("Component Analysis")
        
        component_data = []
        for i in range(gmm.n_components):
            component_data.append({
                "Component": f"Component {i+1}",
                "Weight (π)": f"{gmm.weights_[i]:.3f}",
                "Mean (μ)": f"{gmm.means_[i][0]:.3f}",
                "Std Dev (σ)": f"{np.sqrt(gmm.covariances_[i][0]):.3f}",
                "True Mean": f"{true_params['means'][i]:.3f}",
                "True Std Dev": f"{true_params['stds'][i]:.3f}",
                "True Weight": f"{true_params['weights'][i]:.3f}"
            })
        
        st.dataframe(pd.DataFrame(component_data))
        
    def _render_advanced_topics(self):
        """Render advanced topics with expandable sections."""
        with st.expander("Advanced GMM Topics"):
            st.markdown("""
            ### Model Selection
            
            Choosing the right number of components is crucial in GMM. Too few components 
            can't capture the true complexity of the data (underfitting), while too many 
            can lead to overfitting. Common approaches for model selection include:
            
            1. **Bayesian Information Criterion (BIC)**: Penalizes model complexity to 
            prevent overfitting. The model with the lowest BIC score is preferred.
            
            2. **Akaike Information Criterion (AIC)**: Similar to BIC but with a different 
            penalty term. AIC tends to select more complex models compared to BIC.
            
            3. **Cross-validation**: Evaluates model performance on held-out data to 
            ensure generalization.
            
            ### Covariance Types
            
            GMMs support different covariance structures that affect how components 
            can capture data patterns:
            
            1. **Full**: Each component has its own unrestricted covariance matrix, 
            providing maximum flexibility but requiring more parameters.
            
            2. **Tied**: All components share the same covariance matrix, useful when 
            components are expected to have similar shapes.
            
            3. **Diagonal**: Covariance matrices are diagonal, assuming features are 
            independent within each component.
            
            4. **Spherical**: Covariance matrices are scalar multiples of identity, 
            producing spherical components.
            """)
            
            if st.checkbox("Show Covariance Types Comparison"):
                self._plot_covariance_comparison()
    
    def _plot_covariance_comparison(self):
        """Create comparison plot of different covariance types."""
        # Generate sample data
        data, _ = generate_gmm_data(n_components=2, n_samples=300)
        
        # Compare different covariance types
        covariance_types = ['full', 'tied', 'diagonal', 'spherical']
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        x_plot = np.linspace(min(data) - 1, max(data) + 1, 200)
        
        for idx, cov_type in enumerate(covariance_types):
            gmm = GaussianMixture(
                n_components=2,
                covariance_type=cov_type,
                random_state=42
            )
            gmm.fit(data.reshape(-1, 1))
            
            density = np.exp(gmm.score_samples(x_plot.reshape(-1, 1)))
            axes[idx].hist(data, bins='auto', density=True, 
                          alpha=0.3, color='gray')
            axes[idx].plot(x_plot, density, 'r-', 
                          label=f'{cov_type.capitalize()} Covariance')
            axes[idx].legend()
            axes[idx].set_title(f'{cov_type.capitalize()} Covariance')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### Practical Tips for Using GMMs
        
        1. **Initialization**: The EM algorithm can be sensitive to initialization. 
        Using multiple random starts (n_init parameter) can help find better solutions.
        
        2. **Convergence**: Monitor the convergence of the EM algorithm using the 
        convergence_monitor_ attribute. The algorithm might converge to local optima.
        
        3. **Component Interpretation**: In many applications, individual components 
        have meaningful interpretations (e.g., different subpopulations in your data).
        
        4. **Scaling**: While GMMs are scale-dependent, they're less sensitive to 
        scaling than many other algorithms because each component learns its own variance.
        """)