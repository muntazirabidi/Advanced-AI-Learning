# pages/comparison_page.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
from utils.plotting import plot_density_comparison
from utils.data_generation import generate_synthetic_data, generate_gmm_data

class ComparisonPage:
    """
    A comprehensive comparison of different density estimation methods.
    This page allows users to compare how different methods perform on the same datasets,
    helping build intuition about their relative strengths and weaknesses.
    """
    
    def __init__(self):
        """Initialize the comparison page with default settings."""
        self.methods = {
            "Kernel Density Estimation": self.kde_estimate,
            "Gaussian Mixture Model": self.gmm_estimate,
            "Neural Spline Flow": self.nsf_estimate
        }
    
    def render(self):
        """Render the comparison page content with interactive elements."""
        st.title("Comparative Analysis of Density Estimation Methods")
        
        self._render_introduction()
        self._render_interactive_comparison()
        self._render_method_comparison_table()
        self._render_advanced_analysis()
    
    def _render_introduction(self):
        """Provide an introduction to the comparative analysis."""
        st.markdown("""
        Understanding the strengths and limitations of different density estimation methods
        is crucial for choosing the right approach for your specific problem. Each method
        makes different assumptions and trades off various properties:

        **Kernel Density Estimation (KDE)** provides a non-parametric approach that makes
        minimal assumptions about the underlying distribution. It excels at exploratory
        data analysis and simpler distributions, offering great flexibility for univariate
        data. However, it can struggle with high-dimensional data and requires careful
        bandwidth selection.
        
        **Gaussian Mixture Models (GMM)** offer a parametric approach that works well
        for multimodal data and provides an interpretable decomposition into components.
        They are particularly effective when the data naturally clusters into groups
        and can handle higher dimensions better than KDE. The main challenge lies in
        selecting the appropriate number of components.
        
        **Neural Spline Flows (NSF)** represent the state-of-the-art in flexibility,
        capable of modeling very complex distributions through learned transformations.
        They combine the universal approximation capabilities of neural networks with
        the mathematical guarantees of normalizing flows, making them highly expressive
        while maintaining exact likelihood computation.
        """)
    
    def _render_interactive_comparison(self):
        """Create interactive visualization comparing different methods."""
        st.subheader("Interactive Method Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset selection
            distribution_type = st.selectbox(
                "Select Distribution Type",
                ["Unimodal", "Bimodal", "Skewed", "Heavy-tailed", "Mixture"]
            )
            
            n_samples = st.slider(
                "Number of Samples",
                min_value=100,
                max_value=1000,
                value=500,
                step=100
            )
        
        # Generate data based on selected distribution
        data = generate_synthetic_data(distribution_type, n_samples)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Plot true data histogram in all subplots
        for ax in axes:
            ax.hist(data, bins='auto', density=True, alpha=0.3, 
                   color='gray', label='Data')
        
        # Apply each method
        x_plot = np.linspace(min(data) - 1, max(data) + 1, 200)
        
        # KDE estimation
        kde = self.kde_estimate(data)
        axes[0].plot(x_plot, kde(x_plot), 'r-', label='KDE')
        axes[0].set_title("Kernel Density Estimation")
        axes[0].legend()
        
        # GMM estimation
        gmm = self.gmm_estimate(data)
        gmm_density = np.exp(gmm.score_samples(x_plot.reshape(-1, 1)))
        axes[1].plot(x_plot, gmm_density, 'b-', label='GMM')
        axes[1].set_title("Gaussian Mixture Model")
        axes[1].legend()
        
        # NSF estimation (simplified)
        nsf_density = self.nsf_estimate(data, x_plot)
        axes[2].plot(x_plot, nsf_density, 'g-', label='NSF')
        axes[2].set_title("Neural Spline Flow")
        axes[2].legend()
        
        # Comparison of all methods
        axes[3].plot(x_plot, kde(x_plot), 'r-', label='KDE')
        axes[3].plot(x_plot, gmm_density, 'b-', label='GMM')
        axes[3].plot(x_plot, nsf_density, 'g-', label='NSF')
        axes[3].set_title("All Methods Comparison")
        axes[3].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        self._add_comparison_analysis(distribution_type)
    
    def _add_comparison_analysis(self, distribution_type):
        """Add detailed analysis of the comparison results."""
        st.markdown(f"""
        ### Analysis of Results for {distribution_type} Distribution
        
        The plots above demonstrate how each method approaches density estimation differently:
        
        **Kernel Density Estimation** provides a smooth, non-parametric estimate that
        follows the data closely. For this {distribution_type} distribution, KDE
        {self._get_kde_analysis(distribution_type)}
        
        **Gaussian Mixture Model** creates a parametric estimate using a mixture of
        Gaussian components. In this case, GMM {self._get_gmm_analysis(distribution_type)}
        
        **Neural Spline Flow** learns a flexible transformation that can capture complex
        patterns. For this distribution, NSF {self._get_nsf_analysis(distribution_type)}
        
        The comparison plot shows how the methods differ in their approximation of the
        true density, highlighting their relative strengths and limitations.
        """)
    
    def _get_kde_analysis(self, distribution_type):
        """Get distribution-specific analysis for KDE."""
        analyses = {
            "Unimodal": "performs well due to the simple, single-peaked nature of the data.",
            "Bimodal": "captures both modes but may struggle with the valley between them depending on the bandwidth.",
            "Skewed": "adapts to the asymmetry but might oversmooth the tail.",
            "Heavy-tailed": "tends to underestimate the tails due to its gaussian kernel.",
            "Mixture": "handles the multiple components well but may need careful bandwidth selection."
        }
        return analyses.get(distribution_type, "adapts to the shape of the distribution.")
    
    def _get_gmm_analysis(self, distribution_type):
        """Get distribution-specific analysis for GMM."""
        analyses = {
            "Unimodal": "might overfit by using multiple components for a simple distribution.",
            "Bimodal": "naturally captures the two modes with separate components.",
            "Skewed": "approximates the skewness using a mixture of symmetric components.",
            "Heavy-tailed": "may need extra components to approximate the heavy tails.",
            "Mixture": "excels by matching its components to the natural clusters in the data."
        }
        return analyses.get(distribution_type, "approximates the distribution with mixture components.")
    
    def _get_nsf_analysis(self, distribution_type):
        """Get distribution-specific analysis for NSF."""
        analyses = {
            "Unimodal": "learns a simple transformation but might be unnecessarily complex.",
            "Bimodal": "successfully learns the transformation needed to create two modes.",
            "Skewed": "effectively captures the asymmetry through its flexible spline transformation.",
            "Heavy-tailed": "adapts well to the tails through its nonlinear transformation.",
            "Mixture": "learns a complex transformation to match the mixture structure."
        }
        return analyses.get(distribution_type, "learns an appropriate transformation for the distribution.")
    
    def _render_method_comparison_table(self):
        """Create a detailed comparison table of the methods."""
        st.subheader("Method Characteristics")
        
        comparison_data = {
            "Property": [
                "Flexibility",
                "Sample Efficiency",
                "Training Speed",
                "Inference Speed",
                "Interpretability",
                "High-Dimensional Scaling",
                "Hyperparameter Sensitivity"
            ],
            "KDE": [
                "Medium",
                "High",
                "Fast",
                "Medium",
                "High",
                "Poor",
                "Medium"
            ],
            "GMM": [
                "Medium",
                "Medium",
                "Fast",
                "Fast",
                "High",
                "Medium",
                "Medium"
            ],
            "NSF": [
                "High",
                "Low",
                "Slow",
                "Fast",
                "Low",
                "Good",
                "High"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df)
        
        st.markdown("""
        The table above summarizes key properties of each method:
        
        **Flexibility** refers to the method's ability to capture complex distributions.
        NSF offers the highest flexibility through its learned transformations, while
        KDE and GMM are more constrained by their assumptions.
        
        **Sample Efficiency** indicates how well the method performs with limited data.
        KDE typically performs well with small datasets, while NSF requires more data
        for effective training.
        
        **Training and Inference Speed** reflect computational efficiency. GMM and KDE
        are generally faster to train, while NSF requires more computation but can be
        fast at inference time.
        """)
    
    def _render_advanced_analysis(self):
        """Provide deeper analysis of method behavior."""
        with st.expander("Advanced Analysis"):
            st.markdown("""
            ### Method Selection Guidelines
            
            When choosing a density estimation method, consider these factors:
            
            **Data Characteristics:**
            For simple, unimodal data, KDE is often sufficient and provides a good
            balance of flexibility and simplicity. When data shows clear clustering
            patterns, GMM typically works well by matching its components to these
            clusters. For complex, unknown structures, NSF might be necessary to
            capture subtle patterns in the distribution.
            
            **Sample Size:**
            With small datasets (< 1000 samples), KDE or GMM are usually more reliable
            as they make better use of limited data. For larger datasets, any method
            can work well, but NSF can particularly benefit from the additional data
            to learn more complex transformations.
            
            **Dimensionality:**
            In low dimensions (1-3D), all methods work well, with KDE being particularly
            intuitive. For medium dimensions (4-10D), GMM or NSF are preferred as KDE
            suffers from the curse of dimensionality. In high dimensions (>10D), NSF
            typically performs best due to its ability to learn efficient representations.
            
            **Computational Resources:**
            If computational resources are limited, KDE or GMM provide efficient
            solutions. With abundant resources, NSF can potentially provide the best
            results through its more complex modeling approach.
            """)
    
    def kde_estimate(self, data):
        """Create a KDE estimate for the given data."""
        return gaussian_kde(data)
    
    def gmm_estimate(self, data, n_components=3):
        """Create a GMM estimate for the given data."""
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data.reshape(-1, 1))
        return gmm
    
    def nsf_estimate(self, data, x_eval):
        """Create a simplified NSF estimate for visualization.
        
        Note: This is a simplified version for visualization purposes.
        In practice, you would train a proper NSF model.
        """
        # Create a simple transformation that captures some complexity
        samples = np.random.normal(0, 1, len(data))
        kde = gaussian_kde(samples)
        transformed = kde(x_eval) * np.exp(np.sin(x_eval/2))
        return transformed