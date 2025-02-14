# Project Structure:
#
# density_estimation/
# ├── Home.py                  # Main application entry point
# ├── pages/                   # Page components
# │   ├── __init__.py
# │   ├── comparison_page.py   # Comparison analysis page
# │   ├── gmm_page.py         # Gaussian Mixture Models page
# │   ├── kde_page.py         # Kernel Density Estimation page
# │   ├── maf_page.py         # Masked Autoregressive Flows page
# │   └── nsf_page.py         # Neural Spline Flows page
# └── utils/                   # Utility functions
#     ├── __init__.py
#     ├── data_generation.py   # Data generation utilities
#     └── plotting.py         # Plotting utilities

# Home.py
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pages.kde_page import KDEPage
from pages.gmm_page import GMMPage
from pages.maf_page import MAFPage
from pages.nsf_page import NSFPage
from pages.comparison_page import ComparisonPage
from utils.plotting import configure_plotting_style
from utils.data_generation import generate_synthetic_data

class DensityEstimationApp:
    def __init__(self):
        """Initialize the Density Estimation Learning application."""
        self.setup_app_config()
        self.setup_sidebar()
        self.main_content()

    def setup_app_config(self):
        """Configure the application settings."""
        st.set_page_config(
            page_title="Density Estimation Learning Lab",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        configure_plotting_style()

    def setup_sidebar(self):
        """Initialize the sidebar navigation and controls."""
        st.sidebar.title("Density Estimation Explorer")
        st.sidebar.markdown("""
        Welcome to your journey through density estimation methods. 
        This interactive application will help you understand various 
        approaches to modeling probability distributions.
        """)
        
        self.current_page = st.sidebar.radio(
            "Navigate Topics",
            ["Introduction",
             "Kernel Density Estimation",
             "Gaussian Mixture Models",
             "Masked Autoregressive Flows",
             "Neural Spline Flows",
             "Comparative Analysis"]
        )

    def introduction(self):
        """Render the introduction page with key concepts and motivation."""
        st.title("Understanding Density Estimation")
        self._render_intro_content()
        self._render_interactive_demo()

    def _render_intro_content(self):
        """Render the introductory content."""
        st.markdown("""
        ### What is Density Estimation?
        Density estimation is a fundamental concept in statistics and machine learning 
        that aims to model the probability distribution that generated a dataset. 
        Think of it as reverse engineering the recipe (distribution) from seeing 
        only the final dishes (data points).
        
        ### Why is it Important?
        Density estimation forms the backbone of many machine learning applications:
        - Generative modeling for creating new, realistic data
        - Anomaly detection by identifying unlikely events
        - Understanding data structure and patterns
        - Making probabilistic predictions
        
        ### What You'll Learn
        In this interactive application, you'll explore:
        1. **Kernel Density Estimation**: The most intuitive approach
        2. **Gaussian Mixture Models**: Combining simple distributions
        3. **Masked Autoregressive Flows**: Modern neural approaches
        4. **Neural Spline Flows**: State-of-the-art flexible models
        
        Each section includes:
        - Mathematical intuition with clear explanations
        - Interactive visualizations
        - Practical examples and code
        - Comparative analysis
        """)

    def _render_interactive_demo(self):
        """Render the interactive demonstration section."""
        st.subheader("Interactive Demo: Understanding Data Distributions")
        
        col1, col2 = st.columns(2)
        with col1:
            dist_type = st.selectbox(
                "Select Distribution Type",
                ["Normal", "Mixture of Gaussians", "Bimodal", "Uniform"]
            )
            n_points = st.slider("Number of Data Points", 100, 1000, 500)
        
        data = generate_synthetic_data(dist_type, n_points)
        self._plot_distribution_demo(data)

    def _plot_distribution_demo(self, data):
        """Create an educational visualization of the selected distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot: Histogram with KDE
        sns.histplot(data=data, bins='auto', stat='density', alpha=0.4, ax=ax1)
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 200)
        ax1.plot(x_range, kde(x_range), 'r-', label='KDE')
        ax1.set_title("Data Distribution")
        ax1.legend()
        
        # Second plot: Empirical CDF
        ax2.hist(data, bins=50, density=True, cumulative=True, 
                alpha=0.4, label='Empirical CDF')
        ax2.set_title("Cumulative Distribution")
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        self._add_distribution_explanation()

    def _add_distribution_explanation(self):
        """Add explanatory text for the distribution visualization."""
        st.markdown("""
        The plots above show two ways to visualize a probability distribution:
        
        1. **Probability Density Function (left)**: Shows how likely different values 
        are in the distribution. The histogram shows the empirical distribution of 
        our data, while the red line shows a smooth estimate using Kernel Density 
        Estimation.
        
        2. **Cumulative Distribution Function (right)**: Shows the probability of 
        observing a value less than or equal to each point. This is often more 
        stable than the density function and can reveal different aspects of the 
        distribution.
        """)

    def main_content(self):
        """Route to the appropriate page based on navigation."""
        if self.current_page == "Introduction":
            self.introduction()
        elif self.current_page == "Kernel Density Estimation":
            KDEPage().render()
        elif self.current_page == "Gaussian Mixture Models":
            GMMPage().render()
        elif self.current_page == "Masked Autoregressive Flows":
            MAFPage().render()
        elif self.current_page == "Neural Spline Flows":
            NSFPage().render()
        else:
            ComparisonPage().render()

if __name__ == "__main__":
    app = DensityEstimationApp()