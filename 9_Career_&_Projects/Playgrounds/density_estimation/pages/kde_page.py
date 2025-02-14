# pages/kde_page.py
import streamlit as st
import numpy as np
from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
from utils.plotting import plot_density_comparison
from utils.data_generation import generate_synthetic_data

class KDEPage:
    """
    Page component for Kernel Density Estimation visualization and explanation.
    Provides interactive demonstrations and educational content about KDE.
    """
    
    def render(self):
        """Render the KDE page content and interactive elements."""
        st.title("Kernel Density Estimation (KDE)")
        
        # Introduction and mathematical background
        self._render_introduction()
        
        # Interactive demonstration
        self._render_interactive_demo()
        
        # Technical details and advanced topics
        self._render_technical_details()
        self._render_advanced_topics()
    
    def _render_introduction(self):
        """Render the introductory section explaining KDE concepts."""
        st.markdown("""
        Kernel Density Estimation is one of the most intuitive approaches to density estimation. 
        It works by placing a small probability distribution (kernel) at each data point and then 
        averaging these distributions to create a smooth estimate of the overall density.
        
        ### Mathematical Intuition
        
        The KDE estimator at any point x is given by:
        
        $\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^n K(\frac{x - x_i}{h})$
        
        where:
        - $n$ is the number of data points
        - $h$ is the bandwidth (controls smoothness)
        - $K$ is the kernel function (usually a Gaussian)
        - $x_i$ are the observed data points
        """)
    
    def _render_interactive_demo(self):
        """Create and render the interactive KDE demonstration."""
        st.subheader("Interactive KDE Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            dist_type = st.selectbox(
                "Select Distribution Type",
                ["Normal", "Mixture of Gaussians", "Bimodal", "Skewed"]
            )
            
            n_points = st.slider(
                "Number of Data Points",
                min_value=50,
                max_value=1000,
                value=200,
                step=50
            )
            
            bandwidth = st.slider(
                "Bandwidth (h)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
        
        # Generate and plot data
        data = generate_synthetic_data(dist_type, n_points)
        self._plot_kde_demonstration(data, bandwidth)
    
    def _plot_kde_demonstration(self, data, bandwidth):
        """Create visualizations demonstrating KDE behavior."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot: overall density estimate
        kde = gaussian_kde(data, bw_method=bandwidth)
        plot_density_comparison(
            data,
            kde.evaluate,
            "KDE with Data Histogram",
            ax=ax1
        )
        
        # Second plot: individual kernels
        x_plot = np.linspace(min(data) - 1, max(data) + 1, 200)
        ax2.hist(data, bins='auto', density=True, alpha=0.3, 
                color='gray', label='Data')
        
        # Plot individual kernels for a subset of points
        n_kernels = min(20, len(data))
        selected_points = np.random.choice(data, n_kernels, replace=False)
        for x_i in selected_points:
            kernel = norm.pdf((x_plot - x_i) / bandwidth) / (len(data) * bandwidth)
            ax2.plot(x_plot, kernel, 'b-', alpha=0.2)
        
        # Plot overall density
        ax2.plot(x_plot, kde.evaluate(x_plot), 'r-', 
                label='KDE Estimate')
        ax2.set_title("Individual Kernels")
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        self._add_visualization_explanation()
    
    def _add_visualization_explanation(self):
        """Add explanatory text for the KDE visualization."""
        st.markdown("""
        ### Understanding the Visualization
        
        The plots above demonstrate how KDE works:
        
        1. **Left plot**: Shows the final density estimate (blue line) overlaid on 
           the histogram of the actual data. This represents our best estimate of 
           the true probability density function.
        
        2. **Right plot**: Reveals the "mechanics" of KDE by showing individual 
           kernel functions (blue curves) centered at each data point. The final 
           estimate (red line) is the sum of these individual kernels.
        """)
    
    def _render_technical_details(self):
        """Render the technical explanation section."""
        st.markdown("""
        ### Technical Details
        
        KDE has several important properties:
        
        1. **Bandwidth Selection**: The bandwidth parameter h controls the trade-off 
        between bias and variance:
        - Small bandwidth → Low bias, high variance (more wiggly)
        - Large bandwidth → High bias, low variance (more smooth)
        
        2. **Kernel Choice**: While we used the Gaussian kernel, other options exist:
        - Epanechnikov kernel (optimal in terms of mean squared error)
        - Uniform kernel
        - Triangular kernel
        """)
    
    def _render_advanced_topics(self):
        """Render advanced topics with expandable sections."""
        with st.expander("Advanced KDE Topics"):
            st.markdown("""
            ### Advantages and Limitations
            
            **Advantages:**
            - Non-parametric: No assumptions about the underlying distribution
            - Simple to understand and implement
            - Works well for univariate data
            
            **Limitations:**
            - Bandwidth selection can be tricky
            - Curse of dimensionality in higher dimensions
            - Can struggle with bounded domains
            
            ### Bandwidth Selection Methods
            
            Several methods exist for automatically selecting the bandwidth:
            - Silverman's rule of thumb
            - Scott's rule
            - Cross-validation
            """)
            
            if st.checkbox("Show Bandwidth Selection Methods Comparison"):
                self._plot_bandwidth_comparison()
    
    def _plot_bandwidth_comparison(self):
        """Create comparison plot of different bandwidth selection methods."""
        data = generate_synthetic_data("Normal", 200)
        
        bandwidths = {
            "Scott's Rule": gaussian_kde(data).factor,
            "Custom Small": 0.2,
            "Custom Medium": 0.5,
            "Custom Large": 1.0
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(data, bins='auto', density=True, alpha=0.3, 
               color='gray', label='Data')
        
        # Plot different bandwidth estimates
        x_plot = np.linspace(min(data) - 1, max(data) + 1, 200)
        for method, bw in bandwidths.items():
            kde = gaussian_kde(data, bw_method=bw)
            ax.plot(x_plot, kde.evaluate(x_plot), 
                   label=f'{method} (h={bw:.3f})')
        
        ax.legend()
        ax.set_title("Comparison of Bandwidth Selection Methods")
        st.pyplot(fig)