# pages/nsf_page.py
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.plotting import plot_spline_flow
from utils.data_generation import generate_flow_data

class RationalQuadraticSpline:
    """
    Implementation of rational quadratic splines for Neural Spline Flows.
    Provides a smooth, invertible transformation through piecewise rational
    quadratic functions.
    """
    
    def __init__(self, num_bins=10, bounds=(-3, 3)):
        """
        Initialize the rational quadratic spline.
        
        Args:
            num_bins: Number of bins for the piecewise transformation
            bounds: Tuple of (min, max) values for the transformation domain
        """
        self.num_bins = num_bins
        self.bounds = bounds
        
        # Initialize bin widths, heights, and derivatives with default values
        self.bin_widths = np.ones(num_bins) * (bounds[1] - bounds[0]) / num_bins
        self.bin_heights = np.ones(num_bins)
        self.derivatives = np.ones(num_bins + 1)
    
    def update_parameters(self, widths, heights, derivatives):
        """
        Update the spline parameters while ensuring monotonicity.
        
        Args:
            widths: Log-scale bin widths
            heights: Log-scale bin heights
            derivatives: Unconstrained derivatives at knot points
        """
        # Ensure positivity through exponential/softplus
        self.bin_widths = np.exp(widths)
        self.bin_heights = np.exp(heights)
        self.derivatives = F.softplus(torch.tensor(derivatives)).numpy()
    
    def evaluate(self, x):
        """
        Evaluate the spline transformation at given points.
        
        Args:
            x: Input points to transform
            
        Returns:
            Transformed points through the rational quadratic spline
        """
        x = np.array(x)
        result = np.zeros_like(x)
        
        # Find bin index for each input point
        bin_idx = np.floor((x - self.bounds[0]) / self.bin_widths.sum() * self.num_bins)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1).astype(int)
        
        # Compute cumulative widths and heights
        cum_widths = np.cumsum(np.pad(self.bin_widths, (1, 0))[:-1])
        cum_heights = np.cumsum(np.pad(self.bin_heights, (1, 0))[:-1])
        
        for i in range(len(x)):
            idx = bin_idx[i]
            
            # Compute normalized position within bin
            x_low = self.bounds[0] + cum_widths[idx]
            x_high = x_low + self.bin_widths[idx]
            t = (x[i] - x_low) / (x_high - x_low)
            
            # Apply rational quadratic transformation
            y_low = cum_heights[idx]
            y_high = y_low + self.bin_heights[idx]
            d_low = self.derivatives[idx]
            d_high = self.derivatives[idx + 1]
            
            numerator = self.bin_heights[idx] * (
                d_low * t**2 + 2 * t * (1 - t)
            )
            denominator = d_low + (d_high - d_low) * t
            
            result[i] = y_low + numerator / denominator
        
        return result

class NSFPage:
    """
    Page component for Neural Spline Flows visualization and explanation.
    Provides interactive demonstrations and educational content about NSF.
    """
    
    def __init__(self):
        """Initialize the NSF page with default parameters."""
        self.num_bins_default = 10
        self.bounds = (-3, 3)
        self.spline = RationalQuadraticSpline(
            num_bins=self.num_bins_default,
            bounds=self.bounds
        )
    
    def render(self):
        """Render the NSF page content and interactive elements."""
        st.title("Neural Spline Flows")
        
        self._render_introduction()
        self._render_interactive_demo()
        self._render_technical_details()
        self._render_advanced_topics()
    
    def _render_introduction(self):
        """Render the introductory section explaining NSF concepts."""
        st.markdown("""
        Neural Spline Flows represent a powerful advancement in normalizing flows by 
        combining neural networks with rational quadratic splines. This approach creates 
        flexible, invertible transformations that can model complex probability 
        distributions.

        The key innovation lies in using neural networks to learn the parameters of 
        spline transformations, which ensures both flexibility and mathematical 
        guarantees of invertibility.
        
        ### Core Components
        
        Neural Spline Flows work through three main mechanisms:
        
        1. **Rational Quadratic Splines**: These are piecewise functions that create 
        smooth, monotonic transformations between input and output spaces. Each piece 
        is a rational quadratic function, ensuring smoothness and invertibility.
        
        2. **Neural Parametrization**: A neural network learns optimal parameters for 
        the spline transformation, adapting to the data distribution. This includes 
        bin widths, heights, and derivatives at the knot points.
        
        3. **Normalizing Flow Framework**: The spline transformation is used within 
        the normalizing flow framework to transform a simple base distribution into 
        a complex target distribution.
        """)
    
    def _render_interactive_demo(self):
        """Create and render the interactive spline demonstration."""
        st.subheader("Interactive Spline Transformation Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            num_bins = st.slider(
                "Number of Bins",
                min_value=4,
                max_value=20,
                value=self.num_bins_default,
                help="Controls the complexity of the transformation"
            )
            
            smoothness = st.slider(
                "Smoothness",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Controls how smooth the transformation is"
            )
        
        # Update spline parameters
        self.spline = RationalQuadraticSpline(num_bins=num_bins)
        
        # Generate parameters with controlled randomness
        np.random.seed(42)
        widths = np.random.randn(num_bins) * smoothness
        heights = np.random.randn(num_bins) * smoothness
        derivatives = np.random.randn(num_bins + 1) * smoothness + 1.0
        
        self.spline.update_parameters(widths, heights, derivatives)
        
        # Generate visualization
        x = np.linspace(*self.bounds, 200)
        y = self.spline.evaluate(x)
        
        # Create knot points for visualization
        knots = np.column_stack([
            np.linspace(*self.bounds, num_bins + 1),
            self.spline.evaluate(np.linspace(*self.bounds, num_bins + 1))
        ])
        
        # Plot transformation
        fig = plot_spline_flow(x, y, knots)
        st.pyplot(fig)
        
        self._add_spline_explanation()
    
    def _add_spline_explanation(self):
        """Add explanatory text for the spline visualization."""
        st.markdown("""
        The visualization above shows how the rational quadratic spline transforms 
        the input space. The key elements are:
        
        1. **Knot Points** (red dots): These define the boundaries between different 
        pieces of the spline. The transformation is smooth across these boundaries.
        
        2. **Transformation Function** (blue line): Shows how input values are mapped 
        to output values. The monotonicity of this function ensures invertibility.
        
        3. **Density Transformation** (right plot): Demonstrates how the spline 
        transforms a simple probability density into a more complex one.
        
        Try adjusting the number of bins and smoothness to see how they affect the 
        transformation's flexibility and behavior.
        """)
    
    def _render_technical_details(self):
        """Render the technical explanation section."""
        st.markdown("""
        ### Technical Details
        
        The transformation in Neural Spline Flows is defined by a rational quadratic 
        function within each bin. For a point x in bin k, the transformation is:

        $y(x) = y_k + \\frac{Δy_k}{Δx_k} \\frac{α(ξ^2) + β(ξ)}{1 + (δ_k + δ_{k+1} - 2)ξ(1-ξ)}$

        where:
        - ξ is the normalized position within the bin
        - α and β are quadratic functions
        - δ_k are the derivatives at the knot points
        - Δy_k and Δx_k are the bin width and height
        
        This formulation ensures:
        1. Continuity across bin boundaries
        2. Monotonicity (hence invertibility)
        3. Efficient computation of both forward and inverse transformations
        """)
    
    def _render_advanced_topics(self):
        """Render advanced topics with expandable sections."""
        with st.expander("Advanced Topics"):
            st.markdown("""
            ### Advantages of Neural Spline Flows
            
            NSFs combine several desirable properties:
            
            1. **Expressiveness**: Can represent highly complex transformations while 
            maintaining invertibility. The piecewise nature of the transformation 
            allows it to adapt to local features of the data distribution.
            
            2. **Stability**: The rational quadratic formulation ensures well-behaved 
            gradients and numerical stability, making training more reliable compared 
            to other flow architectures.
            
            3. **Interpretability**: The transformation parameters have clear geometric 
            interpretations, making the model more understandable and easier to 
            debug.
            
            ### Implementation Considerations
            
            When implementing NSFs, several practical considerations are important:
            
            1. **Binning Strategy**: The number and placement of bins can significantly 
            impact performance. Adaptive binning strategies can help focus capacity 
            where it's most needed.
            
            2. **Parameter Constraints**: The neural network must output valid spline 
            parameters (positive widths, heights, and derivatives) to ensure a valid 
            transformation.
            
            3. **Numerical Precision**: Care must be taken in implementing the 
            transformation to maintain numerical stability, especially for extreme 
            values or very narrow bins.
            """)
            
            if st.checkbox("Show Parameter Sensitivity Analysis"):
                self._plot_sensitivity_analysis()
    
    def _plot_sensitivity_analysis(self):
        """Create and render the parameter sensitivity analysis plots."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.linspace(*self.bounds, 200)
        
        # Effect of bin count
        for bins in [5, 10, 15, 20]:
            spline = RationalQuadraticSpline(num_bins=bins)
            y = spline.evaluate(x)
            axes[0].plot(x, y, label=f'{bins} bins')
        axes[0].legend()
        axes[0].set_title("Effect of Bin Count")
        axes[0].grid(True)
        
        # Effect of smoothness
        for smoothness in [0.1, 0.5, 1.0, 2.0]:
            spline = RationalQuadraticSpline(num_bins=10)
            widths = np.random.randn(10) * smoothness
            heights = np.random.randn(10) * smoothness
            derivatives = np.random.randn(11) * smoothness + 1.0
            spline.update_parameters(widths, heights, derivatives)
            y = spline.evaluate(x)
            axes[1].plot(x, y, label=f'smoothness={smoothness:.1f}')
        axes[1].legend()
        axes[1].set_title("Effect of Smoothness")
        axes[1].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        The plots above demonstrate how different parameters affect the spline 
        transformation:
        
        1. **Bin Count** (left): More bins allow for more complex transformations 
        but may require more data to learn effectively.
        
        2. **Smoothness** (right): Higher smoothness values create more dramatic 
        transformations but may make the function harder to learn or less stable.
        """)