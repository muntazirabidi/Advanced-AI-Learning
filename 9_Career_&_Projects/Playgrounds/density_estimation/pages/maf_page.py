# pages/maf_page.py
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.plotting import plot_flow_transformation
from utils.data_generation import generate_flow_data

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.
    This neural network ensures the autoregressive property through masked weights.
    """
    def __init__(self, input_dim, hidden_dims, output_dim_per_input):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim_per_input = output_dim_per_input
        
        # Create masks and layers
        self.masks = []
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [input_dim * output_dim_per_input]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        self.create_masks()
        self.apply_masks()
    
    def create_masks(self):
        """Create masks for autoregressive property."""
        L = len(self.layers)
        degrees = []
        
        # Input degrees
        degrees.append(np.arange(self.input_dim))
        
        # Hidden degrees
        for i in range(len(self.hidden_dims)):
            min_prev_degree = min(degrees[-1])
            degrees.append(np.random.randint(min_prev_degree, 
                                          self.input_dim - 1,
                                          size=self.hidden_dims[i]))
        
        # Output degrees
        degrees.append(np.hstack([np.arange(self.input_dim) 
                                for _ in range(self.output_dim_per_input)]))
        
        # Create masks
        for i in range(L):
            mask = degrees[i+1].reshape(-1, 1) >= degrees[i]
            self.masks.append(mask.float())
    
    def apply_masks(self):
        """Apply masks to network weights."""
        for mask, layer in zip(self.masks, self.layers):
            layer.weight.data *= mask.t()
    
    def forward(self, x):
        """Forward pass with masked weights."""
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

class MAFPage:
    """
    Page component for Masked Autoregressive Flows visualization and explanation.
    Provides interactive demonstrations and educational content about MAF.
    """
    
    def render(self):
        """Render the MAF page content and interactive elements."""
        st.title("Masked Autoregressive Flows (MAF)")
        
        self._render_introduction()
        self._render_interactive_demo()
        self._render_technical_details()
        self._render_advanced_topics()
    
    def _render_introduction(self):
        """Render the introductory section explaining MAF concepts."""
        st.markdown("""
        Masked Autoregressive Flows are a powerful type of normalizing flow that can 
        learn complex probability distributions by transforming a simple base distribution 
        through a series of invertible transformations.
        
        ### Key Concepts
        
        1. **Normalizing Flows**: Transform a simple distribution (like a Gaussian) 
        into a more complex one through a series of invertible transformations.
        
        2. **Autoregressive Property**: Each variable depends only on the previous 
        variables, allowing tractable computation of the transformation's Jacobian.
        
        3. **Change of Variables Formula**: 
        
        $$p_X(x) = p_Z(z) |\det(\frac{\partial f^{-1}}{\partial x})|$$
        
        where:
        - $p_X(x)$ is the target density
        - $p_Z(z)$ is the base density
        - $f$ is the transformation
        """)
    
    def _render_interactive_demo(self):
        """Create and render the interactive MAF demonstration."""
        st.subheader("Interactive MAF Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            flow_type = st.selectbox(
                "Transformation Type",
                ["Affine", "Quadratic", "Sine"]
            )
            
            n_points = st.slider(
                "Number of Data Points",
                min_value=100,
                max_value=1000,
                value=500,
                step=100
            )
        
        # Generate and transform data
        base_samples, transformed_samples = generate_flow_data(
            flow_type, n_points, seed=42
        )
        
        # Create visualization
        fig = plot_flow_transformation(
            base_samples,
            transformed_samples,
            base_density=lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
            title=f"{flow_type} Flow Transformation"
        )
        
        st.pyplot(fig)
        
        self._add_transformation_explanation(flow_type)
    
    def _add_transformation_explanation(self, flow_type):
        """Add explanatory text for the transformation visualization."""
        explanations = {
            "Affine": """
                The affine transformation applies a linear scaling and shift to the 
                base distribution. This is the simplest form of flow but can be 
                powerful when composed in multiple layers.
                """,
            "Quadratic": """
                The quadratic transformation applies a nonlinear squaring operation, 
                showing how flows can capture more complex distributions through 
                nonlinear transformations.
                """,
            "Sine": """
                The sine transformation demonstrates how periodic functions can create 
                complex multi-modal distributions from a simple base distribution.
                """
        }
        
        st.markdown(f"""
        ### Understanding the Transformation
        
        {explanations[flow_type]}
        
        The plots show:
        1. **Left**: The base distribution (standard normal)
        2. **Right**: The transformed distribution after applying the {flow_type} flow
        """)
    
    def _render_technical_details(self):
        """Render the technical explanation section."""
        st.markdown("""
        ### How MAF Works
        
        A Masked Autoregressive Flow transforms data using an autoregressive neural 
        network. The key ideas are:
        
        1. **Autoregressive Transformation**:
        Each dimension is transformed based on previous dimensions:
        
        $x_i = f_i(z_i; h_i(z_{1:i-1}))$
        
        where $h_i$ is a neural network that computes the parameters of the 
        transformation $f_i$.
        
        2. **MADE Architecture**:
        Uses masked weights to ensure the autoregressive property:
        - Input layer: Each unit can only see previous dimensions
        - Hidden layers: Units maintain ordering through careful masking
        - Output layer: Produces parameters for the transformation
        """)
    
    def _render_advanced_topics(self):
        """Render advanced topics with expandable sections."""
        with st.expander("Advanced MAF Topics"):
            st.markdown("""
            ### Comparison with Other Flows
            
            MAF has several advantages and trade-offs:
            
            1. **Advantages**:
            - Tractable likelihood computation
            - Flexible transformations
            - Memory efficient
            
            2. **Limitations**:
            - Sequential generation (slow sampling)
            - Limited parallelization
            - May require many layers for complex distributions
            
            ### Variants and Extensions
            
            Several variants of MAF have been proposed:
            
            1. **Inverse Autoregressive Flow (IAF)**:
            - Fast sampling but slow density evaluation
            - Complementary to MAF
            
            2. **Neural Autoregressive Flow**:
            - Uses monotonic neural networks
            - More flexible but more complex
            
            3. **Block MAF**:
            - Groups dimensions for parallel processing
            - Trade-off between speed and flexibility
            """)
            
            if st.checkbox("Show Architecture Visualization"):
                self._plot_made_architecture()
    
    def _plot_made_architecture(self):
        """Create visualization of the MADE architecture."""
        made_dims = [2, 8, 8, 4]  # Example dimensions
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot units
        for layer in range(len(made_dims)):
            n_units = made_dims[layer]
            for unit in range(n_units):
                x = layer
                y = unit - n_units/2
                circle = plt.Circle((x, y), 0.2, fill=False)
                ax.add_artist(circle)
        
        # Plot connections with masks
        for layer in range(len(made_dims)-1):
            src_units = made_dims[layer]
            dst_units = made_dims[layer+1]
            
            for src in range(src_units):
                for dst in range(dst_units):
                    # Simulate masking based on ordering
                    if dst >= src:
                        x = [layer, layer+1]
                        y = [src - src_units/2, dst - dst_units/2]
                        ax.plot(x, y, 'k-', alpha=0.2)
        
        ax.set_xlim(-0.5, len(made_dims)-0.5)
        ax.set_ylim(-max(made_dims)/2-0.5, max(made_dims)/2+0.5)
        ax.axis('off')
        ax.set_title("MADE Architecture\nArrows show allowed connections")
        
        st.pyplot(fig)
        
        st.markdown("""
        The visualization above shows the MADE (Masked Autoencoder for Distribution 
        Estimation) architecture used in MAF. Each arrow represents a potential 
        connection in the network, where connections are masked to ensure the 
        autoregressive property. This masking scheme guarantees that each output 
        only depends on previous inputs, which is crucial for the tractability of 
        the flow transformation.
        """)