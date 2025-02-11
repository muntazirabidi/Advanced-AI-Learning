import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.latex_utils import write_latex_section

st.set_page_config(
    page_title="BNN vs GP Comparison",
    page_icon="🧠",
    layout="wide"
)

# Main page content
st.title("Bayesian Deep Learning: BNNs vs GPs")

# Introduction
st.header("Theoretical Background")
st.markdown("""
This application explores two powerful approaches to Bayesian machine learning:
Bayesian Neural Networks (BNNs) and Gaussian Processes (GPs). Both methods offer
principled ways to handle uncertainty in machine learning predictions.
""")

# BNN Theory
st.subheader("Bayesian Neural Networks")
st.markdown(write_latex_section("bnn"))

# Visualize BNN prior and posterior
if st.button("Visualize BNN Weight Distribution"):
    fig = go.Figure()
    x = np.linspace(-3, 3, 100)
    prior = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    posterior = np.exp(-0.5 * (x-1)**2 / 0.5**2) / (np.sqrt(2 * np.pi) * 0.5)
    
    fig.add_trace(go.Scatter(x=x, y=prior, name="Prior", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x, y=posterior, name="Posterior", line=dict(color="red")))
    fig.update_layout(title="BNN Weight Distribution", xaxis_title="Weight Value", yaxis_title="Density")
    st.plotly_chart(fig)

# GP Theory
st.subheader("Gaussian Processes")
st.markdown(write_latex_section("gp"))

# Visualize GP prior samples
if st.button("Visualize GP Prior Samples"):
    x = np.linspace(-3, 3, 100)
    K = np.exp(-0.5 * (x[:, None] - x[None, :])**2)
    samples = np.random.multivariate_normal(np.zeros_like(x), K, size=5)
    
    fig = go.Figure()
    for i in range(5):
        fig.add_trace(go.Scatter(x=x, y=samples[i], name=f"Sample {i+1}", opacity=0.6))
    fig.update_layout(title="GP Prior Samples", xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig)