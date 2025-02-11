# pages/1_Demo.py
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.bnn import BayesianNeuralNetwork, BNNConfig
from models.gp import GPModel, GPConfig
import gpytorch
from utils.plotting import plot_comparative_analysis, plot_training_loss

st.title("Interactive Model Demo")

# Data generation
st.header("Data Generation")

# Add function selection
function_type = st.selectbox(
    "Select true function",
    ["sine", "quadratic", "multi-modal"],
    format_func=lambda x: {
        "sine": "Sine function: f(x) = sin(x)",
        "quadratic": "Quadratic function: f(x) = 0.5x² - 1",
        "multi-modal": "Multi-modal: f(x) = sin(x) + 0.5cos(2x)"
    }[x]
)

# Define true function based on selection
def true_function(x):
    if function_type == "sine":
        return np.sin(x)
    elif function_type == "quadratic":
        return 0.5 * x**2 - 1
    else:  # multi-modal
        return np.sin(x) + 0.5 * np.cos(2*x)

n_points = st.slider("Number of training points", 5, 50, 20)
noise_level = st.slider("Noise level", 0.0, 0.5, 0.1, 0.05)

# Generate synthetic data
X_train = np.linspace(-3, 3, n_points)
y_train = true_function(X_train) + np.random.normal(0, noise_level, n_points)
X_test = np.linspace(-4, 4, 100)

# Display the true function
st.subheader("True Function and Training Data")
fig_true, ax_true = plt.subplots(figsize=(10, 6))
ax_true.scatter(X_train, y_train, c='black', label='Training Data', alpha=0.6)
x_plot = np.linspace(-4, 4, 200)
ax_true.plot(x_plot, true_function(x_plot), 'r--', label='True Function', linewidth=2)
ax_true.grid(True, alpha=0.3)
ax_true.legend()
ax_true.set_xlabel('x')
ax_true.set_ylabel('y')
ax_true.set_title('True Function and Training Data')
st.pyplot(fig_true)

# Model configurations
col1, col2 = st.columns(2)

with col1:
    st.subheader("BNN Configuration")
    hidden_size = st.slider("Hidden layer size", 16, 128, 64, 16)
    n_layers = st.slider("Number of hidden layers", 1, 3, 2)
    prior_std = st.slider("Prior standard deviation", 0.1, 2.0, 1.0, 0.1)
    
    bnn_config = BNNConfig(
        hidden_sizes=[hidden_size] * n_layers,
        prior_std=prior_std,
        learning_rate=0.01,
        num_samples=100,
        epochs=2000
    )

with col2:
    st.subheader("GP Configuration")
    kernel_type = st.selectbox("Kernel type", ["rbf", "matern", "periodic"])
    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
    
    gp_config = GPConfig(
        kernel_type=kernel_type,
        learning_rate=learning_rate,
        n_iterations=100,
        noise_prior=noise_level
    )

if st.button("Train Models"):
    with st.spinner("Training models..."):
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
        
        # Train BNN
        bnn_model = BayesianNeuralNetwork(bnn_config)
        optimizer = torch.optim.Adam(bnn_model.parameters(), lr=bnn_config.learning_rate)
        
        progress_bar = st.progress(0)
        losses = []
        
        for epoch in range(bnn_config.epochs):
            optimizer.zero_grad()
            pred = bnn_model(X_train_tensor)
            loss = bnn_model.calculate_loss(pred, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if epoch % 100 == 0:
                progress_bar.progress(epoch / bnn_config.epochs)
        
        # Make BNN predictions
        with torch.no_grad():
            bnn_mean, bnn_std = bnn_model.predict(X_test_tensor)
            bnn_mean = bnn_mean.detach().numpy()
            bnn_std = bnn_std.detach().numpy()
            
            # For plotting, we only need mean and total std
            bnn_results = (bnn_mean, bnn_std, None, None)
        
        # Train GP
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = GPModel(X_train_tensor, y_train_tensor.squeeze(), likelihood, gp_config)
        
        gp_model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(gp_model.parameters(), lr=gp_config.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
        
        for i in range(gp_config.n_iterations):
            optimizer.zero_grad()
            output = gp_model(X_train_tensor)
            loss = -mll(output, y_train_tensor.squeeze())
            loss.backward()
            optimizer.step()
        
        # Make GP predictions
        gp_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = gp_model(X_test_tensor)
            y_pred = likelihood(f_pred)
            gp_mean = y_pred.mean.numpy()
            gp_std = y_pred.stddev.numpy()
        
        gp_results = (gp_mean, gp_std)
        
        # Plot results
        fig = plot_comparative_analysis(X_train, y_train, X_test, bnn_results, gp_results, true_function)
        st.pyplot(fig)
        
        # Plot training loss
        loss_fig = plot_training_loss(losses)
        st.pyplot(loss_fig)