import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
pyro.set_rng_seed(42)


class DampedOscillatorSimulator:
    def __init__(self, mass=1.0, k=1.0, gamma=0.1, dt=0.1, noise_std=0.1):
        """
        Initialize the damped harmonic oscillator simulator with PyTorch tensors.

        Parameters:
            mass (float): mass of the oscillator
            k (float): spring constant
            gamma (float): damping coefficient
            dt (float): time step
            noise_std (float): standard deviation of measurement noise
        """
        self.mass = torch.tensor(float(mass))
        self.k = torch.tensor(float(k))
        self.gamma = torch.tensor(float(gamma))
        self.dt = torch.tensor(float(dt))
        self.noise_std = torch.tensor(float(noise_std))

    def simulate(self, t_max=10.0, initial_position=1.0, initial_velocity=0.0):
        """
        Simulate the damped oscillator.

        Parameters:
            t_max (float): maximum simulation time
            initial_position (float): starting position
            initial_velocity (float): starting velocity

        Returns:
            tuple: (times, noisy_positions)
        """
        # Convert inputs to tensors
        t_max = torch.tensor(float(t_max))
        initial_position = torch.tensor(float(initial_position))
        initial_velocity = torch.tensor(float(initial_velocity))

        # Calculate number of steps and create time array
        n_steps = int(t_max / self.dt)
        times = torch.linspace(0, t_max, n_steps)

        # Calculate natural frequency and damping
        omega0 = torch.sqrt(self.k / self.mass)
        damping = self.gamma / (2 * self.mass)
        # Add a small constant for numerical stability
        omega = torch.sqrt(omega0**2 - damping**2 + 1e-6)

        # Compute the analytical solution (assume initial phase = 0)
        positions = initial_position * torch.exp(-damping * times) * torch.cos(omega * times)

        # Add Gaussian noise
        noise = torch.normal(0.0, self.noise_std, positions.shape)
        noisy_positions = positions + noise

        return times, noisy_positions


def model(times, observations=None):
    """
    Probabilistic model for the damped oscillator.
    
    Parameters:
        times (torch.Tensor): time points
        observations (torch.Tensor, optional): observed positions
    """
    # Prior distributions for parameters
    mass = pyro.sample('mass', dist.LogNormal(torch.tensor(0.0), torch.tensor(0.5)))
    k = pyro.sample('k', dist.LogNormal(torch.tensor(0.0), torch.tensor(0.5)))
    gamma = pyro.sample('gamma', dist.LogNormal(torch.tensor(-2.0), torch.tensor(0.5)))
    noise = pyro.sample('noise', dist.LogNormal(torch.tensor(-2.0), torch.tensor(0.5)))
    
    # Create simulator with sampled parameters
    simulator = DampedOscillatorSimulator(
        mass=mass.item(),
        k=k.item(),
        gamma=gamma.item(),
        noise_std=noise.item()
    )
    
    # Simulate the oscillator using the sampled parameters
    _, positions = simulator.simulate(t_max=times[-1].item())
    
    # Only condition on observed data if observations are provided
    if observations is not None:
        with pyro.plate('data', len(observations)):
            pyro.sample('obs', dist.Normal(positions, noise), obs=observations)
    
    return positions



def guide(times, observations=None):
    """
    Variational guide (approximate posterior) for the model.
    """
    # Variational parameters for mass
    mass_loc = pyro.param('mass_loc', torch.tensor(1.0))
    mass_scale = pyro.param('mass_scale', torch.tensor(0.1),
                              constraint=dist.constraints.positive)

    # Variational parameters for k
    k_loc = pyro.param('k_loc', torch.tensor(1.0))
    k_scale = pyro.param('k_scale', torch.tensor(0.1),
                         constraint=dist.constraints.positive)

    # Variational parameters for gamma
    gamma_loc = pyro.param('gamma_loc', torch.tensor(0.1))
    gamma_scale = pyro.param('gamma_scale', torch.tensor(0.1),
                             constraint=dist.constraints.positive)

    # Variational parameters for noise
    noise_loc = pyro.param('noise_loc', torch.tensor(0.1))
    noise_scale = pyro.param('noise_scale', torch.tensor(0.1),
                             constraint=dist.constraints.positive)

    # Sample from the variational distributions
    pyro.sample('mass', dist.LogNormal(mass_loc, mass_scale))
    pyro.sample('k', dist.LogNormal(k_loc, k_scale))
    pyro.sample('gamma', dist.LogNormal(gamma_loc, gamma_scale))
    pyro.sample('noise', dist.LogNormal(noise_loc, noise_scale))


def train_vi(model, guide, times, observations, num_iterations=1000):
    """
    Train the variational inference model.

    Parameters:
        model: probabilistic model function
        guide: variational guide function
        times (torch.Tensor): time points
        observations (torch.Tensor): observed positions
        num_iterations (int): number of training iterations

    Returns:
        list: training losses
    """
    pyro.clear_param_store()

    adam = Adam({"lr": 0.01})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    losses = []
    for _ in tqdm(range(num_iterations)):
        loss = svi.step(times, observations)
        losses.append(loss)

    return losses


def plot_results(times, true_positions, observations, inferred_positions, losses, noise_std):
    """
    Plot the results of the inference.

    Parameters:
        times (torch.Tensor): time points
        true_positions (torch.Tensor): noise-free simulated positions
        observations (torch.Tensor): noisy observed positions
        inferred_positions (torch.Tensor): inferred positions from the model
        losses (list): training loss over iterations
        noise_std (float): noise standard deviation (used for error bands)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Convert tensors to numpy arrays for plotting
    times_np = times.numpy()
    true_positions_np = true_positions.numpy()
    observations_np = observations.numpy()
    inferred_positions_np = inferred_positions.numpy()

    # Plot trajectories
    ax1.plot(times_np, true_positions_np, 'g-', label='True trajectory')
    ax1.plot(times_np, observations_np, 'k.', alpha=0.3, label='Noisy observations')
    ax1.plot(times_np, inferred_positions_np, 'r--', label='Inferred trajectory')

    # Plot 2σ error bands
    ax1.fill_between(times_np,
                     inferred_positions_np - 2 * noise_std,
                     inferred_positions_np + 2 * noise_std,
                     color='r', alpha=0.2)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()

    # Plot training loss
    ax2.plot(losses)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')

    plt.tight_layout()
    return fig


def plot_corner(samples):
    """
    Create a corner plot of the posterior distributions of the parameters.

    Parameters:
        samples (list of dict): each dict contains posterior samples for 'mass', 'k', 'gamma', and 'noise'
    """
    # Convert the samples to a NumPy array
    samples_np = np.array([
        [s['mass'].item(), s['k'].item(), s['gamma'].item(), s['noise'].item()]
        for s in samples
    ])

    fig = corner.corner(
        samples_np,
        labels=['mass', 'k', 'gamma', 'noise'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    return fig


def main():
    """
    Main function to run simulation and inference.
    """
    # Generate synthetic data using true parameters
    true_simulator = DampedOscillatorSimulator(
        mass=1.2,
        k=0.8,
        gamma=0.15,
        noise_std=0.1
    )
    times, observations = true_simulator.simulate()

    # For plotting the true (noise-free) trajectory, simulate with zero noise
    simulator_no_noise = DampedOscillatorSimulator(
        mass=1.2,
        k=0.8,
        gamma=0.15,
        noise_std=0.0
    )
    _, noise_free_positions = simulator_no_noise.simulate()

    # Train the model using variational inference
    losses = train_vi(model, guide, times, observations)

    # Generate posterior samples using a single Predictive call
    num_samples = 1000
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    posterior = predictive(times, observations)

    # Reconstruct the posterior sample list in a convenient format for the corner plot
    posterior_samples = [{
        'mass': posterior['mass'][i],
        'k': posterior['k'][i],
        'gamma': posterior['gamma'][i],
        'noise': posterior['noise'][i]
    } for i in range(num_samples)]

    # Get the inferred positions from the model (using the current parameter estimates)
    with torch.no_grad():
        inferred_positions = model(times)

    # Plot the results and the posterior distributions
    fig1 = plot_results(times, noise_free_positions, observations, inferred_positions, losses,
                        noise_std=true_simulator.noise_std.item())
    fig2 = plot_corner(posterior_samples)

    plt.show()


if __name__ == "__main__":
    main()
