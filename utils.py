import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hopfield import HopfieldNetwork

##################################################
# Hopfield Utility Functions
##################################################

def print_stability_status(hop, theta=0., use_local_biases=True):
    """
    Prints the stability status of each stored memory in the Hopfield network.
    Parameters:
    hop : (HopfieldNetwork) An instance of the Hopfield network.
    """
    stability = hop.check_stability(theta=theta, use_local_biases=use_local_biases)
    for label, margin in stability.items():
        status = "Stable" if margin >= 0 else "Unstable"
        print(f"Pattern {label}: {status} (margin = {margin:.6f})")

def compare_patterns(inp, out):
    """
    Compares the input and output patterns of the Hopfield network by plotting them side by side.
    Parameters:
    inp : (np.ndarray) The input pattern.
    out : (np.ndarray) The output pattern retrieved by the Hopfield network.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(inp.reshape(28,28), cmap='gray', cbar=False, ax=axs[0]) 
    sns.heatmap(out.reshape(28,28), cmap='gray', cbar=False, ax=axs[1])
    # Set titles
    axs[0].set_title("Input Pattern")
    axs[1].set_title("Retrieved Pattern")
    # Remove ticks
    for i in range(2):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()
        
def plot_energy(history):
    """
    Plots the energy of the Hopfield network over the retrieval process.
    Parameters:
    history : (dict) A dictionary containing the history of the retrieval process,
                     including 'iteration' and 'energy' lists.
    """ 
    # Determine number of features from history
    num_features = np.sum(np.array(history['iteration']) == 0)
        
    # Energy (skip initial state)
    energy = np.array(history['energy'][num_features:])
    step = np.arange(len(energy))
    
    # Define iteration boundaries
    iter = [num_features * i for i in range(len(energy) // num_features + 1)]

    # Plot energy over steps
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(x=step, y=energy, ax=ax)

    # Mark iteration boundaries
    ax.vlines(iter, 0, 1, transform=ax.get_xaxis_transform(), linestyle="--", color="r")

    ax.set_title("Energy of the Hopfield network", size=16)
    ax.set_xlabel("Step", size=14)
    ax.set_ylabel("Energy", size=14)

    plt.show()
    
def plot_weight_matrix(hop):
    """
    Plots the weight matrix of the Hopfield network.
    Parameters:
    hop : (HopfieldNetwork) An instance of the Hopfield network.
    """
    W = hop.weights()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(W, cmap='viridis')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
def add_noise(pattern, p=0.1):
    """
    Add noise to a binary pattern by flipping each bit with probability p.
    """
    noisy_pattern = pattern.copy()
    n_flip = int(p * pattern.size)
    flip_indices = np.random.choice(pattern.size, size=n_flip, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

def hide_pattern(pattern):
    """
    Hide a fraction of the pattern by setting those bits to zero.
    """
    hidden_pattern = pattern.copy().reshape(28,28)
    hidden_pattern[:,0:14] = -1  # Hide top half of the pattern
    return hidden_pattern.reshape(-1)