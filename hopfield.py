import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import colors

##################################################
# Utility functions
##################################################

def sign(x):
    """Sign function returning +1 or -1."""
    
    return np.where(x >= 0, 1, -1)

def load_patterns(filename="patterns8.csv"):
    """
    Reads a CSV file containing one pattern per row.
    The first column is 'name', followed by columns pixel_0..pixel_63.
    Returns a dictionary: {name: np.array([...])}.
    """
    
    patterns_dict = {}
    
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip the header row
        header = next(reader)
        
        for row in reader:
            name = row[0]  
            # Convert the next 64 columns from strings to integers
            pixels = [int(value) for value in row[1:1+64]]
            
            # Store in the dictionary
            patterns_dict[name] = np.array(pixels, dtype=int)
    
    return patterns_dict

def display_pattern(pattern, title=None, cmap=colors.ListedColormap(["lightblue", "orange"]), figname=None, 
                    width=600, height=600, dpi=96):
    """
    Display the 8x8 pattern as a heatmap.
    pattern: 1D or 2D numpy array of shape (64,) or (8,8) with values in {+1, -1}.
    """

    # Reshape the input image
    if pattern.ndim == 1:
        pattern = pattern.reshape(8, 8)

    # Set dimensions divisible by 2
    figsize = (width / dpi, height / dpi)

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.imshow(pattern, cmap=cmap, vmin=-1, vmax=1)
    
    if title:
        ax.set_title(title)    
    plt.axis("off")

    if figname:
        # Save figure
        plt.savefig(figname, format="png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def flatten_pattern(pattern_2d):
    """Convert an 8x8 pattern (with 1 or -1) to a 1D array of length 64."""
    
    return pattern_2d.reshape(-1)

def generate_noisy_pattern(pattern, noise_level=0.1):
    """
    Flip a fraction of pixels in a pattern to generate noise.
    noise_level: fraction of pixels to flip (0.0 - 1.0).
    """
    
    noisy_pattern = pattern.copy()
    n_to_flip = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), n_to_flip, replace=False)
    noisy_pattern[flip_indices] *= -1
    
    return noisy_pattern

##################################################
# Hopfield Network Class
##################################################

class HopfieldNetwork:
    def __init__(self, n_neurons=64):
        self.n_neurons = n_neurons
        # Initialize weight matrix
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        """
        Train the network using Hebbian learning.
        patterns: list of 1D numpy arrays of shape (n_neurons,) with values {+1, -1}.
        """
        
        # Reset weights
        self.W = np.zeros((self.n_neurons, self.n_neurons))

        for p in patterns:
            # Outer product
            self.W += np.outer(p, p)

        # Zero out the diagonal
        np.fill_diagonal(self.W, 0)

        # Normalize weights by number of patterns
        # self.W /= len(patterns)

    def retrieve(self, pattern, max_iterations=50):
        """
        Retrieve a stored pattern starting from an initial state.
        pattern: 1D numpy array of shape (n_neurons,) with values {+1, -1}.
        max_iterations: maximum number of asynchronous update cycles.
        """
        
        state = pattern.copy()

        for _ in range(max_iterations):
            # Asynchronous update: pick neurons in random order
            indices = np.arange(self.n_neurons)
            np.random.shuffle(indices)

            changed = False
            for i in indices:
                # Calculate internal potetial of neuron
                xi = np.dot(self.W[i, :], state)
                # Set new state
                new_state = sign(xi)
                if new_state != state[i]:
                    state[i] =  new_state
                    changed = True

            if not changed:
                break

        return state

    def retrieve_with_history(self, pattern, max_iterations=50):
        """
        Similar to `retrieve`, but also returns the entire history (list of states)
        as the network iterates.
        
        Returns:
            history: List of states (numpy arrays). ``history[0]`` is the
                initial state and ``history[i]`` (``i > 0``) is the state after
                completing iteration ``i`` of asynchronous updates.
        """
        
        state = pattern.copy()    # Input state
        history = [state.copy()]  # State evolution

        for _ in range(max_iterations):
            # Asynchronous update: pick neurons in random order
            indices = np.arange(self.n_neurons)
            np.random.shuffle(indices)

            changed = False
            for i in indices:
                # Calculate internal potential
                xi = np.dot(self.W[i, :], state)
                # Set new state
                new_state = sign(xi)
                if new_state != state[i]:
                    changed = True
                    state[i] = new_state

            # Append state after finishing asynchronous updates for this
            # iteration. This keeps ``history`` aligned with the number of
            # completed update cycles.
            history.append(state.copy())

            # If no value changes, finish the training process
            if not changed:
                break

        return history

    def energy(self, state):
        """
        Compute the Hopfield energy of a given state using:
            E(s) = -1/2 * s^T * W * s
        """
        
        return -.5 * np.dot(state, self.W.dot(state))

    def get_weights(self):
        """
        Return the current weight matrix of the Hopfield network.
        """
        
        return self.W
