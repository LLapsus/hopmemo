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
    
    if x > 0:  return 1
    if x < 0:  return -1
    return x

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

        # Normalize weights by number of neurons
        self.W /= self.n_neurons

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
