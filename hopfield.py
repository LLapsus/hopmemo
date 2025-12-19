import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import colors

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
        # Validate patterns
        if any((p.shape != (self.n_neurons,) or not np.isin(p, (-1, 1)).all()) for p in patterns):
            raise ValueError("patterns must be a list of {+1, -1} vectors with shape (n_neurons,)")
        
        # Reset weights
        self.W = np.zeros((self.n_neurons, self.n_neurons))

        for p in patterns:
            # Outer product
            self.W += np.outer(p, p)

        # Zero out the diagonal
        np.fill_diagonal(self.W, 0)

        # Normalize weights by number of neurons
        self.W /= self.n_neurons
        
    def train_pseudoinverse_centered(self, patterns):
        """
        Train using centered pseudoinverse learning to handle correlated patterns.
        patterns: array-like of shape (num_patterns, n_neurons) with values {+1, -1}.
        """
        patterns = np.asarray(patterns)
        if patterns.ndim != 2 or patterns.shape[1] != self.n_neurons:
            raise ValueError("patterns must be shape (num_patterns, n_neurons)")
        if not np.isin(patterns, (-1, 1)).all():
            raise ValueError("patterns must be {+1, -1}")

        P = patterns.T  # shape (n_neurons, num_patterns)
        mean = P.mean(axis=1, keepdims=True)
        centered = P - mean
        self.W = centered @ np.linalg.pinv(centered.T @ centered) @ centered.T
        np.fill_diagonal(self.W, 0)

    def train_pseudoinverse_damped(self, patterns, lam=0.1, centered=True, zero_diagonal=False):
        """
        Train using a damped pseudoinverse, optionally centered, to improve stability
        on correlated patterns.
        lam: Tikhonov damping parameter (>= 0).
        centered: subtract pixel-wise mean before computing weights.
        zero_diagonal: if True, zero out the diagonal after training.
        """
        patterns = np.asarray(patterns)
        if patterns.ndim != 2 or patterns.shape[1] != self.n_neurons:
            raise ValueError("patterns must be shape (num_patterns, n_neurons)")
        if not np.isin(patterns, (-1, 1)).all():
            raise ValueError("patterns must be {+1, -1}")
        if lam < 0:
            raise ValueError("lam must be non-negative")

        P = patterns.T  # shape (n_neurons, num_patterns)
        X = P - P.mean(axis=1, keepdims=True) if centered else P
        gram = X.T @ X
        gram_damped = gram + lam * np.eye(gram.shape[0])
        self.W = X @ np.linalg.inv(gram_damped) @ X.T
        if zero_diagonal:
            np.fill_diagonal(self.W, 0)

    def retrieve(self, pattern, max_iterations=50):
        """
        Retrieve a stored pattern starting from an initial state.
        pattern: 1D numpy array of shape (n_neurons,) with values {+1, -1}.
        max_iterations: maximum number of asynchronous update cycles.
        """
        # Validate pattern
        if pattern.shape != (self.n_neurons,) or not np.isin(pattern, (-1, 1)).all():
            raise ValueError("pattern must be a {+1, -1} vector with shape (n_neurons,)")
        
        state = pattern.copy()
        
        # List of neuron indices for asynchronous updates
        indices = np.arange(self.n_neurons)

        for iter in range(max_iterations):
            # Asynchronous update: pick neurons in random order
            np.random.shuffle(indices)

            changed = False
            for i in indices:
                # Calculate internal potetial of neuron
                xi = np.dot(self.W[i, :], state)
                if xi != 0:
                    new_state = 1 if xi > 0 else -1
                    if new_state != state[i]:
                        state[i] =  new_state
                        changed = True

            if not changed:
                print(f"Converged after {iter:d} iterations.")
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
        # Validate pattern
        if pattern.shape != (self.n_neurons,) or not np.isin(pattern, (-1, 1)).all():
            raise ValueError("pattern must be a {+1, -1} vector with shape (n_neurons,)")
        
        state = pattern.copy()    # Input state
        history = [state.copy()]  # State evolution
        
        # List of neuron indices for asynchronous updates
        indices = np.arange(self.n_neurons)

        for iter in range(max_iterations):
            # Asynchronous update: pick neurons in random order
            np.random.shuffle(indices)

            changed = False
            for i in indices:
                # Calculate internal potetial of neuron
                xi = np.dot(self.W[i, :], state)
                if xi != 0:
                    new_state = 1 if xi > 0 else -1
                    if new_state != state[i]:
                        changed = True
                        state[i] = new_state

            # Append state after finishing asynchronous updates for this
            # iteration. This keeps ``history`` aligned with the number of
            # completed update cycles.
            history.append(state.copy())

            # If no value changes, finish the training process
            if not changed:
                print(f"Converged after {iter:d} iterations.")
                break

        return history

    def energy(self, state):
        """
        Compute the Hopfield energy of a given state using:
            E(s) = -1/2 * s^T * W * s
        """
        
        return -.5 * np.dot(state, self.W.dot(state))

    def get_margins(self, patterns):
        """
        Calculate margins for each pattern.
        A pattern is fixed point under the current weights, if the margin is non-negative.

        Returns a list of min margin for each pattern.
        """
        results = []
        for p in patterns:
            if p.shape != (self.n_neurons,) or not np.isin(p, (-1, 1)).all():
                raise ValueError("patterns must be {+1, -1} vectors with shape (n_neurons,)")
            h = self.W @ p
            margin = (p * h).min()
            results.append(margin)
        return results

    def get_weights(self):
        """
        Return the current weight matrix of the Hopfield network.
        """
        return self.W
