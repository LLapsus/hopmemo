import numpy as np

##################################################
# Hopfield Network Class
##################################################

class HopfieldNetwork:
    def __init__(self, n_neurons=64, learning_method="hebbian", damped_lam=0.1, damped_centered=True, damped_zero_diagonal=False):
        # Number of neurons in the network
        self.n_neurons = n_neurons
        # Learning strategy for memorize: "hebbian", "centered", or "damped"
        self.learning_method = learning_method            # Method for weight update
        self.damped_lam = damped_lam                      # Damping factor for damped pseudoinverse
        self.damped_centered = damped_centered            # Centering option for damped pseudoinverse
        self.damped_zero_diagonal = damped_zero_diagonal  # Zero diagonal option for damped pseudoinverse
        # Initialize weight matrix
        self.W = np.zeros((n_neurons, n_neurons))         # Weight matrix initialized to zero
        # Stored patterns remembered through memorize
        self.memories = np.empty((0, n_neurons), dtype=int)
        # Optional labels for stored patterns (aligned with memories rows)
        self.memory_labels = np.empty((0,), dtype=object)

    def reset_network(self):
        """
        Reset synaptic weights and stored memories to initial state.
        """
        # Reset stored memories
        self.memories = np.empty((0, self.n_neurons), dtype=int)
        self.memory_labels = np.empty((0,), dtype=object)
        # Reset weight matrix
        self.W = np.zeros((self.n_neurons, self.n_neurons))
        
    def num_memories(self):
        """
        Return the number of patterns currently stored in the network.
        """
        return self.memories.shape[0]

    def memorize(self, patterns, labels=None):
        """
        Store patterns and update weights according to `learning_method`.
        - hebbian: incremental Hebbian update (adds outer products of new patterns).
        - centered: recompute weights via centered pseudoinverse on all memories.
        - damped: recompute weights via damped pseudoinverse on all memories.
        Can be called repeatedly; new patterns are appended to `self.memories`.

        patterns: array-like of shape (num_patterns, n_neurons) or iterable of
            1D arrays with values {+1, -1}.
        labels: optional array-like of length num_patterns with names for each pattern.
        """
        patterns = np.asarray(patterns, dtype=int)
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)
        if patterns.ndim != 2 or patterns.shape[1] != self.n_neurons:
            raise ValueError("patterns must be shape (num_patterns, n_neurons)")
        if not np.isin(patterns, (-1, 1)).all():
            raise ValueError("patterns must be {+1, -1}")
        num_new = patterns.shape[0]

        if labels is None:
            label_arr = np.array([None] * num_new, dtype=object)
        else:
            label_arr = np.asarray(labels, dtype=object)
            if label_arr.shape != (num_new,):
                raise ValueError("labels must have length equal to number of patterns")

        # Store new memories
        if self.memories.size == 0:
            self.memories = patterns.copy()
            self.memory_labels = label_arr.copy()
        else:
            self.memories = np.vstack([self.memories, patterns])
            self.memory_labels = np.hstack([self.memory_labels, label_arr])

        # Update weights according to chosen learning method
        method = self.learning_method
        num_total = self.memories.shape[0]
        # Hebbian update
        if method == "hebbian" or num_total < 2:
            # Print note if falling back to Hebbian
            if method != "hebbian":
                print("Note: Only one pattern stored; falling back to Hebbian update.")
            # Add contributions from new patterns; keep previously memorized weights.
            self.W += patterns.T @ patterns / self.n_neurons
            np.fill_diagonal(self.W, 0)
        # Centered pseudoinverse
        elif method == "centered":
            self._pseudoinverse_centered()
        # Damped pseudoinverse
        elif method == "damped":
            self._pseudoinverse_damped(
                lam=self.damped_lam,
                centered=self.damped_centered,
                zero_diagonal=self.damped_zero_diagonal,
            )
        else:
            raise ValueError(f"Unknown learning_method '{method}'. Use 'hebbian', 'centered', or 'damped'.")
        
    def _pseudoinverse_centered(self):
        """
        Train using centered pseudoinverse learning on all stored memories.
        Assumes inputs were validated in `memorize`.
        """
        patterns = self.memories
        P = patterns.T  # shape (n_neurons, num_patterns)
        mean = P.mean(axis=1, keepdims=True)
        centered = P - mean
        self.W = centered @ np.linalg.pinv(centered.T @ centered) @ centered.T
        np.fill_diagonal(self.W, 0)

    def _pseudoinverse_damped(self, lam=0.1, centered=True, zero_diagonal=False):
        """
        Train using a damped pseudoinverse on all stored memories.
        lam: Tikhonov damping parameter (>= 0).
        centered: subtract pixel-wise mean before computing weights.
        zero_diagonal: if True, zero out the diagonal after training.
        """
        patterns = self.memories
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

    def get_weights(self):
        """
        Return the current weight matrix of the Hopfield network.
        """
        return self.W
    
    def check_stability(self):
        """
        Compute margin for each stored pattern.
        Returns a dict mapping label -> margin (None used if label missing).
        """
        if self.memories.size == 0:
            return {}

        labels = self.memory_labels
        if labels.shape[0] != self.memories.shape[0]:
            # Defensive: align length if somehow mismatched
            labels = np.resize(labels, (self.memories.shape[0],))

        result = {}
        for p, label in zip(self.memories, labels):
            h = self.W @ p
            margin = (p * h).min()
            result[label] = margin
        return result
