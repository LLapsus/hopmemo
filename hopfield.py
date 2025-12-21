import numpy as np

##################################################
# Hopfield Network Class
##################################################

class HopfieldNetwork:
    def __init__(self, n_neurons=64, learning_method="hebbian", damped_lam=0.1, damped_centered=True, damped_zero_diagonal=False):
        # Number of neurons in the network
        self.n_neurons = n_neurons
        # Learning strategy for memorize: "hebbian", "storkey", "centered", or "damped"
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
        # Local biases
        self.theta_loc = np.zeros(n_neurons)
        # Parameters to calculate local biases
        self.beta = 0.5
        self.eps = 1e-3
        self.theta_clip = 3.0

    def reset_network(self):
        """
        Reset synaptic weights and stored memories to initial state.
        """
        # Reset stored memories
        self.memories = np.empty((0, self.n_neurons), dtype=int)
        self.memory_labels = np.empty((0,), dtype=object)
        # Reset weight matrix
        self.W = np.zeros((self.n_neurons, self.n_neurons))
        # Reset local biases
        self.theta_loc = np.zeros(self.n_neurons)
        
    def num_memories(self):
        """
        Return the number of patterns currently stored in the network.
        """
        return self.memories.shape[0]

    def memorize(self, patterns, labels=None):
        """
        Store patterns and update weights according to `learning_method`.
        - hebbian: incremental Hebbian update (adds outer products of new patterns).
        - storkey: incremental Storkey update for each new pattern.
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

        # Prepare labels
        if labels is None:
            label_arr = np.array([None] * num_new, dtype=object)
        else:
            label_arr = np.asarray(labels, dtype=object)
            if label_arr.shape != (num_new,):
                raise ValueError("labels must have length equal to number of patterns")

        # Store new memories
        if self.memories.size == 0:
            # No previous memories are stored
            self.memories = patterns.copy()        # Store new patterns
            self.memory_labels = label_arr.copy()  # Store corresponding labels
        else:
            # Append to existing memories
            self.memories = np.vstack([self.memories, patterns])             # Append new patterns
            self.memory_labels = np.hstack([self.memory_labels, label_arr])  # Append new labels
            
        # Recompute local biases
        self._update_theta_loc()

        # Update weights according to chosen learning method
        method = self.learning_method
        num_total = self.memories.shape[0]
        if method == "hebbian" or (num_total < 2 and method in {"centered", "damped"}):
            # Print note if falling back to Hebbian
            if method != "hebbian":
                print("Note: Only one pattern stored; falling back to Hebbian update.")
            # Hebbian: recompute weights from stored memories.
            self._hebbian(patterns)
        elif method == "storkey":
            self._storkey(patterns)
        elif method == "centered":
            # Centered pseudoinverse
            self._pseudoinverse_centered()
        elif method == "damped":
            # Damped pseudoinverse
            self._pseudoinverse_damped(
                lam=self.damped_lam,
                centered=self.damped_centered,
                zero_diagonal=self.damped_zero_diagonal,
            )
        else:
            raise ValueError(f"Unknown learning_method '{method}'. Use 'hebbian', 'centered', 'damped', or 'storkey'.")
        
    def _update_theta_loc(self):
        """
        Update local biases to be the pixel-wise mean of stored memories.
        If no memories are stored, biases are zero.
        """
        # Parameters for local bias calculation
        beta       = self.beta
        eps        = self.eps
        theta_clip = self.theta_clip
        
        if self.memories.size == 0:
            self.theta_loc = np.zeros(self.n_neurons)
        else:
            m = self.memories.mean(axis=0)                         # calculate mean
            m = np.clip(m, -1 + eps, 1 - eps)                      # avoid atanh blow-up
            b = -beta * np.arctanh(m)                              # external field
            self.theta_loc = np.clip(b, -theta_clip, theta_clip)   # optional safety cap
        
    def _hebbian(self, patterns):
        """
        Incremental Hebbian learning for the provided patterns.
        Assumes inputs were validated in `memorize`.
        """
        self.W += patterns.T @ patterns / self.n_neurons
        self.W = 0.5 * (self.W + self.W.T)  # Ensure symmetry of weights
        np.fill_diagonal(self.W, 0)         # Zero out self-connections
        
    def _storkey(self, patterns):
        """
        Incremental Storkey learning update for provided patterns.
        """
        for p in patterns:
            # Local fields excluding the ij contribution for each pair
            h = self.W @ p
            h_minus_j = h[:, None] - self.W * p               # h_i - w_ij * p_j
            h_minus_i = h[None, :] - (self.W.T * p[:, None])  # h_j - w_ji * p_i
            delta = (np.outer(p, p) - p[:, None] * h_minus_i - h_minus_j * p[None, :]) / self.n_neurons
            self.W += delta
        self.W = 0.5 * (self.W + self.W.T)  # Ensure symmetry of weights
        np.fill_diagonal(self.W, 0)         # Zero out self-connections
        
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

    def retrieve(self, pattern, theta=0., max_iterations=50, 
                 history=False, update_rule="async", use_local_biases=False, random_state=None,
                 verbose=False
                ):
        """
        Retrieve a stored pattern starting from an initial state.
        pattern: 1D numpy array of shape (n_neurons,) with values {+1, -1}.
        theta: uniform bias term added to each neuron's potential.
        max_iterations: maximum number of asynchronous update cycles.
        history: if True, return update history. For async, logs per-neuron updates; for
                 sync, logs iteration-level energy only.
        update_rule: 'async' (default) or 'sync' for asynchronous vs synchronous updates.
        use_local_biases: include neuron-specific biases (theta_loc) if True.
        random_state: seed for random number generator (for async updates).
        """
        # Validate pattern
        if pattern.shape != (self.n_neurons,) or not np.isin(pattern, (-1, 1)).all():
            raise ValueError("pattern must be a {+1, -1} vector with shape (n_neurons,)")
        if update_rule not in {"async", "sync"}:
            raise ValueError("update_rule must be 'async' or 'sync'")
                
        # Initialize state
        state = pattern.copy()
        
        # Initialize tracking lists
        if history:
            if update_rule == "async":
                # History for asynchronous updates: keep all per-neuron updates
                history = {
                    'iteration': [],
                    'neuron': [],
                    'value': [],
                    'energy': []
                }
                E0 = self.energy(state, theta, use_local_biases=use_local_biases)
                for i in range(self.n_neurons):
                    history['iteration'].append(0)     # Iteration number
                    history['neuron'].append(i)        # Neuron index
                    history['value'].append(state[i])  # Value of the neuron
                    history['energy'].append(E0)       # Energy of the network
            else:
                # History for synchronous updates: keep iteration-level energy only
                history = {
                    'iteration': [0],
                    'energy': [self.energy(state, theta, use_local_biases=use_local_biases)]
                }
        
        for iter in range(1, max_iterations+1):
            changed = False
            if update_rule == "async":
                # Asynchronous update: pick neurons in random order
                indices = np.arange(self.n_neurons)
                if random_state is None:
                    np.random.shuffle(indices)
                else:
                    rng = np.random.default_rng(random_state)
                    rng.shuffle(indices)
                np.random.shuffle(indices)

                for i in indices:
                    # Calculate internal potential of neuron
                    bias_i = self.theta_loc[i] if use_local_biases else 0.0
                    xi = np.dot(self.W[i, :], state) - theta - bias_i
                    if xi != 0:
                        new_state = 1 if xi > 0 else -1
                        if new_state != state[i]:
                            state[i] = new_state
                            changed = True
                    # Record the update
                    if history:
                        history['iteration'].append(iter)
                        history['neuron'].append(i)
                        history['value'].append(state[i])
                        history['energy'].append(self.energy(state, theta, use_local_biases=use_local_biases))
            else:
                # Synchronous update: compute all neuron updates from current state
                theta_term = self.theta_loc if use_local_biases else 0.0
                potentials = self.W @ state - theta - theta_term
                new_state = np.where(potentials > 0, 1, np.where(potentials < 0, -1, state))
                changed = not np.array_equal(new_state, state)
                state = new_state
                if history:
                    history['iteration'].append(iter)
                    history['energy'].append(self.energy(state, theta, use_local_biases=use_local_biases))
        
            # If no changes occurred during the iteration, the dynamics have converged
            if not changed:
                if verbose:
                    print(f"Converged after {iter:d} iterations.")
                break

        # Return final state and retrieval history if requested
        if history:
            return state, history
        else:
            return state

    def energy(self, state, theta=0., use_local_biases=False):
        """
        Compute the Hopfield energy of a given state using:
            E(s) = -1/2 * s^T * W * s + theta * sum(s) + theta_loc dot s
        where s is the state vector, W is the weight matrix, theta is the uniform bias,
        and theta_loc are neuron-specific biases.
        """
        theta_term = np.dot(self.theta_loc, state) if use_local_biases else 0.0
        return -.5 * np.dot(state, self.W.dot(state)) + theta * np.sum(state) + theta_term

    def weights(self):
        """
        Return the current weight matrix of the Hopfield network.
        """
        return self.W
    
    def check_stability(self, theta=0., use_local_biases=False):
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
        theta_term = self.theta_loc if use_local_biases else 0.0
        for p, label in zip(self.memories, labels):
            h = self.W @ p
            margin = (p * (h - theta - theta_term)).min()
            result[label] = margin
        return result

    def nearest_memory(self, pattern, metric="hamming"):
        """
        Find the closest stored memory to the given pattern.
        metric: 'hamming' (default) minimizes bit flips; 'overlap' maximizes dot product.
        Returns (memory, label, score) where score is distance for hamming
        and overlap value for overlap.
        """
        pattern = np.asarray(pattern, dtype=int)
        if pattern.shape != (self.n_neurons,) or not np.isin(pattern, (-1, 1)).all():
            raise ValueError("pattern must be a {+1, -1} vector with shape (n_neurons,)")
        if self.memories.size == 0:
            raise ValueError("no memories stored")

        labels = self.memory_labels
        if labels.shape[0] != self.memories.shape[0]:
            labels = np.resize(labels, (self.memories.shape[0],))

        if metric == "hamming":
            # Hamming distance: count differing bits
            scores = (self.memories != pattern).sum(axis=1)  # lower is closer
            best_idx = int(np.argmin(scores))
        elif metric == "overlap":
            # Overlap: dot product
            scores = self.memories @ pattern  # higher is closer
            best_idx = int(np.argmax(scores))
        else:
            raise ValueError("metric must be 'hamming' or 'overlap'")

        best_mem = self.memories[best_idx]
        best_label = labels[best_idx]
        best_score = scores[best_idx]
        return best_mem, best_label, best_score
