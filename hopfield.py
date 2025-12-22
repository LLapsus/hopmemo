import numpy as np

##################################################
# Hopfield Network Class
##################################################

class HopfieldNetwork:
    def __init__(self, n_neurons=64, learning_method="hebbian", damped_lam=0.1, damped_centered=True, damped_zero_diagonal=True):
        # Number of neurons in the network
        self.n_neurons = n_neurons
        # Learning strategy for memorize: "hebbian", "storkey", "pinv_centered", or "pinv_damped"
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
        - hebbian: recompute Hebbian weights from all stored memories.
        - storkey: recompute Storkey weights from all stored memories.
        - pinv_centered: recompute weights via centered pseudoinverse on all memories.
        - pinv_damped: recompute weights via damped pseudoinverse on all memories.
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
        num_memories = self.memories.shape[0]
        if method == "hebbian":
            # Hebbian: recompute weights from stored memories.
            self._hebbian()
        elif method == "storkey":
            self._storkey()
        elif method == "pinv_centered":
            if num_memories < 2:
                # If total number of memories < 2, fall back to Hebbian
                print("Note: Only one pattern stored; falling back to Hebbian update.")
                self._hebbian()
            else:
                # Centered pseudoinverse
                self._pseudoinverse_centered()
        elif method == "pinv_damped":
            if num_memories < 2:
                # If total number of memories < 2, fall back to Hebbian
                print("Note: Only one pattern stored; falling back to Hebbian update.")
                self._hebbian()
            else:
                # Damped pseudoinverse
                self._pseudoinverse_damped(
                    lam=self.damped_lam,
                    centered=self.damped_centered,
                )
        else:
            # Invalid learning method
            raise ValueError(f"Unknown learning_method '{method}'. Use 'hebbian', 'storkey', 'pinv_centered', or 'pinv_damped'.")
        
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
        
    def _hebbian(self, center=True):
        """
        Hebbian learning recomputed from all stored memories.
        If `center` is True, subtract the mean of each neuron before training.
        """
        if self.memories.size == 0:
            self.W = np.zeros((self.n_neurons, self.n_neurons))
            return

        patterns = self.memories.astype(float)
        # For a single pattern, centering would zero out the data; skip centering in that case.
        if center and patterns.shape[0] > 1:
            X = patterns - patterns.mean(axis=0, keepdims=True)
        else:
            X = patterns
        self.W = X.T @ X / self.n_neurons   # Hebbian weight update
        self.W = 0.5 * (self.W + self.W.T)  # Ensure symmetry of weights
        np.fill_diagonal(self.W, 0)         # Zero out self-connections
        
    def _storkey(self):
        """
        Storkey learning recomputed from all stored memories.
        """
        if self.memories.size == 0:
            self.W = np.zeros((self.n_neurons, self.n_neurons))
            return

        self.W = np.zeros((self.n_neurons, self.n_neurons))
        for p in self.memories:
            h = self.W @ p
            delta = (np.outer(p, p) - np.outer(p, h) - np.outer(h, p)) / self.n_neurons
            self.W += delta
        self.W = 0.5 * (self.W + self.W.T)  # Ensure symmetry of weights
        np.fill_diagonal(self.W, 0)         # Zero out self-connections
        
    def _pseudoinverse_centered(self):
        """
        Train using centered pseudoinverse learning on all stored memories.
        Assumes inputs were validated in `memorize`.
        """
        patterns = self.memories.astype(float)
        P = patterns.T  # shape (n_neurons, num_patterns)
        mean = P.mean(axis=1, keepdims=True)
        centered = P - mean
        self.W = centered @ np.linalg.pinv(centered.T @ centered) @ centered.T
        np.fill_diagonal(self.W, 0)

    def _pseudoinverse_damped(self, lam=0.1, centered=True):
        """
        Train using a damped pseudoinverse on all stored memories.
        lam: Tikhonov damping parameter (>= 0).
        centered: subtract pixel-wise mean before computing weights.
        """
        patterns = self.memories.astype(float)
        if lam < 0:
            raise ValueError("lam must be non-negative")

        P = patterns.T  # shape (n_neurons, num_patterns)
        X = P - P.mean(axis=1, keepdims=True) if centered else P
        gram = X.T @ X
        gram_damped = gram + lam * np.eye(gram.shape[0])
        self.W = X @ np.linalg.solve(gram_damped, X.T)
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
        pattern = np.asarray(pattern, dtype=int)
        if pattern.shape != (self.n_neurons,) or not np.isin(pattern, (-1, 1)).all():
            raise ValueError("pattern must be a {+1, -1} vector with shape (n_neurons,)")
        if update_rule not in {"async", "sync"}:
            raise ValueError("update_rule must be 'async' or 'sync'")
                
        # Initialize state
        state = pattern.copy()
        
        # Initialize tracking lists
        track_history = bool(history)
        history_data = None
        if track_history:
            if update_rule == "async":
                # History for asynchronous updates: keep all per-neuron updates
                history_data = {
                    'iteration': [],
                    'neuron': [],
                    'value': [],
                    'energy': []
                }
                E0 = self.energy(state, theta, use_local_biases=use_local_biases)
                for i in range(self.n_neurons):
                    history_data['iteration'].append(0)     # Iteration number
                    history_data['neuron'].append(i)        # Neuron index
                    history_data['value'].append(state[i])  # Value of the neuron
                    history_data['energy'].append(E0)       # Energy of the network
            else:
                # History for synchronous updates: keep iteration-level energy only
                history_data = {
                    'iteration': [0],
                    'energy': [self.energy(state, theta, use_local_biases=use_local_biases)]
                }
        
        rng = np.random.default_rng(random_state) if random_state is not None else None
        for iteration in range(1, max_iterations + 1):
            changed = False
            if update_rule == "async":
                # Asynchronous update: pick neurons in random order
                indices = np.arange(self.n_neurons)
                if rng is None:
                    np.random.shuffle(indices)
                else:
                    rng.shuffle(indices)

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
                    if track_history:
                        history_data['iteration'].append(iteration)
                        history_data['neuron'].append(i)
                        history_data['value'].append(state[i])
                        history_data['energy'].append(self.energy(state, theta, use_local_biases=use_local_biases))
            else:
                # Synchronous update: compute all neuron updates from current state
                theta_term = self.theta_loc if use_local_biases else 0.0
                potentials = self.W @ state - theta - theta_term
                new_state = np.where(potentials > 0, 1, np.where(potentials < 0, -1, state))
                changed = not np.array_equal(new_state, state)
                state = new_state
                if track_history:
                    history_data['iteration'].append(iteration)
                    history_data['energy'].append(self.energy(state, theta, use_local_biases=use_local_biases))
        
            # If no changes occurred during the iteration, the dynamics have converged
            if not changed:
                if verbose:
                    print(f"Converged after {iteration:d} iterations.")
                break

        # Return final state and retrieval history if requested
        return (state, history_data) if track_history else state

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
    
    def biases(self):
        """
        Return the current local biases of the Hopfield network.
        """
        return self.theta_loc
    
    #-------------------------------- Stability Checking -------------------------------
    
    def check_stability(self, theta=0., use_local_biases=False):
        """
        Compute margin for each stored pattern.
        Returns a dict mapping label -> margin (None used if label missing).
        """
        if self.memories.size == 0:
            return {}

        labels = self.memory_labels
        if labels.shape[0] != self.memories.shape[0]:
            # Check align length if somehow mismatched
            labels = np.resize(labels, (self.memories.shape[0],))

        result = {}
        theta_term = self.theta_loc if use_local_biases else 0.0
        for p, label in zip(self.memories, labels):
            h = self.W @ p
            margin = (p * (h - theta - theta_term)).min()
            result[label] = margin
        return result
    
    #-------------------------------- Similarity Measures -------------------------------

    def _hamming_distance(self, pattern1, pattern2):
        """
        Compute the Hamming distance between two patterns.
        Both patterns must be {+1, -1} vectors of shape (n_neurons,).
        Returns the number of differing bits.
        """
        # Ensure inputs are integers
        pattern1 = np.asarray(pattern1, dtype=int)  # First pattern
        pattern2 = np.asarray(pattern2, dtype=int)  # Second pattern
        # Validate shapes
        if pattern1.shape != (self.n_neurons,) or pattern2.shape != (self.n_neurons,):
            raise ValueError("Both patterns must have shape (n_neurons,)")
        # Validate values
        if not (np.isin(pattern1, (-1, 1)).all() and np.isin(pattern2, (-1, 1)).all()):
            raise ValueError("Both patterns must be {+1, -1} vectors")
        # Compute and return Hamming distance
        return np.sum(pattern1 != pattern2)
    
    def _overlap(self, pattern1, pattern2):
        """
        Compute the overlap (dot product) between two patterns.
        Both patterns must be {+1, -1} vectors of shape (n_neurons,).
        Returns the dot product value.
        """
        # Ensure inputs are integers
        pattern1 = np.asarray(pattern1, dtype=int)  # First pattern
        pattern2 = np.asarray(pattern2, dtype=int)  # Second pattern
        # Validate shapes
        if pattern1.shape != (self.n_neurons,) or pattern2.shape != (self.n_neurons,):
            raise ValueError("Both patterns must have shape (n_neurons,)")
        # Validate values
        if not (np.isin(pattern1, (-1, 1)).all() and np.isin(pattern2, (-1, 1)).all()):
            raise ValueError("Both patterns must be {+1, -1} vectors")
        # Compute and return overlap (dot product)
        return np.dot(pattern1, pattern2)
    
    def overlap_matrix(self):
        """
        Compute the overlap matrix between all stored memories.
        Returns a 2D numpy array of shape (num_memories, num_memories)
        where entry (i, j) is the overlap between memory i and memory j.
        """
        # Handle case with no stored memories
        if self.memories.size == 0:
            return np.empty((0, 0), dtype=int)
        # Initialize overlap matrix
        num_memories = self.memories.shape[0]
        overlap_mat = np.zeros((num_memories, num_memories), dtype=int)
        # Compute overlaps
        for i in range(num_memories):
            for j in range(num_memories):
                overlap_mat[i, j] = self._overlap(self.memories[i], self.memories[j])
        # Return the overlap matrix
        return overlap_mat / self.n_neurons  # Normalize by number of neurons
    
    def max_offdiagonal_overlap(self):
        """
        Compute the maximum off-diagonal overlap between stored memories.
        Returns the highest overlap value found between any two distinct memories.
        If fewer than 2 memories are stored, returns None.
        """
        # Get the overlap matrix
        C = self.overlap_matrix()
        if C.shape[0] == 0:
            # No memories stored
            return None  
        m = C.shape[0]
        if m < 2:
            # Only one memory stored
            return float(C[0, 0]) 
        # Mask diagonal entries to ignore self-overlaps
        mask = ~np.eye(m, dtype=bool)
        # Return the maximum off-diagonal overlap
        return float(np.max(np.abs(C[mask])))

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
        # Return best matching memory, its label, and the score
        best_mem   = self.memories[best_idx]
        best_label = labels[best_idx]
        best_score = scores[best_idx]
        return best_mem, best_label, best_score
    
    def mean_memory(self):
        """
        Compute the mean pattern of all stored memories.
        Returns a float array of shape (n_neurons,) with values in [-1, 1].
        If no memories are stored, returns an array of zeros.
        """
        if self.memories.size == 0:
            return np.zeros(self.n_neurons, dtype=float)
        return self.memories.mean(axis=0)
