# Hopfield Network

A clean, interactive implementation of the classical **Hopfield Network** — a recurrent neural network that acts as an associative memory. Store patterns, corrupt them with noise, and watch the network retrieve the originals.

**Live demo:** [hopmemo.streamlit.app](https://hopmemo.streamlit.app/)

---

## What is a Hopfield Network?

A Hopfield Network is a fully-connected recurrent neural network that can memorize a set of binary patterns and later retrieve them from partial or noisy inputs. Each pattern is stored as an attractor in the network's energy landscape. Given a corrupted query, the network iteratively updates its state until it converges to the nearest stored memory.

---

## Features

- **4 learning rules:** Hebbian, Storkey, centered pseudoinverse, damped pseudoinverse
- **2 retrieval modes:** Asynchronous (random neuron order) or synchronous (all neurons at once)
- **Built-in diagnostics:** Network load, pairwise pattern correlations, stability margins
- **3 datasets:** Alphanumeric characters (A–Z, 0–9), icon patterns, and OpenML image datasets (MNIST, Fashion-MNIST, Kuzushiji)
- **Interactive apps:** Streamlit web app and Tkinter desktop app

---

## Run Locally

**Requirements:** Python 3.8+

```bash
git clone https://github.com/LLapsus/hopmemo.git
cd hopmemo
pip install -r requirements.txt
```

**Streamlit web app:**
```bash
streamlit run streamlit_app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Tkinter desktop app:**
```bash
python apps/tk_app.py
```
No extra dependencies needed — Tkinter is part of the Python standard library.

---

## Interactive Apps

### Streamlit App (`streamlit_app.py`)

The web app uses the alphanumeric dataset (A–Z and 0–9, rendered as 28×28 binary images). From the UI you can:

- Click patterns to **memorize** or **forget** them one by one
- Configure the **learning method** and retrieval parameters in the sidebar
- Inspect the **weight matrix** as a live heatmap
- View **diagnostics**: network load (α = P/N), pairwise pattern correlations, and stability status
- Select a stored pattern, apply **noise or inversion**, and run retrieval to see the input → output comparison

### Tkinter App (`tk_app.py`)

A lightweight desktop GUI for hands-on experimentation:

- Draw patterns directly on an interactive pixel grid
- Memorize, retrieve, and reset the network
- Adjustable noise level, theta, max iterations, and update rule
- Side-by-side input/output grids for easy comparison

---

## Python API

The core class is `HopfieldNetwork` in [hopfield/network.py](hopfield/network.py).

### Quick Example

```python
import numpy as np
from hopfield import HopfieldNetwork
from utils import add_noise

# Create a network for 64-dimensional patterns (e.g. 8×8 images)
hop = HopfieldNetwork(n_neurons=64, learning_method="hebbian")

# Define some binary patterns with values in {-1, +1}
rng = np.random.default_rng(42)
patterns = rng.choice([-1, 1], size=(5, 64))

# Memorize them
hop.memorize(patterns, labels=["A", "B", "C", "D", "E"])

# Corrupt one pattern with 20% noise
noisy = add_noise(patterns[0], p=0.2)

# Retrieve the closest stored memory
retrieved = hop.retrieve(noisy, max_iterations=50, update_rule="async")

# Check how many bits differ from the original
errors = np.sum(retrieved != patterns[0])
print(f"Bit errors after retrieval: {errors}")  # Should be 0 if load is low

# Identify which stored memory was recovered
memory, label, score = hop.nearest_memory(retrieved, metric="hamming")
print(f"Retrieved: '{label}' (Hamming distance: {score})")
```

### Constructor

```python
HopfieldNetwork(
    n_neurons=64,           # Number of neurons (= dimensionality of patterns)
    learning_method="hebbian",  # "hebbian" | "storkey" | "pinv_centered" | "pinv_damped"
    damped_lam=0.1,         # Regularization for "pinv_damped"
    damped_centered=True,   # Center patterns before pseudoinverse
    damped_zero_diagonal=True   # Zero the weight matrix diagonal
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `memorize(patterns, labels=None)` | Store an array of patterns `(P, N)` and update weights |
| `retrieve(pattern, theta=0., max_iterations=50, update_rule="async", history=False)` | Retrieve a stored pattern from a query; returns final state (and optional history dict) |
| `reset_network()` | Clear all memories and weights |
| `energy(state)` | Compute the Hopfield energy of a state |
| `weights()` | Return the weight matrix `(N, N)` |
| `check_stability()` | Return per-pattern stability margins |
| `overlap_matrix()` | Compute pairwise overlaps between stored memories |
| `nearest_memory(pattern, metric="hamming")` | Find the closest stored memory to a query |
| `num_memories()` | Number of currently stored patterns |

### Learning Methods

| Method | Notes |
|--------|-------|
| `"hebbian"` | Classic outer-product rule; simple and fast |
| `"storkey"` | Improved capacity over Hebbian; considers existing weights during training |
| `"pinv_centered"` | Pseudoinverse rule with centering; optimal for near-orthogonal patterns |
| `"pinv_damped"` | Tikhonov-regularized pseudoinverse; tunable via `damped_lam` |

### Datasets

```python
from datasets.alnum import generate_alnum_dataset
from datasets.icons import generate_icons
from datasets.openml import ImageDataset28

# Alphanumeric: 36 patterns of shape (28, 28) with values in {-1, +1}
X, y = generate_alnum_dataset()          # X: (36, 28, 28), y: labels

# Icons: ~50 procedurally-generated icon patterns
X, y = generate_icons(n=50, seed=0)      # X: (50, 28, 28), y: labels

# MNIST (requires internet on first run; cached afterwards)
ds = ImageDataset28.load(dataset="mnist", bin_method="threshold", thr=64)
X, y = ds.subset(n=100, flatten=True)   # X: (100, 784), values in {-1, +1}
```

### Utilities

```python
from hopfield.utils import add_noise, hide_pattern, compare_patterns, plot_energy

noisy  = add_noise(pattern, p=0.15)      # Flip 15% of bits randomly
hidden = hide_pattern(pattern)           # Set the bottom half to zero

# Visualize side-by-side input vs. output
compare_patterns(noisy, retrieved)

# Retrieve with history and plot energy convergence
retrieved, history = hop.retrieve(noisy, history=True)
plot_energy(history)
```

---

## Repository Structure

```
hopmemo/
├── hopfield/                  # Core library (importable package)
│   ├── __init__.py
│   ├── network.py             # HopfieldNetwork class
│   └── utils.py               # Noise, distance, and visualization helpers
├── datasets/                  # Pattern generators
│   ├── alnum.py               # Alphanumeric patterns (A–Z, 0–9)
│   ├── icons.py               # Procedural icon patterns
│   └── openml.py              # MNIST / Fashion-MNIST / Kuzushiji loader
├── apps/                      # Interactive frontends
│   ├── streamlit_app.py       # Web app (Streamlit) — also mirrored at root
│   └── tk_app.py              # Desktop GUI (Tkinter)
├── notebooks/                 # Demo notebooks
│   ├── hopfield_demo.ipynb
│   ├── alnum_dataset_demo.ipynb
│   ├── icon_dataset_demo.ipynb
│   └── openml_datasets_demo.ipynb
├── streamlit_app.py           # Streamlit Cloud entry point (root-level)
├── requirements.txt           # All dependencies
└── README.md
```

---

## Dependencies

```bash
pip install -r requirements.txt
```
