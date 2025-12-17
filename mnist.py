#-------------------------------------------------------------------------------
# MNIST Data Loader and Utility Functions
#-------------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------------------
# MNIST Data Loading and Preprocessing Utilities
#-------------------------------------------------------------------------------

def load_mnist(
    data_home: str | Path = "~/.cache/",
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST (28x28) via scikit-learn / OpenML.

    Returns:
        X : (n_samples, 28, 28) uint8, values 0..255
        y : (n_samples,) int labels 0..9
    """
    data_home = Path(data_home).expanduser()
    data_home.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import fetch_openml

    # load data
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        cache=cache,
        data_home=str(data_home),
        parser="auto",
    )

    # assign features and labels
    X = mnist.data
    y = mnist.target

    # safety & normalization
    X = np.asarray(X, dtype=np.float32)
    if X.max() <= 1.0:
        X *= 255.0

    X = X.reshape(-1, 28, 28).astype(np.uint8)
    y = y.astype(int)

    return X, y



def binarize_mnist(img: np.ndarray, method: str = "threshold", thr: int = 64) -> np.ndarray:
    """
    img: (H,W) uint8 in [0,255]
    returns: binary mask uint8 {0,1} where 1 = ink/foreground
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if method == "threshold":
        # simple, robust default
        return (img >= thr).astype(np.uint8)

    if method == "percentile":
        # good for prototypes or varying contrast
        t = np.percentile(img, 75)  # tweak 70-85 if needed
        return (img >= t).astype(np.uint8)

    if method == "mean_std":
        # mild adaptive threshold based on image stats
        m = float(img.mean())
        s = float(img.std())
        t = m + 0.2 * s
        return (img >= t).astype(np.uint8)

    raise ValueError(f"Unknown method={method!r}")

def mnist_to_state(
    img: np.ndarray,
    *,
    bin_method: str = "threshold",
    thr: int = 64,
    invert: bool = False,
) -> np.ndarray:
    """
    Convert MNIST-like grayscale image to Hopfield state vector in {-1,+1}.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img.shape}")

    # In MNIST, background is ~0 and ink is high values -> mask = ink
    mask = binarize_mnist(img, method=bin_method, thr=thr)

    if invert:
        mask = 1 - mask

    # map {0,1} -> {-1,+1}
    state = np.where(mask > 0, 1, -1).astype(np.int8).reshape(-1)
    return state


#-------------------------------------------------------------------------------
# MNIST Plotting Utilities
#-------------------------------------------------------------------------------

def plot_mnist_sample(image: np.ndarray, figsize: tuple = (4, 4)) -> None:
    """
    Plot given mnist sample.

    Args:
        image : (28, 28) uint8
        figsize : (4, 4) figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(image, cmap="gray", cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    plt.show()
