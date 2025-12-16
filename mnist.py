#-------------------------------------------------------------------------------
# MNIST Data Loader and Utility Functions
#-------------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np

def load_mnist_openml(
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

#-------------------------------------------------------------------------------
# MNIST Plotting Utilities
#-------------------------------------------------------------------------------

def plot_mnist_sample(image: np.ndarray, figsize: np.ndarray = (4, 4)) -> None:
    """
    Plot given mnist sample.

    Args:
        image : (28, 28) uint8
        figsize: ndarray (4, 4) figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(image, cmap="gray", cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    plt.show()


#-------------------------------------------------------------------------------
# MNIST Image Processing Utilities
#-------------------------------------------------------------------------------

def binarize_mnist(img: np.ndarray, method: str = "threshold", thr: int = 64, perc: int = 75) -> np.ndarray:
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
        t = np.percentile(img, perc)  # tweak 70-85 if needed
        return (img >= t).astype(np.uint8)

    if method == "mean_std":
        # mild adaptive threshold based on image stats
        m = float(img.mean())
        s = float(img.std())
        t = m + 0.2 * s
        return (img >= t).astype(np.uint8)

    raise ValueError(f"Unknown method={method!r}")


def center_by_centroid(mask: np.ndarray, fill: int = 0) -> np.ndarray:
    """
    Shift a binary mask so that its centroid is at the image center.
    mask: (H,W) {0,1}
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        # empty -> return as-is
        return mask.copy()

    cy = ys.mean()
    cx = xs.mean()

    # desired centroid:
    ty = (H - 1) / 2.0
    tx = (W - 1) / 2.0

    dy = int(round(ty - cy))
    dx = int(round(tx - cx))

    out = np.full((H, W), fill_value=fill, dtype=mask.dtype)

    # compute source/dest slices safely
    y0_src = max(0, -dy)
    y1_src = min(H, H - dy)
    x0_src = max(0, -dx)
    x1_src = min(W, W - dx)

    y0_dst = max(0, dy)
    y1_dst = min(H, H + dy)
    x0_dst = max(0, dx)
    x1_dst = min(W, W + dx)

    out[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return out


def mnist_to_state(
    img: np.ndarray,
    *,
    bin_method: str = "threshold",
    thr: int = 64,
    center: bool = True,
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

    if center:
        mask = center_by_centroid(mask)

    # map {0,1} -> {-1,+1}
    state = np.where(mask > 0, 1, -1).astype(np.int8).reshape(-1)
    return state
