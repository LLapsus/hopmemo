#-------------------------------------------------------------------------------
# MNIST Data Loader
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
