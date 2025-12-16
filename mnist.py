#-------------------------------------------------------------------------------
# MNIST Dataset Loader
#-------------------------------------------------------------------------------

from __future__ import annotations
import gzip
import struct
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

#-------------------------------------------------------------------------------
# MNIST dataset files

MNIST_URL = "https://yann.lecun.com/exdb/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

#-------------------------------------------------------------------------------

# Download MNIST files if not present
def _download_mnist(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        out = dst / fname
        if out.exists():
            continue
        urlretrieve(MNIST_URL + fname, out)

# Read IDX formatted files
def _read_idx_images(path_gz: Path) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic for images: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)

# Read IDX formatted files
def _read_idx_labels(path_gz: Path) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic for labels: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)

# Main function to load MNIST dataset
def load_mnist(root: str | Path = ".cache/mnist") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    _download_mnist(root)
    Xtr = _read_idx_images(root / FILES["train_images"])
    ytr = _read_idx_labels(root / FILES["train_labels"])
    Xte = _read_idx_images(root / FILES["test_images"])
    yte = _read_idx_labels(root / FILES["test_labels"])
    return Xtr, ytr, Xte, yte
