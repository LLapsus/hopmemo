from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# Dataset names
# "mnist"         : Standard MNIST digits
# "fashion-mnist" : Zalando's Fashion-MNIST clothes
# "kmnist"        : Kuzushiji-MNIST Japanese characters
DatasetName = Literal["mnist", "fashion-mnist", "kmnist"]

# Binarization methods
# "threshold"  : simple fixed thresholding
# "percentile" : per-image threshold at given percentile
# "mean_std"   : threshold = mean + 0.2*std
BinarizeMethod = Literal["threshold", "percentile", "mean_std"]

@dataclass
class ImageDataset28:
    """
    Holds a binarized 28x28 image dataset in Hopfield-friendly format.

    Attributes
    ----------
    images : np.ndarray | None
        Shape (N, 28, 28), dtype int8, values in {-1, +1}.
    labels : np.ndarray | None
        Shape (N,), dtype int64.
    name : str | None
        Resolved OpenML dataset name.
    """
    images: np.ndarray | None = None
    labels: np.ndarray | None = None
    name: str | None = None

    def load(
        self,
        dataset: DatasetName,
        *,
        bin_method: BinarizeMethod = "threshold",
        thr: int = 64,
        percentile: float = 85.0,
        cache_dir: str | Path = "~/.cache/hopfield_datasets",
        openml_version: int = 1,
        limit: int | None = None,
        seed: int | None = None,
    ) -> "ImageDataset28":
        """
        Download a 28x28 grayscale dataset from OpenML and convert it to {-1, +1}.

        Parameters
        ----------
        dataset:
            Which dataset to load.
            - "mnist"         -> OpenML "mnist_784"
            - "fashion-mnist" -> OpenML "Fashion-MNIST"
            - "kmnist"        -> OpenML "Kuzushiji-MNIST"

        bin_method:
            How to binarize grayscale pixels into a binary ink mask (0/1) before
            mapping to {-1, +1}.
            - "threshold":  mask = (pixel >= thr)
            - "percentile": per-image threshold at the given percentile (good when
                            contrast varies or when images are "soft")
            - "mean_std": threshold = mean + 0.2*std (mild adaptive threshold)

        thr:
            Pixel threshold used when bin_method="threshold". Typical range: 64-96.

        percentile:
            Percentile used when bin_method="percentile". Typical range: 80-92.
            Higher percentile -> thinner digits/clothes (less ink).

        cache_dir:
            Directory used by scikit-learn/OpenML caching. The dataset will be
            downloaded only once and reused afterwards.

        openml_version:
            Dataset version on OpenML (default: 1). Keeping this fixed improves
            reproducibility.

        limit:
            If set, keep only `limit` samples (after loading). Useful for quick demos
            and faster iteration. If None, keep all samples.

        seed:
            Random seed used only when `limit` is set (samples are then chosen randomly).

        Returns
        -------
        self:
            The dataset instance (for chaining).
        """
        X, y, resolved = self._fetch_openml_28x28(
            dataset=dataset,
            cache_dir=cache_dir,
            version=openml_version,
        )

        if limit is not None:
            rng = np.random.default_rng(seed)
            if limit <= 0 or limit > X.shape[0]:
                raise ValueError(f"limit must be in [1, {X.shape[0]}], got {limit}")
            idx = rng.choice(X.shape[0], size=limit, replace=False)
            X = X[idx]
            y = y[idx]

        masks = self._binarize_batch(X, method=bin_method, thr=thr, percentile=percentile)
        
        # Map binary masks {0,1} to {-1,+1}
        self.images = np.where(masks > 0, 1, -1).astype(np.int8)
        self.labels = y.astype(np.int64)
        self.name = resolved
        return self

    def subset(
        self,
        n: int,
        *,
        flatten: bool = False,
        seed: int | None = None,
        stratify: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a random subset of `n` samples.

        Parameters
        ----------
        n:
            Number of samples to return.
        flatten:
            If False -> returns (n, 28, 28). If True -> returns (n, 784).
        seed:
            RNG seed for reproducibility.
        stratify:
            If True, attempts to sample roughly equally from each class.
            (Works best for 10-class datasets like MNIST/Fashion/KMNIST.)

        Returns
        -------
        Xs, ys:
            Subset images and labels.
        """
        if self.images is None or self.labels is None:
            raise RuntimeError("Dataset not loaded. Call .load(...) first.")

        rng = np.random.default_rng(seed)
        N = self.images.shape[0]
        if n > N:
            raise ValueError(f"Requested n={n} but dataset has only {N} samples.")

        # Data stratification
        if stratify:
            y = self.labels
            classes = np.unique(y)
            per = max(1, n // len(classes))
            idx_list = []
            for c in classes:
                inds = np.flatnonzero(y == c)
                if inds.size == 0:
                    continue
                take = min(per, inds.size)
                idx_list.append(rng.choice(inds, size=take, replace=False))
            idx = np.concatenate(idx_list) if idx_list else rng.choice(N, size=n, replace=False)

            if idx.size < n:
                remaining = np.setdiff1d(np.arange(N), idx, assume_unique=False)
                extra = rng.choice(remaining, size=(n - idx.size), replace=False)
                idx = np.concatenate([idx, extra])

            rng.shuffle(idx)
            idx = idx[:n]
        # Non-stratified random sampling
        else:
            idx = rng.choice(N, size=n, replace=False)

        Xs = self.images[idx]
        ys = self.labels[idx]
        if flatten:
            Xs = Xs.reshape(n, -1)
        return Xs, ys

    # ---------------- internals ----------------

    # Use scikit-learn / OpenML to fetch 28x28 datasets
    def _fetch_openml_28x28(
        self,
        *,
        dataset: DatasetName,
        cache_dir: str | Path,
        version: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        from sklearn.datasets import fetch_openml

        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        if dataset == "mnist":
            name = "mnist_784"
        elif dataset == "fashion-mnist":
            name = "Fashion-MNIST"
        elif dataset == "kmnist":
            name = "Kuzushiji-MNIST"
        else:
            raise ValueError(f"Unknown dataset={dataset!r}")

        bunch = fetch_openml(
            name=name,
            version=version,
            as_frame=False,
            cache=True,
            data_home=str(cache_dir),
            parser="auto",
        )

        X = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target)

        if y.dtype.kind in {"U", "S", "O"}:
            y = y.astype(int)

        if X.max() <= 1.0:
            X *= 255.0
        X = np.clip(X, 0, 255).astype(np.uint8)

        if X.shape[1] != 784:
            raise ValueError(f"Expected 784 features for 28x28, got {X.shape[1]}")

        X = X.reshape(-1, 28, 28)
        return X, y.astype(np.int64), name

    # Binarize a batch of images: ( N, 28, 28 ) -> ( N, 28, 28 ) binary mask
    def _binarize_batch(
        self,
        X: np.ndarray,
        *,
        method: BinarizeMethod,
        thr: int,
        percentile: float,
    ) -> np.ndarray:
        if method == "threshold":
            return (X >= np.uint8(thr)).astype(np.uint8)

        if method == "percentile":
            t = np.percentile(X, percentile, axis=(1, 2), keepdims=True)
            return (X >= t).astype(np.uint8)

        if method == "mean_std":
            m = X.mean(axis=(1, 2), keepdims=True)
            s = X.std(axis=(1, 2), keepdims=True)
            t = m + 0.2 * s
            return (X >= t).astype(np.uint8)

        raise ValueError(f"Unknown binarization method={method!r}")
