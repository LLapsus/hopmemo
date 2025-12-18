from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# List of available datasets
# mnist: Original MNIST (hand written digits)
# fashion-mnist: Fashion-MNIST (clothing items)
# kmnist: Kuzushiji-MNIST (Japanese characters)
# emnist: Extended MNIST (letters and digits)
DatasetName = Literal["mnist", "fashion-mnist", "kmnist", "emnist"]  

# Dataclass to hold the dataset of images (28, 28) in binary format {-1, +1} and labels
@dataclass
class ImageDataset28:
    """
    Holds a binarized 28x28 image dataset in Hopfield-friendly binary format.

    images: (N, 28, 28) int8 values in {-1, +1}
    labels: (N,) int64
    name:   dataset identifier
    """
    images: np.ndarray | None = None
    labels: np.ndarray | None = None
    name: str | None = None

    # ---------- Public API ----------

    def load(
        self,
        dataset: DatasetName,
        *,
        bin_method: Literal["threshold", "percentile", "mean_std"] = "threshold",
        thr: int = 64,
        percentile: float = 80.0,
        cache_dir: str | Path = "~/.cache/hopfield_datasets",
        openml_id: int | None = None,
        openml_name: str | None = None,
        openml_version: int | None = None,
    ) -> "ImageDataset28":
        """
        Download dataset via OpenML (sklearn fetch_openml) and binarize to {-1, +1}.

        Notes:
          - MNIST:           OpenML name 'mnist_784' (id 554). :contentReference[oaicite:3]{index=3}
          - Fashion-MNIST:   OpenML name 'Fashion-MNIST' (id 40996). :contentReference[oaicite:4]{index=4}
          - Kuzushiji-MNIST: OpenML name 'Kuzushiji-MNIST' (id 41982). :contentReference[oaicite:5]{index=5}

        EMNIST: OpenML naming varies by split; use openml_id or openml_name if you want it.
        """
        X, y, resolved_name = self._fetch_openml_28x28(
            dataset=dataset,
            cache_dir=cache_dir,
            openml_id=openml_id,
            openml_name=openml_name,
            openml_version=openml_version,
        )

        # binarize -> mask {0,1}
        masks = self._binarize_batch(
            X,
            method=bin_method,
            thr=thr,
            percentile=percentile,
        )

        # {0,1} -> {-1,+1}
        imgs = np.where(masks > 0, 1, -1).astype(np.int8)

        self.images = imgs
        self.labels = y.astype(np.int64)
        self.name = resolved_name
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
        Return a random subset of n samples.

        flatten:
            False -> (n, 28, 28)
            True  -> (n, 784)

        stratify:
            If True, tries to sample roughly equally per class (10-class datasets).
        """
        self._require_loaded()
        assert self.images is not None and self.labels is not None

        rng = np.random.default_rng(seed)

        N = self.images.shape[0]
        if n > N:
            raise ValueError(f"Requested n={n} but dataset has only {N} samples.")

        if stratify:
            # simple stratified sampler (best-effort)
            y = self.labels
            classes = np.unique(y)
            per = max(1, n // len(classes))
            idx = []
            for c in classes:
                inds = np.flatnonzero(y == c)
                if len(inds) == 0:
                    continue
                take = min(per, len(inds))
                idx.append(rng.choice(inds, size=take, replace=False))
            idx = np.concatenate(idx) if idx else rng.choice(N, size=n, replace=False)
            if idx.size < n:
                # top up randomly
                remaining = np.setdiff1d(np.arange(N), idx, assume_unique=False)
                extra = rng.choice(remaining, size=(n - idx.size), replace=False)
                idx = np.concatenate([idx, extra])
            rng.shuffle(idx)
            idx = idx[:n]
        else:
            idx = rng.choice(N, size=n, replace=False)

        Xs = self.images[idx]
        ys = self.labels[idx]

        if flatten:
            Xs = Xs.reshape(n, -1)

        return Xs, ys

    # ---------- Internals ----------

    def _require_loaded(self) -> None:
        if self.images is None or self.labels is None:
            raise RuntimeError("Dataset not loaded. Call .load(...) first.")

    def _fetch_openml_28x28(
        self,
        *,
        dataset: DatasetName,
        cache_dir: str | Path,
        openml_id: int | None,
        openml_name: str | None,
        openml_version: int | None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Returns:
          X: (N, 28, 28) uint8 in [0,255]
          y: (N,) labels
          resolved_name: str
        """
        from sklearn.datasets import fetch_openml  # scikit-learn :contentReference[oaicite:6]{index=6}

        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve OpenML dataset identifier
        if openml_id is not None:
            spec = {"data_id": int(openml_id)}
            resolved = f"openml_id={openml_id}"
        elif openml_name is not None:
            spec = {"name": str(openml_name)}
            if openml_version is not None:
                spec["version"] = int(openml_version)
            resolved = f"openml_name={openml_name}"
        else:
            if dataset == "mnist":
                spec = {"name": "mnist_784", "version": 1}
                resolved = "mnist_784"
            elif dataset == "fashion-mnist":
                # OpenML canonical name is 'Fashion-MNIST' :contentReference[oaicite:7]{index=7}
                spec = {"name": "Fashion-MNIST", "version": 1}
                resolved = "Fashion-MNIST"
            elif dataset == "kmnist":
                spec = {"name": "Kuzushiji-MNIST", "version": 1}
                resolved = "Kuzushiji-MNIST"
            elif dataset == "emnist":
                raise ValueError(
                    "EMNIST has multiple splits and is not reliably addressable by a single "
                    "OpenML name here. Pass openml_id=... or openml_name=... explicitly."
                )
            else:
                raise ValueError(f"Unknown dataset={dataset!r}")

        bunch = fetch_openml(
            as_frame=False,
            cache=True,
            data_home=str(cache_dir),
            parser="auto",
            **spec,
        )

        X = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target)

        # OpenML sometimes returns target as strings
        if y.dtype.kind in {"U", "S", "O"}:
            # best effort numeric cast (works for MNIST/Fashion/KMNIST)
            y = y.astype(int)

        # normalize to 0..255 if needed
        if X.max() <= 1.0:
            X *= 255.0
        X = np.clip(X, 0, 255).astype(np.uint8)

        if X.shape[1] != 784:
            raise ValueError(f"Expected 784 features for 28x28, got {X.shape[1]}")

        X = X.reshape(-1, 28, 28)
        return X, y.astype(np.int64), resolved

    def _binarize_batch(
        self,
        X: np.ndarray,
        *,
        method: str,
        thr: int,
        percentile: float,
    ) -> np.ndarray:
        """
        X: (N, 28, 28) uint8
        returns masks: (N, 28, 28) uint8 in {0,1}, where 1 = ink/foreground
        """
        if method == "threshold":
            return (X >= np.uint8(thr)).astype(np.uint8)

        if method == "percentile":
            # per-image percentile threshold
            # shape: (N,1,1) for broadcasting
            t = np.percentile(X, percentile, axis=(1, 2), keepdims=True)
            return (X >= t).astype(np.uint8)

        if method == "mean_std":
            m = X.mean(axis=(1, 2), keepdims=True)
            s = X.std(axis=(1, 2), keepdims=True)
            t = m + 0.2 * s
            return (X >= t).astype(np.uint8)

        raise ValueError(f"Unknown binarization method={method!r}")

    def _center_by_centroid_batch(self, masks: np.ndarray) -> np.ndarray:
        """
        Center each binary mask by its centroid (ink pixels).
        masks: (N,28,28) {0,1}
        """
        N = masks.shape[0]
        out = np.empty_like(masks)
        H, W = 28, 28
        ty = (H - 1) / 2.0
        tx = (W - 1) / 2.0

        for i in range(N):
            m = masks[i]
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                out[i] = m
                continue

            cy = ys.mean()
            cx = xs.mean()
            dy = int(round(ty - cy))
            dx = int(round(tx - cx))

            shifted = np.zeros((H, W), dtype=m.dtype)

            y0_src = max(0, -dy)
            y1_src = min(H, H - dy)
            x0_src = max(0, -dx)
            x1_src = min(W, W - dx)

            y0_dst = max(0, dy)
            y1_dst = min(H, H + dy)
            x0_dst = max(0, dx)
            x1_dst = min(W, W + dx)

            shifted[y0_dst:y1_dst, x0_dst:x1_dst] = m[y0_src:y1_src, x0_src:x1_src]
            out[i] = shifted

        return out
