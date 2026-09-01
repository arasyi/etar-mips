from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .metrics import exact_topk


def generate_dataset(kind: str, n: int, d: int, q: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "dense_gaussian":
        X = rng.normal(size=(n, d)).astype(np.float32)
        Q = rng.normal(size=(q, d)).astype(np.float32)
    elif kind == "sparse10":
        X = rng.normal(size=(n, d)).astype(np.float32)
        Q = rng.normal(size=(q, d)).astype(np.float32)
        X *= (rng.random((n, d)) < 0.10)
        Q *= (rng.random((q, d)) < 0.10)
    elif kind == "mixed_dropout_heavytail":
        X = rng.standard_t(df=3, size=(n, d)).astype(np.float32)
        Q = rng.standard_t(df=3, size=(q, d)).astype(np.float32)
        X *= (rng.random((n, d)) > 0.35)
        Q *= (rng.random((q, d)) > 0.35)
    elif kind == "norm_heavy":
        X = rng.normal(size=(n, d)).astype(np.float32)
        Q = rng.normal(size=(q, d)).astype(np.float32)
        X *= rng.lognormal(mean=0.0, sigma=1.25, size=(n, 1)).astype(np.float32)
    elif kind == "low_margin":
        base = rng.normal(size=(1, d)).astype(np.float32)
        X = base + 0.05 * rng.normal(size=(n, d)).astype(np.float32)
        Q = base + 0.05 * rng.normal(size=(q, d)).astype(np.float32)
    elif kind == "adversarial_tail":
        X = 0.05 * rng.normal(size=(n, d)).astype(np.float32)
        Q = np.zeros((q, d), dtype=np.float32)
        head = max(1, d // 8)
        Q[:, :head] = rng.normal(size=(q, head)).astype(np.float32)
        tail = rng.integers(head, d, size=q)
        for i, j in enumerate(tail):
            Q[i, j] = 0.4
            rows = rng.choice(n, size=max(1, n // 20), replace=False)
            X[rows, j] += rng.normal(8.0, 0.5, size=rows.size).astype(np.float32)
    else:
        raise ValueError(f"unknown dataset kind {kind}")
    return np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(Q, dtype=np.float32)


def write_hdf5(path: str | Path, X: np.ndarray, Q: np.ndarray, kind: str, seed: int, k: int = 100) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    neighbors, distances = exact_topk(X, Q, min(k, X.shape[0]))
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=X, compression="gzip")
        f.create_dataset("test", data=Q, compression="gzip")
        f.create_dataset("neighbors", data=neighbors, compression="gzip")
        f.create_dataset("distances", data=distances, compression="gzip")
        f.attrs["distance"] = "dot"
        f.attrs["kind"] = kind
        f.attrs["seed"] = seed
        f.attrs["dimension"] = X.shape[1]
        f.attrs["synthetic"] = True


def generate_preset(preset: str, out_dir: str | Path = "data/generated", seed: int = 123) -> list[Path]:
    out_dir = Path(out_dir)
    if preset == "smoke":
        specs = [
            ("dense_gaussian", 1500, 64, 50),
            ("sparse10", 1500, 64, 50),
            ("mixed_dropout_heavytail", 1500, 64, 50),
        ]
    elif preset == "synthetic":
        kinds = ["dense_gaussian", "sparse10", "mixed_dropout_heavytail", "norm_heavy", "low_margin", "adversarial_tail"]
        specs = [(kind, 20000, 128, 200) for kind in kinds]
    elif preset == "synthetic-large":
        specs = [("dense_gaussian", 500000, 64, 100)]
    else:
        raise ValueError("preset must be smoke, synthetic, or synthetic-large")
    paths = []
    for offset, (kind, n, d, q) in enumerate(specs):
        X, Q = generate_dataset(kind, n, d, q, seed + offset)
        path = out_dir / preset / f"{kind}-N{n}-d{d}.hdf5"
        write_hdf5(path, X, Q, kind, seed + offset, k=100)
        paths.append(path)
    return paths


def load_hdf5(path: str | Path, limit_n: int | None = None, limit_q: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        X = np.asarray(f["train"][: limit_n], dtype=np.float32)
        Q = np.asarray(f["test"][: limit_q], dtype=np.float32)
    if X.shape[1] != Q.shape[1]:
        d = min(X.shape[1], Q.shape[1])
        X, Q = X[:, :d], Q[:, :d]
    return np.ascontiguousarray(X), np.ascontiguousarray(Q)

