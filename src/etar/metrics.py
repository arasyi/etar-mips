from __future__ import annotations

import numpy as np


def topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k ids and scores with deterministic score-tie handling.

    NumPy's argpartition does not define which ids are returned when multiple
    rows have the same kth score. Several ANN-Benchmarks datasets contain
    duplicate or tied vectors, so deterministic ties are important for making
    candidate coverage and exact reranking comparable.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        return _topk_1d(scores, k)
    if scores.ndim != 2:
        raise ValueError("scores must be a 1D or 2D array")
    k = min(int(k), scores.shape[1])
    if k <= 0:
        return (
            np.empty((scores.shape[0], 0), dtype=np.int64),
            np.empty((scores.shape[0], 0), dtype=np.float32),
        )
    ids = np.empty((scores.shape[0], k), dtype=np.int64)
    vals = np.empty((scores.shape[0], k), dtype=np.float32)
    for i, row in enumerate(scores):
        row_ids, row_vals = _topk_1d(row, k)
        ids[i] = row_ids
        vals[i] = row_vals
    return ids, vals


def _topk_1d(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    if k == scores.shape[0]:
        ids = np.arange(scores.shape[0], dtype=np.int64)
    else:
        part = np.argpartition(-scores, kth=k - 1)[:k]
        kth_score = np.min(scores[part])
        above = np.flatnonzero(scores > kth_score)
        ties = np.flatnonzero(scores == kth_score)
        need = k - above.size
        if above.size:
            above = above[np.lexsort((above, -scores[above]))]
        ids = above if need <= 0 else np.concatenate([above, np.sort(ties)[:need]])
    ids = ids[np.lexsort((ids, -scores[ids]))][:k].astype(np.int64, copy=False)
    return ids, scores[ids].astype(np.float32, copy=False)


def exact_topk(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    max_score_bytes: int = 512 * 1024 * 1024,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    k = min(int(k), X.shape[0])
    if k <= 0:
        return np.empty((Q.shape[0], 0), dtype=np.int64), np.empty((Q.shape[0], 0), dtype=np.float32)
    bytes_per_query = max(1, X.shape[0] * np.dtype(np.float32).itemsize)
    query_batch = max(1, min(Q.shape[0], int(max_score_bytes // bytes_per_query)))
    ids_out = []
    scores_out = []
    for start in range(0, Q.shape[0], query_batch):
        scores = Q[start : start + query_batch] @ X.T
        ids, vals = topk_from_scores(scores, k)
        ids_out.append(ids.astype(np.int64))
        scores_out.append(vals.astype(np.float32))
    return np.vstack(ids_out), np.vstack(scores_out)


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    vals = []
    for p, t in zip(pred[:, :k], truth[:, :k]):
        vals.append(len(set(map(int, p)) & set(map(int, t))) / float(k))
    return float(np.mean(vals)) if vals else 0.0


def candidate_coverage(candidates: list[np.ndarray], truth: np.ndarray, k: int) -> float:
    vals = []
    for c, t in zip(candidates, truth[:, :k]):
        vals.append(len(set(map(int, c)) & set(map(int, t))) / float(k))
    return float(np.mean(vals)) if vals else 0.0


def score_regret(returned_scores: np.ndarray, truth_scores: np.ndarray, eta: float = 1e-12) -> float:
    if returned_scores.size == 0:
        return float("nan")
    k_scores = returned_scores[:, -1]
    truth_k = truth_scores[:, min(truth_scores.shape[1] - 1, returned_scores.shape[1] - 1)]
    return float(np.mean((truth_k - k_scores) / (np.abs(truth_k) + eta)))


def hot_bytes_per_vector(d: int, residuals: bool = False) -> int:
    return int(d + 4 + 4 + (4 if residuals else 0) + 1)
