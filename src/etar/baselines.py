from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .profiling import rss_mb
from .threads import configure_faiss_threads
from .timing import latency_summary


def _valid_pq_m(d: int, requested: int) -> int:
    requested = max(1, min(int(requested), int(d)))
    if d % requested == 0:
        return requested
    for m in range(requested - 1, 0, -1):
        if d % m == 0:
            return m
    return 1


def _build_faiss_index(X: np.ndarray, kind: str = "flat", nlist: int = 64, **params):
    import faiss

    configure_faiss_threads()
    d = X.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, int(params.get("M", 32)), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = int(params.get("ef_search", params.get("efSearch", 64)))
    elif kind == "ivf":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, min(nlist, max(1, X.shape[0] // 10)), faiss.METRIC_INNER_PRODUCT)
        index.train(X)
        index.nprobe = min(int(params.get("nprobe", 8)), index.nlist)
    elif kind == "ivfpq":
        m = _valid_pq_m(d, int(params.get("m", max(1, min(16, d // 4)))))
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, min(nlist, max(1, X.shape[0] // 10)), m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(X)
        index.nprobe = min(int(params.get("nprobe", 8)), index.nlist)
    else:
        raise ValueError(kind)
    index.add(X)
    return index


def run_faiss(X: np.ndarray, Q: np.ndarray, k: int, kind: str = "flat", nlist: int = 64, mode: str = "online", **params) -> dict[str, Any]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    rss_before_build = rss_mb()
    t0 = perf_counter()
    index = _build_faiss_index(X, kind, nlist, **params)
    build_ms = (perf_counter() - t0) * 1000
    rss_after_build = rss_mb()
    if mode == "batch":
        t1 = perf_counter()
        scores, ids = index.search(Q, k)
        total_ms = (perf_counter() - t1) * 1000
        summary = latency_summary([total_ms / max(Q.shape[0], 1)])
        summary["batch_qps"] = float(Q.shape[0] / max(total_ms / 1000.0, 1e-12))
    else:
        ids = np.empty((Q.shape[0], k), dtype=np.int64)
        scores = np.empty((Q.shape[0], k), dtype=np.float32)
        latencies = []
        for i, q in enumerate(Q):
            t1 = perf_counter()
            s, got = index.search(q[None, :], k)
            latencies.append((perf_counter() - t1) * 1000)
            ids[i], scores[i] = got[0], s[0]
        summary = latency_summary(latencies)
        summary["batch_qps"] = np.nan
    summary["query_ms"] = summary["latency_ms_mean"]
    return {
        "name": f"faiss_{kind}",
        "ids": ids.astype(np.int64),
        "scores": scores.astype(np.float32),
        "build_ms": build_ms,
        "index_bytes": int(getattr(index, "ntotal", X.shape[0]) * X.shape[1] * 4),
        "rss_before_build_mb": rss_before_build,
        "rss_after_build_mb": rss_after_build,
        "rss_build_delta_mb": rss_after_build - rss_before_build,
        **summary,
    }


def run_hnswlib(X: np.ndarray, Q: np.ndarray, k: int, mode: str = "online", M: int = 16, ef_construction: int = 100, ef_search: int = 64) -> dict[str, Any]:
    import hnswlib

    X = np.asarray(X, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    t0 = perf_counter()
    index = hnswlib.Index(space="ip", dim=X.shape[1])
    index.init_index(max_elements=X.shape[0], ef_construction=int(ef_construction), M=int(M))
    index.add_items(X, np.arange(X.shape[0]), num_threads=1)
    index.set_ef(int(ef_search))
    build_ms = (perf_counter() - t0) * 1000
    if mode == "batch":
        t1 = perf_counter()
        ids, distances = index.knn_query(Q, k=k, num_threads=1)
        total_ms = (perf_counter() - t1) * 1000
        summary = latency_summary([total_ms / max(Q.shape[0], 1)])
        summary["batch_qps"] = float(Q.shape[0] / max(total_ms / 1000.0, 1e-12))
    else:
        ids = np.empty((Q.shape[0], k), dtype=np.int64)
        distances = np.empty((Q.shape[0], k), dtype=np.float32)
        latencies = []
        for i, q in enumerate(Q):
            t1 = perf_counter()
            got, dist = index.knn_query(q[None, :], k=k, num_threads=1)
            latencies.append((perf_counter() - t1) * 1000)
            ids[i], distances[i] = got[0], dist[0]
        summary = latency_summary(latencies)
        summary["batch_qps"] = np.nan
    summary["query_ms"] = summary["latency_ms_mean"]
    return {"name": "hnswlib", "ids": ids.astype(np.int64), "scores": -distances.astype(np.float32), "build_ms": build_ms, "index_bytes": 0, **summary}


def run_annoy(X: np.ndarray, Q: np.ndarray, k: int, trees: int = 20, search_k: int = -1, mode: str = "online") -> dict[str, Any]:
    from annoy import AnnoyIndex

    X = np.asarray(X, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    index = AnnoyIndex(X.shape[1], "dot")
    t0 = perf_counter()
    for i, row in enumerate(X):
        index.add_item(i, row)
    index.build(trees)
    build_ms = (perf_counter() - t0) * 1000
    ids = np.empty((Q.shape[0], k), dtype=np.int64)
    scores = np.empty((Q.shape[0], k), dtype=np.float32)
    latencies = []
    for i, q in enumerate(Q):
        t1 = perf_counter()
        got_ids, got_scores = index.get_nns_by_vector(q, k, search_k=int(search_k), include_distances=True)
        latencies.append((perf_counter() - t1) * 1000)
        ids[i, : len(got_ids)] = got_ids
        scores[i, : len(got_scores)] = got_scores
    summary = latency_summary(latencies)
    summary["batch_qps"] = np.nan
    summary["query_ms"] = summary["latency_ms_mean"]
    return {"name": "annoy", "ids": ids, "scores": scores, "build_ms": build_ms, "index_bytes": 0, **summary}


def run_scann(X: np.ndarray, Q: np.ndarray, k: int, mode: str = "online", leaves: int | None = None, leaves_to_search: int = 20, reorder: int | None = None) -> dict[str, Any]:
    import scann

    X = np.asarray(X, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    t0 = perf_counter()
    searcher = scann.scann_ops_pybind.builder(X, k, "dot_product").tree(
        num_leaves=int(leaves or min(200, max(1, X.shape[0] // 20))),
        num_leaves_to_search=int(leaves_to_search),
    ).score_ah(2).reorder(int(reorder or k * 2)).build()
    build_ms = (perf_counter() - t0) * 1000
    ids = np.empty((Q.shape[0], k), dtype=np.int64)
    scores = np.empty((Q.shape[0], k), dtype=np.float32)
    if mode == "batch" and hasattr(searcher, "search_batched"):
        t1 = perf_counter()
        ids, scores = searcher.search_batched(Q, final_num_neighbors=k)
        total_ms = (perf_counter() - t1) * 1000
        summary = latency_summary([total_ms / max(Q.shape[0], 1)])
        summary["batch_qps"] = float(Q.shape[0] / max(total_ms / 1000.0, 1e-12))
    else:
        latencies = []
        for i, q in enumerate(Q):
            t1 = perf_counter()
            ids[i], scores[i] = searcher.search(q, final_num_neighbors=k)
            latencies.append((perf_counter() - t1) * 1000)
        summary = latency_summary(latencies)
    summary["batch_qps"] = np.nan
    summary["query_ms"] = summary["latency_ms_mean"]
    return {"name": "scann", "ids": ids, "scores": scores, "build_ms": build_ms, "index_bytes": 0, **summary}
