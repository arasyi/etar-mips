from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import run_annoy, run_faiss, run_hnswlib, run_scann
from .core import (
    EtarIndex,
    adaptive_r_budget,
    cold_rerank_diagnostics,
    corpus_omega,
    resolve_etar_options,
    select_head_policy,
    tail_alpha,
)
from .data import load_hdf5
from .metrics import candidate_coverage, exact_topk, hot_bytes_per_vector, recall_at_k, score_regret, topk_from_scores
from .profiling import etar_storage_bytes, exact_storage_bytes, rss_mb
from .storage_native import CppStorageIndex
from .threads import configure_faiss_threads, set_single_thread_env
from .timing import latency_summary


INDEX_FREE_METHODS = ["exact_f32", "full_i8", "reduced_fixed", "reduced_energy", "etar", "etar_qrisk", "etar_iq", "etar_block_prune"]


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_nanpercentile(values, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanpercentile(arr, q))


def _cold_rerank_config(method_cfg: dict) -> dict:
    raw = method_cfg.get("cold_rerank", {})
    if isinstance(raw, bool):
        cold_cfg = {"enabled": raw}
    elif raw is None:
        cold_cfg = {}
    elif isinstance(raw, dict):
        cold_cfg = raw
    else:
        cold_cfg = {"enabled": bool(raw)}

    def get(name: str, default):
        flat = f"cold_rerank_{name}"
        if flat in method_cfg:
            return method_cfg[flat]
        return cold_cfg.get(name, default)

    sort_default = cold_cfg.get("sort", method_cfg.get("cold_rerank_sort", False))
    return {
        "cold_rerank_enabled": bool(get("enabled", False)),
        "cold_rerank_apply_to_wall_time": bool(get("apply_to_wall_time", False)),
        "cold_rerank_bandwidth_gbps": float(get("bandwidth_gbps", 0.0)),
        "cold_rerank_row_latency_ns": float(get("row_latency_ns", 0.0)),
        "cold_rerank_page_latency_ns": float(get("page_latency_ns", 0.0)),
        "cold_rerank_page_size": int(get("page_size", 4096)),
        "cold_rerank_sort_candidates": bool(get("sort_candidates", sort_default)),
    }


def _add_cold_rerank_summary(row: dict, values: dict) -> None:
    def arr(key: str) -> np.ndarray:
        return np.asarray(values.get(key, np.array([np.nan])), dtype=np.float64)

    row["cold_rows_per_query"] = _safe_nanmean(arr("cold_rerank_rows"))
    row["cold_pages_per_query"] = _safe_nanmean(arr("cold_rerank_pages"))
    bytes_mean = _safe_nanmean(arr("cold_rerank_bytes"))
    if not np.isnan(bytes_mean):
        row["cold_bytes_per_query"] = int(round(bytes_mean))
    row["cold_rerank_model_ms_mean"] = _safe_nanmean(arr("cold_rerank_model_ms"))
    row["cold_rerank_model_ms_p95"] = _safe_nanpercentile(arr("cold_rerank_model_ms"), 95)
    row["cold_rerank_penalty_ms_mean"] = _safe_nanmean(arr("cold_rerank_penalty_ms"))
    row["cold_rerank_penalty_ms_p95"] = _safe_nanpercentile(arr("cold_rerank_penalty_ms"), 95)
    row["cold_rerank_bandwidth_ms_mean"] = _safe_nanmean(arr("cold_rerank_bandwidth_ms"))
    row["cold_rerank_latency_ms_mean"] = _safe_nanmean(arr("cold_rerank_latency_ms"))


def _store_residuals(cfg: dict) -> bool:
    if "store_residuals" in cfg:
        return bool(cfg["store_residuals"])
    if "residuals" in cfg:
        return bool(cfg["residuals"])
    return float(cfg.get("alpha_q", 0.0) or 0.0) != 0.0


def _store_float32(cfg: dict) -> bool:
    preset = _storage_preset(cfg)
    if preset in {"8+32", "32+32"}:
        return True
    if preset in {"8+8", "8-shared", "32-shared"}:
        return False
    if "store_float32" in cfg:
        return bool(cfg["store_float32"])
    if "keep_float32" in cfg:
        return bool(cfg["keep_float32"])
    rerank_dtype = str(cfg.get("rerank_dtype", "float32")).lower()
    return rerank_dtype not in {"int8", "i8", "8bit", "quantized"}


def _store_codes_row(cfg: dict) -> bool:
    preset = _storage_preset(cfg)
    if preset == "8+8":
        return True
    if preset in {"8+32", "8-shared", "32+32", "32-shared"}:
        return False
    if "store_codes_row" in cfg:
        return bool(cfg["store_codes_row"])
    return bool(cfg.get("dual_8bit_storage", False))


def _storage_preset(cfg: dict) -> str:
    raw = cfg.get("storage_preset", cfg.get("storage_view", ""))
    value = str(raw).strip().lower().replace("_", "-") if raw is not None else ""
    aliases = {
        "8+32": "8+32",
        "8p32": "8+32",
        "i8-f32": "8+32",
        "int8-float32": "8+32",
        "i8col-f32row": "8+32",
        "8+8": "8+8",
        "8p8": "8+8",
        "i8-i8": "8+8",
        "int8-int8": "8+8",
        "i8col-i8row": "8+8",
        "8-shared": "8-shared",
        "8shared": "8-shared",
        "i8-shared": "8-shared",
        "int8-shared": "8-shared",
        "32+32": "32+32",
        "32p32": "32+32",
        "f32-f32": "32+32",
        "float32-float32": "32+32",
        "32-shared": "32-shared",
        "32shared": "32-shared",
        "f32-shared": "32-shared",
        "float32-shared": "32-shared",
    }
    return aliases.get(value, "")


def _query_quantized(cfg: dict) -> bool:
    mode = str(cfg.get("query_mode", cfg.get("scan_query_mode", "float"))).lower()
    return mode in {"int8", "i8", "8bit", "quantized", "q8"}


def _rerank_query_quantized(cfg: dict) -> bool:
    mode = str(cfg.get("rerank_query_mode", cfg.get("rerank_query_quantization", cfg.get("query_mode", "float")))).lower()
    return mode in {"int8", "i8", "8bit", "quantized", "q8"}


def _rerank_dtype_from_store(store_float32: bool) -> str:
    return "float32" if store_float32 else "int8"


def _etar_hot_bytes_per_vector(d: int, scan_dtype: str, scan_layout: str, residuals: bool = False) -> int:
    if scan_dtype == "float32":
        copies = 2 if scan_layout == "column" else 1
        return int(copies * 4 * d + 4 + 1)
    return hot_bytes_per_vector(d, residuals=residuals)


def _index_storage_profile(idx: EtarIndex) -> dict[str, int]:
    capacity = int(getattr(idx, "_capacity", idx.scan_capacity))
    return etar_storage_bytes(
        n=idx.n,
        d=idx.d,
        capacity=capacity,
        scan_dtype=idx.scan_dtype,
        residuals=bool(getattr(idx, "residuals_enabled", True)),
    )


def _add_memory_profile(row: dict, *, storage_backend: str, profile: dict[str, int], rss_before: float, rss_after: float) -> None:
    row["storage_backend"] = storage_backend
    row.update(profile)
    row["rss_before_build_mb"] = rss_before
    row["rss_after_build_mb"] = rss_after
    row["rss_build_delta_mb"] = rss_after - rss_before if not (np.isnan(rss_before) or np.isnan(rss_after)) else np.nan


def run_method_online(dataset_path: str | Path, method_cfg: dict, k: int = 10, limit_n: int | None = None, limit_q: int | None = None) -> pd.DataFrame:
    set_single_thread_env()
    configure_faiss_threads()
    X, Q = load_hdf5(dataset_path, limit_n, limit_q)
    truth_ids, truth_scores = exact_topk(X, Q, k)
    return run_method_online_loaded(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k)


def run_method_online_loaded(dataset_path: str | Path, X: np.ndarray, Q: np.ndarray, truth_ids: np.ndarray, truth_scores: np.ndarray, method_cfg: dict, k: int = 10) -> pd.DataFrame:
    set_single_thread_env()
    configure_faiss_threads()
    family = method_cfg.get("family", "etar")
    if family == "etar":
        return _run_etar_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k, mode="online")
    out = _run_ann_method(X, Q, k, method_cfg, mode="online")
    row = _ann_row(dataset_path, X, Q, truth_ids, out, method_cfg, "online", k)
    return pd.DataFrame([row])


def run_method_batch(dataset_path: str | Path, method_cfg: dict, k: int = 10, limit_n: int | None = None, limit_q: int | None = None) -> pd.DataFrame:
    set_single_thread_env()
    configure_faiss_threads()
    X, Q = load_hdf5(dataset_path, limit_n, limit_q)
    truth_ids, truth_scores = exact_topk(X, Q, k)
    return run_method_batch_loaded(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k)


def run_method_batch_loaded(dataset_path: str | Path, X: np.ndarray, Q: np.ndarray, truth_ids: np.ndarray, truth_scores: np.ndarray, method_cfg: dict, k: int = 10) -> pd.DataFrame:
    set_single_thread_env()
    configure_faiss_threads()
    family = method_cfg.get("family", "etar")
    if family == "etar":
        return _run_etar_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k, mode="batch")
    out = _run_ann_method(X, Q, k, method_cfg, mode="batch")
    row = _ann_row(dataset_path, X, Q, truth_ids, out, method_cfg, "batch", k)
    return pd.DataFrame([row])


def _run_etar_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k: int, mode: str) -> pd.DataFrame:
    method = method_cfg.get("method", "etar")
    if method == "exact_f32":
        return _run_exact_f32_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k, mode)
    if str(method_cfg.get("storage_backend", "numpy")).lower() == "cpp":
        return _run_cpp_storage_etar_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k, mode)
    rss_before_build = rss_mb()
    build_t0 = perf_counter()
    idx = EtarIndex(
        X,
        backend=method_cfg.get("backend", "auto"),
        residuals=_store_residuals(method_cfg),
        scan_quantization=method_cfg.get("scan_quantization"),
        scan_dtype=str(method_cfg.get("scan_dtype", "int8")),
        scan_layout=method_cfg.get("scan_layout"),
        scan_insert_policy=method_cfg.get("scan_insert_policy", "auto"),
        scan_growth_factor=float(method_cfg.get("scan_growth_factor", 2.0)),
        scan_initial_capacity_factor=float(method_cfg.get("scan_initial_capacity_factor", 1.25)),
        scan_shrink_threshold=float(method_cfg.get("scan_shrink_threshold", 0.25)),
        scan_shrink_factor=float(method_cfg.get("scan_shrink_factor", 0.5)),
        scan_compact_shrink=bool(method_cfg.get("scan_compact_shrink", True)),
        layout=method_cfg.get("layout", "identity"),
    )
    index_build_ms = (perf_counter() - build_t0) * 1000.0
    rss_after_build = rss_mb()

    alpha_tail_fixed = None if method_cfg.get("alpha_tail_fixed") is None else float(method_cfg.get("alpha_tail_fixed"))
    opts = resolve_etar_options(
        method,
        head_policy=method_cfg.get("head_policy"),
        tail_policy=method_cfg.get("tail_policy"),
        rerank=method_cfg.get("rerank") if "rerank" in method_cfg else None,
        r_policy=method_cfg.get("r_policy"),
        query_mode=method_cfg.get("query_mode", "float"),
        alpha_tail_fixed=alpha_tail_fixed,
        alpha_q=float(method_cfg.get("alpha_q", 0.0)),
        block_prune=bool(method_cfg.get("block_prune", False)),
    )
    r = int(method_cfg.get("r", method_cfg.get("R", 100)))
    if opts.force_r_to_k:
        r = k
    rho = float(method_cfg.get("rho", 0.9))
    h_value = method_cfg.get("h")
    h_value = None if h_value is None else int(h_value)
    h_max = method_cfg.get("h_max", h_value if h_value is not None else 64)
    h_max = None if h_max is None else int(h_max)
    common = {
        "k": k,
        "r": r,
        "rho": rho,
        "h_max": h_max,
        "h": h_value,
        "method": method,
        "head_policy": opts.head_policy,
        "tail_policy": opts.tail_policy,
        "rerank": opts.rerank,
        "r_policy": opts.r_policy,
        "lambda_tail": float(method_cfg.get("lambda_tail", 0.75)),
        "alpha_min": float(method_cfg.get("alpha_min", 0.05)),
        "alpha_max": float(method_cfg.get("alpha_max", 0.5)),
        "alpha_tail_fixed": alpha_tail_fixed,
        "alpha_q": 0.0 if idx.scan_dtype == "float32" else opts.alpha_q,
        "query_mode": opts.query_mode,
        "scan_dtype": idx.scan_dtype,
        "scan_layout": idx.scan_layout,
        "scan_quantization": idx.scan_quantization,
        "scan_insert_policy": idx.scan_insert_policy,
        "scan_growth_factor": idx.scan_growth_factor,
        "scan_initial_capacity_factor": idx.scan_initial_capacity_factor,
        "scan_shrink_threshold": idx.scan_shrink_threshold,
        "scan_shrink_factor": idx.scan_shrink_factor,
        "scan_compact_shrink": idx.scan_compact_shrink,
        "scan_stride": idx.scan_stride,
        "scan_capacity": idx.scan_capacity,
        "adaptive_r_target_recall": float(method_cfg.get("adaptive_r_target_recall", method_cfg.get("target_recall", 0.95))),
        "adaptive_r_probability": float(method_cfg.get("adaptive_r_probability", method_cfg.get("probability", 0.95))),
        "adaptive_r_mode": str(method_cfg.get("adaptive_r_mode", "event")),
        "prune_lower_bound": bool(method_cfg.get("prune_lower_bound", False)),
        "prune_lower_mode": str(method_cfg.get("prune_lower_mode", "global")),
        "prune_lower_scale": float(method_cfg.get("prune_lower_scale", 1.0)),
        "block_prune": opts.block_prune,
        "block_prune_scout_h": int(method_cfg.get("block_prune_scout_h", 16)),
        "block_prune_beam_factor": float(method_cfg.get("block_prune_beam_factor", 32.0)),
        "block_prune_gate": float(method_cfg.get("block_prune_gate", 0.75)),
        "block_prune_block_size": int(method_cfg.get("block_prune_block_size", 256)),
        "block_prune_cv_threshold": float(method_cfg.get("block_prune_cv_threshold", 0.0)),
        **_cold_rerank_config(method_cfg),
    }
    search_common = {
        k: v for k, v in common.items()
        if k not in {
            "scan_dtype",
            "scan_layout",
            "scan_quantization",
            "scan_insert_policy",
            "scan_growth_factor",
            "scan_initial_capacity_factor",
            "scan_shrink_threshold",
            "scan_shrink_factor",
            "scan_compact_shrink",
            "scan_stride",
            "scan_capacity",
        }
    }

    if mode == "batch":
        t0 = perf_counter()
        ids, scores, cand_ids, _, timings = idx.search_batch(Q, **search_common)
        total_ms = (perf_counter() - t0) * 1000
        measured_per_query = total_ms / max(Q.shape[0], 1)
        cold_penalty = _safe_nanmean(timings.get("cold_rerank_penalty_ms", np.array([0.0])))
        if np.isnan(cold_penalty):
            cold_penalty = 0.0
        simulated_per_query = measured_per_query + cold_penalty
        reported_per_query = simulated_per_query if common.get("cold_rerank_apply_to_wall_time") else measured_per_query
        reported_total_ms = reported_per_query * max(Q.shape[0], 1)
        lat = latency_summary([reported_per_query])
        row = _etar_base_row(dataset_path, X, Q, truth_ids, truth_scores, ids, scores, [c for c in cand_ids], method_cfg, idx.backend_name, common, mode)
        row["scan_dtype"] = idx.scan_dtype
        row["scan_layout"] = idx.scan_layout
        row["scan_quantization"] = idx.scan_quantization
        row["scan_insert_policy"] = idx.scan_insert_policy
        row["scan_stride"] = idx.scan_stride
        row["scan_capacity"] = idx.scan_capacity
        row.update(lat)
        row["query_ms"] = lat["latency_ms_mean"]
        row["measured_latency_ms_mean"] = measured_per_query
        row["simulated_latency_ms_mean"] = simulated_per_query
        row["batch_qps"] = float(Q.shape[0] / max(reported_total_ms / 1000.0, 1e-12))
        row["index_build_ms"] = index_build_ms
        row["native_scan_ms_mean"] = _safe_nanmean(timings.get("native_scan_ms", np.array([np.nan])))
        row["native_top_r_ms_mean"] = _safe_nanmean(timings.get("native_top_r_ms", np.array([np.nan])))
        row["native_rerank_ms_mean"] = _safe_nanmean(timings.get("native_rerank_ms", np.array([np.nan])))
        row["native_total_ms_mean"] = _safe_nanmean(timings.get("native_total_ms", np.array([np.nan])))
        row["simulated_native_total_ms_mean"] = _safe_nanmean(timings.get("simulated_native_total_ms", np.array([np.nan])))
        row["block_prune_survival_mean"] = _safe_nanmean(timings.get("native_block_prune_survival", np.array([np.nan])))
        row["block_prune_used_rate"] = _safe_nanmean(timings.get("native_block_prune_used", np.array([np.nan])))
        row["block_prune_fallback_rate"] = _safe_nanmean(timings.get("native_block_prune_fallback", np.array([np.nan])))
        row["mean_r"] = _safe_nanmean(timings.get("r", np.array([common["r"]], dtype=np.float32)))
        row["p95_r"] = float(np.nanpercentile(timings.get("r", np.array([common["r"]], dtype=np.float32)), 95))
        _add_memory_profile(row, storage_backend="numpy", profile=_index_storage_profile(idx), rss_before=rss_before_build, rss_after=rss_after_build)
        _add_cold_rerank_summary(row, timings)
        return pd.DataFrame([row])

    pred, pred_scores, candidates, diags = [], [], [], []
    for q in Q:
        res = idx.search(q, **search_common)
        pred.append(res.ids)
        pred_scores.append(res.scores)
        candidates.append(res.candidate_ids)
        diags.append(res.diagnostics)
    ids = np.vstack(pred)
    scores = np.vstack(pred_scores)
    lat = latency_summary([d["total_ms"] for d in diags])
    row = _etar_base_row(dataset_path, X, Q, truth_ids, truth_scores, ids, scores, candidates, method_cfg, idx.backend_name, common, mode)
    row["scan_dtype"] = idx.scan_dtype
    row["scan_layout"] = idx.scan_layout
    row["scan_quantization"] = idx.scan_quantization
    row["scan_insert_policy"] = idx.scan_insert_policy
    row["scan_stride"] = idx.scan_stride
    row["scan_capacity"] = idx.scan_capacity
    row.update(lat)
    row["query_ms"] = lat["latency_ms_mean"]
    row["batch_qps"] = np.nan
    row["index_build_ms"] = index_build_ms
    row["measured_latency_ms_mean"] = _safe_nanmean([d.get("measured_total_ms", d.get("total_ms", np.nan)) for d in diags])
    row["simulated_latency_ms_mean"] = _safe_nanmean([d.get("simulated_total_ms", d.get("total_ms", np.nan)) for d in diags])
    row["mean_h"] = float(np.mean([d["h"] for d in diags]))
    row["mean_retained_mass"] = float(np.mean([d.get("retained_mass", 1.0) for d in diags]))
    row["p50_h"] = float(np.percentile([d["h"] for d in diags], 50))
    row["p95_h"] = float(np.percentile([d["h"] for d in diags], 95))
    row["mean_r"] = float(np.mean([d.get("r", common["r"]) for d in diags]))
    row["p95_r"] = float(np.percentile([d.get("r", common["r"]) for d in diags], 95))
    row["mean_candidates"] = float(np.mean([len(c) for c in candidates]))
    row["p95_candidates"] = float(np.percentile([len(c) for c in candidates], 95))
    if common.get("r_policy") == "adaptive" or common.get("prune_lower_bound"):
        row["cold_bytes_per_query"] = int(4 * row["mean_candidates"] * X.shape[1])
    row["native_scan_ms_mean"] = _safe_nanmean([d.get("native_scan_ms", np.nan) for d in diags])
    row["native_top_r_ms_mean"] = _safe_nanmean([d.get("native_top_r_ms", np.nan) for d in diags])
    row["native_rerank_ms_mean"] = _safe_nanmean([d.get("native_rerank_ms", np.nan) for d in diags])
    row["native_total_ms_mean"] = _safe_nanmean([d.get("native_total_ms", np.nan) for d in diags])
    row["block_prune_survival_mean"] = _safe_nanmean([d.get("native_block_prune_survival", np.nan) for d in diags])
    row["block_prune_used_rate"] = _safe_nanmean([d.get("native_block_prune_used", np.nan) for d in diags])
    row["block_prune_fallback_rate"] = _safe_nanmean([d.get("native_block_prune_fallback", np.nan) for d in diags])
    _add_memory_profile(row, storage_backend="numpy", profile=_index_storage_profile(idx), rss_before=rss_before_build, rss_after=rss_after_build)
    _add_cold_rerank_summary(row, {key: [d.get(key, np.nan) for d in diags] for key in [
        "cold_rerank_rows",
        "cold_rerank_bytes",
        "cold_rerank_pages",
        "cold_rerank_bandwidth_ms",
        "cold_rerank_latency_ms",
        "cold_rerank_model_ms",
        "cold_rerank_penalty_ms",
    ]})
    return pd.DataFrame([row])


def _run_cpp_storage_etar_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k: int, mode: str) -> pd.DataFrame:
    if mode not in {"online", "batch"}:
        raise NotImplementedError("C++ storage ETAR supports online and batch modes only")
    if str(method_cfg.get("scan_dtype", "int8")) != "int8":
        raise NotImplementedError("C++ storage ETAR supports scan_dtype=int8 only")
    if str(method_cfg.get("scan_quantization", "row")) != "row":
        raise NotImplementedError("C++ storage ETAR supports row quantization only")
    if str(method_cfg.get("scan_layout", "column")) != "column":
        raise NotImplementedError("C++ storage ETAR supports column scan layout only")
    query_quantized = _query_quantized(method_cfg)
    rerank_query_quantized = _rerank_query_quantized(method_cfg)

    rss_before_build = rss_mb()
    build_t0 = perf_counter()
    idx = CppStorageIndex(
        X,
        initial_capacity_factor=float(method_cfg.get("scan_initial_capacity_factor", 1.25)),
        growth_factor=float(method_cfg.get("scan_growth_factor", 2.0)),
        store_residuals=_store_residuals(method_cfg),
        store_float32=_store_float32(method_cfg),
        store_codes_row=_store_codes_row(method_cfg),
    )
    index_build_ms = (perf_counter() - build_t0) * 1000.0
    rss_after_build = rss_mb()

    method = str(method_cfg.get("method", "etar"))
    alpha_tail_fixed = None if method_cfg.get("alpha_tail_fixed") is None else float(method_cfg.get("alpha_tail_fixed"))
    opts = resolve_etar_options(
        method,
        head_policy=method_cfg.get("head_policy"),
        tail_policy=method_cfg.get("tail_policy"),
        rerank=method_cfg.get("rerank") if "rerank" in method_cfg else None,
        r_policy=method_cfg.get("r_policy"),
        query_mode=method_cfg.get("query_mode", "float"),
        alpha_tail_fixed=alpha_tail_fixed,
        alpha_q=0.0,
        block_prune=bool(method_cfg.get("block_prune", False)),
    )
    if opts.alpha_q != 0.0:
        raise NotImplementedError("C++ storage ETAR does not implement query quantization risk")

    r = int(method_cfg.get("r", method_cfg.get("R", 100)))
    if opts.force_r_to_k:
        r = k
    rho = float(method_cfg.get("rho", 0.9))
    h_value = method_cfg.get("h")
    h_value = None if h_value is None else int(h_value)
    h_max = method_cfg.get("h_max", h_value if h_value is not None else 64)
    h_max = None if h_max is None else int(h_max)
    omega = corpus_omega(X)
    norms = np.linalg.norm(np.asarray(X, dtype=np.float32), axis=1)
    mean_norm = float(np.mean(norms)) if norms.size else 0.0
    norm_cv = float(np.std(norms) / mean_norm) if mean_norm > 0.0 else 0.0
    common = {
        "k": k,
        "r": r,
        "rho": rho,
        "h_max": h_max,
        "h": h_value,
        "method": method,
        "head_policy": opts.head_policy,
        "tail_policy": opts.tail_policy,
        "rerank": opts.rerank,
        "r_policy": opts.r_policy,
        "lambda_tail": float(method_cfg.get("lambda_tail", 0.75)),
        "alpha_min": float(method_cfg.get("alpha_min", 0.05)),
        "alpha_max": float(method_cfg.get("alpha_max", 0.5)),
        "alpha_tail_fixed": alpha_tail_fixed,
        "alpha_q": 0.0,
        "residuals": idx.residuals_enabled,
        "store_float32": idx.store_float32,
        "store_codes_row": idx.store_codes_row,
        "storage_preset": _storage_preset(method_cfg) or ("8+32" if idx.store_float32 else ("8+8" if idx.store_codes_row else "8-shared")),
        "rerank_dtype": idx.rerank_dtype,
        "rerank_layout": idx.rerank_layout,
        "query_mode": "int8" if query_quantized else "float",
        "query_quantization": "int8" if query_quantized else "none",
        "rerank_query_mode": "int8" if rerank_query_quantized else "float",
        "rerank_query_quantization": "int8" if rerank_query_quantized else "none",
        "scan_dtype": "int8",
        "scan_layout": "column",
        "scan_quantization": "row",
        "scan_insert_policy": "cpp_storage",
        "scan_growth_factor": float(method_cfg.get("scan_growth_factor", 2.0)),
        "scan_initial_capacity_factor": float(method_cfg.get("scan_initial_capacity_factor", 1.25)),
        "scan_shrink_threshold": method_cfg.get("scan_shrink_threshold", np.nan),
        "scan_shrink_factor": method_cfg.get("scan_shrink_factor", np.nan),
        "scan_compact_shrink": method_cfg.get("scan_compact_shrink", np.nan),
        "scan_stride": idx.capacity,
        "scan_capacity": idx.capacity,
        "adaptive_r_target_recall": float(method_cfg.get("adaptive_r_target_recall", method_cfg.get("target_recall", 0.95))),
        "adaptive_r_probability": float(method_cfg.get("adaptive_r_probability", method_cfg.get("probability", 0.95))),
        "adaptive_r_mode": str(method_cfg.get("adaptive_r_mode", "event")),
        "prune_lower_bound": bool(method_cfg.get("prune_lower_bound", False)),
        "prune_lower_mode": str(method_cfg.get("prune_lower_mode", "global")),
        "prune_lower_scale": float(method_cfg.get("prune_lower_scale", 1.0)),
        "block_prune": opts.block_prune,
        "block_prune_scout_h": int(method_cfg.get("block_prune_scout_h", 16)),
        "block_prune_beam_factor": float(method_cfg.get("block_prune_beam_factor", 32.0)),
        "block_prune_gate": float(method_cfg.get("block_prune_gate", 0.75)),
        "block_prune_block_size": int(method_cfg.get("block_prune_block_size", 256)),
        "block_prune_cv_threshold": float(method_cfg.get("block_prune_cv_threshold", 0.0)),
        **_cold_rerank_config(method_cfg),
    }

    pred, pred_scores, candidates, diags = [], [], [], []
    for q in np.asarray(Q, dtype=np.float32):
        t0 = perf_counter()
        t_pre0 = perf_counter()
        head, retained, tail_norm_value, h = select_head_policy(q, rho, h_max, opts.head_policy)
        r_query = r
        if opts.r_policy == "adaptive":
            r_query = min(
                idx.live_count,
                adaptive_r_budget(
                    q,
                    head,
                    omega,
                    idx.n,
                    k,
                    r,
                    target_recall=common["adaptive_r_target_recall"],
                    probability=common["adaptive_r_probability"],
                    mode=common["adaptive_r_mode"],
                ),
            )
        use_tail = opts.tail_policy != "none"
        alpha_tail = 0.0
        if use_tail:
            if opts.tail_policy == "fixed":
                if alpha_tail_fixed is None:
                    raise ValueError("alpha_tail_fixed is required when tail_policy='fixed'")
                alpha_tail = float(alpha_tail_fixed)
            else:
                alpha_tail = tail_alpha(
                    q,
                    head,
                    omega,
                    idx.live_count,
                    r_query,
                    tail_norm_value,
                    common["lambda_tail"],
                    common["alpha_min"],
                    common["alpha_max"],
                )
        preprocess_ms = (perf_counter() - t_pre0) * 1000.0
        prune_code = 0
        if common["prune_lower_bound"]:
            prune_code = 2 if common["prune_lower_mode"] in {"candidate", "candidates", "topr"} else 1
        block_prune_runtime = bool(opts.block_prune and norm_cv >= float(common["block_prune_cv_threshold"]))
        ids, scores, cand_ids, _, timings = idx.search_profile(
            q,
            head,
            k=k,
            r=r_query,
            use_tail=use_tail,
            alpha_tail=alpha_tail,
            tail_norm=tail_norm_value,
            no_rerank=not opts.rerank,
            prune_lower_bound=prune_code,
            prune_lower_scale=common["prune_lower_scale"],
            block_prune=block_prune_runtime,
            block_prune_scout_h=common["block_prune_scout_h"],
            block_prune_beam_factor=common["block_prune_beam_factor"],
            block_prune_gate=common["block_prune_gate"],
            block_prune_block_size=common["block_prune_block_size"],
            sort_candidates=bool(common.get("cold_rerank_sort_candidates", False)),
            quantize_query=query_quantized,
            rerank_quantize_query=rerank_query_quantized,
        )
        total_ms = (perf_counter() - t0) * 1000.0
        cold_stats = cold_rerank_diagnostics(
            cand_ids if opts.rerank else np.empty(0, dtype=np.int64),
            X.shape[1],
            float(timings.get("native_rerank_ms", 0.0)),
            enabled=bool(common["cold_rerank_enabled"] and opts.rerank and common["rerank_dtype"] == "float32"),
            bandwidth_gbps=float(common["cold_rerank_bandwidth_gbps"]),
            row_latency_ns=float(common["cold_rerank_row_latency_ns"]),
            page_latency_ns=float(common["cold_rerank_page_latency_ns"]),
            page_size=int(common["cold_rerank_page_size"]),
        )
        simulated_total_ms = total_ms + float(cold_stats["cold_rerank_penalty_ms"])
        pred.append(ids)
        pred_scores.append(scores)
        candidates.append(cand_ids)
        diags.append({
            "total_ms": simulated_total_ms if common["cold_rerank_apply_to_wall_time"] else total_ms,
            "measured_total_ms": total_ms,
            "simulated_total_ms": simulated_total_ms,
            "h": h,
            "retained_mass": retained,
            "r": r_query,
            "native_scan_ms": timings.get("native_scan_ms", np.nan),
            "native_top_r_ms": timings.get("native_top_r_ms", np.nan),
            "native_rerank_ms": timings.get("native_rerank_ms", np.nan),
            "native_total_ms": timings.get("native_total_ms", np.nan),
            "native_block_prune_survival": timings.get("native_block_prune_survival", np.nan),
            "native_block_prune_used": timings.get("native_block_prune_used", np.nan),
            "native_block_prune_fallback": timings.get("native_block_prune_fallback", np.nan),
            "preprocess_ms": preprocess_ms,
            "python_overhead_ms": max(0.0, total_ms - preprocess_ms - float(timings.get("native_total_ms", 0.0))),
            "adaptive_r": bool(opts.r_policy == "adaptive"),
            "r_max": int(r),
            "norm_cv": norm_cv,
            **cold_stats,
        })

    ids = np.vstack(pred)
    scores = np.vstack(pred_scores)
    lat = latency_summary([d["total_ms"] for d in diags])
    row = _etar_base_row(dataset_path, X, Q, truth_ids, truth_scores, ids, scores, candidates, method_cfg, "cpp_storage", common, mode)
    row.update(lat)
    row["query_ms"] = lat["latency_ms_mean"]
    row["batch_qps"] = (
        float(Q.shape[0] / max(sum(d["total_ms"] for d in diags) / 1000.0, 1e-12))
        if mode == "batch"
        else np.nan
    )
    row["index_build_ms"] = index_build_ms
    row["measured_latency_ms_mean"] = _safe_nanmean([d.get("measured_total_ms", np.nan) for d in diags])
    row["simulated_latency_ms_mean"] = _safe_nanmean([d.get("simulated_total_ms", np.nan) for d in diags])
    row["mean_h"] = float(np.mean([d["h"] for d in diags]))
    row["mean_retained_mass"] = float(np.mean([d.get("retained_mass", 1.0) for d in diags]))
    row["p50_h"] = float(np.percentile([d["h"] for d in diags], 50))
    row["p95_h"] = float(np.percentile([d["h"] for d in diags], 95))
    row["mean_r"] = float(np.mean([d.get("r", common["r"]) for d in diags]))
    row["p95_r"] = float(np.percentile([d.get("r", common["r"]) for d in diags], 95))
    row["mean_candidates"] = float(np.mean([len(c) for c in candidates]))
    row["p95_candidates"] = float(np.percentile([len(c) for c in candidates], 95))
    if common.get("r_policy") == "adaptive" or common.get("prune_lower_bound"):
        row["cold_bytes_per_query"] = int(4 * row["mean_candidates"] * X.shape[1])
    row["native_scan_ms_mean"] = _safe_nanmean([d.get("native_scan_ms", np.nan) for d in diags])
    row["native_top_r_ms_mean"] = _safe_nanmean([d.get("native_top_r_ms", np.nan) for d in diags])
    row["native_rerank_ms_mean"] = _safe_nanmean([d.get("native_rerank_ms", np.nan) for d in diags])
    row["native_total_ms_mean"] = _safe_nanmean([d.get("native_total_ms", np.nan) for d in diags])
    row["simulated_native_total_ms_mean"] = _safe_nanmean([
        d.get("native_total_ms", np.nan) + d.get("cold_rerank_penalty_ms", 0.0)
        for d in diags
    ])
    row["block_prune_survival_mean"] = _safe_nanmean([d.get("native_block_prune_survival", np.nan) for d in diags])
    row["block_prune_used_rate"] = _safe_nanmean([d.get("native_block_prune_used", np.nan) for d in diags])
    row["block_prune_fallback_rate"] = _safe_nanmean([d.get("native_block_prune_fallback", np.nan) for d in diags])
    _add_memory_profile(row, storage_backend="cpp", profile=idx.storage_profile, rss_before=rss_before_build, rss_after=rss_after_build)
    _add_cold_rerank_summary(row, {key: [d.get(key, np.nan) for d in diags] for key in [
        "cold_rerank_rows",
        "cold_rerank_bytes",
        "cold_rerank_pages",
        "cold_rerank_bandwidth_ms",
        "cold_rerank_latency_ms",
        "cold_rerank_model_ms",
        "cold_rerank_penalty_ms",
    ]})
    if common["rerank_dtype"] == "int8" and opts.rerank:
        row["cold_bytes_per_query"] = int(round(row["mean_candidates"] * X.shape[1]))
    idx.close()
    return pd.DataFrame([row])


def _run_exact_f32_method(dataset_path, X, Q, truth_ids, truth_scores, method_cfg, k: int, mode: str) -> pd.DataFrame:
    rss_before_build = rss_mb()
    rss_after_build = rss_mb()
    ids = np.empty((Q.shape[0], k), dtype=np.int64)
    scores = np.empty((Q.shape[0], k), dtype=np.float32)
    latencies = []
    if mode == "batch":
        t0 = perf_counter()
        ids, scores = exact_topk(X, Q, k)
        total_ms = (perf_counter() - t0) * 1000.0
        lat = latency_summary([total_ms / max(Q.shape[0], 1)])
        batch_qps = float(Q.shape[0] / max(total_ms / 1000.0, 1e-12))
    else:
        for i, q in enumerate(Q):
            t0 = perf_counter()
            all_scores = X @ q
            cand, cand_scores = topk_from_scores(all_scores, k)
            latencies.append((perf_counter() - t0) * 1000.0)
            ids[i] = cand.astype(np.int64)
            scores[i] = cand_scores.astype(np.float32)
        lat = latency_summary(latencies)
        batch_qps = np.nan
    common = {
        "method": "exact_f32",
        "k": k,
        "r": k,
        "rho": 1.0,
        "h_max": X.shape[1],
        "h": X.shape[1],
        "head_policy": "full",
        "tail_policy": "none",
        "rerank": True,
        "r_policy": "fixed",
        "scan_dtype": "float32",
        "scan_layout": "row",
        "scan_quantization": "none",
    }
    row = _etar_base_row(dataset_path, X, Q, truth_ids, truth_scores, ids, scores, [c for c in ids], method_cfg, "numpy", common, mode)
    row.update(lat)
    row["query_ms"] = lat["latency_ms_mean"]
    row["batch_qps"] = batch_qps
    row["index_build_ms"] = 0.0
    row["mean_h"] = X.shape[1]
    row["mean_retained_mass"] = 1.0
    row["p50_h"] = X.shape[1]
    row["p95_h"] = X.shape[1]
    row["native_scan_ms_mean"] = np.nan
    row["native_top_r_ms_mean"] = np.nan
    row["native_rerank_ms_mean"] = np.nan
    _add_memory_profile(row, storage_backend="numpy", profile=exact_storage_bytes(n=X.shape[0], d=X.shape[1]), rss_before=rss_before_build, rss_after=rss_after_build)
    return pd.DataFrame([row])


def _etar_base_row(dataset_path, X, Q, truth_ids, truth_scores, ids, scores, candidates, method_cfg, backend, common, mode):
    return {
        "dataset": str(dataset_path),
        "method": method_cfg.get("name", common["method"]),
        "method_impl": common["method"],
        "family": "etar",
        "config_id": method_cfg.get("config_id", method_cfg.get("name", common["method"])),
        "backend": backend,
        "storage_backend": common.get("storage_backend", method_cfg.get("storage_backend", "numpy")),
        "scan_dtype": common.get("scan_dtype", "int8"),
        "scan_layout": common.get("scan_layout", "row"),
        "scan_quantization": common.get("scan_quantization", method_cfg.get("scan_quantization", "row")),
        "scan_insert_policy": common.get("scan_insert_policy", method_cfg.get("scan_insert_policy", "auto")),
        "scan_growth_factor": common.get("scan_growth_factor", method_cfg.get("scan_growth_factor", np.nan)),
        "scan_initial_capacity_factor": common.get("scan_initial_capacity_factor", method_cfg.get("scan_initial_capacity_factor", np.nan)),
        "scan_shrink_threshold": common.get("scan_shrink_threshold", method_cfg.get("scan_shrink_threshold", np.nan)),
        "scan_shrink_factor": common.get("scan_shrink_factor", method_cfg.get("scan_shrink_factor", np.nan)),
        "scan_compact_shrink": common.get("scan_compact_shrink", method_cfg.get("scan_compact_shrink", np.nan)),
        "scan_stride": common.get("scan_stride", np.nan),
        "scan_capacity": common.get("scan_capacity", np.nan),
        "store_float32": bool(common.get("store_float32", True)),
        "store_codes_row": bool(common.get("store_codes_row", False)),
        "storage_preset": common.get("storage_preset", method_cfg.get("storage_preset", "")),
        "rerank_dtype": common.get("rerank_dtype", _rerank_dtype_from_store(bool(common.get("store_float32", True)))),
        "rerank_layout": common.get("rerank_layout", common.get("rerank_dtype", _rerank_dtype_from_store(bool(common.get("store_float32", True))))),
        "query_mode": common.get("query_mode", method_cfg.get("query_mode", "float")),
        "query_quantization": common.get("query_quantization", "none"),
        "rerank_query_mode": common.get("rerank_query_mode", method_cfg.get("rerank_query_mode", method_cfg.get("query_mode", "float"))),
        "rerank_query_quantization": common.get("rerank_query_quantization", "none"),
        "layout": method_cfg.get("layout", "identity"),
        "N": X.shape[0],
        "d": X.shape[1],
        "Q": Q.shape[0],
        "K": common["k"],
        "R": common["r"],
        "rho": common["rho"],
        "h_max": common["h_max"],
        "h": common.get("h"),
        "head_policy": common.get("head_policy"),
        "tail_policy": common.get("tail_policy"),
        "r_policy": common.get("r_policy"),
        "alpha_tail_fixed": common.get("alpha_tail_fixed"),
        "alpha_q": common.get("alpha_q", 0.0),
        "block_prune": common.get("block_prune", False),
        "block_prune_scout_h": common.get("block_prune_scout_h", np.nan),
        "block_prune_beam_factor": common.get("block_prune_beam_factor", np.nan),
        "block_prune_gate": common.get("block_prune_gate", np.nan),
        "block_prune_block_size": common.get("block_prune_block_size", np.nan),
        "block_prune_cv_threshold": common.get("block_prune_cv_threshold", np.nan),
        "cold_rerank_enabled": common.get("cold_rerank_enabled", False),
        "cold_rerank_apply_to_wall_time": common.get("cold_rerank_apply_to_wall_time", False),
        "cold_rerank_bandwidth_gbps": common.get("cold_rerank_bandwidth_gbps", 0.0),
        "cold_rerank_row_latency_ns": common.get("cold_rerank_row_latency_ns", 0.0),
        "cold_rerank_page_latency_ns": common.get("cold_rerank_page_latency_ns", 0.0),
        "cold_rerank_page_size": common.get("cold_rerank_page_size", 4096),
        "cold_rerank_sort_candidates": common.get("cold_rerank_sort_candidates", False),
        "rerank": bool(common.get("rerank", common["method"] not in {"reduced_fixed_no_rerank", "reduced_energy_no_rerank", "etar_no_rerank"})),
        "recall_at_k": recall_at_k(ids, truth_ids, common["k"]),
        "top1_recall": recall_at_k(ids, truth_ids, 1),
        "candidate_coverage": candidate_coverage(candidates, truth_ids, common["k"]),
        "score_regret": score_regret(scores, truth_scores),
        "online_or_batch": mode,
        "hot_bytes_per_vector": _etar_hot_bytes_per_vector(
            X.shape[1],
            common.get("scan_dtype", "int8"),
            common.get("scan_layout", "row"),
            bool(common.get("residuals", False)),
        ),
        "cold_bytes_per_query": (
            0
            if not bool(common.get("rerank", True))
            else int((1 if common.get("rerank_dtype", "float32") == "int8" else 4) * common["r"] * X.shape[1])
        ),
    }


def _run_ann_method(X, Q, k: int, method_cfg: dict, mode: str):
    family = method_cfg.get("family")
    if family == "faiss":
        params = {k: v for k, v in method_cfg.items() if k not in {"name", "family", "index", "modes", "config_id", "nlist"}}
        return run_faiss(X, Q, k, kind=method_cfg.get("index", "flat"), nlist=int(method_cfg.get("nlist", 64)), mode=mode, **params)
    if family == "hnswlib":
        return run_hnswlib(X, Q, k, mode=mode, M=int(method_cfg.get("M", 16)), ef_construction=int(method_cfg.get("ef_construction", 100)), ef_search=int(method_cfg.get("ef_search", 64)))
    if family == "annoy":
        return run_annoy(X, Q, k, trees=int(method_cfg.get("trees", 20)), search_k=int(method_cfg.get("search_k", -1)), mode=mode)
    if family == "scann":
        return run_scann(X, Q, k, mode=mode, leaves=method_cfg.get("leaves"), leaves_to_search=int(method_cfg.get("leaves_to_search", 20)), reorder=method_cfg.get("reorder"))
    raise ValueError(f"unknown benchmark family {family!r}")


def _ann_row(dataset_path, X, Q, truth_ids, out, method_cfg, mode: str, k: int) -> dict:
    strict_recall = recall_at_k(out["ids"], truth_ids, k)
    strict_top1 = recall_at_k(out["ids"], truth_ids, 1)
    exact_exhaustive = method_cfg.get("family") == "faiss" and method_cfg.get("index", "flat") == "flat"
    reported_recall = 1.0 if exact_exhaustive else strict_recall
    reported_top1 = 1.0 if exact_exhaustive else strict_top1
    index_payload_bytes = int(out.get("index_bytes", 0))
    rss_before = out.get("rss_before_build_mb", np.nan)
    rss_after = out.get("rss_after_build_mb", np.nan)
    return {
        "dataset": str(dataset_path),
        "method": method_cfg.get("name", out["name"]),
        "method_impl": out["name"],
        "family": method_cfg.get("family"),
        "config_id": method_cfg.get("config_id", method_cfg.get("name", out["name"])),
        "backend": "ann",
        "storage_backend": f"{method_cfg.get('family')}:{method_cfg.get('index', out['name'])}",
        "N": X.shape[0],
        "d": X.shape[1],
        "Q": Q.shape[0],
        "K": k,
        "R": k,
        "rho": np.nan,
        "h_max": np.nan,
        "recall_at_k": reported_recall,
        "top1_recall": reported_top1,
        "candidate_coverage": reported_recall,
        "strict_recall_at_k": strict_recall,
        "strict_top1_recall": strict_top1,
        "tie_adjusted_exact": bool(exact_exhaustive and strict_recall < 1.0),
        "query_ms": out["query_ms"],
        "latency_ms_mean": out["latency_ms_mean"],
        "latency_ms_p50": out["latency_ms_p50"],
        "latency_ms_p95": out["latency_ms_p95"],
        "latency_ms_min": out["latency_ms_min"],
        "latency_ms_max": out["latency_ms_max"],
        "online_or_batch": mode,
        "batch_qps": out.get("batch_qps", np.nan),
        "build_ms": out["build_ms"],
        "index_build_ms": out["build_ms"],
        "index_payload_bytes_estimate": index_payload_bytes,
        "index_payload_mb_estimate": float(index_payload_bytes / (1024.0 * 1024.0)),
        "rss_before_build_mb": rss_before,
        "rss_after_build_mb": rss_after,
        "rss_build_delta_mb": out.get(
            "rss_build_delta_mb",
            rss_after - rss_before if not (np.isnan(rss_before) or np.isnan(rss_after)) else np.nan,
        ),
        "hot_bytes_per_vector": index_payload_bytes / max(X.shape[0], 1),
        "cold_bytes_per_query": 0,
        "status": "ok",
        "mean_h": out.get("mean_h", np.nan),
        "native_top_r_ms_mean": out.get("native_top_r_ms_mean", np.nan),
        "native_rerank_ms_mean": out.get("native_rerank_ms_mean", np.nan),
        "native_total_ms_mean": out.get("native_total_ms_mean", np.nan),
        "wall_latency_ms_mean": out.get("wall_latency_ms_mean", np.nan),
    }


def run_index_free(
    dataset_path: str | Path,
    backend: str = "auto",
    k: int = 10,
    r_values: Iterable[int] = (10, 50, 100),
    rho_values: Iterable[float] = (0.8, 0.9),
    h_max_values: Iterable[int] = (16, 32, 64),
    limit_n: int | None = None,
    limit_q: int | None = 50,
) -> pd.DataFrame:
    set_single_thread_env()
    X, Q = load_hdf5(dataset_path, limit_n, limit_q)
    truth_ids, truth_scores = exact_topk(X, Q, k)
    idx = EtarIndex(X, backend=backend, residuals=True)
    rows = []
    for method in INDEX_FREE_METHODS:
        rs = [k] if method == "exact_f32" else r_values
        rhos = [0.9] if method in {"exact_f32", "full_i8", "reduced_fixed"} else rho_values
        hs = [X.shape[1]] if method in {"exact_f32", "full_i8"} else h_max_values
        for r in rs:
            for rho in rhos:
                for h_max in hs:
                    pred, pred_scores, candidates, diags = [], [], [], []
                    for q in Q:
                        res = idx.search(q, k=k, r=r, rho=rho, h_max=h_max, method=method)
                        pred.append(res.ids)
                        pred_scores.append(res.scores)
                        candidates.append(res.candidate_ids)
                        diags.append(res.diagnostics)
                    pred_arr = np.vstack(pred)
                    pred_scores_arr = np.vstack(pred_scores)
                    lat = latency_summary([d["total_ms"] for d in diags])
                    native_scan = [d.get("native_scan_ms", np.nan) for d in diags]
                    native_topr = [d.get("native_top_r_ms", np.nan) for d in diags]
                    native_rerank = [d.get("native_rerank_ms", np.nan) for d in diags]
                    rows.append({
                        "dataset": str(dataset_path),
                        "method": method,
                        "backend": idx.backend_name,
                        "N": X.shape[0],
                        "d": X.shape[1],
                        "Q": Q.shape[0],
                        "K": k,
                        "R": r,
                        "rho": rho,
                        "h_max": h_max,
                        "mean_h": float(np.mean([d["h"] for d in diags])),
                        "mean_retained_mass": float(np.mean([d.get("retained_mass", 1.0) for d in diags])),
                        "p50_h": float(np.percentile([d["h"] for d in diags], 50)),
                        "p95_h": float(np.percentile([d["h"] for d in diags], 95)),
                        "recall_at_k": recall_at_k(pred_arr, truth_ids, k),
                        "top1_recall": recall_at_k(pred_arr, truth_ids, 1),
                        "candidate_coverage": candidate_coverage(candidates, truth_ids, k),
                        "score_regret": score_regret(pred_scores_arr, truth_scores),
                        "query_ms": lat["latency_ms_mean"],
                        **lat,
                        "online_or_batch": "online",
                        "batch_qps": np.nan,
                        "native_scan_ms_mean": _safe_nanmean(native_scan),
                        "native_top_r_ms_mean": _safe_nanmean(native_topr),
                        "native_rerank_ms_mean": _safe_nanmean(native_rerank),
                        "hot_bytes_per_vector": hot_bytes_per_vector(X.shape[1], residuals=True),
                        "cold_bytes_per_query": int(4 * r * X.shape[1]),
                    })
    return pd.DataFrame(rows)


def run_ann_baselines(dataset_path: str | Path, k: int = 10, limit_n: int | None = None, limit_q: int | None = 50) -> pd.DataFrame:
    set_single_thread_env()
    configure_faiss_threads()
    X, Q = load_hdf5(dataset_path, limit_n, limit_q)
    truth_ids, _ = exact_topk(X, Q, k)
    rows = []
    runners = [
        lambda: run_faiss(X, Q, k, "flat", mode="online"),
        lambda: run_faiss(X, Q, k, "hnsw", mode="online"),
        lambda: run_faiss(X, Q, k, "ivf", mode="online"),
        lambda: run_faiss(X, Q, k, "ivfpq", mode="online"),
        lambda: run_hnswlib(X, Q, k, mode="online"),
        lambda: run_annoy(X, Q, k, mode="online"),
        lambda: run_scann(X, Q, k, mode="online"),
    ]
    for runner in runners:
        try:
            out = runner()
            rows.append({
                "dataset": str(dataset_path),
                "method": out["name"],
                "backend": "ann",
                "N": X.shape[0],
                "d": X.shape[1],
                "Q": Q.shape[0],
                "K": k,
                "R": k,
                "rho": np.nan,
                "h_max": np.nan,
                "mean_h": np.nan,
                "mean_retained_mass": np.nan,
                "recall_at_k": recall_at_k(out["ids"], truth_ids, k),
                "top1_recall": recall_at_k(out["ids"], truth_ids, 1),
                "candidate_coverage": recall_at_k(out["ids"], truth_ids, k),
                "query_ms": out["query_ms"],
                "latency_ms_mean": out["latency_ms_mean"],
                "latency_ms_p50": out["latency_ms_p50"],
                "latency_ms_p95": out["latency_ms_p95"],
                "latency_ms_min": out["latency_ms_min"],
                "latency_ms_max": out["latency_ms_max"],
                "online_or_batch": "online",
                "batch_qps": np.nan,
                "build_ms": out["build_ms"],
                "hot_bytes_per_vector": out["index_bytes"] / max(X.shape[0], 1),
                "cold_bytes_per_query": 0,
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"dataset": str(dataset_path), "method": getattr(runner, "__name__", "ann"), "status": f"failed: {exc}"})
    return pd.DataFrame(rows)


def run_batch_throughput(dataset_path: str | Path, backend: str = "auto", k: int = 10, r: int = 100, rho: float = 0.9, h_max: int = 64, limit_n: int | None = None, limit_q: int | None = 50) -> pd.DataFrame:
    set_single_thread_env()
    configure_faiss_threads()
    X, Q = load_hdf5(dataset_path, limit_n, limit_q)
    truth_ids, truth_scores = exact_topk(X, Q, k)
    rows = []
    idx = EtarIndex(X, backend=backend, residuals=True)
    for method in ["exact_f32", "full_i8", "reduced_energy", "etar", "etar_iq"]:
        t0 = perf_counter()
        ids, scores, cand_ids, _, timings = idx.search_batch(Q, k=k, r=r, rho=rho, h_max=h_max, method=method)
        total_ms = (perf_counter() - t0) * 1000
        per_q = total_ms / max(Q.shape[0], 1)
        lat = latency_summary([per_q])
        rows.append({
            "dataset": str(dataset_path),
            "method": method,
            "backend": idx.backend_name,
            "N": X.shape[0],
            "d": X.shape[1],
            "Q": Q.shape[0],
            "K": k,
            "R": r,
            "rho": rho,
            "h_max": h_max,
            "recall_at_k": recall_at_k(ids, truth_ids, k),
            "top1_recall": recall_at_k(ids, truth_ids, 1),
            "candidate_coverage": candidate_coverage([c for c in cand_ids], truth_ids, k),
            "score_regret": score_regret(scores, truth_scores),
            "query_ms": per_q,
            **lat,
            "online_or_batch": "batch",
            "batch_qps": float(Q.shape[0] / max(total_ms / 1000.0, 1e-12)),
            "native_scan_ms_mean": _safe_nanmean(timings.get("native_scan_ms", np.array([np.nan]))),
            "native_top_r_ms_mean": _safe_nanmean(timings.get("native_top_r_ms", np.array([np.nan]))),
            "native_rerank_ms_mean": _safe_nanmean(timings.get("native_rerank_ms", np.array([np.nan]))),
            "hot_bytes_per_vector": hot_bytes_per_vector(X.shape[1], residuals=True),
            "cold_bytes_per_query": int(4 * r * X.shape[1]),
        })
    ann_runners = [
        lambda: run_faiss(X, Q, k, "flat", mode="batch"),
        lambda: run_faiss(X, Q, k, "ivf", mode="batch"),
        lambda: run_hnswlib(X, Q, k, mode="batch"),
        lambda: run_scann(X, Q, k, mode="batch"),
    ]
    for runner in ann_runners:
        try:
            out = runner()
            rows.append({
                "dataset": str(dataset_path),
                "method": out["name"],
                "backend": "ann",
                "N": X.shape[0],
                "d": X.shape[1],
                "Q": Q.shape[0],
                "K": k,
                "R": k,
                "recall_at_k": recall_at_k(out["ids"], truth_ids, k),
                "top1_recall": recall_at_k(out["ids"], truth_ids, 1),
                "candidate_coverage": recall_at_k(out["ids"], truth_ids, k),
                "query_ms": out["query_ms"],
                "latency_ms_mean": out["latency_ms_mean"],
                "latency_ms_p50": out["latency_ms_p50"],
                "latency_ms_p95": out["latency_ms_p95"],
                "latency_ms_min": out["latency_ms_min"],
                "latency_ms_max": out["latency_ms_max"],
                "online_or_batch": "batch",
                "batch_qps": out["batch_qps"],
                "build_ms": out["build_ms"],
                "hot_bytes_per_vector": out["index_bytes"] / max(X.shape[0], 1),
                "cold_bytes_per_query": 0,
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"dataset": str(dataset_path), "method": "batch_ann", "online_or_batch": "batch", "status": f"failed: {exc}"})
    return pd.DataFrame(rows)


def run_dynamic_smoke(n: int = 1000, d: int = 64, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    idx = EtarIndex(X, backend="auto")
    rows = []
    x_new = rng.normal(size=d).astype(np.float32)
    t0 = perf_counter()
    row = idx.insert(x_new)
    rows.append({"operation": "insert", "ms": (perf_counter() - t0) * 1000, "N": idx.n, "row_id": row})
    t0 = perf_counter()
    idx.delete(0)
    rows.append({"operation": "delete", "ms": (perf_counter() - t0) * 1000, "N": idx.n, "tombstone_fraction": 1.0 - float(np.mean(idx.alive))})
    t0 = perf_counter()
    row = idx.update(1, x_new)
    rows.append({"operation": "update_by_append", "ms": (perf_counter() - t0) * 1000, "N": idx.n, "row_id": row})
    t0 = perf_counter()
    idx.compact()
    rows.append({"operation": "compact", "ms": (perf_counter() - t0) * 1000, "N": idx.n, "tombstone_fraction": 0.0})
    return pd.DataFrame(rows)


def run_dynamic_baselines(n: int = 1000, d: int = 64, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    x_new = rng.normal(size=(1, d)).astype(np.float32)
    rows = []

    rows.extend(run_dynamic_smoke(n, d, seed).assign(method="etar").to_dict("records"))

    try:
        import hnswlib

        index = hnswlib.Index(space="ip", dim=d)
        index.init_index(max_elements=n + 1, ef_construction=100, M=16, allow_replace_deleted=True)
        index.add_items(X, np.arange(n))
        t0 = perf_counter()
        index.add_items(x_new, np.array([n]))
        rows.append({"method": "hnswlib", "operation": "insert", "ms": (perf_counter() - t0) * 1000, "N": n + 1})
        t0 = perf_counter()
        index.mark_deleted(0)
        rows.append({"method": "hnswlib", "operation": "delete", "ms": (perf_counter() - t0) * 1000, "N": n + 1})
    except Exception as exc:
        rows.append({"method": "hnswlib", "operation": "dynamic", "status": f"failed: {exc}"})

    try:
        import faiss

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
        index.add_with_ids(X, np.arange(n, dtype=np.int64))
        t0 = perf_counter()
        index.add_with_ids(x_new, np.array([n], dtype=np.int64))
        rows.append({"method": "faiss_flat", "operation": "insert", "ms": (perf_counter() - t0) * 1000, "N": n + 1})
        t0 = perf_counter()
        index.remove_ids(np.array([0], dtype=np.int64))
        rows.append({"method": "faiss_flat", "operation": "delete", "ms": (perf_counter() - t0) * 1000, "N": n})
    except Exception as exc:
        rows.append({"method": "faiss_flat", "operation": "dynamic", "status": f"failed: {exc}"})

    for method in ("annoy", "scann"):
        rows.append({"method": method, "operation": "dynamic", "status": "static/rebuild baseline"})
    return pd.DataFrame(rows)


def run_suite(paths: list[str | Path], out: str | Path, backend: str = "auto", ann: bool = True, limit_n: int | None = None, limit_q: int | None = 50) -> Path:
    frames = []
    for path in paths:
        frames.append(run_index_free(path, backend=backend, limit_n=limit_n, limit_q=limit_q))
        if ann:
            frames.append(run_ann_baselines(path, limit_n=limit_n, limit_q=limit_q))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out
