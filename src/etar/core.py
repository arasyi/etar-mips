from __future__ import annotations

from dataclasses import dataclass, replace
import bisect
from functools import lru_cache
import math
from time import perf_counter
from typing import Any

import numpy as np

from .native import CppBackend, NumbaBackend


METHODS = {
    "exact_f32",
    "full_i8",
    "reduced_fixed",
    "reduced_energy",
    "reduced_fixed_no_rerank",
    "reduced_energy_no_rerank",
    "etar",
    "etar_no_rerank",
    "etar_qrisk",
    "etar_iq",
    "etar_adaptive_r",
    "etar_block_prune",
}

HEAD_POLICIES = {"full", "fixed", "energy"}
TAIL_POLICIES = {"none", "fixed", "derived"}
R_POLICIES = {"fixed", "adaptive"}
SCAN_DTYPES = {"int8", "float32"}
SCAN_LAYOUTS = {"row", "column"}
SCAN_QUANTIZATIONS = {"row", "column"}
SCAN_INSERT_POLICIES = {"auto", "appendable", "rebuild"}


@dataclass
class SearchResult:
    ids: np.ndarray
    scores: np.ndarray
    candidate_ids: np.ndarray
    candidate_scores: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EtarResolvedOptions:
    method: str
    legacy_method: str
    head_policy: str
    tail_policy: str
    rerank: bool
    r_policy: str
    query_mode: str
    alpha_q: float
    block_prune: bool
    force_r_to_k: bool


def _as_optional_bool(value: bool | str | int | None, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "on"}:
            return True
        if lower in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be boolean")


def _normalize_policy(value: str | None, allowed: set[str], name: str, aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    aliases = aliases or {}
    normalized = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return normalized


def _normalize_scan_config(
    scan_dtype: str = "int8",
    scan_layout: str | None = None,
    scan_quantization: str | None = "row",
) -> tuple[str, str, str]:
    dtype_aliases = {"i8": "int8", "quantized": "int8", "float": "float32", "f32": "float32"}
    dtype = dtype_aliases.get(str(scan_dtype).strip().lower(), str(scan_dtype).strip().lower())
    if dtype not in SCAN_DTYPES:
        raise ValueError("scan_dtype must be 'int8' or 'float32'")
    quant_raw = None if scan_quantization is None else str(scan_quantization).strip().lower()
    layout_raw = None if scan_layout is None else str(scan_layout).strip().lower()
    if layout_raw is None:
        layout_raw = "column" if dtype == "int8" else "row"
    if layout_raw not in SCAN_LAYOUTS:
        raise ValueError("scan_layout must be 'row' or 'column'")
    if dtype == "int8":
        quantization = "row" if quant_raw is None else quant_raw
        if quantization not in SCAN_QUANTIZATIONS:
            raise ValueError("scan_quantization must be 'row' or 'column' for int8 scan")
    else:
        if quant_raw is not None and quant_raw not in {"none", *SCAN_QUANTIZATIONS}:
            raise ValueError("scan_quantization must be 'none', 'row', or 'column'")
        quantization = "none"
    return dtype, layout_raw, quantization


def _normalize_scan_insert_policy(policy: str | None, scan_dtype: str, scan_layout: str, scan_quantization: str) -> str:
    raw = "auto" if policy is None else str(policy).strip().lower()
    if raw not in SCAN_INSERT_POLICIES:
        raise ValueError("scan_insert_policy must be 'auto', 'appendable', or 'rebuild'")
    supported = scan_layout == "column" and (scan_dtype == "float32" or (scan_dtype == "int8" and scan_quantization == "row"))
    if raw == "auto":
        return "appendable" if supported else "rebuild"
    if raw == "appendable" and not supported:
        raise ValueError("scan_insert_policy='appendable' requires column layout with float32 scan or int8 row quantization")
    return raw


def resolve_etar_options(
    method: str = "etar",
    *,
    head_policy: str | None = None,
    tail_policy: str | None = None,
    rerank: bool | str | int | None = None,
    r_policy: str | None = None,
    query_mode: str = "float",
    alpha_tail_fixed: float | None = None,
    alpha_q: float = 0.0,
    block_prune: bool = False,
) -> EtarResolvedOptions:
    method = str(method)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "exact_f32":
        return EtarResolvedOptions(
            method="exact_f32",
            legacy_method=method,
            head_policy="full",
            tail_policy="none",
            rerank=True,
            r_policy="fixed",
            query_mode="float",
            alpha_q=0.0,
            block_prune=False,
            force_r_to_k=False,
        )

    legacy_head = {
        "full_i8": "full",
        "reduced_fixed": "fixed",
        "reduced_fixed_no_rerank": "fixed",
        "reduced_energy": "energy",
        "reduced_energy_no_rerank": "energy",
        "etar": "energy",
        "etar_no_rerank": "energy",
        "etar_qrisk": "energy",
        "etar_iq": "energy",
        "etar_adaptive_r": "energy",
        "etar_block_prune": "energy",
    }[method]
    legacy_tail_methods = {"etar", "etar_no_rerank", "etar_qrisk", "etar_iq", "etar_adaptive_r", "etar_block_prune"}
    legacy_no_rerank_methods = {"reduced_fixed_no_rerank", "reduced_energy_no_rerank", "etar_no_rerank"}

    resolved_head = _normalize_policy(
        head_policy,
        HEAD_POLICIES,
        "head_policy",
        {"full_i8": "full", "reduced_fixed": "fixed", "reduced_energy": "energy", "rho": "energy"},
    ) or legacy_head
    if tail_policy is None:
        resolved_tail = "fixed" if method in legacy_tail_methods and alpha_tail_fixed is not None else (
            "derived" if method in legacy_tail_methods else "none"
        )
    else:
        resolved_tail = _normalize_policy(
            tail_policy,
            TAIL_POLICIES,
            "tail_policy",
            {"off": "none", "false": "none", "no_tail": "none", "tail": "derived", "auto": "derived"},
        )
    resolved_rerank = _as_optional_bool(rerank, "rerank")
    if resolved_rerank is None:
        resolved_rerank = method not in legacy_no_rerank_methods
    resolved_r_policy = _normalize_policy(
        r_policy,
        R_POLICIES,
        "r_policy",
        {"derived": "adaptive", "auto": "adaptive", "adaptive_r": "adaptive"},
    ) or ("adaptive" if method == "etar_adaptive_r" else "fixed")
    resolved_query_mode = "int8" if method == "etar_iq" else str(query_mode)
    if resolved_query_mode not in {"float", "int8"}:
        raise ValueError("query_mode must be float or int8")
    resolved_alpha_q = float(alpha_q)
    resolved_block_prune = bool(block_prune or method == "etar_block_prune")
    force_r_to_k = method in legacy_no_rerank_methods and rerank is None and r_policy is None
    return EtarResolvedOptions(
        method="etar",
        legacy_method=method,
        head_policy=resolved_head,
        tail_policy=resolved_tail,
        rerank=bool(resolved_rerank),
        r_policy=resolved_r_policy,
        query_mode=resolved_query_mode,
        alpha_q=resolved_alpha_q,
        block_prune=resolved_block_prune,
        force_r_to_k=force_r_to_k,
    )


def cold_rerank_diagnostics(
    candidate_ids: np.ndarray,
    d: int,
    native_rerank_ms: float = 0.0,
    *,
    enabled: bool = False,
    bandwidth_gbps: float = 0.0,
    row_latency_ns: float = 0.0,
    page_latency_ns: float = 0.0,
    page_size: int = 4096,
) -> dict[str, Any]:
    ids = np.asarray(candidate_ids, dtype=np.int64)
    ids = ids[ids >= 0]
    rows = int(ids.size)
    row_bytes = int(d) * np.dtype(np.float32).itemsize
    bytes_total = int(rows * row_bytes)
    page_size = max(int(page_size), 1)
    pages = 0
    if rows > 0 and row_bytes > 0:
        starts = ids.astype(np.int64) * np.int64(row_bytes)
        first_pages = starts // page_size
        last_pages = (starts + row_bytes - 1) // page_size
        if np.array_equal(first_pages, last_pages):
            pages = int(np.unique(first_pages).size)
        else:
            touched: set[int] = set()
            for first, last in zip(first_pages.tolist(), last_pages.tolist()):
                touched.update(range(int(first), int(last) + 1))
            pages = len(touched)

    bandwidth_ms = 0.0
    latency_ms = 0.0
    if enabled:
        if bandwidth_gbps > 0.0:
            bandwidth_ms = float(bytes_total) / (float(bandwidth_gbps) * 1_000_000.0)
        latency_ms = (float(rows) * max(float(row_latency_ns), 0.0) + float(pages) * max(float(page_latency_ns), 0.0)) * 1e-6
    model_ms = bandwidth_ms + latency_ms
    native_ms = max(float(native_rerank_ms), 0.0)
    penalty_ms = max(0.0, model_ms - native_ms) if enabled else 0.0
    return {
        "cold_rerank_enabled": bool(enabled),
        "cold_rerank_rows": rows,
        "cold_rerank_bytes": bytes_total,
        "cold_rerank_pages": pages,
        "cold_rerank_bandwidth_ms": bandwidth_ms,
        "cold_rerank_latency_ms": latency_ms,
        "cold_rerank_model_ms": model_ms if enabled else 0.0,
        "cold_rerank_native_ms": native_ms,
        "cold_rerank_penalty_ms": penalty_ms,
        "cold_rerank_page_size": page_size,
    }


def cold_rerank_batch_diagnostics(
    candidate_ids: np.ndarray,
    d: int,
    native_rerank_ms: np.ndarray | float = 0.0,
    *,
    enabled: bool = False,
    bandwidth_gbps: float = 0.0,
    row_latency_ns: float = 0.0,
    page_latency_ns: float = 0.0,
    page_size: int = 4096,
) -> dict[str, np.ndarray]:
    cand = np.asarray(candidate_ids, dtype=np.int64)
    if cand.ndim == 1:
        cand = cand[None, :]
    native = np.asarray(native_rerank_ms, dtype=np.float32)
    if native.ndim == 0:
        native = np.full(cand.shape[0], float(native), dtype=np.float32)
    rows: list[int] = []
    bytes_: list[int] = []
    pages: list[int] = []
    bandwidth: list[float] = []
    latency: list[float] = []
    model: list[float] = []
    penalty: list[float] = []
    for i, ids in enumerate(cand):
        stats = cold_rerank_diagnostics(
            ids,
            d,
            float(native[i]) if i < native.size else 0.0,
            enabled=enabled,
            bandwidth_gbps=bandwidth_gbps,
            row_latency_ns=row_latency_ns,
            page_latency_ns=page_latency_ns,
            page_size=page_size,
        )
        rows.append(int(stats["cold_rerank_rows"]))
        bytes_.append(int(stats["cold_rerank_bytes"]))
        pages.append(int(stats["cold_rerank_pages"]))
        bandwidth.append(float(stats["cold_rerank_bandwidth_ms"]))
        latency.append(float(stats["cold_rerank_latency_ms"]))
        model.append(float(stats["cold_rerank_model_ms"]))
        penalty.append(float(stats["cold_rerank_penalty_ms"]))
    return {
        "cold_rerank_rows": np.asarray(rows, dtype=np.float32),
        "cold_rerank_bytes": np.asarray(bytes_, dtype=np.float32),
        "cold_rerank_pages": np.asarray(pages, dtype=np.float32),
        "cold_rerank_bandwidth_ms": np.asarray(bandwidth, dtype=np.float32),
        "cold_rerank_latency_ms": np.asarray(latency, dtype=np.float32),
        "cold_rerank_model_ms": np.asarray(model, dtype=np.float32),
        "cold_rerank_penalty_ms": np.asarray(penalty, dtype=np.float32),
    }


def _as_f32_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array")
    return np.ascontiguousarray(x)


def _orient_scan_codes(codes: np.ndarray, layout: str) -> np.ndarray:
    if layout == "row":
        return np.ascontiguousarray(codes)
    if layout == "column":
        return np.ascontiguousarray(codes.T)
    raise ValueError("scan_layout must be 'row' or 'column'")


def quantize_rowwise(x: np.ndarray, residuals: bool = False, layout: str = "column") -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = _as_f32_matrix(x)
    max_abs = np.max(np.abs(x), axis=1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    codes = np.rint(x / scales[:, None]).clip(-127, 127).astype(np.int8)
    c_scan = _orient_scan_codes(codes, layout)
    eta = None
    if residuals:
        eta = np.linalg.norm(x - codes.astype(np.float32) * scales[:, None], axis=1).astype(np.float32)
    return c_scan, scales, eta


def quantize_colwise(x: np.ndarray, residuals: bool = False, layout: str = "column") -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = _as_f32_matrix(x)
    max_abs = np.max(np.abs(x), axis=0)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    codes = np.rint(x / scales[None, :]).clip(-127, 127).astype(np.int8)
    c_scan = _orient_scan_codes(codes, layout)
    eta = None
    if residuals:
        eta = np.linalg.norm(x - codes.astype(np.float32) * scales[None, :], axis=1).astype(np.float32)
    return c_scan, scales, eta


def quantize_row(x: np.ndarray, residuals: bool = False) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    scale = np.float32(max_abs / 127.0 if max_abs > 0.0 else 1.0)
    codes = np.rint(x / scale).clip(-127, 127).astype(np.int8)
    residual = float(np.linalg.norm(x - codes.astype(np.float32) * scale)) if residuals else 0.0
    return codes, float(scale), residual


def corpus_omega(x: np.ndarray) -> np.ndarray:
    x = _as_f32_matrix(x)
    norms = np.linalg.norm(x, axis=1)
    out = np.zeros(x.shape[1], dtype=np.float32)
    mask = norms > 0
    if np.any(mask):
        y = x[mask] / norms[mask, None]
        out = np.mean(y * y, axis=0).astype(np.float32)
    return out


def select_head(q: np.ndarray, rho: float, h_max: int | None, method: str) -> tuple[np.ndarray, float, float, int]:
    q = np.asarray(q, dtype=np.float32)
    d = q.size
    order = np.argsort(-(q * q), kind="stable").astype(np.int64)
    total = float(np.dot(q, q))
    if method == "full_i8":
        h = d
    elif method == "reduced_fixed":
        h = min(d, int(h_max if h_max is not None else d))
    else:
        if total == 0.0:
            h_rho = 1
        else:
            csum = np.cumsum((q[order] * q[order]).astype(np.float64))
            h_rho = int(np.searchsorted(csum / total, rho, side="left") + 1)
        h = min(d, h_rho)
        if h_max is not None:
            h = min(h, int(h_max))
    head = np.ascontiguousarray(order[: max(1, h)])
    retained = float(np.dot(q[head], q[head]) / total) if total > 0 else 1.0
    tail_norm = float(np.sqrt(max(total * (1.0 - retained), 0.0)))
    return head, retained, tail_norm, h


def select_head_policy(q: np.ndarray, rho: float, h_max: int | None, head_policy: str) -> tuple[np.ndarray, float, float, int]:
    legacy_method = {
        "full": "full_i8",
        "fixed": "reduced_fixed",
        "energy": "etar",
    }[head_policy]
    return select_head(q, rho, h_max, legacy_method)


def tail_alpha(
    q: np.ndarray,
    head: np.ndarray,
    omega: np.ndarray,
    n: int,
    r: int,
    tail_norm: float,
    lambda_tail: float,
    alpha_min: float,
    alpha_max: float,
    eta: float = 1e-12,
) -> float:
    q2 = np.asarray(q, dtype=np.float32) ** 2
    mask = np.ones(q2.size, dtype=bool)
    mask[head] = False
    sigma = float(np.sqrt(np.sum(q2[mask] * omega[mask])))
    z = float(np.sqrt(max(0.0, 2.0 * np.log(max(float(n) / max(float(r), 1.0), 1.0)))))
    return float(np.clip(lambda_tail * z * sigma / (tail_norm + eta), alpha_min, alpha_max))


def _normal_sf(x: float) -> float:
    return 0.5 * float(math.erfc(float(x) / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    # Peter J. Acklam's inverse-normal approximation. Accurate enough for
    # adaptive budget calibration and avoids a SciPy dependency in the hot path.
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return float((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                     ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return float((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                     (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return float(-(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))


def adaptive_r_budget(
    q: np.ndarray,
    head: np.ndarray,
    omega: np.ndarray,
    n: int,
    k: int,
    r_max: int,
    target_recall: float = 0.95,
    probability: float = 0.95,
    mode: str = "event",
) -> int:
    k = max(1, int(k))
    r_max = max(k, int(r_max))
    n = max(r_max, int(n))
    q2 = np.asarray(q, dtype=np.float32) ** 2
    omega = np.asarray(omega, dtype=np.float32)
    full = float(np.sum(q2 * omega))
    if full <= 0.0:
        return r_max
    retained = float(np.sum(q2[np.asarray(head, dtype=np.int64)] * omega[np.asarray(head, dtype=np.int64)]))
    c = float(np.sqrt(np.clip(retained / full, 0.0, 0.999999)))
    return _adaptive_r_from_c(c, n, k, r_max, float(target_recall), float(probability), mode)


@lru_cache(maxsize=128)
def _adaptive_r_thresholds(
    n: int,
    k: int,
    r_max: int,
    target_recall: float,
    probability: float,
    mode: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ranks = range(1, k + 1)
    a = [_normal_ppf(1.0 - float(rank) / float(n)) for rank in ranks]
    need = min(max(int(math.ceil(float(target_recall) * k)), 1), k)
    target = float(probability)

    def ok_for_c(r: int, c: float) -> bool:
        if c <= 0.0:
            return False
        denom = math.sqrt(max(1.0 - c * c, 1e-8))
        b = _normal_ppf(1.0 - float(r) / float(n))
        p = [_normal_sf((b - c * ar) / denom) for ar in a]
        if mode == "mean":
            return float(sum(p) / len(p)) >= float(target_recall)
        probs = [0.0] * (k + 1)
        probs[0] = 1.0
        for pr in p:
            for m in range(k, 0, -1):
                probs[m] = probs[m] * (1.0 - pr) + probs[m - 1] * pr
            probs[0] *= 1.0 - pr
        return float(sum(probs[need:])) >= target

    budgets: list[int] = []
    thresholds: list[float] = []
    for r in range(k, r_max + 1):
        if not ok_for_c(r, 0.999999):
            continue
        lo, hi = 0.0, 0.999999
        for _ in range(28):
            mid = (lo + hi) * 0.5
            if ok_for_c(r, mid):
                hi = mid
            else:
                lo = mid
        budgets.append(r)
        thresholds.append(hi)
    return tuple(budgets), tuple(-x for x in thresholds)


def _adaptive_r_from_c(
    c: float,
    n: int,
    k: int,
    r_max: int,
    target_recall: float,
    probability: float,
    mode: str,
) -> int:
    budgets, neg_thresholds = _adaptive_r_thresholds(
        int(n),
        int(k),
        int(r_max),
        round(float(target_recall), 6),
        round(float(probability), 6),
        str(mode),
    )
    if not budgets:
        return int(r_max)
    idx = bisect.bisect_left(neg_thresholds, -float(c))
    if idx < len(budgets):
        return int(budgets[idx])
    return int(r_max)


class EtarIndex:
    def __init__(
        self,
        X: np.ndarray,
        backend: str = "auto",
        keep_float: bool = True,
        residuals: bool = False,
        scan_quantization: str | None = "row",
        scan_dtype: str = "int8",
        scan_layout: str | None = None,
        scan_insert_policy: str | None = "auto",
        scan_growth_factor: float = 2.0,
        scan_initial_capacity_factor: float = 1.25,
        scan_shrink_threshold: float = 0.25,
        scan_shrink_factor: float = 0.5,
        scan_compact_shrink: bool = True,
        omega_update_policy: str = "lazy",
        layout: str = "identity",
    ):
        scan_dtype, scan_layout, scan_quantization = _normalize_scan_config(scan_dtype, scan_layout, scan_quantization)
        scan_insert_policy = _normalize_scan_insert_policy(scan_insert_policy, scan_dtype, scan_layout, scan_quantization)
        scan_growth_factor = float(scan_growth_factor)
        scan_initial_capacity_factor = float(scan_initial_capacity_factor)
        scan_shrink_threshold = float(scan_shrink_threshold)
        scan_shrink_factor = float(scan_shrink_factor)
        if scan_growth_factor <= 1.0:
            raise ValueError("scan_growth_factor must be > 1.0")
        if scan_initial_capacity_factor < 1.0:
            raise ValueError("scan_initial_capacity_factor must be >= 1.0")
        if not 0.0 < scan_shrink_threshold <= 1.0:
            raise ValueError("scan_shrink_threshold must be in (0, 1]")
        if not 0.0 < scan_shrink_factor < 1.0:
            raise ValueError("scan_shrink_factor must be in (0, 1)")
        if omega_update_policy not in {"lazy", "exact_live"}:
            raise ValueError("omega_update_policy must be 'lazy' or 'exact_live'")
        if layout not in {"identity", "norm_desc"}:
            raise ValueError("layout must be 'identity' or 'norm_desc'")
        self.keep_float = keep_float
        self.residuals_enabled = residuals
        self.scan_dtype = scan_dtype
        self.scan_layout = scan_layout
        self.scan_quantization = scan_quantization
        self.scan_insert_policy = scan_insert_policy
        self.scan_growth_factor = scan_growth_factor
        self.scan_initial_capacity_factor = scan_initial_capacity_factor
        self.scan_shrink_threshold = scan_shrink_threshold
        self.scan_shrink_factor = scan_shrink_factor
        self.scan_compact_shrink = bool(scan_compact_shrink)
        self.scan_stride = 0
        self.scan_growth_events = 0
        self.scan_rebuild_events = 0
        self.omega_update_policy = omega_update_policy
        self.layout = layout
        x0 = _as_f32_matrix(X)
        initial_ids = np.arange(x0.shape[0], dtype=np.int64)
        if layout == "norm_desc" and x0.shape[0] > 0:
            order = np.argsort(-np.linalg.norm(x0, axis=1), kind="stable")
            x0 = np.ascontiguousarray(x0[order])
            initial_ids = initial_ids[order]
        self._n = int(x0.shape[0])
        self._capacity = max(1, int(math.ceil(max(self._n, 1) * self.scan_initial_capacity_factor)))
        self._X = np.empty((self._capacity, x0.shape[1]), dtype=np.float32)
        self._X[: self._n] = x0
        self._alive = np.zeros(self._capacity, dtype=np.uint8)
        self._alive[: self._n] = 1
        self._row_ids = np.full(self._capacity, -1, dtype=np.int64)
        self._row_ids[: self._n] = initial_ids
        self._next_row_id = int(initial_ids.max() + 1) if initial_ids.size else 0
        self._orig_to_phys = {int(orig): int(pos) for pos, orig in enumerate(initial_ids)}
        self.live_count = self._n
        self.has_tombstones = False
        self._base_n = self._n
        self._delta_ids: list[int] = []
        self._delta_codes: list[np.ndarray] = []
        self._delta_scales: list[float] = []
        self._delta_norms: list[float] = []
        self._delta_residuals: list[float] = []
        self._omega_sum = np.zeros(x0.shape[1], dtype=np.float64)
        self._omega_count = 0
        self._rebuild_metadata()
        self.backend_name = self._choose_backend(backend)
        self.backend = CppBackend() if self.backend_name == "cpp" else NumbaBackend()

    @property
    def X(self) -> np.ndarray:
        return self._X[: self._n]

    @property
    def alive(self) -> np.ndarray:
        return self._alive[: self._n]

    @property
    def n(self) -> int:
        return int(self._n)

    @property
    def d(self) -> int:
        return int(self._X.shape[1])

    @property
    def scan_capacity(self) -> int:
        return int(self.scan_stride if self.scan_layout == "column" else self._capacity)

    @property
    def scan_slack_fraction(self) -> float:
        return float(1.0 - (self._n / max(self.scan_capacity, 1)))

    @property
    def tombstone_count(self) -> int:
        return max(0, int(self._n) - int(self.live_count))

    @property
    def tombstone_fraction(self) -> float:
        return float(self.tombstone_count / max(int(self._n), 1))

    def should_compact(
        self,
        *,
        tombstone_fraction: float | None = None,
        min_tombstones: int = 1,
        shrink: bool = True,
    ) -> bool:
        tombstones = self.tombstone_count
        if tombstones < int(min_tombstones):
            return False
        if tombstone_fraction is not None and float(tombstone_fraction) >= 0.0:
            if self.tombstone_fraction >= float(tombstone_fraction):
                return True
        if shrink and self.scan_compact_shrink and self._uses_appendable_column_scan():
            shrink_at = int(math.floor(self._capacity * self.scan_shrink_threshold))
            if self.live_count <= shrink_at and self._capacity > max(1, self.live_count):
                return True
        return False

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        old_capacity = self._capacity
        new_capacity = max(needed, int(math.ceil(self._capacity * self.scan_growth_factor)))
        new_x = np.empty((new_capacity, self.d), dtype=np.float32)
        new_x[: self._n] = self.X
        new_alive = np.zeros(new_capacity, dtype=np.uint8)
        new_alive[: self._n] = self.alive
        new_row_ids = np.full(new_capacity, -1, dtype=np.int64)
        new_row_ids[: self._n] = self._row_ids[: self._n]
        self._X = new_x
        self._alive = new_alive
        self._row_ids = new_row_ids
        self._capacity = new_capacity
        self._resize_appendable_scan(old_capacity, new_capacity)

    def _uses_appendable_column_scan(self) -> bool:
        return self.scan_insert_policy == "appendable" and self.scan_layout == "column"

    def _resize_appendable_scan(self, old_capacity: int, new_capacity: int) -> None:
        if not self._uses_appendable_column_scan() or not hasattr(self, "C_col_scan"):
            return
        copy_cols = min(int(old_capacity), int(new_capacity), int(self._n))
        if self.scan_dtype == "int8":
            new_codes = np.zeros((self.d, new_capacity), dtype=np.int8)
            if copy_cols > 0:
                new_codes[:, :copy_cols] = self.C_col_scan[:, :copy_cols]
            self.C_col_scan = new_codes
            self.C_col = self.C_col_scan
        else:
            new_scan = np.zeros((self.d, new_capacity), dtype=np.float32)
            if copy_cols > 0:
                new_scan[:, :copy_cols] = self.X_scan[:, :copy_cols]
            self.X_scan = new_scan
        for name in ("scales", "scan_scales", "norms", "residual_norms"):
            if not hasattr(self, name):
                continue
            arr = getattr(self, name)
            if arr.ndim != 1:
                continue
            target = self.d if name == "scan_scales" else new_capacity
            if arr.shape[0] == target:
                continue
            new_arr = np.zeros(target, dtype=arr.dtype)
            take = min(arr.shape[0], target, copy_cols if name != "scan_scales" else arr.shape[0])
            if take > 0:
                new_arr[:take] = arr[:take]
            if name == "scan_scales" and take < target:
                new_arr[take:] = 1.0
            setattr(self, name, new_arr)
        self.scan_stride = int(new_capacity)
        self.scan_growth_events += 1

    def _physical_id(self, row_id: int) -> int:
        try:
            return int(self._orig_to_phys[int(row_id)])
        except KeyError as exc:
            raise IndexError(f"unknown row_id {row_id}") from exc

    def _map_row_ids(self, ids: np.ndarray) -> np.ndarray:
        mapped = np.asarray(ids, dtype=np.int64).copy()
        mask = mapped >= 0
        if np.any(mask):
            mapped[mask] = self._row_ids[mapped[mask]]
        return mapped

    def _refresh_omega_from_sum(self) -> None:
        if self._omega_count > 0:
            self.omega = (self._omega_sum / float(self._omega_count)).astype(np.float32)
        else:
            self.omega = np.zeros(self.d, dtype=np.float32)

    def _omega_contribution(self, x: np.ndarray) -> tuple[np.ndarray | None, int]:
        norm = float(np.linalg.norm(x))
        if norm <= 0.0:
            return None, 0
        y = (np.asarray(x, dtype=np.float32) / norm).astype(np.float64)
        return y * y, 1

    def _add_omega_row(self, x: np.ndarray) -> None:
        contribution, count = self._omega_contribution(x)
        if count:
            self._omega_sum += contribution
            self._omega_count += count
            self._refresh_omega_from_sum()

    def _remove_omega_row(self, x: np.ndarray) -> None:
        contribution, count = self._omega_contribution(x)
        if count:
            self._omega_sum -= contribution
            self._omega_count = max(0, self._omega_count - count)
            np.maximum(self._omega_sum, 0.0, out=self._omega_sum)
            self._refresh_omega_from_sum()

    def _choose_backend(self, backend: str) -> str:
        if backend not in {"auto", "cpp", "numba"}:
            raise ValueError("backend must be auto, cpp, or numba")
        if backend == "numba":
            return "numba"
        if CppBackend.available():
            return "cpp"
        if backend == "cpp":
            raise RuntimeError("C++ backend requested but native library is unavailable")
        return "numba"

    def _rebuild_metadata(self) -> None:
        x = self.X
        appendable_column = self._uses_appendable_column_scan()
        column_stride = self._capacity if appendable_column and self.scan_layout == "column" else self._n
        self.scan_stride = int(column_stride if self.scan_layout == "column" else self.d)
        if self.scan_dtype == "int8":
            if self.scan_quantization == "row":
                c_scan, scales, residuals = quantize_rowwise(
                    x,
                    self.residuals_enabled,
                    layout=self.scan_layout,
                )
                self.scales = np.ones(self._capacity, dtype=np.float32) if appendable_column else scales
                self.scales[: self._n] = scales
                self.residual_norms = np.zeros(self._capacity, dtype=np.float32) if appendable_column else residuals
                if residuals is not None:
                    self.residual_norms[: self._n] = residuals
                self.scan_scales = np.ones(self.d, dtype=np.float32)
                self.scan_mode = 0 if self.scan_layout == "column" else 4
            else:
                c_scan, self.scan_scales, self.residual_norms = quantize_colwise(
                    x,
                    self.residuals_enabled,
                    layout=self.scan_layout,
                )
                self.scales = np.ones(self._n, dtype=np.float32)
                self.scan_mode = 1 if self.scan_layout == "column" else 5
            if self.scan_layout == "column" and appendable_column:
                self.C_col_scan = np.zeros((self.d, self._capacity), dtype=np.int8)
                self.C_col_scan[:, : self._n] = c_scan
            else:
                self.C_col_scan = c_scan
            self.C_col = self.C_col_scan
            self.X_scan = np.ascontiguousarray(x.T if self.scan_layout == "column" else x, dtype=np.float32)
        else:
            self.C_col = np.zeros((1, 1), dtype=np.int8)
            self.C_col_scan = self.C_col
            self.scales = np.ones(self._capacity if appendable_column else self._n, dtype=np.float32)
            self.scan_scales = np.ones(self.d, dtype=np.float32)
            self.residual_norms = np.zeros(self._capacity if appendable_column else self._n, dtype=np.float32)
            if self.scan_layout == "column" and appendable_column:
                self.X_scan = np.zeros((self.d, self._capacity), dtype=np.float32)
                self.X_scan[:, : self._n] = x.T
            else:
                self.X_scan = np.ascontiguousarray(x.T if self.scan_layout == "column" else x, dtype=np.float32)
            self.scan_mode = 3 if self.scan_layout == "column" else 2
        norms = np.linalg.norm(x, axis=1).astype(np.float32)
        if appendable_column:
            self.norms = np.zeros(self._capacity, dtype=np.float32)
            self.norms[: self._n] = norms
        else:
            self.norms = norms
        live_norms = self.norms[: self._n][self.alive.astype(bool)]
        mean_norm = float(np.mean(live_norms)) if live_norms.size else 0.0
        self.norm_cv = float(np.std(live_norms) / mean_norm) if mean_norm > 0.0 else 0.0
        logical_norms = self.norms[: self._n]
        mask = (self.alive.astype(bool)) & (logical_norms > 0)
        if np.any(mask):
            y = x[mask] / logical_norms[mask, None]
            self._omega_sum = np.sum((y * y).astype(np.float64), axis=0)
            self._omega_count = int(mask.sum())
        else:
            self._omega_sum = np.zeros(self.d, dtype=np.float64)
            self._omega_count = 0
        self._refresh_omega_from_sum()
        self._base_n = self._n
        self._delta_ids = []
        self._delta_codes = []
        self._delta_scales = []
        self._delta_norms = []
        self._delta_residuals = []
        if self._alive.shape[0] < self._n:
            self._ensure_capacity(self._n)

    def insert(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},)")
        if self._uses_appendable_column_scan():
            self._ensure_capacity(self._n + 1)
            phys_id = self._n
            row_id = self._next_row_id
            self._next_row_id += 1
            self._X[phys_id] = x
            self._alive[phys_id] = 1
            self._row_ids[phys_id] = row_id
            self._orig_to_phys[int(row_id)] = int(phys_id)
            self._n += 1
            self.live_count += 1
            norm = float(np.linalg.norm(x))
            self.norms[phys_id] = norm
            if self.scan_dtype == "int8":
                codes, scale, residual = quantize_row(x, self.residuals_enabled)
                self.C_col_scan[:, phys_id] = codes
                self.C_col = self.C_col_scan
                self.scales[phys_id] = scale
                self.residual_norms[phys_id] = residual
            else:
                self.X_scan[:, phys_id] = x
                self.scales[phys_id] = 1.0
                self.residual_norms[phys_id] = 0.0
            if norm > 0.0:
                self._add_omega_row(x)
            self._base_n = self._n
            return row_id
        if self.scan_layout == "column" or (self.scan_dtype == "int8" and self.scan_quantization == "column"):
            self._ensure_capacity(self._n + 1)
            phys_id = self._n
            row_id = self._next_row_id
            self._next_row_id += 1
            self._X[phys_id] = x
            self._alive[phys_id] = 1
            self._row_ids[phys_id] = row_id
            self._orig_to_phys[int(row_id)] = int(phys_id)
            self._n += 1
            self.live_count += 1
            self._rebuild_metadata()
            self.scan_rebuild_events += 1
            return row_id
        self._ensure_capacity(self._n + 1)
        phys_id = self._n
        row_id = self._next_row_id
        self._next_row_id += 1
        self._X[phys_id] = x
        self._alive[phys_id] = 1
        self._row_ids[phys_id] = row_id
        self._orig_to_phys[int(row_id)] = int(phys_id)
        self._n += 1
        self.live_count += 1
        if self.scan_dtype == "int8":
            codes, scale, residual = quantize_row(x, self.residuals_enabled)
        else:
            codes = np.empty(0, dtype=np.int8)
            scale = 1.0
            residual = 0.0
        norm = float(np.linalg.norm(x))
        self._delta_ids.append(phys_id)
        self._delta_codes.append(codes)
        self._delta_scales.append(scale)
        self._delta_norms.append(norm)
        self._delta_residuals.append(residual)
        if norm > 0.0:
            self._add_omega_row(x)
        return row_id

    def delete(self, row_id: int) -> None:
        phys_id = self._physical_id(int(row_id))
        if self._alive[phys_id] != 0:
            if self.omega_update_policy == "exact_live":
                self._remove_omega_row(self._X[phys_id])
            self._alive[phys_id] = 0
            self.live_count -= 1
            self.has_tombstones = True

    def update(self, row_id: int, x: np.ndarray) -> int:
        self.delete(row_id)
        return self.insert(x)

    def compact(self) -> None:
        old_capacity = self._capacity
        mask = self.alive.astype(bool)
        x = np.ascontiguousarray(self.X[mask])
        if self.layout == "identity":
            row_ids = np.arange(x.shape[0], dtype=np.int64)
        else:
            row_ids = np.ascontiguousarray(self._row_ids[: self._n][mask])
        self._n = int(x.shape[0])
        if self._uses_appendable_column_scan() and self.scan_compact_shrink:
            if self._n == 0:
                self._capacity = 1
            elif self._n <= int(math.floor(old_capacity * self.scan_shrink_threshold)):
                self._capacity = max(self._n, int(math.ceil(old_capacity * self.scan_shrink_factor)))
            else:
                self._capacity = max(1, old_capacity)
        else:
            self._capacity = max(1, self._n)
        self._X = np.empty((self._capacity, x.shape[1]), dtype=np.float32)
        self._X[: self._n] = x
        self._alive = np.zeros(self._capacity, dtype=np.uint8)
        self._alive[: self._n] = 1
        self._row_ids = np.full(self._capacity, -1, dtype=np.int64)
        self._row_ids[: self._n] = row_ids
        self._orig_to_phys = {int(orig): int(pos) for pos, orig in enumerate(row_ids)}
        if self.layout == "identity":
            self._next_row_id = self._n
        self.live_count = self._n
        self.has_tombstones = False
        self._rebuild_metadata()

    def _base_live_count(self) -> int:
        if self._base_n == 0:
            return 0
        return int(np.sum(self._alive[: self._base_n] != 0))

    def _base_scan_matrix(self) -> np.ndarray:
        if self.scan_mode == 3:
            if self._base_n == self._n:
                return self.X_scan
            return np.ascontiguousarray(self._X[: self._base_n].T, dtype=np.float32)
        return self._X[: self._base_n]

    def _delta_candidate_scores(
        self,
        q: np.ndarray,
        head: np.ndarray,
        use_tail: bool,
        alpha_tail: float,
        tail_norm: float,
        alpha_q: float,
        int8_query: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        live_ids = []
        scores = []
        for pos, row_id in enumerate(self._delta_ids):
            if row_id >= self._n or self._alive[row_id] == 0:
                continue
            if self.scan_dtype == "float32":
                partial = float(np.dot(q[head], self._X[row_id, head]))
            else:
                codes = self._delta_codes[pos]
                if int8_query:
                    qmax = float(np.max(np.abs(q[head]))) if head.size else 0.0
                    qscale = qmax / 127.0 if qmax > 0.0 else 1.0
                    qcodes = np.rint(q[head] / qscale).clip(-127, 127).astype(np.int16)
                    partial = float(np.dot(qcodes, codes[head].astype(np.int16)) * qscale * self._delta_scales[pos])
                else:
                    partial = float(np.dot(q[head], codes[head].astype(np.float32)) * self._delta_scales[pos])
            score = partial
            if use_tail:
                score += float(alpha_tail) * float(tail_norm) * self._delta_norms[pos]
            if alpha_q != 0.0:
                score += float(alpha_q) * float(np.linalg.norm(q[head])) * self._delta_residuals[pos] * float(np.sqrt(head.size / max(self.d, 1)))
            live_ids.append(row_id)
            scores.append(score)
        if not live_ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.asarray(live_ids, dtype=np.int64), np.asarray(scores, dtype=np.float32)

    @staticmethod
    def _top_by_score(ids: np.ndarray, scores: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        if ids.size == 0 or limit <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        take = min(int(limit), ids.size)
        if ids.size > take:
            part = np.argpartition(-scores, kth=take - 1)[:take]
        else:
            part = np.arange(ids.size)
        order = np.argsort(-scores[part], kind="stable")
        picked = part[order]
        return ids[picked].astype(np.int64), scores[picked].astype(np.float32)

    def _backend_search_base(
        self,
        q: np.ndarray,
        head: np.ndarray,
        k: int,
        r: int,
        use_tail: bool,
        alpha_t: float,
        tail_norm: float,
        aq: float,
        int8_query: bool,
        no_rerank: bool,
        prune_lower_bound: bool,
        prune_lower_scale: float,
        block_prune: bool,
        block_prune_scout_h: int,
        block_prune_beam_factor: float,
        block_prune_gate: float,
        block_prune_block_size: int,
        sort_rerank_candidates: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        base_live = self._base_live_count()
        if self._base_n == 0 or base_live == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                {"native_scan_ms": 0.0, "native_top_r_ms": 0.0, "native_rerank_ms": 0.0, "native_total_ms": 0.0},
            )
        base_k = min(int(k), base_live)
        base_r = min(max(int(r), base_k), base_live)
        alive_base = self._alive[: self._base_n]
        alive_backend = alive_base if (self.has_tombstones or self.backend_name != "cpp" or self._base_n != self._n) else None
        return self.backend.search_profile(
            self._X[: self._base_n],
            self._base_scan_matrix(),
            self.C_col_scan,
            self.scales,
            self.scan_scales,
            self.norms,
            self.residual_norms,
            alive_backend,
            q,
            head,
            self.scan_stride,
            base_k,
            base_r,
            use_tail,
            alpha_t,
            tail_norm,
            aq,
            int8_query,
            self.scan_mode,
            no_rerank,
            prune_lower_bound,
            prune_lower_scale,
            block_prune,
            block_prune_scout_h,
            block_prune_beam_factor,
            block_prune_gate,
            block_prune_block_size,
            sort_rerank_candidates,
        )

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        r: int = 100,
        rho: float = 0.9,
        h_max: int | None = 64,
        h: int | None = None,
        method: str = "etar",
        head_policy: str | None = None,
        tail_policy: str | None = None,
        rerank: bool | str | int | None = None,
        r_policy: str | None = None,
        lambda_tail: float = 0.75,
        alpha_min: float = 0.05,
        alpha_max: float = 0.5,
        alpha_tail_fixed: float | None = None,
        alpha_q: float = 0.0,
        query_mode: str = "float",
        adaptive_r_target_recall: float = 0.95,
        adaptive_r_probability: float = 0.95,
        adaptive_r_mode: str = "event",
        prune_lower_bound: bool = False,
        prune_lower_mode: str = "global",
        prune_lower_scale: float = 1.0,
        block_prune: bool = False,
        block_prune_scout_h: int = 16,
        block_prune_beam_factor: float = 32.0,
        block_prune_gate: float = 0.75,
        block_prune_block_size: int = 256,
        block_prune_cv_threshold: float = 0.0,
        cold_rerank_enabled: bool = False,
        cold_rerank_apply_to_wall_time: bool = False,
        cold_rerank_bandwidth_gbps: float = 0.0,
        cold_rerank_row_latency_ns: float = 0.0,
        cold_rerank_page_latency_ns: float = 0.0,
        cold_rerank_page_size: int = 4096,
        cold_rerank_sort_candidates: bool = False,
    ) -> SearchResult:
        opts = resolve_etar_options(
            method,
            head_policy=head_policy,
            tail_policy=tail_policy,
            rerank=rerank,
            r_policy=r_policy,
            query_mode=query_mode,
            alpha_tail_fixed=alpha_tail_fixed,
            alpha_q=alpha_q,
            block_prune=block_prune,
        )
        if self.scan_dtype == "float32" and opts.query_mode == "int8":
            opts = replace(opts, query_mode="float")
        adaptive_r = opts.r_policy == "adaptive"
        no_rerank = not opts.rerank
        h_limit = h if h is not None else h_max
        h_limit = None if h_limit is None else int(h_limit)
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32))
        if q.shape != (self.d,):
            raise ValueError(f"q must have shape ({self.d},)")
        k = min(int(k), self.live_count)
        r = min(max(int(r), k), self.live_count)
        if opts.force_r_to_k:
            r = k

        t0 = perf_counter()
        if opts.method == "exact_f32":
            ids, scores = self._exact(q, k)
            diag = {
                "backend": "numpy",
                "method": method,
                "canonical_method": "exact_f32",
                "head_policy": opts.head_policy,
                "tail_policy": opts.tail_policy,
                "r_policy": opts.r_policy,
                "scan_dtype": self.scan_dtype,
                "scan_layout": self.scan_layout,
                "scan_quantization": self.scan_quantization,
                "scan_insert_policy": self.scan_insert_policy,
                "scan_stride": int(self.scan_stride),
                "scan_capacity": int(self.scan_capacity),
                "scan_slack_fraction": float(self.scan_slack_fraction),
                "scan_growth_events": int(self.scan_growth_events),
                "scan_rebuild_events": int(self.scan_rebuild_events),
                "h": self.d,
                "retained_mass": 1.0,
            }
            diag["total_ms"] = (perf_counter() - t0) * 1000.0
            mapped = self._map_row_ids(ids)
            return SearchResult(mapped, scores, mapped.copy(), scores.copy(), diag)

        t_pre0 = perf_counter()
        head, retained, tail_norm, selected_h = select_head_policy(q, rho, h_limit, opts.head_policy)
        r_max = r
        if adaptive_r:
            r = min(
                self.live_count,
                adaptive_r_budget(
                    q,
                    head,
                    self.omega,
                    self.n,
                    k,
                    r_max,
                    target_recall=adaptive_r_target_recall,
                    probability=adaptive_r_probability,
                    mode=adaptive_r_mode,
                ),
            )
        alpha_t = 0.0
        use_tail = opts.tail_policy != "none"
        if use_tail:
            if opts.tail_policy == "fixed":
                if alpha_tail_fixed is None:
                    raise ValueError("alpha_tail_fixed is required when tail_policy='fixed'")
                alpha_t = float(alpha_tail_fixed)
            else:
                alpha_t = tail_alpha(q, head, self.omega, self.n, r, tail_norm, lambda_tail, alpha_min, alpha_max)
        aq = float(opts.alpha_q)
        if self.scan_dtype == "float32":
            aq = 0.0
        preprocess_ms = (perf_counter() - t_pre0) * 1000.0

        prune_code = 0
        if prune_lower_bound:
            prune_code = 2 if prune_lower_mode in {"candidate", "candidates", "topr"} else 1
        block_prune_requested = bool(opts.block_prune)
        block_prune_runtime = bool(
            block_prune_requested
            and self.backend_name == "cpp"
            and self.scan_mode == 0
            and opts.query_mode == "float"
            and self.norm_cv >= float(block_prune_cv_threshold)
        )
        base_ids, base_scores, base_cand_ids, base_cand_scores, native_timings = self._backend_search_base(
            q,
            head,
            k,
            r,
            use_tail,
            alpha_t,
            tail_norm,
            aq,
            opts.query_mode == "int8",
            no_rerank,
            prune_code,
            prune_lower_scale,
            block_prune_runtime,
            int(block_prune_scout_h),
            float(block_prune_beam_factor),
            float(block_prune_gate),
            int(block_prune_block_size),
            bool(cold_rerank_sort_candidates),
        )
        delta_ids, delta_scores = self._delta_candidate_scores(
            q, head, use_tail, alpha_t, tail_norm, aq, opts.query_mode == "int8"
        )
        if delta_ids.size:
            merged_cand_ids = np.concatenate([base_cand_ids, delta_ids])
            merged_cand_scores = np.concatenate([base_cand_scores, delta_scores])
            cand_ids, cand_scores = self._top_by_score(merged_cand_ids, merged_cand_scores, r)
            if no_rerank:
                ids, scores = self._top_by_score(cand_ids, cand_scores, k)
            elif cand_ids.size:
                if cold_rerank_sort_candidates:
                    order = np.argsort(cand_ids, kind="stable")
                    exact_scores = np.empty(cand_ids.size, dtype=np.float32)
                    exact_scores[order] = (self.X[cand_ids[order]] @ q).astype(np.float32)
                else:
                    exact_scores = self.X[cand_ids] @ q
                ids, scores = self._top_by_score(cand_ids, exact_scores.astype(np.float32), k)
            else:
                ids, scores = base_ids, base_scores
        else:
            cand_ids, cand_scores = base_cand_ids, base_cand_scores
            ids, scores = base_ids, base_scores
        total_ms = (perf_counter() - t0) * 1000.0
        diag = {
            "backend": self.backend_name,
            "method": method,
            "canonical_method": opts.method,
            "legacy_method": opts.legacy_method,
            "head_policy": opts.head_policy,
            "tail_policy": opts.tail_policy,
            "r_policy": opts.r_policy,
            "query_mode": opts.query_mode,
            "scan_dtype": self.scan_dtype,
            "scan_layout": self.scan_layout,
            "scan_quantization": self.scan_quantization,
            "scan_insert_policy": self.scan_insert_policy,
            "scan_stride": int(self.scan_stride),
            "scan_capacity": int(self.scan_capacity),
            "scan_slack_fraction": float(self.scan_slack_fraction),
            "scan_growth_events": int(self.scan_growth_events),
            "scan_rebuild_events": int(self.scan_rebuild_events),
            "h": int(selected_h),
            "retained_mass": retained,
            "tail_norm": tail_norm,
            "alpha_tail": alpha_t,
            "alpha_q": aq,
            "r": int(r),
            "r_max": int(r_max),
            "adaptive_r": bool(adaptive_r),
            "adaptive_r_target_recall": float(adaptive_r_target_recall) if adaptive_r else None,
            "adaptive_r_probability": float(adaptive_r_probability) if adaptive_r else None,
            "adaptive_r_mode": adaptive_r_mode if adaptive_r else None,
            "prune_lower_bound": bool(prune_lower_bound),
            "prune_lower_mode": prune_lower_mode if prune_lower_bound else None,
            "prune_lower_scale": float(prune_lower_scale),
            "layout": self.layout,
            "norm_cv": float(self.norm_cv),
            "block_prune_requested": bool(block_prune_requested),
            "block_prune": bool(block_prune_runtime),
            "block_prune_scout_h": int(block_prune_scout_h),
            "block_prune_beam_factor": float(block_prune_beam_factor),
            "block_prune_gate": float(block_prune_gate),
            "block_prune_block_size": int(block_prune_block_size),
            "block_prune_cv_threshold": float(block_prune_cv_threshold),
            "rerank": not no_rerank,
            "preprocess_ms": preprocess_ms,
            "total_ms": total_ms,
            "measured_total_ms": total_ms,
            "python_overhead_ms": max(0.0, total_ms - preprocess_ms - float(native_timings.get("native_total_ms", 0.0))),
            "base_n": int(self._base_n),
            "delta_n": int(sum(1 for row_id in self._delta_ids if row_id < self._n and self._alive[row_id] != 0)),
            "cold_rerank_sort_candidates": bool(cold_rerank_sort_candidates),
            "cold_rerank_apply_to_wall_time": bool(cold_rerank_apply_to_wall_time),
            "cold_rerank_bandwidth_gbps": float(cold_rerank_bandwidth_gbps),
            "cold_rerank_row_latency_ns": float(cold_rerank_row_latency_ns),
            "cold_rerank_page_latency_ns": float(cold_rerank_page_latency_ns),
        }
        diag.update(native_timings)
        cold_stats = cold_rerank_diagnostics(
            cand_ids if not no_rerank else np.empty(0, dtype=np.int64),
            self.d,
            float(diag.get("native_rerank_ms", 0.0)),
            enabled=bool(cold_rerank_enabled and not no_rerank),
            bandwidth_gbps=float(cold_rerank_bandwidth_gbps),
            row_latency_ns=float(cold_rerank_row_latency_ns),
            page_latency_ns=float(cold_rerank_page_latency_ns),
            page_size=int(cold_rerank_page_size),
        )
        diag.update(cold_stats)
        simulated_total_ms = total_ms + float(cold_stats["cold_rerank_penalty_ms"])
        diag["simulated_total_ms"] = simulated_total_ms
        if cold_rerank_apply_to_wall_time:
            diag["total_ms"] = simulated_total_ms
        return SearchResult(self._map_row_ids(ids), scores, self._map_row_ids(cand_ids), cand_scores, diag)

    def search_batch(
        self,
        Q: np.ndarray,
        k: int = 10,
        r: int = 100,
        rho: float = 0.9,
        h_max: int | None = 64,
        h: int | None = None,
        method: str = "etar",
        head_policy: str | None = None,
        tail_policy: str | None = None,
        rerank: bool | str | int | None = None,
        r_policy: str | None = None,
        lambda_tail: float = 0.75,
        alpha_min: float = 0.05,
        alpha_max: float = 0.5,
        alpha_tail_fixed: float | None = None,
        alpha_q: float = 0.0,
        query_mode: str = "float",
        adaptive_r_target_recall: float = 0.95,
        adaptive_r_probability: float = 0.95,
        adaptive_r_mode: str = "event",
        prune_lower_bound: bool = False,
        prune_lower_mode: str = "global",
        prune_lower_scale: float = 1.0,
        block_prune: bool = False,
        block_prune_scout_h: int = 16,
        block_prune_beam_factor: float = 32.0,
        block_prune_gate: float = 0.75,
        block_prune_block_size: int = 256,
        block_prune_cv_threshold: float = 0.0,
        cold_rerank_enabled: bool = False,
        cold_rerank_apply_to_wall_time: bool = False,
        cold_rerank_bandwidth_gbps: float = 0.0,
        cold_rerank_row_latency_ns: float = 0.0,
        cold_rerank_page_latency_ns: float = 0.0,
        cold_rerank_page_size: int = 4096,
        cold_rerank_sort_candidates: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        opts = resolve_etar_options(
            method,
            head_policy=head_policy,
            tail_policy=tail_policy,
            rerank=rerank,
            r_policy=r_policy,
            query_mode=query_mode,
            alpha_tail_fixed=alpha_tail_fixed,
            alpha_q=alpha_q,
            block_prune=block_prune,
        )
        if self.scan_dtype == "float32" and opts.query_mode == "int8":
            opts = replace(opts, query_mode="float")
        h_limit = h if h is not None else h_max
        h_limit = None if h_limit is None else int(h_limit)
        k = min(int(k), self.live_count)
        r = min(max(int(r), k), self.live_count)
        has_live_delta = any(row_id < self._n and self._alive[row_id] != 0 for row_id in self._delta_ids)
        if opts.method == "exact_f32":
            scores = np.asarray(Q, dtype=np.float32) @ self.X.T
            scores[:, self.alive == 0] = -np.inf
            ids = np.argpartition(-scores, kth=min(k - 1, self.n - 1), axis=1)[:, :k]
            row = np.arange(ids.shape[0])[:, None]
            order = np.argsort(-scores[row, ids], axis=1)
            ids = ids[row, order].astype(np.int64)
            mapped = self._map_row_ids(ids)
            return mapped, scores[row, ids].astype(np.float32), mapped.copy(), scores[row, ids].astype(np.float32), {}
        if has_live_delta or opts.r_policy == "adaptive":
            results = [
                self.search(
                    q,
                    k=k,
                    r=r,
                    rho=rho,
                    h_max=h_limit,
                    method=method,
                    head_policy=head_policy,
                    tail_policy=tail_policy,
                    rerank=rerank,
                    r_policy=r_policy,
                    lambda_tail=lambda_tail,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    alpha_tail_fixed=alpha_tail_fixed,
                    alpha_q=alpha_q,
                    query_mode=query_mode,
                    adaptive_r_target_recall=adaptive_r_target_recall,
                    adaptive_r_probability=adaptive_r_probability,
                    adaptive_r_mode=adaptive_r_mode,
                    prune_lower_bound=prune_lower_bound,
                    prune_lower_mode=prune_lower_mode,
                    prune_lower_scale=prune_lower_scale,
                    block_prune=block_prune,
                    block_prune_scout_h=block_prune_scout_h,
                    block_prune_beam_factor=block_prune_beam_factor,
                    block_prune_gate=block_prune_gate,
                    block_prune_block_size=block_prune_block_size,
                    block_prune_cv_threshold=block_prune_cv_threshold,
                    cold_rerank_enabled=cold_rerank_enabled,
                    cold_rerank_apply_to_wall_time=cold_rerank_apply_to_wall_time,
                    cold_rerank_bandwidth_gbps=cold_rerank_bandwidth_gbps,
                    cold_rerank_row_latency_ns=cold_rerank_row_latency_ns,
                    cold_rerank_page_latency_ns=cold_rerank_page_latency_ns,
                    cold_rerank_page_size=cold_rerank_page_size,
                    cold_rerank_sort_candidates=cold_rerank_sort_candidates,
                )
                for q in np.asarray(Q, dtype=np.float32)
            ]
            max_cand = max((res.candidate_ids.size for res in results), default=0)
            ids = np.full((len(results), k), -1, dtype=np.int64)
            scores = np.full((len(results), k), -np.inf, dtype=np.float32)
            cand_ids = np.full((len(results), max_cand), -1, dtype=np.int64)
            cand_scores = np.full((len(results), max_cand), -np.inf, dtype=np.float32)
            for i, res in enumerate(results):
                ids[i, : res.ids.size] = res.ids
                scores[i, : res.scores.size] = res.scores
                cand_ids[i, : res.candidate_ids.size] = res.candidate_ids
                cand_scores[i, : res.candidate_scores.size] = res.candidate_scores
            timings = {
                "dynamic_loop": True,
                "r": np.asarray([res.diagnostics.get("r", np.nan) for res in results], dtype=np.float32),
                "native_scan_ms": np.asarray([res.diagnostics.get("native_scan_ms", np.nan) for res in results], dtype=np.float32),
                "native_top_r_ms": np.asarray([res.diagnostics.get("native_top_r_ms", np.nan) for res in results], dtype=np.float32),
                "native_rerank_ms": np.asarray([res.diagnostics.get("native_rerank_ms", np.nan) for res in results], dtype=np.float32),
                "native_total_ms": np.asarray([res.diagnostics.get("native_total_ms", np.nan) for res in results], dtype=np.float32),
                "native_block_prune_survival": np.asarray([res.diagnostics.get("native_block_prune_survival", np.nan) for res in results], dtype=np.float32),
                "native_block_prune_used": np.asarray([res.diagnostics.get("native_block_prune_used", np.nan) for res in results], dtype=np.float32),
                "native_block_prune_fallback": np.asarray([res.diagnostics.get("native_block_prune_fallback", np.nan) for res in results], dtype=np.float32),
                "measured_total_ms": np.asarray([res.diagnostics.get("measured_total_ms", np.nan) for res in results], dtype=np.float32),
                "simulated_total_ms": np.asarray([res.diagnostics.get("simulated_total_ms", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_rows": np.asarray([res.diagnostics.get("cold_rerank_rows", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_bytes": np.asarray([res.diagnostics.get("cold_rerank_bytes", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_pages": np.asarray([res.diagnostics.get("cold_rerank_pages", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_bandwidth_ms": np.asarray([res.diagnostics.get("cold_rerank_bandwidth_ms", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_latency_ms": np.asarray([res.diagnostics.get("cold_rerank_latency_ms", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_model_ms": np.asarray([res.diagnostics.get("cold_rerank_model_ms", np.nan) for res in results], dtype=np.float32),
                "cold_rerank_penalty_ms": np.asarray([res.diagnostics.get("cold_rerank_penalty_ms", np.nan) for res in results], dtype=np.float32),
            }
            return ids, scores, cand_ids, cand_scores, timings
        no_rerank = not opts.rerank
        if opts.force_r_to_k:
            r = k
        Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float32))
        heads = []
        offsets = [0]
        use_tail_arr = []
        alpha_tail_arr = []
        tail_norm_arr = []
        alpha_q_arr = []
        int8_query_arr = []
        no_rerank_arr = []
        prune_lower_bound_arr = []
        prune_lower_scale_arr = []
        for q in Q:
            head, _, tail_norm, _ = select_head_policy(q, rho, h_limit, opts.head_policy)
            use_tail = opts.tail_policy != "none"
            at = 0.0
            if use_tail:
                if opts.tail_policy == "fixed":
                    if alpha_tail_fixed is None:
                        raise ValueError("alpha_tail_fixed is required when tail_policy='fixed'")
                    at = float(alpha_tail_fixed)
                else:
                    at = tail_alpha(q, head, self.omega, self.n, r, tail_norm, lambda_tail, alpha_min, alpha_max)
            heads.append(head)
            offsets.append(offsets[-1] + head.size)
            use_tail_arr.append(int(use_tail))
            alpha_tail_arr.append(at)
            tail_norm_arr.append(tail_norm)
            alpha_q_arr.append(0.0 if self.scan_dtype == "float32" else float(opts.alpha_q))
            int8_query_arr.append(int(opts.query_mode == "int8"))
            no_rerank_arr.append(int(no_rerank))
            prune_lower_bound_arr.append(int(prune_lower_bound))
            if prune_lower_bound and prune_lower_mode in {"candidate", "candidates", "topr"}:
                prune_lower_bound_arr[-1] = 2
            prune_lower_scale_arr.append(float(prune_lower_scale))
        flat_heads = np.ascontiguousarray(np.concatenate(heads).astype(np.int64))
        alive_backend = self.alive if (self.has_tombstones or self.backend_name != "cpp") else None
        block_prune_requested = bool(opts.block_prune)
        block_prune_runtime = bool(
            block_prune_requested
            and self.backend_name == "cpp"
            and self.scan_mode == 0
            and opts.query_mode == "float"
            and self.norm_cv >= float(block_prune_cv_threshold)
        )
        ids, scores, cand_ids, cand_scores, timings = self.backend.search_batch(
            self.X, self.X_scan if self.scan_mode == 3 else self.X, self.C_col_scan, self.scales, self.scan_scales, self.norms, self.residual_norms, alive_backend,
            Q, flat_heads, np.asarray(offsets, dtype=np.int64), self.scan_stride, k, r,
            np.asarray(use_tail_arr, dtype=np.int32), np.asarray(alpha_tail_arr, dtype=np.float32),
            np.asarray(tail_norm_arr, dtype=np.float32), np.asarray(alpha_q_arr, dtype=np.float32),
            np.asarray(int8_query_arr, dtype=np.int32), self.scan_mode, np.asarray(no_rerank_arr, dtype=np.int32),
            np.asarray(prune_lower_bound_arr, dtype=np.int32),
            np.asarray(prune_lower_scale_arr, dtype=np.float32),
            block_prune_runtime,
            int(block_prune_scout_h),
            float(block_prune_beam_factor),
            float(block_prune_gate),
            int(block_prune_block_size),
            bool(cold_rerank_sort_candidates),
        )
        timings.update(
            cold_rerank_batch_diagnostics(
                cand_ids if not no_rerank else np.empty((Q.shape[0], 0), dtype=np.int64),
                self.d,
                timings.get("native_rerank_ms", 0.0),
                enabled=bool(cold_rerank_enabled and not no_rerank),
                bandwidth_gbps=float(cold_rerank_bandwidth_gbps),
                row_latency_ns=float(cold_rerank_row_latency_ns),
                page_latency_ns=float(cold_rerank_page_latency_ns),
                page_size=int(cold_rerank_page_size),
            )
        )
        native_total = np.asarray(timings.get("native_total_ms", np.zeros(Q.shape[0], dtype=np.float32)), dtype=np.float32)
        penalty = np.asarray(timings.get("cold_rerank_penalty_ms", np.zeros(Q.shape[0], dtype=np.float32)), dtype=np.float32)
        timings["simulated_native_total_ms"] = native_total + penalty
        return self._map_row_ids(ids), scores, self._map_row_ids(cand_ids), cand_scores, timings

    def _exact(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.X @ q
        scores = np.where(self.alive.astype(bool), scores, -np.inf)
        ids = np.argpartition(-scores, kth=min(k - 1, scores.size - 1))[:k]
        ids = ids[np.argsort(-scores[ids], kind="stable")]
        return ids.astype(np.int64), scores[ids].astype(np.float32)
