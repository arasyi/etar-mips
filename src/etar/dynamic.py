from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .core import EtarIndex, adaptive_r_budget, resolve_etar_options, select_head_policy, tail_alpha
from .metrics import candidate_coverage, recall_at_k
from .profiling import etar_storage_bytes, rss_mb
from .storage_native import CppStorageIndex
from .threads import configure_faiss_threads, set_single_thread_env
from .timing import latency_summary


@dataclass(frozen=True)
class StreamOp:
    step: int
    operation: str


@dataclass(frozen=True)
class DynamicMethodSpec:
    name: str
    params: dict[str, Any]


def _sample_vectors(regime: str, n: int, d: int, rng: np.random.Generator, drift: float = 0.0) -> np.ndarray:
    shift = np.full((1, d), drift, dtype=np.float32)
    if regime in {"dense_gaussian", "stream_drift"}:
        return (rng.normal(size=(n, d)).astype(np.float32) + shift).astype(np.float32)
    if regime == "norm_heavy":
        x = rng.normal(size=(n, d)).astype(np.float32)
        x *= rng.lognormal(mean=0.0 + drift, sigma=1.0, size=(n, 1)).astype(np.float32)
        return x
    if regime == "bursty_events":
        x = rng.normal(scale=0.25, size=(n, d)).astype(np.float32) + shift
        cols = rng.integers(0, d, size=n)
        x[np.arange(n), cols] += rng.normal(6.0 + drift, 1.0, size=n).astype(np.float32)
        return x
    raise ValueError(f"unknown streaming regime {regime!r}")


def operation_sequence(steps: int, mix: dict[str, float], seed: int, sliding_window: bool = False) -> list[StreamOp]:
    if sliding_window:
        pattern = ["insert", "delete", "query"]
        return [StreamOp(i, pattern[i % len(pattern)]) for i in range(int(steps))]
    return mixed_operation_sequence(steps, mix, seed)


def mixed_operation_sequence(steps: int, mix: dict[str, float], seed: int, allowed_ops: list[str] | None = None) -> list[StreamOp]:
    ops = allowed_ops or ["query", "insert", "update", "delete"]
    weights = np.array([float(mix.get(op, 0.0)) for op in ops], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("operation_mix must have positive total weight")
    weights = weights / weights.sum()
    raw = weights * int(steps)
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: int(steps) - int(counts.sum())]:
        counts[i] += 1
    seq = []
    for op, count in zip(ops, counts):
        seq.extend([op] * int(count))
    rng = np.random.default_rng(seed)
    rng.shuffle(seq)
    return [StreamOp(i, op) for i, op in enumerate(seq)]


def workload_operation_sequence(cfg: dict[str, Any], seed: int) -> list[StreamOp]:
    steps = int(cfg.get("steps", 1000))
    workload_type = str(cfg.get("type", "mixed"))
    name = str(cfg.get("name", "dynamic"))
    if workload_type == "sliding_window" or name == "sliding_window":
        return operation_sequence(steps, {}, seed, sliding_window=True)
    if workload_type == "append_only":
        mix = cfg.get("operation_mix", {"query": 0.8, "insert": 0.2})
        return mixed_operation_sequence(steps, mix, seed, ["query", "insert"])
    if workload_type == "burst_ingest":
        query_mix = cfg.get("query_mix", {"query": 0.95, "insert": 0.05})
        burst_mix = cfg.get("burst_mix", {"query": 0.1, "insert": 0.9})
        burst_every = max(1, int(cfg.get("burst_every", 1000)))
        burst_steps = max(1, int(cfg.get("burst_steps", 100)))
        ops: list[StreamOp] = []
        offset = 0
        phase_idx = 0
        while offset < steps:
            in_burst = offset > 0 and offset % burst_every == 0
            length = min(burst_steps if in_burst else min(burst_every - (offset % burst_every), steps - offset), steps - offset)
            mix = burst_mix if in_burst else query_mix
            phase_ops = mixed_operation_sequence(length, mix, seed + phase_idx, ["query", "insert", "update", "delete"])
            ops.extend(StreamOp(offset + op.step, op.operation) for op in phase_ops)
            offset += length
            phase_idx += 1
        return ops
    if workload_type == "phased":
        phases = cfg.get("phases", [])
        if not phases:
            raise ValueError("phased dynamic workload requires non-empty phases")
        ops = []
        offset = 0
        for i, phase in enumerate(phases):
            if offset >= steps:
                break
            length = min(int(phase.get("steps", steps - offset)), steps - offset)
            phase_type = str(phase.get("type", "mixed"))
            if phase_type == "append_only":
                allowed = ["query", "insert"]
                mix = phase.get("operation_mix", {"query": 0.8, "insert": 0.2})
            else:
                allowed = ["query", "insert", "update", "delete"]
                mix = phase.get("operation_mix", cfg.get("operation_mix", {"query": 0.7, "insert": 0.1, "update": 0.1, "delete": 0.1}))
            phase_ops = mixed_operation_sequence(length, mix, seed + i, allowed)
            ops.extend(StreamOp(offset + op.step, op.operation) for op in phase_ops)
            offset += length
        if offset < steps:
            tail = mixed_operation_sequence(steps - offset, cfg.get("operation_mix", {"query": 0.7, "insert": 0.1, "update": 0.1, "delete": 0.1}), seed + len(phases))
            ops.extend(StreamOp(offset + op.step, op.operation) for op in tail)
        return ops
    return mixed_operation_sequence(steps, cfg.get("operation_mix", {"query": 0.7, "insert": 0.1, "update": 0.1, "delete": 0.1}), seed)


class StreamingWorkload:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = dict(cfg)
        self.name = str(cfg.get("name", "dynamic"))
        self.regime = str(cfg.get("regime", "stream_drift"))
        self.initial_n = int(cfg.get("initial_n", 1000))
        self.stream_n = int(cfg.get("stream_n", max(1, cfg.get("steps", 100))))
        self.d = int(cfg.get("d", 64))
        self.steps = int(cfg.get("steps", 1000))
        self.query_count = int(cfg.get("queries", self.steps))
        self.seed = int(cfg.get("seed", 123))
        self.k = int(cfg.get("k", 10))
        self.window_size = int(cfg.get("window_size", self.initial_n))
        self.recall_every = int(cfg.get("recall_every", 25))
        self.eval_after_events = set(cfg.get("eval_after_events", ["insert", "update", "delete", "compact", "rebuild"]))
        self.eval_every_mutations = max(1, int(cfg.get("eval_every_mutations", 1)))
        self.eval_query_count = int(cfg.get("eval_query_count", min(16, self.query_count)))
        self.eval_query_seed = int(cfg.get("eval_query_seed", self.seed + 1009))
        self.eval_mode = str(cfg.get("eval_mode", "single_query"))
        self.victim_policy = str(cfg.get("victim_policy", "fifo"))
        self.drift = float(cfg.get("drift", 0.5 if self.regime == "stream_drift" else 0.0))
        self.sliding = self.name == "sliding_window" or str(cfg.get("type", "")) == "sliding_window"
        self.rng = np.random.default_rng(self.seed + 7919)
        rng = np.random.default_rng(self.seed)
        self.initial = _sample_vectors(self.regime, self.initial_n, self.d, rng, drift=0.0)
        self.stream = _sample_vectors(self.regime, max(self.stream_n, self.steps + 8), self.d, rng, drift=self.drift)
        self.queries = _sample_vectors(self.regime, self.query_count, self.d, rng, drift=self.drift)
        eval_rng = np.random.default_rng(self.eval_query_seed)
        self.eval_queries = _sample_vectors(self.regime, self.eval_query_count, self.d, eval_rng, drift=self.drift)
        self.ops = workload_operation_sequence(cfg, self.seed)


def exact_live_topk(vectors: np.ndarray, ids: list[int], q: np.ndarray, k: int) -> np.ndarray:
    if len(ids) == 0:
        return np.empty((1, 0), dtype=np.int64)
    scores = vectors[np.asarray(ids)] @ q
    take = min(k, len(ids))
    order = np.argsort(-scores, kind="stable")[:take]
    return np.asarray(ids, dtype=np.int64)[order][None, :]


def should_eval_checkpoint(op: StreamOp, mutation_count: int, workload_cfg: dict[str, Any]) -> bool:
    triggers = set(workload_cfg.get("eval_after_events", ["insert", "update", "delete", "compact", "rebuild"]))
    every = max(1, int(workload_cfg.get("eval_every_mutations", 1)))
    return op.operation in triggers and mutation_count > 0 and mutation_count % every == 0


def _live_matrix(vectors: np.ndarray | dict[int, np.ndarray], live_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(live_ids, dtype=np.int64)
    if len(labels) == 0:
        return np.empty((0, 0), dtype=np.float32), labels
    if isinstance(vectors, dict):
        return np.vstack([vectors[int(i)] for i in labels]).astype(np.float32), labels
    return vectors[labels].astype(np.float32), labels


def _truth_for_queries(vectors: np.ndarray | dict[int, np.ndarray], live_ids: list[int], queries: np.ndarray, k: int) -> np.ndarray:
    mat, labels = _live_matrix(vectors, live_ids)
    if mat.size == 0:
        return np.empty((queries.shape[0], 0), dtype=np.int64)
    scores = queries @ mat.T
    take = min(k, len(labels))
    idx = np.argpartition(-scores, kth=take - 1, axis=1)[:, :take]
    row = np.arange(queries.shape[0])[:, None]
    order = np.argsort(-scores[row, idx], axis=1)
    return labels[idx[row, order]]


def _omega_contribution(x: np.ndarray) -> tuple[np.ndarray | None, int]:
    norm = float(np.linalg.norm(x))
    if norm <= 0.0:
        return None, 0
    y = (np.asarray(x, dtype=np.float32) / norm).astype(np.float64)
    return y * y, 1


def _initial_omega_state(x: np.ndarray) -> tuple[np.ndarray, int]:
    mat = np.asarray(x, dtype=np.float32)
    if mat.size == 0:
        return np.zeros(mat.shape[1] if mat.ndim == 2 else 0, dtype=np.float64), 0
    norms = np.linalg.norm(mat, axis=1)
    mask = norms > 0
    if not np.any(mask):
        return np.zeros(mat.shape[1], dtype=np.float64), 0
    y = mat[mask] / norms[mask, None]
    return np.sum((y * y).astype(np.float64), axis=0), int(mask.sum())


def _omega_value(omega_sum: np.ndarray, omega_count: int) -> np.ndarray:
    if omega_count <= 0:
        return np.zeros_like(omega_sum, dtype=np.float32)
    return (omega_sum / float(omega_count)).astype(np.float32)


def _pop_victim(
    live_ids: list[int],
    vectors: np.ndarray | dict[int, np.ndarray],
    policy: str,
    rng: np.random.Generator,
) -> int | None:
    if not live_ids:
        return None
    if policy in {"fifo", "oldest"}:
        return live_ids.pop(0)
    if policy in {"lifo", "newest"}:
        return live_ids.pop()
    if policy == "random":
        pos = int(rng.integers(0, len(live_ids)))
        return live_ids.pop(pos)
    if policy in {"high_norm", "highest_norm"}:
        if isinstance(vectors, dict):
            norms = [float(np.linalg.norm(vectors[int(i)])) for i in live_ids]
        else:
            norms = [float(np.linalg.norm(vectors[int(i)])) for i in live_ids]
        pos = int(np.argmax(norms))
        return live_ids.pop(pos)
    raise ValueError(f"unknown victim_policy {policy!r}")


def _dynamic_method_specs(workload_cfg: dict[str, Any]) -> list[DynamicMethodSpec]:
    raw_methods = workload_cfg.get("methods", ["etar"])
    specs = []
    for raw in raw_methods:
        if isinstance(raw, str):
            params = dict(workload_cfg.get(raw, {}))
            if raw == "etar" or raw.startswith("etar_"):
                params = {**dict(workload_cfg.get("etar", {})), **params}
            specs.append(DynamicMethodSpec(raw, params))
        elif isinstance(raw, dict):
            params = dict(raw)
            name = str(params.pop("name", params.get("method", params.get("family", ""))))
            if not name:
                raise ValueError("dynamic method mapping requires a name")
            base = dict(workload_cfg.get(name, {}))
            if name == "etar" or name.startswith("etar_"):
                base = {**dict(workload_cfg.get("etar", {})), **base}
            specs.append(DynamicMethodSpec(name, {**base, **params}))
        else:
            raise ValueError("dynamic methods must be strings or mappings")
    return specs


def _should_periodic_rebuild(op: StreamOp, mutation_count: int, last_rebuild_mutation: int, rebuild_every: int, rebuild_unit: str) -> bool:
    if rebuild_every <= 0:
        return False
    if rebuild_unit == "mutations":
        return mutation_count > last_rebuild_mutation and (mutation_count - last_rebuild_mutation) >= rebuild_every
    if rebuild_unit == "steps":
        return (op.step + 1) % rebuild_every == 0
    raise ValueError(f"unknown rebuild_unit {rebuild_unit!r}")


def evaluate_checkpoint(
    workload: StreamingWorkload,
    method: str,
    trigger_op: StreamOp,
    checkpoint_id: int,
    mutation_count: int,
    live_ids: list[int],
    total_n: int,
    vectors: np.ndarray | dict[int, np.ndarray],
    query_fn,
    status: str = "ok",
) -> dict[str, Any]:
    truth = _truth_for_queries(vectors, live_ids, workload.eval_queries, workload.k)
    pred = []
    candidates = []
    latencies = []
    for q in workload.eval_queries:
        t0 = perf_counter()
        out = query_fn(q)
        latencies.append((perf_counter() - t0) * 1000)
        if isinstance(out, tuple):
            ids, cand = out
        else:
            ids, cand = out, None
        pred.append(np.asarray(ids, dtype=np.int64)[: truth.shape[1]])
        if cand is not None:
            candidates.append(np.asarray(cand, dtype=np.int64))
    pred_arr = np.vstack(pred) if pred else np.empty_like(truth)
    lat = latency_summary(latencies)
    row = {
        "row_type": "checkpoint",
        "workload": workload.name,
        "method": method,
        "operation": "eval_checkpoint",
        "trigger_operation": trigger_op.operation,
        "trigger_step": trigger_op.step,
        "mutation_count": mutation_count,
        "checkpoint_id": checkpoint_id,
        "eval_query_count": workload.eval_queries.shape[0],
        "live_n": len(live_ids),
        "total_n": total_n,
        "tombstone_fraction": 1.0 - (len(live_ids) / max(total_n, 1)),
        "eval_recall_at_k": recall_at_k(pred_arr, truth, truth.shape[1]) if truth.shape[1] else np.nan,
        "eval_candidate_coverage": candidate_coverage(candidates, truth, truth.shape[1]) if candidates and truth.shape[1] else np.nan,
        "status": status,
    }
    for key, value in lat.items():
        row[f"eval_{key}"] = value
    return row


def _summary(rows: list[dict[str, Any]], workload: str, method: str) -> list[dict[str, Any]]:
    out = []

    def _safe_nanmean(values: list[Any]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanmean(arr))

    total_ms = sum(float(r.get("latency_ms", 0.0) or 0.0) for r in rows if r.get("row_type") == "event")
    total_ops = sum(1 for r in rows if r.get("row_type") == "event")
    query_rows = [r for r in rows if r.get("operation") == "query" and r.get("row_type") == "event"]
    if query_rows:
        lat = latency_summary([r["latency_ms"] for r in query_rows])
        recalls = [r["recall_at_k"] for r in query_rows if not pd.isna(r.get("recall_at_k", np.nan))]
        out.append({
            "row_type": "summary",
            "workload": workload,
            "method": method,
            "operation": "query",
            **lat,
            "recall_at_k": float(np.mean(recalls)) if recalls else np.nan,
            "ops_per_sec": float(total_ops / max(total_ms / 1000.0, 1e-12)),
            "queries_per_sec": float(len(query_rows) / max(sum(r["latency_ms"] for r in query_rows) / 1000.0, 1e-12)),
        })
    for op in ["insert", "update", "delete", "compact", "rebuild"]:
        part = [r for r in rows if r.get("operation") == op and r.get("row_type") == "event"]
        if part:
            out.append({"row_type": "summary", "workload": workload, "method": method, "operation": op, **latency_summary([r["latency_ms"] for r in part])})
    checkpoint_rows = [r for r in rows if r.get("row_type") == "checkpoint"]
    if checkpoint_rows:
        lat = latency_summary([r["eval_latency_ms_mean"] for r in checkpoint_rows])
        out.append({
            "row_type": "summary",
            "workload": workload,
            "method": method,
            "operation": "eval_checkpoint",
            **{f"eval_{k}": v for k, v in lat.items()},
            "eval_recall_at_k": _safe_nanmean([r.get("eval_recall_at_k", np.nan) for r in checkpoint_rows]),
            "eval_candidate_coverage": _safe_nanmean([r.get("eval_candidate_coverage", np.nan) for r in checkpoint_rows]),
        })
    return out


def _build_record(
    workload: StreamingWorkload,
    method: str,
    build_ms: float,
    rss_before_build: float,
    rss_after_build: float,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "workload": workload.name,
        "method": method,
        "operation": "build",
        "initial_n": workload.initial_n,
        "d": workload.d,
        "build_ms": float(build_ms),
        "rss_before_build_mb": rss_before_build,
        "rss_after_build_mb": rss_after_build,
        "rss_build_delta_mb": (
            rss_after_build - rss_before_build
            if not (np.isnan(rss_before_build) or np.isnan(rss_after_build))
            else np.nan
        ),
        "status": "ok",
    }
    row.update(extra)
    return row


def _split_stream_result(result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(result, tuple):
        rows, build_rows = result
        return list(rows), list(build_rows)
    return list(result), []


def run_dynamic_workload_with_build(
    workload_cfg: dict[str, Any],
    progress_fn=None,
    workload_index: int = 1,
    workload_total: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workload = StreamingWorkload(workload_cfg)
    methods = _dynamic_method_specs(workload_cfg)
    frames = []
    build_frames = []
    for method_index, spec in enumerate(methods, start=1):
        method = spec.name
        if progress_fn is not None:
            progress_fn(
                f"dynamic workload {workload_index}/{workload_total} "
                f"method {method_index}/{len(methods)}: workload={workload.name} method={method}"
            )
        if method == "etar" or method.startswith("etar_"):
            rows, build_rows = _split_stream_result(_run_etar_stream(workload, spec.params, method_name=method))
        elif method == "hnswlib":
            rows, build_rows = _split_stream_result(_run_hnswlib_stream(workload, spec.params))
        elif method == "faiss_flat":
            rows, build_rows = _split_stream_result(_run_faiss_flat_stream(workload))
        elif method == "faiss_hnsw":
            rebuild_every = int(spec.params.get("rebuild_after_mutations", spec.params.get("rebuild_every", 0)))
            rebuild_unit = str(spec.params.get("rebuild_unit", "mutations" if "rebuild_after_mutations" in spec.params else workload_cfg.get("rebuild_unit", "steps")))
            rows, build_rows = _split_stream_result(_run_faiss_hnsw_stream(workload, spec.params, rebuild_every=rebuild_every, rebuild_unit=rebuild_unit))
        elif method in {"faiss_ivf", "faiss_ivfpq"}:
            rebuild_every = int(spec.params.get("rebuild_after_mutations", spec.params.get("rebuild_every", 0)))
            rebuild_unit = str(spec.params.get("rebuild_unit", "mutations" if "rebuild_after_mutations" in spec.params else workload_cfg.get("rebuild_unit", "steps")))
            rows, build_rows = _split_stream_result(_run_faiss_inverted_stream(workload, method, spec.params, rebuild_every=rebuild_every, rebuild_unit=rebuild_unit))
        elif method in {"annoy_rebuild", "scann_static"}:
            rebuild_every = int(spec.params.get("rebuild_after_mutations", spec.params.get("rebuild_every", workload_cfg.get("rebuild_every", 100))))
            rebuild_unit = str(spec.params.get("rebuild_unit", "mutations" if "rebuild_after_mutations" in spec.params else workload_cfg.get("rebuild_unit", "steps")))
            rows, build_rows = _split_stream_result(_run_rebuild_static_stream(workload, method, rebuild_every=rebuild_every, rebuild_unit=rebuild_unit))
        else:
            rows = [{"row_type": "summary", "workload": workload.name, "method": method, "operation": "dynamic", "status": "unsupported"}]
            build_rows = []
        frames.append(pd.DataFrame(rows + _summary(rows, workload.name, method)))
        if build_rows:
            build_frames.append(pd.DataFrame(build_rows))
        if progress_fn is not None:
            checkpoints = sum(1 for r in rows if r.get("row_type") == "checkpoint")
            progress_fn(
                f"dynamic workload {workload_index}/{workload_total} "
                f"method {method_index}/{len(methods)}: done workload={workload.name} "
                f"method={method} rows={len(rows)} checkpoints={checkpoints}"
            )
    dynamic = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    build = pd.concat(build_frames, ignore_index=True) if build_frames else pd.DataFrame()
    return dynamic, build


def run_dynamic_workload(workload_cfg: dict[str, Any], progress_fn=None, workload_index: int = 1, workload_total: int = 1) -> pd.DataFrame:
    dynamic, _build = run_dynamic_workload_with_build(
        workload_cfg,
        progress_fn=progress_fn,
        workload_index=workload_index,
        workload_total=workload_total,
    )
    return dynamic


def _store_residuals(params: dict[str, Any]) -> bool:
    if "store_residuals" in params:
        return bool(params["store_residuals"])
    if "residuals" in params:
        return bool(params["residuals"])
    return float(params.get("alpha_q", 0.0) or 0.0) != 0.0


def _store_float32(params: dict[str, Any]) -> bool:
    preset = _storage_preset(params)
    if preset in {"8+32", "32+32"}:
        return True
    if preset in {"8+8", "8-shared", "32-shared"}:
        return False
    if "store_float32" in params:
        return bool(params["store_float32"])
    if "keep_float32" in params:
        return bool(params["keep_float32"])
    rerank_dtype = str(params.get("rerank_dtype", "float32")).lower()
    return rerank_dtype not in {"int8", "i8", "8bit", "quantized"}


def _store_codes_row(params: dict[str, Any]) -> bool:
    preset = _storage_preset(params)
    if preset == "8+8":
        return True
    if preset in {"8+32", "8-shared", "32+32", "32-shared"}:
        return False
    if "store_codes_row" in params:
        return bool(params["store_codes_row"])
    return bool(params.get("dual_8bit_storage", False))


def _storage_preset(params: dict[str, Any]) -> str:
    raw = params.get("storage_preset", params.get("storage_view", ""))
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


def _query_quantized(params: dict[str, Any]) -> bool:
    mode = str(params.get("query_mode", params.get("scan_query_mode", "float"))).lower()
    return mode in {"int8", "i8", "8bit", "quantized", "q8"}


def _rerank_query_quantized(params: dict[str, Any]) -> bool:
    mode = str(params.get("rerank_query_mode", params.get("rerank_query_quantization", params.get("query_mode", "float")))).lower()
    return mode in {"int8", "i8", "8bit", "quantized", "q8"}


def _event(workload: StreamingWorkload, method: str, op: StreamOp, latency_ms: float, live_ids: list[int], total_n: int, recall: float | None = None, status: str = "ok") -> dict[str, Any]:
    return {
        "row_type": "event",
        "workload": workload.name,
        "method": method,
        "operation": op.operation,
        "step": op.step,
        "live_n": len(live_ids),
        "total_n": total_n,
        "tombstone_fraction": 1.0 - (len(live_ids) / max(total_n, 1)),
        "latency_ms": float(latency_ms),
        "recall_at_k": np.nan if recall is None else float(recall),
        "status": status,
    }


def _run_etar_stream(workload: StreamingWorkload, params: dict[str, Any], method_name: str = "etar") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(params.get("storage_backend", "numpy")).lower() == "cpp":
        return _run_etar_cpp_storage_stream(workload, params, method_name)

    rss_before_build = rss_mb()
    t_build = perf_counter()
    idx = EtarIndex(
        workload.initial,
        backend=params.get("backend", "auto"),
        residuals=_store_residuals(params),
        scan_quantization=params.get("scan_quantization"),
        scan_dtype=str(params.get("scan_dtype", "int8")),
        scan_layout=params.get("scan_layout"),
        scan_insert_policy=params.get("scan_insert_policy", "auto"),
        scan_growth_factor=float(params.get("scan_growth_factor", 2.0)),
        scan_initial_capacity_factor=float(params.get("scan_initial_capacity_factor", 1.25)),
        scan_shrink_threshold=float(params.get("scan_shrink_threshold", 0.25)),
        scan_shrink_factor=float(params.get("scan_shrink_factor", 0.5)),
        scan_compact_shrink=bool(params.get("scan_compact_shrink", True)),
        omega_update_policy=str(params.get("omega_update_policy", workload.cfg.get("omega_update_policy", "lazy"))),
    )
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_profile = etar_storage_bytes(
        n=idx.n,
        d=idx.d,
        capacity=int(getattr(idx, "_capacity", idx.scan_capacity)),
        scan_dtype=idx.scan_dtype,
        residuals=bool(getattr(idx, "residuals_enabled", False)),
    )
    build_row = _build_record(
        workload,
        method_name,
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend="numpy",
        scan_dtype=idx.scan_dtype,
        scan_layout=idx.scan_layout,
        scan_quantization=idx.scan_quantization,
        scan_insert_policy=idx.scan_insert_policy,
        scan_capacity=int(idx.scan_capacity),
        scan_slack_fraction=float(idx.scan_slack_fraction),
        **build_profile,
    )
    live = list(range(workload.initial_n))
    stream_pos = 0
    query_pos = 0
    rows = []
    compact_every = int(params.get("compact_every", 0) or 0)
    compact_on_delete = bool(params.get("compact_on_delete", params.get("compact_after_delete", False)))
    compact_tombstone_raw = params.get("compact_tombstone_fraction", None)
    compact_tombstone_fraction = None if compact_tombstone_raw is None else float(compact_tombstone_raw)
    compact_min_tombstones = int(params.get("compact_min_tombstones", 1))
    compact_on_shrink = bool(params.get("compact_on_shrink", True))
    mutation_count = 0
    checkpoint_id = 0

    def with_scan_state(row: dict[str, Any]) -> dict[str, Any]:
        profile = etar_storage_bytes(
            n=idx.n,
            d=idx.d,
            capacity=int(getattr(idx, "_capacity", idx.scan_capacity)),
            scan_dtype=idx.scan_dtype,
            residuals=bool(getattr(idx, "residuals_enabled", False)),
        )
        row.update({
            "storage_backend": "numpy",
            "scan_dtype": idx.scan_dtype,
            "scan_layout": idx.scan_layout,
            "scan_quantization": idx.scan_quantization,
            "scan_insert_policy": idx.scan_insert_policy,
            "scan_stride": int(idx.scan_stride),
            "scan_capacity": int(idx.scan_capacity),
            "scan_slack_fraction": float(idx.scan_slack_fraction),
            "scan_growth_events": int(idx.scan_growth_events),
            "scan_rebuild_events": int(idx.scan_rebuild_events),
            **profile,
        })
        return row

    def search_params() -> dict[str, Any]:
        out = {
            "k": workload.k,
            "r": int(params.get("r", params.get("R", 100))),
            "rho": float(params.get("rho", 0.9)),
            "h_max": int(params.get("h_max", 64)),
            "method": params.get("method", "etar"),
            "lambda_tail": float(params.get("lambda_tail", 0.75)),
            "alpha_min": float(params.get("alpha_min", 0.05)),
            "alpha_max": float(params.get("alpha_max", 0.5)),
            "alpha_q": float(params.get("alpha_q", 0.0)),
            "query_mode": str(params.get("query_mode", "float")),
        }
        for key in ("h", "head_policy", "tail_policy", "rerank", "r_policy"):
            if key in params:
                out[key] = params[key]
        if params.get("alpha_tail_fixed") is not None:
            out["alpha_tail_fixed"] = float(params["alpha_tail_fixed"])
        return out

    def compact_index(op: StreamOp, reason: str) -> None:
        nonlocal live, checkpoint_id
        old_live = np.flatnonzero(idx.alive)
        mapping = {int(old): new for new, old in enumerate(old_live)}
        t0 = perf_counter()
        idx.compact()
        ms = (perf_counter() - t0) * 1000
        live = [mapping[x] for x in live if x in mapping]
        compact_op = StreamOp(op.step, "compact")
        row = with_scan_state(_event(workload, method_name, compact_op, ms, live, idx.n))
        row["compact_reason"] = reason
        rows.append(row)
        if should_eval_checkpoint(compact_op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, method_name, compact_op, checkpoint_id, mutation_count, live, idx.n, idx.X,
                lambda q: (
                    (res := idx.search(q, **search_params())).ids,
                    res.candidate_ids,
                ),
            ))

    for op in workload.ops:
        recall = None
        compacted_this_step = False
        if op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            res = idx.search(q, **search_params())
            ms = (perf_counter() - t0) * 1000
            if op.step % workload.recall_every == 0:
                truth = exact_live_topk(idx.X, live, q, workload.k)
                recall = recall_at_k(res.ids[None, :], truth, min(workload.k, truth.shape[1]))
        elif op.operation == "insert":
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            t0 = perf_counter()
            live.append(idx.insert(x))
            ms = (perf_counter() - t0) * 1000
            mutation_count += 1
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, idx.X, workload.victim_policy, workload.rng)
            t0 = perf_counter()
            idx.delete(victim)
            ms = (perf_counter() - t0) * 1000
            mutation_count += 1
        elif op.operation == "update" and live:
            victim = _pop_victim(live, idx.X, workload.victim_policy, workload.rng)
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            t0 = perf_counter()
            live.append(idx.update(victim, x))
            ms = (perf_counter() - t0) * 1000
            mutation_count += 1
        else:
            ms = 0.0
        rows.append(with_scan_state(_event(workload, method_name, op, ms, live, idx.n, recall)))
        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, method_name, op, checkpoint_id, mutation_count, live, idx.n, idx.X,
                lambda q: (
                    (res := idx.search(q, **search_params())).ids,
                    res.candidate_ids,
                ),
            ))
        if (
            compact_on_delete
            and op.operation in {"delete", "update"}
            and idx.should_compact(
                tombstone_fraction=compact_tombstone_fraction,
                min_tombstones=compact_min_tombstones,
                shrink=compact_on_shrink,
            )
        ):
            compact_index(op, "conditional_delete")
            compacted_this_step = True
        if compact_every and (op.step + 1) % compact_every == 0:
            if not compacted_this_step:
                compact_index(op, "periodic")
    return rows, [build_row]


def _run_etar_cpp_storage_stream(workload: StreamingWorkload, params: dict[str, Any], method_name: str = "etar") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(params.get("scan_dtype", "int8")) != "int8":
        raise NotImplementedError("C++ storage ETAR supports scan_dtype=int8 only")
    if str(params.get("scan_quantization", "row")) != "row":
        raise NotImplementedError("C++ storage ETAR supports row quantization only")
    if str(params.get("scan_layout", "column")) != "column":
        raise NotImplementedError("C++ storage ETAR supports column scan layout only")
    query_quantized = _query_quantized(params)
    rerank_query_quantized = _rerank_query_quantized(params)

    rss_before_build = rss_mb()
    t_build = perf_counter()
    idx = CppStorageIndex(
        workload.initial,
        initial_capacity_factor=float(params.get("scan_initial_capacity_factor", 1.25)),
        growth_factor=float(params.get("scan_growth_factor", 2.0)),
        store_residuals=_store_residuals(params),
        store_float32=_store_float32(params),
        store_codes_row=_store_codes_row(params),
    )
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_row = _build_record(
        workload,
        method_name,
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend="cpp",
        scan_dtype=idx.scan_dtype,
        scan_layout=idx.scan_layout,
        scan_quantization=idx.scan_quantization,
        scan_insert_policy=idx.scan_insert_policy,
        scan_capacity=int(idx.scan_capacity),
        scan_slack_fraction=float(idx.scan_slack_fraction),
        scan_growth_events=int(idx.scan_growth_events),
        scan_rebuild_events=int(idx.scan_rebuild_events),
        store_float32=bool(idx.store_float32),
        store_codes_row=bool(idx.store_codes_row),
        **idx.storage_profile,
    )
    vectors: dict[int, np.ndarray] = {int(i): workload.initial[i].copy() for i in range(workload.initial_n)}
    live = list(range(workload.initial_n))
    stream_pos = 0
    query_pos = 0
    rows: list[dict[str, Any]] = []
    compact_every = int(params.get("compact_every", 0) or 0)
    compact_on_delete = bool(params.get("compact_on_delete", params.get("compact_after_delete", False)))
    compact_tombstone_raw = params.get("compact_tombstone_fraction", None)
    compact_tombstone_fraction = None if compact_tombstone_raw is None else float(compact_tombstone_raw)
    compact_min_tombstones = int(params.get("compact_min_tombstones", 1))
    compact_on_shrink = bool(params.get("compact_on_shrink", True))
    mutation_count = 0
    checkpoint_id = 0
    omega_update_policy = str(params.get("omega_update_policy", workload.cfg.get("omega_update_policy", "lazy")))
    if omega_update_policy not in {"lazy", "exact_live"}:
        raise ValueError("omega_update_policy must be 'lazy' or 'exact_live'")
    omega_sum, omega_count = _initial_omega_state(workload.initial)

    method = str(params.get("method", "etar"))
    alpha_tail_fixed = None if params.get("alpha_tail_fixed") is None else float(params.get("alpha_tail_fixed"))
    opts = resolve_etar_options(
        method,
        head_policy=params.get("head_policy"),
        tail_policy=params.get("tail_policy"),
        rerank=params.get("rerank") if "rerank" in params else None,
        r_policy=params.get("r_policy"),
        query_mode=params.get("query_mode", "float"),
        alpha_tail_fixed=alpha_tail_fixed,
        alpha_q=0.0,
        block_prune=bool(params.get("block_prune", False)),
    )
    if opts.alpha_q != 0.0:
        raise NotImplementedError("C++ storage ETAR does not implement query quantization risk")

    def add_omega(x: np.ndarray) -> None:
        nonlocal omega_sum, omega_count
        contribution, count = _omega_contribution(x)
        if count:
            omega_sum += contribution
            omega_count += count

    def remove_omega(x: np.ndarray) -> None:
        nonlocal omega_sum, omega_count
        contribution, count = _omega_contribution(x)
        if count:
            omega_sum -= contribution
            np.maximum(omega_sum, 0.0, out=omega_sum)
            omega_count = max(0, omega_count - count)

    def with_scan_state(row: dict[str, Any]) -> dict[str, Any]:
        row.update({
            "storage_backend": "cpp",
            "rss_before_build_mb": rss_before_build,
            "rss_after_build_mb": rss_after_build,
            "rss_build_delta_mb": rss_after_build - rss_before_build if not (np.isnan(rss_before_build) or np.isnan(rss_after_build)) else np.nan,
            "scan_dtype": idx.scan_dtype,
            "scan_layout": idx.scan_layout,
            "scan_quantization": idx.scan_quantization,
            "scan_insert_policy": idx.scan_insert_policy,
            "scan_stride": int(idx.scan_stride),
            "scan_capacity": int(idx.scan_capacity),
            "scan_slack_fraction": float(idx.scan_slack_fraction),
            "scan_growth_events": int(idx.scan_growth_events),
            "scan_rebuild_events": int(idx.scan_rebuild_events),
            "store_float32": bool(idx.store_float32),
            "store_codes_row": bool(idx.store_codes_row),
            "storage_preset": _storage_preset(params) or ("8+32" if idx.store_float32 else ("8+8" if idx.store_codes_row else "8-shared")),
            "rerank_dtype": idx.rerank_dtype,
            "rerank_layout": idx.rerank_layout,
            "query_mode": "int8" if query_quantized else "float",
            "query_quantization": "int8" if query_quantized else "none",
            "rerank_query_mode": "int8" if rerank_query_quantized else "float",
            "rerank_query_quantization": "int8" if rerank_query_quantized else "none",
            **idx.storage_profile,
        })
        return row

    def search_cpp(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        head, retained, tail_norm_value, h = select_head_policy(
            q,
            float(params.get("rho", 0.9)),
            int(params.get("h_max", 64)),
            opts.head_policy,
        )
        r = int(params.get("r", params.get("R", 100)))
        if opts.force_r_to_k:
            r = workload.k
        omega = _omega_value(omega_sum, omega_count)
        r_query = r
        if opts.r_policy == "adaptive":
            r_query = min(
                idx.live_count,
                adaptive_r_budget(
                    q,
                    head,
                    omega,
                    idx.n,
                    workload.k,
                    r,
                    target_recall=float(params.get("adaptive_r_target_recall", params.get("target_recall", 0.95))),
                    probability=float(params.get("adaptive_r_probability", params.get("probability", 0.95))),
                    mode=str(params.get("adaptive_r_mode", "event")),
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
                    float(params.get("lambda_tail", 0.75)),
                    float(params.get("alpha_min", 0.05)),
                    float(params.get("alpha_max", 0.5)),
                )
        prune_code = 0
        if bool(params.get("prune_lower_bound", False)):
            prune_code = 2 if str(params.get("prune_lower_mode", "global")) in {"candidate", "candidates", "topr"} else 1
        ids, scores, cand_ids, _, timings = idx.search_profile(
            q,
            head,
            k=workload.k,
            r=r_query,
            use_tail=use_tail,
            alpha_tail=alpha_tail,
            tail_norm=tail_norm_value,
            no_rerank=not opts.rerank,
            prune_lower_bound=prune_code,
            prune_lower_scale=float(params.get("prune_lower_scale", 1.0)),
            block_prune=bool(opts.block_prune),
            block_prune_scout_h=int(params.get("block_prune_scout_h", 16)),
            block_prune_beam_factor=float(params.get("block_prune_beam_factor", 32.0)),
            block_prune_gate=float(params.get("block_prune_gate", 0.75)),
            block_prune_block_size=int(params.get("block_prune_block_size", 256)),
            sort_candidates=bool(params.get("cold_rerank_sort_candidates", False)),
            quantize_query=query_quantized,
            rerank_quantize_query=rerank_query_quantized,
        )
        diag = {
            "mean_h": int(h),
            "mean_retained_mass": float(retained),
            "R": int(r_query),
            "native_scan_ms": timings.get("native_scan_ms", np.nan),
            "native_top_r_ms": timings.get("native_top_r_ms", np.nan),
            "native_rerank_ms": timings.get("native_rerank_ms", np.nan),
            "native_total_ms": timings.get("native_total_ms", np.nan),
        }
        return ids, cand_ids, diag

    def compact_index(op: StreamOp, reason: str) -> None:
        nonlocal checkpoint_id
        t0 = perf_counter()
        idx.compact()
        ms = (perf_counter() - t0) * 1000
        compact_op = StreamOp(op.step, "compact")
        row = with_scan_state(_event(workload, method_name, compact_op, ms, live, idx.n))
        row["compact_reason"] = reason
        rows.append(row)
        if should_eval_checkpoint(compact_op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(with_scan_state(evaluate_checkpoint(
                workload,
                method_name,
                compact_op,
                checkpoint_id,
                mutation_count,
                live,
                idx.n,
                vectors,
                lambda q: search_cpp(q)[:2],
            )))

    for op in workload.ops:
        recall = None
        compacted_this_step = False
        extra: dict[str, Any] = {}
        if op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            ids, _, diag = search_cpp(q)
            ms = (perf_counter() - t0) * 1000
            extra.update(diag)
            if op.step % workload.recall_every == 0 and live:
                truth = _truth_for_queries(vectors, live, q[None, :], workload.k)
                recall = recall_at_k(ids[None, :], truth, min(workload.k, truth.shape[1]))
        elif op.operation == "insert":
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            t0 = perf_counter()
            label = idx.insert(x)
            ms = (perf_counter() - t0) * 1000
            live.append(label)
            vectors[label] = x.copy()
            add_omega(x)
            mutation_count += 1
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            old = vectors.get(int(victim))
            t0 = perf_counter()
            idx.delete(victim)
            ms = (perf_counter() - t0) * 1000
            if old is not None and omega_update_policy == "exact_live":
                remove_omega(old)
            vectors.pop(int(victim), None)
            mutation_count += 1
        elif op.operation == "update" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            old = vectors.get(int(victim))
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            t0 = perf_counter()
            label = idx.update(victim, x)
            ms = (perf_counter() - t0) * 1000
            if old is not None and omega_update_policy == "exact_live":
                remove_omega(old)
            vectors.pop(int(victim), None)
            live.append(label)
            vectors[label] = x.copy()
            add_omega(x)
            mutation_count += 1
        else:
            ms = 0.0
        row = with_scan_state(_event(workload, method_name, op, ms, live, idx.n, recall))
        row.update(extra)
        rows.append(row)
        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(with_scan_state(evaluate_checkpoint(
                workload,
                method_name,
                op,
                checkpoint_id,
                mutation_count,
                live,
                idx.n,
                vectors,
                lambda q: search_cpp(q)[:2],
            )))
        if (
            compact_on_delete
            and op.operation in {"delete", "update"}
            and idx.should_compact(
                tombstone_fraction=compact_tombstone_fraction,
                min_tombstones=compact_min_tombstones,
                shrink=compact_on_shrink,
            )
        ):
            compact_index(op, "conditional_delete")
            compacted_this_step = True
        if compact_every and (op.step + 1) % compact_every == 0:
            if not compacted_this_step:
                compact_index(op, "periodic")
    idx.close()
    return rows, [build_row]


def _run_hnswlib_stream(workload: StreamingWorkload, params: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import hnswlib

    params = params or {}
    rss_before_build = rss_mb()
    t_build = perf_counter()
    index = hnswlib.Index(space="ip", dim=workload.d)
    max_elements = workload.initial_n + workload.steps + 32
    m = int(params.get("M", 16))
    ef_construction = int(params.get("ef_construction", 100))
    ef_search = int(params.get("ef_search", params.get("ef", 64)))
    index.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=m,
        allow_replace_deleted=True,
    )
    labels = np.arange(workload.initial_n)
    index.add_items(workload.initial, labels, num_threads=1)
    index.set_ef(ef_search)
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_row = _build_record(
        workload,
        "hnswlib",
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend="hnswlib",
        M=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        max_elements=max_elements,
    )
    vectors = {int(i): workload.initial[i].copy() for i in labels}
    live = list(map(int, labels))
    next_id = workload.initial_n
    rows = []
    stream_pos = query_pos = 0
    mutation_count = 0
    checkpoint_id = 0
    for op in workload.ops:
        recall = None
        if op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            ids, _ = index.knn_query(q[None, :], k=min(workload.k, len(live)), num_threads=1)
            ms = (perf_counter() - t0) * 1000
            if op.step % workload.recall_every == 0 and live:
                mat = np.vstack([vectors[i] for i in live])
                truth_labels = np.asarray(live)[np.argsort(-(mat @ q), kind="stable")[: min(workload.k, len(live))]][None, :]
                recall = recall_at_k(ids, truth_labels, min(workload.k, truth_labels.shape[1]))
        elif op.operation == "insert":
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            index.add_items(x[None, :], np.array([label]), num_threads=1)
            ms = (perf_counter() - t0) * 1000
            live.append(label)
            vectors[label] = x.copy()
            mutation_count += 1
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            t0 = perf_counter()
            index.mark_deleted(victim)
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            mutation_count += 1
        elif op.operation == "update" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            index.mark_deleted(victim)
            index.add_items(x[None, :], np.array([label]), num_threads=1)
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            live.append(label)
            vectors[label] = x.copy()
            mutation_count += 1
        else:
            ms = 0.0
        rows.append(_event(workload, "hnswlib", op, ms, live, next_id, recall))
        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, "hnswlib", op, checkpoint_id, mutation_count, live, next_id, vectors,
                lambda q: index.knn_query(q[None, :], k=min(workload.k, len(live)), num_threads=1)[0][0],
            ))
    return rows, [build_row]


def _run_faiss_flat_stream(workload: StreamingWorkload) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import faiss

    configure_faiss_threads()
    rss_before_build = rss_mb()
    t_build = perf_counter()
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(workload.d))
    labels = np.arange(workload.initial_n, dtype=np.int64)
    index.add_with_ids(workload.initial, labels)
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_row = _build_record(
        workload,
        "faiss_flat",
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend="faiss:flat",
        index_ntotal=int(index.ntotal),
    )
    vectors = {int(i): workload.initial[i].copy() for i in labels}
    live = list(map(int, labels))
    next_id = workload.initial_n
    rows = []
    stream_pos = query_pos = 0
    mutation_count = 0
    checkpoint_id = 0
    for op in workload.ops:
        recall = None
        if op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            _, ids = index.search(q[None, :], min(workload.k, len(live)))
            ms = (perf_counter() - t0) * 1000
            if op.step % workload.recall_every == 0 and live:
                mat = np.vstack([vectors[i] for i in live])
                truth_labels = np.asarray(live)[np.argsort(-(mat @ q), kind="stable")[: min(workload.k, len(live))]][None, :]
                recall = recall_at_k(ids, truth_labels, min(workload.k, truth_labels.shape[1]))
        elif op.operation == "insert":
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            index.add_with_ids(x[None, :], np.array([label], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            live.append(label)
            vectors[label] = x.copy()
            mutation_count += 1
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            t0 = perf_counter()
            index.remove_ids(np.array([victim], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            mutation_count += 1
        elif op.operation == "update" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            index.remove_ids(np.array([victim], dtype=np.int64))
            index.add_with_ids(x[None, :], np.array([label], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            live.append(label)
            vectors[label] = x.copy()
            mutation_count += 1
        else:
            ms = 0.0
        rows.append(_event(workload, "faiss_flat", op, ms, live, next_id, recall))
        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, "faiss_flat", op, checkpoint_id, mutation_count, live, next_id, vectors,
                lambda q: index.search(q[None, :], min(workload.k, len(live)))[1][0],
            ))
    return rows, [build_row]


def _run_faiss_hnsw_stream(workload: StreamingWorkload, params: dict[str, Any], rebuild_every: int = 0, rebuild_unit: str = "steps") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import faiss

    configure_faiss_threads()
    m = int(params.get("M", params.get("m", 32)))
    ef_search = int(params.get("ef_search", params.get("efSearch", 64)))
    overfetch = max(1, int(params.get("overfetch", 10)))

    def build_index(vectors: dict[int, np.ndarray], live_ids: list[int]):
        base = faiss.IndexHNSWFlat(workload.d, m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efSearch = ef_search
        index = faiss.IndexIDMap2(base)
        if live_ids:
            labels = np.asarray(live_ids, dtype=np.int64)
            mat = np.vstack([vectors[int(i)] for i in labels]).astype(np.float32)
            index.add_with_ids(mat, labels)
        return index

    vectors = {int(i): workload.initial[i].copy() for i in range(workload.initial_n)}
    live = list(range(workload.initial_n))
    live_set = set(live)
    rss_before_build = rss_mb()
    t_build = perf_counter()
    index = build_index(vectors, live)
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_row = _build_record(
        workload,
        "faiss_hnsw",
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend="faiss:hnsw",
        M=m,
        ef_search=ef_search,
        overfetch=overfetch,
        index_ntotal=int(index.ntotal),
    )
    next_id = workload.initial_n
    rows = []
    stream_pos = query_pos = 0
    mutation_count = 0
    checkpoint_id = 0
    last_rebuild_mutation = 0

    def query_live(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = min(workload.k, len(live))
        if target <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        stale_count = max(0, int(index.ntotal) - len(live))
        fetch = min(int(index.ntotal), max(target * overfetch, target + stale_count))
        _, raw_ids = index.search(q[None, :], fetch)
        candidates = [int(i) for i in raw_ids[0] if int(i) >= 0 and int(i) in live_set]
        # Preserve FAISS HNSW order after filtering stale labels.
        seen = set()
        filtered = []
        for label in candidates:
            if label not in seen:
                seen.add(label)
                filtered.append(label)
            if len(filtered) >= target:
                break
        ids = np.full(target, -1, dtype=np.int64)
        ids[: len(filtered)] = np.asarray(filtered, dtype=np.int64)
        return ids, np.asarray(candidates, dtype=np.int64)

    for op in workload.ops:
        recall = None
        if op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            ids, _ = query_live(q)
            ms = (perf_counter() - t0) * 1000
            if op.step % workload.recall_every == 0 and live:
                mat = np.vstack([vectors[i] for i in live])
                truth_labels = np.asarray(live)[np.argsort(-(mat @ q), kind="stable")[: min(workload.k, len(live))]][None, :]
                recall = recall_at_k(ids[None, :], truth_labels, min(workload.k, truth_labels.shape[1]))
            rows.append(_event(workload, "faiss_hnsw", op, ms, live, next_id, recall, status="tombstone_filter"))
        elif op.operation == "insert":
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            index.add_with_ids(x[None, :], np.array([label], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            live.append(label)
            live_set.add(label)
            vectors[label] = x.copy()
            mutation_count += 1
            rows.append(_event(workload, "faiss_hnsw", op, ms, live, next_id, status="native_insert"))
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            t0 = perf_counter()
            live_set.discard(int(victim))
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            mutation_count += 1
            rows.append(_event(workload, "faiss_hnsw", op, ms, live, next_id, status="external_tombstone"))
        elif op.operation == "update" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            live_set.discard(int(victim))
            index.add_with_ids(x[None, :], np.array([label], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            live.append(label)
            live_set.add(label)
            vectors[label] = x.copy()
            mutation_count += 1
            rows.append(_event(workload, "faiss_hnsw", op, ms, live, next_id, status="tombstone_filter_insert"))
        else:
            rows.append(_event(workload, "faiss_hnsw", op, 0.0, live, next_id, status="skipped"))

        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, "faiss_hnsw", op, checkpoint_id, mutation_count, live, next_id, vectors,
                query_live,
                status="tombstone_filter",
            ))
        if _should_periodic_rebuild(op, mutation_count, last_rebuild_mutation, rebuild_every, rebuild_unit):
            t0 = perf_counter()
            index = build_index(vectors, live)
            ms = (perf_counter() - t0) * 1000
            last_rebuild_mutation = mutation_count
            rebuild_op = StreamOp(op.step, "rebuild")
            rows.append(_event(workload, "faiss_hnsw", rebuild_op, ms, live, next_id, status="rebuild"))
            if should_eval_checkpoint(rebuild_op, mutation_count, workload.cfg):
                checkpoint_id += 1
                rows.append(evaluate_checkpoint(
                    workload, "faiss_hnsw", rebuild_op, checkpoint_id, mutation_count, live, next_id, vectors,
                    query_live,
                    status="rebuild",
                ))
    return rows, [build_row]


def _run_faiss_inverted_stream(workload: StreamingWorkload, method: str, params: dict[str, Any], rebuild_every: int = 0, rebuild_unit: str = "steps") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import faiss

    configure_faiss_threads()
    nlist = int(params.get("nlist", 128))
    nprobe = int(params.get("nprobe", 16))
    pq_m = int(params.get("m", max(1, min(16, workload.d // 4))))
    overfetch = max(1, int(params.get("overfetch", 10)))

    def valid_pq_m(d: int, requested: int) -> int:
        requested = max(1, min(int(requested), int(d)))
        if d % requested == 0:
            return requested
        for candidate in range(requested - 1, 0, -1):
            if d % candidate == 0:
                return candidate
        return 1

    def build_index(vectors: dict[int, np.ndarray], live_ids: list[int]):
        labels = np.asarray(live_ids, dtype=np.int64)
        mat = np.vstack([vectors[int(i)] for i in labels]).astype(np.float32) if live_ids else np.empty((0, workload.d), dtype=np.float32)
        local_nlist = min(max(1, nlist), max(1, mat.shape[0]))
        if method == "faiss_ivf":
            base = faiss.IndexIVFFlat(faiss.IndexFlatIP(workload.d), workload.d, local_nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            base = faiss.IndexIVFPQ(
                faiss.IndexFlatIP(workload.d),
                workload.d,
                local_nlist,
                valid_pq_m(workload.d, pq_m),
                8,
                faiss.METRIC_INNER_PRODUCT,
            )
        if mat.shape[0] > 0:
            base.train(mat)
        base.nprobe = min(max(1, nprobe), base.nlist)
        index = faiss.IndexIDMap2(base)
        if mat.shape[0] > 0:
            index.add_with_ids(mat, labels)
        return index

    vectors = {int(i): workload.initial[i].copy() for i in range(workload.initial_n)}
    live = list(range(workload.initial_n))
    live_set = set(live)
    rss_before_build = rss_mb()
    t_build = perf_counter()
    index = build_index(vectors, live)
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_row = _build_record(
        workload,
        method,
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend=f"faiss:{method.replace('faiss_', '')}",
        nlist=nlist,
        nprobe=nprobe,
        overfetch=overfetch,
        index_ntotal=int(index.ntotal),
    )
    next_id = workload.initial_n
    rows = []
    stream_pos = query_pos = 0
    mutation_count = 0
    checkpoint_id = 0
    last_rebuild_mutation = 0

    def query_index(q: np.ndarray) -> np.ndarray:
        if not live:
            return np.empty(0, dtype=np.int64)
        target = min(workload.k, len(live))
        stale_count = max(0, int(index.ntotal) - len(live))
        fetch = min(int(index.ntotal), max(target * overfetch, target + stale_count))
        _, raw_ids = index.search(q[None, :], fetch)
        seen = set()
        filtered = []
        for label in raw_ids[0]:
            label = int(label)
            if label >= 0 and label in live_set and label not in seen:
                seen.add(label)
                filtered.append(label)
            if len(filtered) >= target:
                break
        ids = np.full(target, -1, dtype=np.int64)
        ids[: len(filtered)] = np.asarray(filtered, dtype=np.int64)
        return ids

    for op in workload.ops:
        recall = None
        if op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            ids = query_index(q)
            ms = (perf_counter() - t0) * 1000
            if op.step % workload.recall_every == 0 and live:
                mat = np.vstack([vectors[i] for i in live])
                truth_labels = np.asarray(live)[np.argsort(-(mat @ q), kind="stable")[: min(workload.k, len(live))]][None, :]
                recall = recall_at_k(ids[None, :], truth_labels, min(workload.k, truth_labels.shape[1]))
            rows.append(_event(workload, method, op, ms, live, next_id, recall))
        elif op.operation == "insert":
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            index.add_with_ids(x[None, :], np.array([label], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            live.append(label)
            live_set.add(label)
            vectors[label] = x.copy()
            mutation_count += 1
            rows.append(_event(workload, method, op, ms, live, next_id, status="native_insert"))
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            t0 = perf_counter()
            live_set.discard(int(victim))
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            mutation_count += 1
            rows.append(_event(workload, method, op, ms, live, next_id, status="external_tombstone"))
        elif op.operation == "update" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            x = workload.stream[stream_pos % workload.stream.shape[0]]
            stream_pos += 1
            label = next_id
            next_id += 1
            t0 = perf_counter()
            live_set.discard(int(victim))
            index.add_with_ids(x[None, :], np.array([label], dtype=np.int64))
            ms = (perf_counter() - t0) * 1000
            vectors.pop(victim, None)
            live.append(label)
            live_set.add(label)
            vectors[label] = x.copy()
            mutation_count += 1
            rows.append(_event(workload, method, op, ms, live, next_id, status="tombstone_filter_insert"))
        else:
            rows.append(_event(workload, method, op, 0.0, live, next_id, status="skipped"))

        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, method, op, checkpoint_id, mutation_count, live, next_id, vectors,
                query_index,
            ))
        if _should_periodic_rebuild(op, mutation_count, last_rebuild_mutation, rebuild_every, rebuild_unit):
            t0 = perf_counter()
            index = build_index(vectors, live)
            ms = (perf_counter() - t0) * 1000
            last_rebuild_mutation = mutation_count
            rebuild_op = StreamOp(op.step, "rebuild")
            rows.append(_event(workload, method, rebuild_op, ms, live, next_id, status="retrain_rebuild"))
            if should_eval_checkpoint(rebuild_op, mutation_count, workload.cfg):
                checkpoint_id += 1
                rows.append(evaluate_checkpoint(
                    workload, method, rebuild_op, checkpoint_id, mutation_count, live, next_id, vectors,
                    query_index,
                    status="retrain_rebuild",
                ))
    return rows, [build_row]


def _run_rebuild_static_stream(workload: StreamingWorkload, method: str, rebuild_every: int = 100, rebuild_unit: str = "steps") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rss_before_build = rss_mb()
    t_build = perf_counter()
    vectors = {i: workload.initial[i].copy() for i in range(workload.initial_n)}
    live = list(range(workload.initial_n))
    indexed_live = list(live)
    indexed_vectors = {i: vectors[i].copy() for i in indexed_live}
    if indexed_live:
        _ = np.vstack([indexed_vectors[i] for i in indexed_live])
    build_ms = (perf_counter() - t_build) * 1000
    rss_after_build = rss_mb()
    build_row = _build_record(
        workload,
        method,
        build_ms,
        rss_before_build,
        rss_after_build,
        storage_backend="static_rebuild_proxy",
        rebuild_every=rebuild_every,
        rebuild_unit=rebuild_unit,
    )
    rows = []
    stream_pos = query_pos = 0
    next_id = workload.initial_n
    mutation_count = 0
    checkpoint_id = 0
    last_rebuild_mutation = 0

    def query_snapshot(q: np.ndarray) -> np.ndarray:
        if not indexed_live:
            return np.empty(0, dtype=np.int64)
        mat = _live_matrix(indexed_vectors, indexed_live)[0]
        return np.asarray(indexed_live, dtype=np.int64)[np.argsort(-(mat @ q), kind="stable")[: min(workload.k, len(indexed_live))]]

    for op in workload.ops:
        if op.operation == "insert":
            vectors[next_id] = workload.stream[stream_pos % workload.stream.shape[0]].copy()
            stream_pos += 1
            live.append(next_id)
            next_id += 1
            rows.append(_event(workload, method, op, 0.0, live, next_id, status="metadata_only"))
            mutation_count += 1
        elif op.operation == "delete" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            vectors.pop(victim, None)
            rows.append(_event(workload, method, op, 0.0, live, next_id, status="metadata_only"))
            mutation_count += 1
        elif op.operation == "update" and live:
            victim = _pop_victim(live, vectors, workload.victim_policy, workload.rng)
            vectors.pop(victim, None)
            vectors[next_id] = workload.stream[stream_pos % workload.stream.shape[0]].copy()
            stream_pos += 1
            live.append(next_id)
            next_id += 1
            rows.append(_event(workload, method, op, 0.0, live, next_id, status="metadata_only"))
            mutation_count += 1
        elif op.operation == "query":
            q = workload.queries[query_pos % workload.queries.shape[0]]
            query_pos += 1
            t0 = perf_counter()
            _ = query_snapshot(q)
            ms = (perf_counter() - t0) * 1000
            rows.append(_event(workload, method, op, ms, live, next_id, status="static_or_rebuild_proxy"))
        if should_eval_checkpoint(op, mutation_count, workload.cfg):
            checkpoint_id += 1
            rows.append(evaluate_checkpoint(
                workload, method, op, checkpoint_id, mutation_count, live, next_id, vectors,
                query_snapshot,
                status="static_or_rebuild_proxy",
            ))
        if _should_periodic_rebuild(op, mutation_count, last_rebuild_mutation, rebuild_every, rebuild_unit):
            t0 = perf_counter()
            indexed_live = list(live)
            indexed_vectors = {i: vectors[i].copy() for i in indexed_live}
            if indexed_live:
                _ = np.vstack([indexed_vectors[i] for i in indexed_live])
            ms = (perf_counter() - t0) * 1000
            last_rebuild_mutation = mutation_count
            rebuild_op = StreamOp(op.step, "rebuild")
            rows.append(_event(workload, method, rebuild_op, ms, live, next_id, status="rebuild_proxy"))
            if should_eval_checkpoint(rebuild_op, mutation_count, workload.cfg):
                checkpoint_id += 1
                rows.append(evaluate_checkpoint(
                    workload, method, rebuild_op, checkpoint_id, mutation_count, live, next_id, vectors,
                    query_snapshot,
                    status="rebuild_proxy",
                ))
    return rows, [build_row]


def run_streaming_dynamic_with_build(dynamic_cfg: dict[str, Any] | None, progress_fn=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_single_thread_env()
    if not dynamic_cfg or not dynamic_cfg.get("enabled", False):
        return pd.DataFrame(), pd.DataFrame()
    workloads = dynamic_cfg.get("workloads", [])
    frames = []
    build_frames = []
    for i, workload in enumerate(workloads, start=1):
        dynamic, build = run_dynamic_workload_with_build(
            workload,
            progress_fn=progress_fn,
            workload_index=i,
            workload_total=len(workloads),
        )
        if not dynamic.empty:
            frames.append(dynamic)
        if not build.empty:
            build_frames.append(build)
    dynamic = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    build = pd.concat(build_frames, ignore_index=True) if build_frames else pd.DataFrame()
    return dynamic, build


def run_streaming_dynamic(dynamic_cfg: dict[str, Any] | None, progress_fn=None) -> pd.DataFrame:
    dynamic, _build = run_streaming_dynamic_with_build(dynamic_cfg, progress_fn=progress_fn)
    return dynamic
