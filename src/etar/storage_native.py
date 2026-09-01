from __future__ import annotations

import ctypes
import sysconfig
from pathlib import Path

import numpy as np

from .profiling import etar_storage_bytes


def _native_path() -> Path | None:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    here = Path(__file__).resolve().parent
    matches = list(here.glob("_etar_storage_cpp*.so")) + list(here.glob(f"_etar_storage_cpp*{suffix}"))
    return matches[0] if matches else None


class CppStorageIndex:
    @staticmethod
    def available() -> bool:
        return _native_path() is not None

    def __init__(
        self,
        X: np.ndarray,
        *,
        initial_capacity_factor: float = 1.25,
        growth_factor: float = 2.0,
        store_residuals: bool = False,
        store_float32: bool = True,
        store_codes_row: bool = False,
    ) -> None:
        path = _native_path()
        if path is None:
            raise RuntimeError("native C++ storage library not found")
        self.lib = ctypes.CDLL(str(path))
        self._configure()
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D float32 array")
        self.d = int(X.shape[1])
        self.scan_dtype = "int8"
        self.scan_layout = "column"
        self.scan_quantization = "row"
        self.scan_insert_policy = "cpp_storage"
        self.residuals_enabled = bool(store_residuals)
        self.store_float32 = bool(store_float32)
        self.store_codes_row = bool(store_codes_row)
        self.rerank_dtype = "float32" if self.store_float32 else "int8"
        self.rerank_layout = "float32" if self.store_float32 else ("row_int8" if self.store_codes_row else "column_int8")
        self.row_id_bytes = 4
        self.scan_growth_events = 0
        self.scan_rebuild_events = 0
        self.compaction_events = 0
        self.handle = self.lib.etar_storage_create(
            X.ctypes.data_as(_fptr),
            int(X.shape[0]),
            int(X.shape[1]),
            float(initial_capacity_factor),
            float(growth_factor),
            int(self.residuals_enabled),
            int(self.store_float32),
            int(self.store_codes_row),
        )
        if not self.handle:
            raise RuntimeError("failed to create native C++ storage index")

    def _configure(self) -> None:
        self.lib.etar_storage_create.restype = ctypes.c_void_p
        self.lib.etar_storage_create.argtypes = [
            _fptr,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.lib.etar_storage_free.restype = None
        self.lib.etar_storage_free.argtypes = [ctypes.c_void_p]
        self.lib.etar_storage_n.restype = ctypes.c_int64
        self.lib.etar_storage_n.argtypes = [ctypes.c_void_p]
        self.lib.etar_storage_live_count.restype = ctypes.c_int64
        self.lib.etar_storage_live_count.argtypes = [ctypes.c_void_p]
        self.lib.etar_storage_capacity.restype = ctypes.c_int64
        self.lib.etar_storage_capacity.argtypes = [ctypes.c_void_p]
        self.lib.etar_storage_insert.restype = ctypes.c_int64
        self.lib.etar_storage_insert.argtypes = [ctypes.c_void_p, _fptr]
        self.lib.etar_storage_delete.restype = None
        self.lib.etar_storage_delete.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.etar_storage_update.restype = ctypes.c_int64
        self.lib.etar_storage_update.argtypes = [ctypes.c_void_p, ctypes.c_int64, _fptr]
        self.lib.etar_storage_compact.restype = None
        self.lib.etar_storage_compact.argtypes = [ctypes.c_void_p]
        self.lib.etar_storage_search_profile.restype = _Status
        self.lib.etar_storage_search_profile.argtypes = [
            ctypes.c_void_p,
            _fptr,
            _i64ptr,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            _i64ptr,
            _fptr,
            _i64ptr,
            _fptr,
            ctypes.POINTER(_Profile),
        ]

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.etar_storage_free(handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def n(self) -> int:
        return int(self.lib.etar_storage_n(self.handle))

    @property
    def live_count(self) -> int:
        return int(self.lib.etar_storage_live_count(self.handle))

    @property
    def capacity(self) -> int:
        return int(self.lib.etar_storage_capacity(self.handle))

    @property
    def scan_stride(self) -> int:
        return self.capacity

    @property
    def scan_capacity(self) -> int:
        return self.capacity

    @property
    def scan_slack_fraction(self) -> float:
        return float(1.0 - (self.n / max(self.capacity, 1)))

    @property
    def tombstone_count(self) -> int:
        return max(0, self.n - self.live_count)

    @property
    def tombstone_fraction(self) -> float:
        return float(self.tombstone_count / max(self.n, 1))

    @property
    def storage_profile(self) -> dict[str, int]:
        return etar_storage_bytes(
            n=self.n,
            d=self.d,
            capacity=self.capacity,
            scan_dtype="int8",
            residuals=self.residuals_enabled,
            row_id_bytes=self.row_id_bytes,
            store_float32=self.store_float32,
            store_codes_row=self.store_codes_row,
        )

    def should_compact(
        self,
        *,
        tombstone_fraction: float | None = None,
        min_tombstones: int = 1,
        shrink: bool = True,
    ) -> bool:
        del shrink
        if self.tombstone_count < int(min_tombstones):
            return False
        if tombstone_fraction is not None and float(tombstone_fraction) >= 0.0:
            return self.tombstone_fraction >= float(tombstone_fraction)
        return self.tombstone_count > 0

    def insert(self, x: np.ndarray) -> int:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},)")
        before = self.capacity
        row_id = int(self.lib.etar_storage_insert(self.handle, x.ctypes.data_as(_fptr)))
        if self.capacity > before:
            self.scan_growth_events += 1
        return row_id

    def delete(self, row_id: int) -> None:
        self.lib.etar_storage_delete(self.handle, int(row_id))

    def update(self, row_id: int, x: np.ndarray) -> int:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},)")
        before = self.capacity
        new_id = int(self.lib.etar_storage_update(self.handle, int(row_id), x.ctypes.data_as(_fptr)))
        if self.capacity > before:
            self.scan_growth_events += 1
        return new_id

    def compact(self) -> None:
        self.compaction_events += 1
        self.lib.etar_storage_compact(self.handle)

    def search_profile(
        self,
        q: np.ndarray,
        head: np.ndarray,
        *,
        k: int,
        r: int,
        use_tail: bool,
        alpha_tail: float,
        tail_norm: float,
        no_rerank: bool = False,
        prune_lower_bound: int = 0,
        prune_lower_scale: float = 1.0,
        block_prune: bool = False,
        block_prune_scout_h: int = 16,
        block_prune_beam_factor: float = 32.0,
        block_prune_gate: float = 0.75,
        block_prune_block_size: int = 256,
        sort_candidates: bool = False,
        quantize_query: bool = False,
        rerank_quantize_query: bool | None = None,
    ):
        if rerank_quantize_query is None:
            rerank_quantize_query = quantize_query
        q = np.ascontiguousarray(q, dtype=np.float32)
        head = np.ascontiguousarray(head, dtype=np.int64)
        k = min(int(k), self.live_count)
        r = min(max(int(r), k), self.live_count)
        out_ids = np.full(k, -1, dtype=np.int64)
        out_scores = np.full(k, -np.inf, dtype=np.float32)
        cand_ids = np.full(r, -1, dtype=np.int64)
        cand_scores = np.full(r, -np.inf, dtype=np.float32)
        profile = _Profile()
        status = self.lib.etar_storage_search_profile(
            self.handle,
            q.ctypes.data_as(_fptr),
            head.ctypes.data_as(_i64ptr),
            int(head.size),
            k,
            r,
            int(use_tail),
            float(alpha_tail),
            float(tail_norm),
            int(no_rerank),
            int(prune_lower_bound),
            float(prune_lower_scale),
            int(block_prune),
            int(block_prune_scout_h),
            float(block_prune_beam_factor),
            float(block_prune_gate),
            int(block_prune_block_size),
            int(sort_candidates),
            int(quantize_query),
            int(rerank_quantize_query),
            out_ids.ctypes.data_as(_i64ptr),
            out_scores.ctypes.data_as(_fptr),
            cand_ids.ctypes.data_as(_i64ptr),
            cand_scores.ctypes.data_as(_fptr),
            ctypes.byref(profile),
        )
        timings = {
            "native_scan_ms": float(profile.scan_ms),
            "native_top_r_ms": float(profile.top_r_ms),
            "native_rerank_ms": float(profile.rerank_ms),
            "native_total_ms": float(profile.total_native_ms),
            "native_block_prune_survival": float(profile.block_prune_survival),
            "native_block_prune_used": float(profile.block_prune_used),
            "native_block_prune_fallback": float(profile.block_prune_fallback),
        }
        return (
            out_ids[: status.results],
            out_scores[: status.results],
            cand_ids[: status.candidates],
            cand_scores[: status.candidates],
            timings,
        )


class _Status(ctypes.Structure):
    _fields_ = [("candidates", ctypes.c_int32), ("results", ctypes.c_int32)]


class _Profile(ctypes.Structure):
    _fields_ = [
        ("scan_ms", ctypes.c_float),
        ("top_r_ms", ctypes.c_float),
        ("rerank_ms", ctypes.c_float),
        ("total_native_ms", ctypes.c_float),
        ("block_prune_survival", ctypes.c_float),
        ("block_prune_used", ctypes.c_float),
        ("block_prune_fallback", ctypes.c_float),
    ]


_fptr = ctypes.POINTER(ctypes.c_float)
_i64ptr = ctypes.POINTER(ctypes.c_int64)
