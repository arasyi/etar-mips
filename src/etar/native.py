from __future__ import annotations

import ctypes
import sysconfig
from pathlib import Path

import numpy as np


def _native_path() -> Path | None:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    here = Path(__file__).resolve().parent
    matches = list(here.glob("_etar_cpp*.so")) + list(here.glob(f"_etar_cpp*{suffix}"))
    return matches[0] if matches else None


class CppBackend:
    @staticmethod
    def available() -> bool:
        return _native_path() is not None

    def __init__(self) -> None:
        path = _native_path()
        if path is None:
            raise RuntimeError("native C++ library not found")
        self.lib = ctypes.CDLL(str(path))
        self.lib.etar_search.restype = _Status
        self.lib.etar_search.argtypes = [
            _fptr, _fptr, _i8ptr, _fptr, _fptr, _fptr, _fptr, _u8ptr, _fptr, _i64ptr,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
            ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_float, ctypes.c_int64,
            ctypes.c_int32,
            _i64ptr, _fptr, _i64ptr, _fptr,
        ]
        self.lib.etar_search_profile.restype = _Status
        self.lib.etar_search_profile.argtypes = [
            _fptr, _fptr, _i8ptr, _fptr, _fptr, _fptr, _fptr, _u8ptr, _fptr, _i64ptr,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
            ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_float, ctypes.c_int64,
            ctypes.c_int32,
            _i64ptr, _fptr, _i64ptr, _fptr, ctypes.POINTER(_Profile),
        ]
        self.lib.etar_search_batch.restype = None
        self.lib.etar_search_batch.argtypes = [
            _fptr, _fptr, _i8ptr, _fptr, _fptr, _fptr, _fptr, _u8ptr, _fptr, _i64ptr, _i64ptr,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            _i32ptr, _fptr, _fptr, _fptr, _i32ptr, _i32ptr, _i32ptr, _fptr,
            ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_float, ctypes.c_int64,
            ctypes.c_int32,
            _i64ptr, _fptr, _i64ptr, _fptr, ctypes.c_int32, ctypes.POINTER(_Profile),
        ]

    def search(
        self,
        X,
        X_scan,
        C_col,
        scales,
        scan_scales,
        norms,
        residuals,
        alive,
        q,
        head,
        scan_stride,
        k,
        r,
        use_tail,
        alpha_tail,
        tail_norm,
        alpha_q,
        int8_query,
        scan_mode=0,
        no_rerank=False,
        prune_lower_bound=False,
        prune_lower_scale=1.0,
        block_prune=False,
        block_prune_scout_h=16,
        block_prune_beam_factor=32.0,
        block_prune_gate=0.75,
        block_prune_block_size=256,
        sort_rerank_candidates=False,
    ):
        X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head = _prepare_arrays(
            X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head
        )
        out_ids = np.full(k, -1, dtype=np.int64)
        out_scores = np.full(k, -np.inf, dtype=np.float32)
        cand_ids = np.full(r, -1, dtype=np.int64)
        cand_scores = np.full(r, -np.inf, dtype=np.float32)
        status = self.lib.etar_search(
            X.ctypes.data_as(_fptr),
            X_scan.ctypes.data_as(_fptr),
            C_col.ctypes.data_as(_i8ptr),
            scales.ctypes.data_as(_fptr),
            scan_scales.ctypes.data_as(_fptr),
            norms.ctypes.data_as(_fptr),
            residuals.ctypes.data_as(_fptr),
            _ptr_or_none(alive, _u8ptr),
            q.ctypes.data_as(_fptr),
            head.ctypes.data_as(_i64ptr),
            head.size,
            X.shape[0],
            X.shape[1],
            int(scan_stride),
            k,
            r,
            int(use_tail),
            float(alpha_tail),
            float(tail_norm),
            float(alpha_q),
            int(int8_query),
            int(scan_mode),
            int(no_rerank),
            int(prune_lower_bound),
            float(prune_lower_scale),
            int(block_prune),
            int(block_prune_scout_h),
            float(block_prune_beam_factor),
            float(block_prune_gate),
            int(block_prune_block_size),
            int(sort_rerank_candidates),
            out_ids.ctypes.data_as(_i64ptr),
            out_scores.ctypes.data_as(_fptr),
            cand_ids.ctypes.data_as(_i64ptr),
            cand_scores.ctypes.data_as(_fptr),
        )
        return out_ids[: status.results], out_scores[: status.results], cand_ids[: status.candidates], cand_scores[: status.candidates]

    def search_profile(
        self,
        X,
        X_scan,
        C_col,
        scales,
        scan_scales,
        norms,
        residuals,
        alive,
        q,
        head,
        scan_stride,
        k,
        r,
        use_tail,
        alpha_tail,
        tail_norm,
        alpha_q,
        int8_query,
        scan_mode=0,
        no_rerank=False,
        prune_lower_bound=False,
        prune_lower_scale=1.0,
        block_prune=False,
        block_prune_scout_h=16,
        block_prune_beam_factor=32.0,
        block_prune_gate=0.75,
        block_prune_block_size=256,
        sort_rerank_candidates=False,
    ):
        X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head = _prepare_arrays(
            X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head
        )
        out_ids = np.full(k, -1, dtype=np.int64)
        out_scores = np.full(k, -np.inf, dtype=np.float32)
        cand_ids = np.full(r, -1, dtype=np.int64)
        cand_scores = np.full(r, -np.inf, dtype=np.float32)
        profile = _Profile()
        status = self.lib.etar_search_profile(
            X.ctypes.data_as(_fptr),
            X_scan.ctypes.data_as(_fptr),
            C_col.ctypes.data_as(_i8ptr),
            scales.ctypes.data_as(_fptr),
            scan_scales.ctypes.data_as(_fptr),
            norms.ctypes.data_as(_fptr),
            residuals.ctypes.data_as(_fptr),
            _ptr_or_none(alive, _u8ptr),
            q.ctypes.data_as(_fptr),
            head.ctypes.data_as(_i64ptr),
            head.size,
            X.shape[0],
            X.shape[1],
            int(scan_stride),
            k,
            r,
            int(use_tail),
            float(alpha_tail),
            float(tail_norm),
            float(alpha_q),
            int(int8_query),
            int(scan_mode),
            int(no_rerank),
            int(prune_lower_bound),
            float(prune_lower_scale),
            int(block_prune),
            int(block_prune_scout_h),
            float(block_prune_beam_factor),
            float(block_prune_gate),
            int(block_prune_block_size),
            int(sort_rerank_candidates),
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
        return out_ids[: status.results], out_scores[: status.results], cand_ids[: status.candidates], cand_scores[: status.candidates], timings

    def search_batch(
        self,
        X,
        X_scan,
        C_col,
        scales,
        scan_scales,
        norms,
        residuals,
        alive,
        Q,
        heads,
        head_offsets,
        scan_stride,
        k,
        r,
        use_tail,
        alpha_tail,
        tail_norm,
        alpha_q,
        int8_query,
        scan_mode=0,
        no_rerank=None,
        prune_lower_bound=None,
        prune_lower_scale=None,
        block_prune=False,
        block_prune_scout_h=16,
        block_prune_beam_factor=32.0,
        block_prune_gate=0.75,
        block_prune_block_size=256,
        sort_rerank_candidates=False,
    ):
        X = np.ascontiguousarray(X, dtype=np.float32)
        X_scan = np.ascontiguousarray(X_scan, dtype=np.float32)
        C_col = np.ascontiguousarray(C_col, dtype=np.int8)
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        scan_scales = np.ascontiguousarray(scan_scales, dtype=np.float32)
        norms = np.ascontiguousarray(norms, dtype=np.float32)
        alive = None if alive is None else np.ascontiguousarray(alive, dtype=np.uint8)
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        heads = np.ascontiguousarray(heads, dtype=np.int64)
        head_offsets = np.ascontiguousarray(head_offsets, dtype=np.int64)
        if residuals is None:
            residuals = np.zeros(X.shape[0], dtype=np.float32)
        residuals = np.ascontiguousarray(residuals, dtype=np.float32)
        use_tail = np.ascontiguousarray(use_tail, dtype=np.int32)
        alpha_tail = np.ascontiguousarray(alpha_tail, dtype=np.float32)
        tail_norm = np.ascontiguousarray(tail_norm, dtype=np.float32)
        alpha_q = np.ascontiguousarray(alpha_q, dtype=np.float32)
        int8_query = np.ascontiguousarray(int8_query, dtype=np.int32)
        if no_rerank is None:
            no_rerank = np.zeros(Q.shape[0], dtype=np.int32)
        no_rerank = np.ascontiguousarray(no_rerank, dtype=np.int32)
        if prune_lower_bound is None:
            prune_lower_bound = np.zeros(Q.shape[0], dtype=np.int32)
        prune_lower_bound = np.ascontiguousarray(prune_lower_bound, dtype=np.int32)
        if prune_lower_scale is None:
            prune_lower_scale = np.ones(Q.shape[0], dtype=np.float32)
        prune_lower_scale = np.ascontiguousarray(prune_lower_scale, dtype=np.float32)
        qn = Q.shape[0]
        out_ids = np.full((qn, k), -1, dtype=np.int64)
        out_scores = np.full((qn, k), -np.inf, dtype=np.float32)
        cand_ids = np.full((qn, r), -1, dtype=np.int64)
        cand_scores = np.full((qn, r), -np.inf, dtype=np.float32)
        profiles = (_Profile * qn)()
        self.lib.etar_search_batch(
            X.ctypes.data_as(_fptr),
            X_scan.ctypes.data_as(_fptr),
            C_col.ctypes.data_as(_i8ptr),
            scales.ctypes.data_as(_fptr),
            scan_scales.ctypes.data_as(_fptr),
            norms.ctypes.data_as(_fptr),
            residuals.ctypes.data_as(_fptr),
            _ptr_or_none(alive, _u8ptr),
            Q.ctypes.data_as(_fptr),
            heads.ctypes.data_as(_i64ptr),
            head_offsets.ctypes.data_as(_i64ptr),
            qn,
            X.shape[0],
            X.shape[1],
            int(scan_stride),
            k,
            r,
            use_tail.ctypes.data_as(_i32ptr),
            alpha_tail.ctypes.data_as(_fptr),
            tail_norm.ctypes.data_as(_fptr),
            alpha_q.ctypes.data_as(_fptr),
            int8_query.ctypes.data_as(_i32ptr),
            no_rerank.ctypes.data_as(_i32ptr),
            prune_lower_bound.ctypes.data_as(_i32ptr),
            prune_lower_scale.ctypes.data_as(_fptr),
            int(block_prune),
            int(block_prune_scout_h),
            float(block_prune_beam_factor),
            float(block_prune_gate),
            int(block_prune_block_size),
            int(sort_rerank_candidates),
            out_ids.ctypes.data_as(_i64ptr),
            out_scores.ctypes.data_as(_fptr),
            cand_ids.ctypes.data_as(_i64ptr),
            cand_scores.ctypes.data_as(_fptr),
            int(scan_mode),
            profiles,
        )
        timings = {
            "native_scan_ms": np.array([p.scan_ms for p in profiles], dtype=np.float32),
            "native_top_r_ms": np.array([p.top_r_ms for p in profiles], dtype=np.float32),
            "native_rerank_ms": np.array([p.rerank_ms for p in profiles], dtype=np.float32),
            "native_total_ms": np.array([p.total_native_ms for p in profiles], dtype=np.float32),
            "native_block_prune_survival": np.array([p.block_prune_survival for p in profiles], dtype=np.float32),
            "native_block_prune_used": np.array([p.block_prune_used for p in profiles], dtype=np.float32),
            "native_block_prune_fallback": np.array([p.block_prune_fallback for p in profiles], dtype=np.float32),
        }
        return out_ids, out_scores, cand_ids, cand_scores, timings


class NumbaBackend:
    def search(
        self,
        X,
        X_scan,
        C_col,
        scales,
        scan_scales,
        norms,
        residuals,
        alive,
        q,
        head,
        scan_stride,
        k,
        r,
        use_tail,
        alpha_tail,
        tail_norm,
        alpha_q,
        int8_query,
        scan_mode=0,
        no_rerank=False,
        prune_lower_bound=False,
        prune_lower_scale=1.0,
        block_prune=False,
        block_prune_scout_h=16,
        block_prune_beam_factor=32.0,
        block_prune_gate=0.75,
        block_prune_block_size=256,
        sort_rerank_candidates=False,
    ):
        if prune_lower_bound:
            raise NotImplementedError("lower-bound pruning requires the C++ backend")
        if block_prune:
            raise NotImplementedError("block pruning requires the C++ backend")
        from .numba_backend import search_numba

        if residuals is None:
            residuals = np.zeros(X.shape[0], dtype=np.float32)
        return search_numba(
            np.ascontiguousarray(X, dtype=np.float32),
            np.ascontiguousarray(X_scan, dtype=np.float32),
            np.ascontiguousarray(C_col, dtype=np.int8),
            np.ascontiguousarray(scales, dtype=np.float32),
            np.ascontiguousarray(scan_scales, dtype=np.float32),
            np.ascontiguousarray(norms, dtype=np.float32),
            np.ascontiguousarray(residuals, dtype=np.float32),
            np.ascontiguousarray(alive, dtype=np.uint8),
            np.ascontiguousarray(q, dtype=np.float32),
            np.ascontiguousarray(head, dtype=np.int64),
            int(scan_stride),
            int(k),
            int(r),
            bool(use_tail),
            float(alpha_tail),
            float(tail_norm),
            float(alpha_q),
            bool(int8_query),
            int(scan_mode),
            bool(no_rerank),
        )

    def search_profile(self, *args, **kwargs):
        ids, scores, cand_ids, cand_scores = self.search(*args, **kwargs)
        return ids, scores, cand_ids, cand_scores, {}

    def search_batch(
        self,
        X,
        X_scan,
        C_col,
        scales,
        scan_scales,
        norms,
        residuals,
        alive,
        Q,
        heads,
        head_offsets,
        scan_stride,
        k,
        r,
        use_tail,
        alpha_tail,
        tail_norm,
        alpha_q,
        int8_query,
        scan_mode=0,
        no_rerank=None,
        prune_lower_bound=None,
        prune_lower_scale=None,
        block_prune=False,
        block_prune_scout_h=16,
        block_prune_beam_factor=32.0,
        block_prune_gate=0.75,
        block_prune_block_size=256,
        sort_rerank_candidates=False,
    ):
        if prune_lower_bound is not None and np.any(prune_lower_bound):
            raise NotImplementedError("lower-bound pruning requires the C++ backend")
        if block_prune:
            raise NotImplementedError("block pruning requires the C++ backend")
        ids, scores, cand_ids, cand_scores = [], [], [], []
        for i, q in enumerate(Q):
            h0, h1 = head_offsets[i], head_offsets[i + 1]
            nr = bool(no_rerank[i]) if no_rerank is not None else False
            out = self.search(
                X,
                X_scan,
                C_col,
                scales,
                scan_scales,
                norms,
                residuals,
                alive,
                q,
                heads[h0:h1],
                scan_stride,
                k,
                r,
                bool(use_tail[i]),
                float(alpha_tail[i]),
                float(tail_norm[i]),
                float(alpha_q[i]),
                bool(int8_query[i]),
                int(scan_mode),
                nr,
                False,
                1.0,
                False,
                16,
                32.0,
                0.75,
                256,
                sort_rerank_candidates,
            )
            ids.append(out[0])
            scores.append(out[1])
            cand_ids.append(out[2])
            cand_scores.append(out[3])
        return np.vstack(ids), np.vstack(scores), np.vstack(cand_ids), np.vstack(cand_scores), {}


def _prepare_arrays(X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head):
    X = np.ascontiguousarray(X, dtype=np.float32)
    X_scan = np.ascontiguousarray(X_scan, dtype=np.float32)
    C_col = np.ascontiguousarray(C_col, dtype=np.int8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    scan_scales = np.ascontiguousarray(scan_scales, dtype=np.float32)
    norms = np.ascontiguousarray(norms, dtype=np.float32)
    alive = None if alive is None else np.ascontiguousarray(alive, dtype=np.uint8)
    q = np.ascontiguousarray(q, dtype=np.float32)
    head = np.ascontiguousarray(head, dtype=np.int64)
    if residuals is None:
        residuals = np.zeros(X.shape[0], dtype=np.float32)
    residuals = np.ascontiguousarray(residuals, dtype=np.float32)
    return X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head


def _ptr_or_none(array, ptr_type):
    return None if array is None else array.ctypes.data_as(ptr_type)


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
_i8ptr = ctypes.POINTER(ctypes.c_int8)
_u8ptr = ctypes.POINTER(ctypes.c_uint8)
_i64ptr = ctypes.POINTER(ctypes.c_int64)
_i32ptr = ctypes.POINTER(ctypes.c_int32)
