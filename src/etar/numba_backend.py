from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _score_row(X, X_scan, C_col, scales, scan_scales, norms, residuals, q, head, row, n, d, scan_stride, use_tail, alpha_tail, tail_norm, alpha_q, int8_query, scan_mode):
    h = head.size
    if scan_mode == 2:
        tq = 1.0
        if int8_query:
            max_abs = 0.0
            for t in range(h):
                v = abs(q[head[t]])
                if v > max_abs:
                    max_abs = v
            tq = max_abs / 127.0 if max_abs > 0.0 else 1.0 / 127.0
        s = 0.0
        for t in range(h):
            j = head[t]
            qj = q[j]
            if int8_query:
                b = int(np.rint(qj / tq))
                if b > 127:
                    b = 127
                elif b < -127:
                    b = -127
                qj = tq * b
            s += qj * X[row, j]
    elif scan_mode == 3:
        tq = 1.0
        if int8_query:
            max_abs = 0.0
            for t in range(h):
                v = abs(q[head[t]])
                if v > max_abs:
                    max_abs = v
            tq = max_abs / 127.0 if max_abs > 0.0 else 1.0 / 127.0
        s = 0.0
        for t in range(h):
            j = head[t]
            qj = q[j]
            if int8_query:
                b = int(np.rint(qj / tq))
                if b > 127:
                    b = 127
                elif b < -127:
                    b = -127
                qj = tq * b
            s += qj * X_scan[j, row]
    elif scan_mode == 4 or scan_mode == 5:
        tq = 1.0
        if int8_query:
            max_abs = 0.0
            for t in range(h):
                v = abs(q[head[t]])
                if v > max_abs:
                    max_abs = v
            tq = max_abs / 127.0 if max_abs > 0.0 else 1.0 / 127.0
        s = 0.0
        for t in range(h):
            j = head[t]
            qj = q[j]
            if int8_query:
                b = int(np.rint(qj / tq))
                if b > 127:
                    b = 127
                elif b < -127:
                    b = -127
                qj = tq * b
            if scan_mode == 5:
                qj *= scan_scales[j]
            s += qj * C_col[row, j]
        if scan_mode == 4:
            s *= scales[row]
    elif int8_query:
        max_abs = 0.0
        for t in range(h):
            v = abs(q[head[t]])
            if v > max_abs:
                max_abs = v
        if max_abs == 0.0:
            s = 0.0
        else:
            tq = max_abs / 127.0
            acc = 0
            accf = 0.0
            for t in range(h):
                j = head[t]
                b = int(np.rint(q[j] / tq))
                if b > 127:
                    b = 127
                elif b < -127:
                    b = -127
                if scan_mode == 1:
                    accf += b * int(C_col[j, row]) * scan_scales[j]
                else:
                    acc += b * int(C_col[j, row])
            if scan_mode == 1:
                s = tq * accf
            else:
                s = tq * acc
                s *= scales[row]
    else:
        acc = 0.0
        for t in range(h):
            j = head[t]
            qj = q[j]
            if scan_mode == 1:
                qj *= scan_scales[j]
            acc += qj * C_col[j, row]
        s = acc
        if scan_mode != 1:
            s *= scales[row]
    if use_tail:
        s += alpha_tail * tail_norm * norms[row]
    if alpha_q != 0.0:
        qhn = 0.0
        for t in range(h):
            qhn += q[head[t]] * q[head[t]]
        s += alpha_q * np.sqrt(qhn) * residuals[row] * np.sqrt(h / d)
    return s


@njit(cache=True)
def _exact_dot(X, q, row):
    s = 0.0
    for j in range(X.shape[1]):
        s += X[row, j] * q[j]
    return s


def search_numba(X, X_scan, C_col, scales, scan_scales, norms, residuals, alive, q, head, scan_stride, k, r, use_tail, alpha_tail, tail_norm, alpha_q, int8_query, scan_mode, no_rerank=False):
    n, d = X.shape
    scan_scores = np.full(n, -np.inf, dtype=np.float32)
    for i in range(n):
        if alive[i] != 0:
            scan_scores[i] = _score_row(X, X_scan, C_col, scales, scan_scales, norms, residuals, q, head, i, n, d, scan_stride, use_tail, alpha_tail, tail_norm, alpha_q, int8_query, scan_mode)
    cand = np.argpartition(-scan_scores, kth=min(r - 1, n - 1))[:r]
    cand = cand[np.argsort(-scan_scores[cand], kind="stable")]
    cand = cand[scan_scores[cand] != -np.inf].astype(np.int64)
    if no_rerank:
        out = cand[:k]
        return out.astype(np.int64), scan_scores[out].astype(np.float32), cand, scan_scores[cand].astype(np.float32)
    exact = np.empty(cand.size, dtype=np.float32)
    for t, row in enumerate(cand):
        exact[t] = _exact_dot(X, q, int(row))
    order = np.argsort(-exact, kind="stable")[:k]
    return cand[order].astype(np.int64), exact[order].astype(np.float32), cand, scan_scores[cand].astype(np.float32)
