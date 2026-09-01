from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")


def latency_summary(latencies_ms: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "latency_ms_mean": float("nan"),
            "latency_ms_p50": float("nan"),
            "latency_ms_p95": float("nan"),
            "latency_ms_min": float("nan"),
            "latency_ms_max": float("nan"),
        }
    return {
        "latency_ms_mean": float(np.mean(arr)),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_min": float(np.min(arr)),
        "latency_ms_max": float(np.max(arr)),
    }


def timed_call(fn: Callable[[], T]) -> tuple[T, float]:
    t0 = perf_counter()
    out = fn()
    return out, (perf_counter() - t0) * 1000.0

