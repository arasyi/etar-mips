from __future__ import annotations

import os


def rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return float("nan")


def etar_storage_bytes(
    *,
    n: int,
    d: int,
    capacity: int | None = None,
    scan_dtype: str = "int8",
    residuals: bool = True,
    row_ids: bool = True,
    row_id_bytes: int = 8,
    store_float32: bool = True,
    store_codes_row: bool = False,
) -> dict[str, int]:
    cap = max(int(capacity if capacity is not None else n), 1)
    dim = int(d)
    full_precision = (4 * cap * dim) if store_float32 else 0
    scan = (cap * dim) if str(scan_dtype) == "int8" else (4 * cap * dim)
    rerank_codes = (cap * dim) if store_codes_row and str(scan_dtype) == "int8" else 0
    id_bytes = max(0, int(row_id_bytes)) if row_ids else 0
    metadata = cap * (4 + 4 + (4 if residuals else 0) + 1 + id_bytes)
    total = full_precision + scan + rerank_codes + metadata
    return {
        "storage_capacity": cap,
        "full_precision_bytes_estimate": int(full_precision),
        "scan_bytes_estimate": int(scan),
        "rerank_code_bytes_estimate": int(rerank_codes),
        "metadata_bytes_estimate": int(metadata),
        "storage_bytes_estimate": int(total),
        "storage_mb_estimate": float(total / (1024.0 * 1024.0)),
    }


def exact_storage_bytes(*, n: int, d: int) -> dict[str, int]:
    total = int(4 * int(n) * int(d))
    return {
        "storage_capacity": int(n),
        "full_precision_bytes_estimate": total,
        "scan_bytes_estimate": 0,
        "metadata_bytes_estimate": 0,
        "storage_bytes_estimate": total,
        "storage_mb_estimate": float(total / (1024.0 * 1024.0)),
    }
