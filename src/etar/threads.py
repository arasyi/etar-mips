from __future__ import annotations

import os


THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def set_single_thread_env() -> dict[str, str]:
    for key, value in THREAD_ENV_VARS.items():
        os.environ[key] = value
    return dict(THREAD_ENV_VARS)


def configure_faiss_threads() -> bool:
    try:
        import faiss

        faiss.omp_set_num_threads(1)
        return True
    except Exception:
        return False

