# `etar` Package

This package contains the ETAR index, native and fallback backends, the C++ dynamic-table wrapper used by the WF-IoT configs, benchmark dispatchers, dataset generation, dynamic workloads, metrics, and result tables.

## Public API

```python
from etar import EtarIndex
```

`EtarIndex(X, backend="auto", keep_float=True, residuals=False)` builds:

- row-wise int8 corpus codes,
- row scales,
- row norms,
- optional residual norms,
- a column-major hot code matrix for reduced-coordinate scans.

`search(...)` returns a `SearchResult` containing result ids, scores, candidate ids, and diagnostics. Reranked methods return exact float32 scores; no-rerank methods return reduced int8 scores.

The default `method="etar"` remains the index-free reduced scan with exact float32 rerank. `method="etar_no_rerank"` uses the same tail-aware reduced score but returns top-`K` directly from that score.

## Backend Selection

`backend="auto"` tries the C++ scan extension first. If it is unavailable, the package falls back to the NumPy/Numba implementation. The WF-IoT configurations set `storage_backend: cpp`, which stores the mutable scan table in the C++ extension while keeping the same ETAR algorithmic parameters.

Use explicit backends when comparing correctness or performance:

```python
EtarIndex(X, backend="cpp")
EtarIndex(X, backend="numba")
```

## Important Modules

- `core.py`: public index object and dynamic operations.
- `native.py`: C++ backend wrapper.
- `numba_backend.py`: fallback backend.
- `baselines.py`: FAISS, hnswlib, Annoy, and ScaNN baselines.
- `experiments.py`: online and batch benchmark dispatch.
- `dynamic.py`: streaming corpus workloads and checkpoint evaluation.
- `config_runner.py`: YAML-driven orchestration.
- `config_schema.py`: config normalization and sweep expansion.
- `data.py`: synthetic/HDF5 dataset generation and loading.
- `timing.py`: latency summaries.
- `metrics.py`: recall, coverage, regret, and exact top-k helpers.
- `tables.py`: table generation.

## Command Entrypoints

Installed console scripts:

- `etar-generate-data`
- `etar-smoke`
- `etar-run`
- `etar-run-config`

Module entry point:

```bash
python -m etar run-config configs/wfiot_smoke.yaml --out results/wfiot_smoke
```
