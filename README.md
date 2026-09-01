# WF-IoT 2026 ETAR Reproducibility Artifact

This repository contains the artifact for the accepted WF-IoT 2026 paper:

> Index-Free Dynamic Edge Retrieval with Energy-Tail-Aware Partial Scans

It provides the ETAR implementation, experiment configurations, five-repeat reproduction workflow, and Android benchmark. Experiment outputs are generated locally under `results/` and are not committed.

## Repository layout

```text
.
├── analysis/                 # repeated-run aggregation and RSS profiling
├── android/                  # native Android benchmark app
├── configs/                  # WF-IoT 2026 experiment configurations
├── cpp/                      # ETAR C++ scan and dynamic-table backends
├── data/                     # external dataset placement notes
├── scripts/                  # build and reproduction entry points
├── src/etar/                 # Python benchmark package
├── environment.yml           # conda environment
├── setup.py                  # native-extension build
└── pyproject.toml
```

## Environment

Create the conda environment and build the C++ extensions from the repository root:

```bash
conda env create -f environment.yml
conda activate etar-wfiot
python setup.py build_ext --inplace
```

`environment.yml` defines the complete experiment environment, including the ANN baseline libraries. `pyproject.toml` defines the installable Python package, build backend, runtime dependencies, and command entry points.

All shell entry points default to the `etar-wfiot` environment. Set `CONDA_ENV` only when using a differently named environment:

```bash
CONDA_ENV=my-environment scripts/run_wfiot_once.sh smoke
```

## Data

The configurations generate the synthetic datasets. Place the ANN-Benchmarks HDF5 datasets at these paths:

```text
data/ann/sift-128-euclidean.hdf5
data/ann/glove-100-angular.hdf5
data/ann/lastfm-64-dot.hdf5
data/ann/fashion-mnist-784-euclidean.hdf5
data/ann/nytimes-256-angular.hdf5
```

See [`data/README.md`](data/README.md) for dataset details.

## Installation check

Run one smoke-test replicate:

```bash
scripts/run_wfiot_once.sh smoke
```

The command writes its output under `results/wfiot_smoke`.

## Five-repeat reproduction

The paper reports means over five repeated runs. `reproduce_wfiot.sh` defaults to five isolated repetitions, keeps each run in a separate directory, and writes aggregate means, standard deviations, counts, and approximate 95% confidence intervals.

Run the full five-repeat workflow:

```bash
scripts/reproduce_wfiot.sh all
```

Individual targets are also available:

```bash
scripts/reproduce_wfiot.sh static
scripts/reproduce_wfiot.sh dynamic
scripts/reproduce_wfiot.sh compaction
scripts/reproduce_wfiot.sh scaling
scripts/reproduce_wfiot.sh scaling-rss
```

Pass an explicit repetition count or output directory after the target:

```bash
scripts/reproduce_wfiot.sh static 5 results/static_reproduction
```

Existing repeat directories are preserved. Set `OVERWRITE_REPEATS=1` to replace a previously generated batch.

The compaction and scaling targets include Faiss HNSW rebuilds and cases with one million rows.

## Single-replicate runs

`run_wfiot_once.sh` runs one replicate for diagnostics or development:

```bash
scripts/run_wfiot_once.sh static
scripts/run_wfiot_once.sh dynamic
scripts/run_wfiot_once.sh compaction
scripts/run_wfiot_once.sh scaling
scripts/run_wfiot_once.sh scaling-rss
```

These commands do not reproduce the five-repeat statistics reported in the paper.

## Outputs

Single-replicate outputs are written to target-specific directories such as:

```text
results/wfiot_static/
results/wfiot_dynamic/
results/wfiot_compaction_stress/
results/wfiot_scaling/
results/wfiot_scaling_fresh_rss/
```

Five-repeat outputs use `results/repeats/<target>/run_XX`. Aggregated CSV files are written to `results/repeats/summary` by `analysis/aggregate_repeated_results.py`.

## Android benchmark

The `android/` module runs exact float32 scanning, exhaustive full-coordinate int8 scanning, ETAR, and ETAR-LM on an Android device.

Build and install the benchmark APK from the repository root:

```bash
gradle -p android :app:assembleBenchmark
adb install -r android/app/build/outputs/apk/benchmark/app-benchmark.apk
```

The app exports JSON containing the device and experiment configuration, build time, Recall@K, latency distribution, memory measurements, and available energy and thermal readings. See [`android/README.md`](android/README.md) for the measurement procedure and native host test.

## Storage presets

The artifact uses the C++ dynamic-table backend for ETAR:

- `ETAR` uses an 8-bit column-major scan view and a 32-bit row-major reranking view.
- `ETAR-LM` uses an 8-bit column-major scan view and an 8-bit row-major reranking view.

The query remains in single precision for coordinate selection, retained-energy computation, tail scoring, and 32-bit reranking. For an 8-bit scan or reranking view, the implementation can quantize the selected query coordinates for integer arithmetic.

## Memory reporting

The result CSVs include analytical representation fields and measured process RSS fields. The scaling profile reports RSS build delta because it measures the additional process memory observed while building each method. RSS also includes the Python runtime, loaded libraries, allocator behavior, datasets, and temporary arrays.

The static scaling configuration sets `scan_initial_capacity_factor: 1.0` so ETAR is measured without insertion reserve. Streaming configurations retain spare capacity for insertions.
