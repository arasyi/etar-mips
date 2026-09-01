# Data

The WF-IoT configs use two data sources.

## Generated Synthetic Data

The static config generates four synthetic datasets:

- dense Gaussian,
- 10% sparse Gaussian,
- mixed dropout/heavy-tail,
- norm-heavy.

Each uses `N=50,000`, `d=256`, and `Q=1,000` in the paper config. Generated files are written to the selected output directory under `data/`.

## ANN-Benchmarks Data

The paper also uses five ANN-Benchmarks HDF5 files, capped to `N=50,000` and `Q=1,000` by the config:

```text
data/ann/sift-128-euclidean.hdf5
data/ann/glove-100-angular.hdf5
data/ann/lastfm-64-dot.hdf5
data/ann/fashion-mnist-784-euclidean.hdf5
data/ann/nytimes-256-angular.hdf5
```

These files are not duplicated in the artifact directory. Place them at the paths above before running the full static experiment.
