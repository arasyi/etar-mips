# ETAR Edge Benchmark for Android

This module builds an installable Android benchmark app for the four
index-free methods used in the paper:

- exact float32 scan;
- exhaustive full-coordinate int8 scan (`full_i8_scan`), without reranking;
- ETAR with a column-major int8 candidate scan and float32 reranking;
- ETAR-LM with column-major int8 scanning and row-major int8 reranking.

The UI starts with the WF-IoT artifact's static defaults (`N=50,000`, `d=256`,
`Q=1,000`, `K=10`, `R=100`, `rho=0.90`, `h_max=128`, `seed=123`). Every value
can be changed on the device and is retained in `SharedPreferences`; changing
an experiment does not require rebuilding the APK.

## Build

Run the build from the repository root with the Android SDK, NDK, CMake, and
Gradle installed:

```bash
gradle -p android :app:assembleBenchmark
adb install -r android/app/build/outputs/apk/benchmark/app-benchmark.apk
```

The benchmark APK is signed with the Android debug key because it is a research
instrument, not a store release. The Gradle configuration builds `arm64-v8a`
and `x86_64` native libraries.

## Measurement workflow

1. Install the benchmark APK on a physical ARM64 phone.
2. Set the corpus, query, and ETAR parameters in the app.
3. Use a zero-second trace duration for correctness smoke tests. Use 30–60
   seconds per method when collecting power data.
4. Rotate the method execution order between runs to reduce thermal/order bias.
   Leave CPU core at `-1` for normal Android scheduling, or select an allowed
   core index for controlled big-/little-core experiments.
5. Start an Android System Trace / Power Profiler recording and run the app.
6. Integrate power over the `ETAR_BENCH_exact_f32`,
   `ETAR_BENCH_full_i8_scan`, `ETAR_BENCH_etar`, and `ETAR_BENCH_etar_lm`
   trace sections.
7. Share the JSON result from the app. It records device/SoC details, Android
   version, configuration, build time, Recall@K, latency distribution,
   analytical representation/workspace bytes, process PSS snapshots, per-method
   joules and joules/query when Android exposes a usable battery energy counter,
   and thermal, charging, battery-capacity, and instantaneous-current state
   before and after the run.

The in-app energy value is the decrease in Android's
`BATTERY_PROPERTY_ENERGY_COUNTER` (remaining energy in nWh, converted to joules)
across each measured trace section. After the four methods, the app records an
equal-duration `ETAR_BENCH_idle` section with the same screen state, then reports
raw and idle-adjusted joules, joules/query, and average power. Values are `null`
when the property is unsupported, does not decrease because the device is
charging, or is too coarse for the run duration. A zero-second quick check skips
the idle baseline. For publication-quality CPU/memory-rail measurements,
integrate the Power Profiler rails over each trace section, subtract the idle
rail power, and divide by `measured_queries`; the in-app battery-counter result
is a cross-check, not a substitute for calibrated rail measurements.

For repeated trials, rotate the method order and export one JSON file per run.
Compute confidence intervals across those independent files rather than across
queries from one thermally correlated batch.

The selectable synthetic datasets use the definitions in
`src/etar/data.py`: `dense_gaussian`, `sparse10`,
`mixed_dropout_heavytail` (Student's t with `df=3`, 35% dropout), and
`norm_heavy` (log-normal corpus row scales with `sigma=1.25`). They are
generated natively from the configured seed. The statistical definitions
match NumPy, although the C++ standard-library RNG stream is not byte-identical
to NumPy's PCG64 stream. Corpus generation and warm-up are excluded from query
latency. All four methods in an Android run receive identical deterministic
data.

The reported representation size is the primary per-method memory comparison.
The process PSS values cover the whole app and native allocator and are included
as implementation-level diagnostics, not as replacements for analytical bytes.

## Native portability

The core is ordinary C++17. ARM64 builds use a NEON candidate-scan kernel;
other architectures use the scalar implementation. No `-march=native`, AVX,
Python, NumPy, or third-party ANN dependency is used by the Android app.

The small host test can be run without Android tools:

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -Iandroid/app/src/main/cpp \
  android/app/src/main/cpp/etar_core.cpp \
  android/native_test/etar_core_test.cpp \
  -o /tmp/etar_core_test
/tmp/etar_core_test
```
