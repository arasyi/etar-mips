from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_METHODS = [
    "exact_f32",
    "etar_8p32_scaling_high_recall",
    "etar_8p8_scaling_high_recall",
    "faiss_hnsw",
    "hnswlib",
    "faiss_ivf",
]

DEFAULT_N_VALUES = [50_000, 100_000, 250_000, 500_000, 1_000_000]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cfg


def _method_by_name(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = base.get("benchmarks", {}).get("methods", [])
    out: dict[str, dict[str, Any]] = {}
    for method in methods:
        if isinstance(method, dict) and method.get("name"):
            out[str(method["name"])] = dict(method)
    return out


def _dataset_path(data_dir: Path, n: int, d: int, q: int, seed: int) -> Path:
    return data_dir / f"wfiot_scaling_dense_gaussian_N{n}_d{d}_q{q}_s{seed}.hdf5"


def _case_config(
    *,
    base: dict[str, Any],
    method: dict[str, Any],
    dataset_path: Path,
    n: int,
    d: int,
    q: int,
    seed: int,
) -> dict[str, Any]:
    method_cfg = dict(method)
    method_cfg["modes"] = ["online"]
    return {
        "name": f"wfiot_scaling_fresh_rss_{method_cfg['name']}_N{n}",
        "output_dir": "unused",
        "seed": seed,
        "k": int(base.get("k", 10)),
        "ground_truth_k": int(base.get("ground_truth_k", 10)),
        "stages": ["env", "online"],
        "limits": {"q": q},
        "datasets": {
            "items": [
                {
                    "name": f"wfiot_scaling_dense_gaussian_N{n}_d{d}_q{q}_s{seed}",
                    "source": "hdf5",
                    "path": str(dataset_path),
                    "limit_q": q,
                }
            ]
        },
        "benchmarks": {"methods": [method_cfg]},
        "dynamic": {"enabled": False},
    }


def _read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _method_order(name: str) -> int:
    try:
        return DEFAULT_METHODS.index(name)
    except ValueError:
        return len(DEFAULT_METHODS)


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def _make_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for row in sorted(rows, key=lambda r: (int(float(r["N"])), _method_order(str(r["method"])))):
        summary.append(
            {
                "N": int(float(row["N"])),
                "method": row["method"],
                "recall_at_k": _float(row, "recall_at_k"),
                "latency_ms_mean": _float(row, "latency_ms_mean"),
                "index_build_s": _float(row, "index_build_ms") / 1000.0,
                "storage_mb_estimate": _float(row, "storage_mb_estimate"),
                "rss_after_build_mb": _float(row, "rss_after_build_mb"),
                "rss_build_delta_mb": _float(row, "rss_build_delta_mb"),
                "full_precision_bytes_estimate": row.get("full_precision_bytes_estimate", ""),
                "scan_bytes_estimate": row.get("scan_bytes_estimate", ""),
                "rerank_code_bytes_estimate": row.get("rerank_code_bytes_estimate", ""),
                "metadata_bytes_estimate": row.get("metadata_bytes_estimate", ""),
            }
        )
    return summary


def _make_build_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda r: (int(float(r["N"])), _method_order(str(r["method"])))):
        index_build_s = _float(row, "index_build_ms") / 1000.0
        out.append(
            {
                "N": int(float(row["N"])),
                "method": row["method"],
                "operation": "build",
                "index_build_ms": _float(row, "index_build_ms"),
                "index_build_s": index_build_s,
                "rss_before_build_mb": _float(row, "rss_before_build_mb"),
                "rss_after_build_mb": _float(row, "rss_after_build_mb"),
                "rss_build_delta_mb": _float(row, "rss_build_delta_mb"),
                "storage_mb_estimate": _float(row, "storage_mb_estimate"),
                "full_precision_bytes_estimate": row.get("full_precision_bytes_estimate", ""),
                "scan_bytes_estimate": row.get("scan_bytes_estimate", ""),
                "rerank_code_bytes_estimate": row.get("rerank_code_bytes_estimate", ""),
                "metadata_bytes_estimate": row.get("metadata_bytes_estimate", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WFIoT scaling memory profile with one fresh process per method/scale.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/wfiot_scaling.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("results/wfiot_scaling/data"))
    parser.add_argument("--out", type=Path, default=Path("results/wfiot_scaling_fresh_rss"))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--q", type=int, default=50)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base = _load_yaml(args.base_config)
    methods = _method_by_name(base)
    missing = [name for name in args.methods if name not in methods]
    if missing:
        raise KeyError(f"methods not found in {args.base_config}: {', '.join(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    config_dir = args.out / "case_configs"
    case_root = args.out / "cases"
    config_dir.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("PYTHONUNBUFFERED", "1")
    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[var] = "1"
    env.setdefault("XDG_CACHE_HOME", "/tmp/etar-xdg-cache")
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(args.n_values) * len(args.methods)
    step = 0
    for n in args.n_values:
        dataset_path = _dataset_path(args.data_dir, n, args.d, args.q, args.seed)
        if not dataset_path.exists():
            raise FileNotFoundError(f"missing scaling dataset: {dataset_path}")
        for method_name in args.methods:
            step += 1
            case_name = f"N{n}_{method_name}"
            case_out = case_root / case_name
            result_csv = case_out / "results" / "online_results.csv"
            if args.skip_existing and result_csv.exists():
                print(f"[fresh-rss] {step}/{total} reuse {case_name}", flush=True)
            else:
                if case_out.exists():
                    shutil.rmtree(case_out)
                case_cfg = _case_config(
                    base=base,
                    method=methods[method_name],
                    dataset_path=dataset_path.resolve(),
                    n=n,
                    d=args.d,
                    q=args.q,
                    seed=args.seed,
                )
                config_path = config_dir / f"{case_name}.yaml"
                config_path.write_text(yaml.safe_dump(case_cfg, sort_keys=False), encoding="utf-8")
                print(f"[fresh-rss] {step}/{total} run {case_name}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "etar",
                        "run-config",
                        str(config_path),
                        "--out",
                        str(case_out),
                    ],
                    check=True,
                    env=env,
                )
            row = _read_one_row(result_csv)
            row["fresh_process_profile"] = "true"
            rows.append(row)
            _write_csv(args.out / "results" / "fresh_rss_results.csv", rows)
            _write_csv(args.out / "results" / "fresh_rss_summary.csv", _make_summary(rows))
            _write_csv(args.out / "results" / "fresh_rss_build_results.csv", _make_build_results(rows))

    print(args.out / "results" / "fresh_rss_summary.csv")


if __name__ == "__main__":
    main()
