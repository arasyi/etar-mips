from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from .config_schema import normalize_config
from .data import generate_dataset, generate_preset, load_hdf5, write_hdf5
from .dynamic import run_streaming_dynamic_with_build
from .experiments import run_dynamic_baselines, run_method_batch_loaded, run_method_online_loaded
from .metrics import exact_topk
from .tables import make_tables
from .threads import set_single_thread_env


STAGES = ["env", "data", "ground_truth", "online", "batch", "dynamic", "ablations", "analyze", "tables"]


def progress(message: str) -> None:
    print(f"[etar] {message}", flush=True)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    return cfg


def stage_list(stage: str, cfg: dict) -> list[str]:
    if stage == "all":
        return list(cfg.get("stages", STAGES))
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return [stage]


def resolve_output_dir(cfg: dict, override: str | None) -> Path:
    return Path(override or cfg.get("output_dir", "results/etar_run"))


def setup_dirs(out: Path) -> None:
    for sub in ["logs", "results", "tables", "data", "ground_truth"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def write_env_report(out: Path, cfg_path: Path, cfg: dict) -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "thread_env": set_single_thread_env(),
        "config_path": str(cfg_path),
        "docs": ["README.md"],
    }
    (out / "logs" / "environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def build_cpp_if_needed(out: Path) -> None:
    targets = [
        ("_etar_cpp", Path("cpp/etar_cpp.cpp")),
        ("_etar_storage_cpp", Path("cpp/etar_storage_cpp.cpp")),
    ]
    stale = False
    for stem, source in targets:
        so_files = list(Path("src/etar").glob(f"{stem}*.so"))
        if not so_files or any(source.stat().st_mtime > so.stat().st_mtime for so in so_files):
            stale = True
            break
    if stale:
        progress("building C++ extensions")
        log = out / "logs" / "build_cpp.log"
        with log.open("w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True, stdout=f, stderr=subprocess.STDOUT)
        progress(f"C++ extension build finished; log={log}")
    else:
        progress("C++ extensions are up to date")


def prepare_datasets(cfg: dict, out: Path) -> list[Path]:
    paths: list[Path] = []
    data_cfg = cfg.get("datasets", {})
    items = data_cfg.get("items", [])
    records = []
    for idx, item in enumerate(items, start=1):
        source = item.get("source", "hdf5")
        label = item.get("name") or item.get("path") or item.get("kind") or f"dataset_{idx}"
        progress(f"data {idx}/{len(items)}: preparing {label} ({source})")
        if source == "generated":
            name = item.get("name") or f"{item['kind']}_N{item['n']}_d{item['d']}_s{item.get('seed', cfg.get('seed', 123))}"
            path = Path(item.get("path") or out / "data" / f"{name}.hdf5")
            X, Q = generate_dataset(
                str(item["kind"]),
                int(item["n"]),
                int(item["d"]),
                int(item.get("queries", item.get("q", 100))),
                int(item.get("seed", cfg.get("seed", 123))),
            )
            write_hdf5(path, X, Q, str(item["kind"]), int(item.get("seed", cfg.get("seed", 123))), k=int(cfg.get("ground_truth_k", 100)))
        elif source in {"hdf5", "ann"}:
            path = Path(item["path"])
            if not path.exists():
                if not item.get("download_url"):
                    raise FileNotFoundError(f"missing dataset {path}; add download_url to dataset item to enable download")
                raise NotImplementedError("dataset download is config-gated but not implemented for this source yet")
        else:
            raise ValueError(f"unknown dataset source {source!r}")
        paths.append(path)
        rec = dict(item)
        rec["path"] = str(path)
        rec.setdefault("name", path.stem)
        records.append(rec)
        progress(f"data {idx}/{len(items)}: ready {path}")
    manifest = pd.DataFrame(records or [{"path": str(p)} for p in paths])
    manifest.to_csv(out / "results" / "datasets.csv", index=False)
    progress(f"data manifest written: {out / 'results' / 'datasets.csv'}")
    return paths


def dataset_paths(cfg: dict, out: Path) -> list[Path]:
    manifest = out / "results" / "datasets.csv"
    if manifest.exists():
        return [Path(p) for p in pd.read_csv(manifest)["path"].tolist()]
    return prepare_datasets(cfg, out)


def compute_ground_truth(paths: list[Path], cfg: dict, out: Path) -> None:
    k = int(cfg.get("k", 10))
    dataset_items = cfg.get("datasets", {}).get("items", [])
    for idx, (path, item) in enumerate(zip(paths, dataset_items), start=1):
        progress(f"ground_truth {idx}/{len(paths)}: computing {path.stem}")
        X, Q = load_hdf5(path, item.get("limit_n", cfg.get("limits", {}).get("n")), item.get("limit_q", cfg.get("limits", {}).get("q")))
        ids, scores = exact_topk(X, Q, max(k, int(cfg.get("ground_truth_k", 100))))
        name = path.stem.replace("/", "_")
        pd.DataFrame(ids).to_csv(out / "ground_truth" / f"{name}_ids.csv", index=False)
        pd.DataFrame(scores).to_csv(out / "ground_truth" / f"{name}_scores.csv", index=False)
        progress(f"ground_truth {idx}/{len(paths)}: wrote {name}_ids.csv and {name}_scores.csv")


def _write_result_frames(frames: list[pd.DataFrame], dst: Path) -> None:
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(dst, index=False)


def _online_build_frame(results: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "dataset",
        "method",
        "method_impl",
        "family",
        "index",
        "config_id",
        "backend",
        "storage_backend",
        "N",
        "d",
        "Q",
        "K",
        "R",
        "r",
        "rho",
        "h",
        "h_max",
        "storage_preset",
        "tail_policy",
        "rerank",
        "rerank_dtype",
        "query_mode",
        "scan_dtype",
        "scan_layout",
        "scan_quantization",
        "online_or_batch",
        "status",
    ]
    build_cols = [
        "index_build_ms",
        "build_ms",
        "rss_before_build_mb",
        "rss_after_build_mb",
        "rss_build_delta_mb",
        "storage_bytes_estimate",
        "storage_mb_estimate",
        "index_payload_bytes_estimate",
        "index_payload_mb_estimate",
        "full_precision_bytes_estimate",
        "scan_bytes_estimate",
        "rerank_code_bytes_estimate",
        "metadata_bytes_estimate",
        "scan_capacity",
        "scan_slack_fraction",
    ]
    cols = [c for c in meta_cols + build_cols if c in results.columns]
    if not cols:
        return pd.DataFrame()
    build = results[cols].copy()
    if "index_build_ms" in build.columns:
        build["index_build_s"] = pd.to_numeric(build["index_build_ms"], errors="coerce") / 1000.0
    if "build_ms" in build.columns:
        build["build_s"] = pd.to_numeric(build["build_ms"], errors="coerce") / 1000.0
    return build


def run_online(paths: list[Path], cfg: dict, out: Path) -> Path:
    frames = []
    dst = out / "results" / "online_results.csv"
    build_dst = out / "results" / "online_build_results.csv"
    dataset_items = cfg.get("datasets", {}).get("items", [])
    methods = [m for m in cfg.get("benchmarks", {}).get("methods", []) if "online" in m.get("modes", ["online"])]
    total = max(len(paths) * len(methods), 1)
    step = 0
    for path, item in zip(paths, dataset_items):
        limit_n = item.get("limit_n", cfg.get("limits", {}).get("n"))
        limit_q = item.get("limit_q", cfg.get("limits", {}).get("q"))
        progress(f"online: loading dataset={path.stem} limit_n={limit_n} limit_q={limit_q}")
        X, Q = load_hdf5(path, limit_n, limit_q)
        progress(f"online: computing exact top-{int(cfg.get('k', 10))} ground truth for dataset={path.stem} N={X.shape[0]} Q={Q.shape[0]}")
        truth_ids, truth_scores = exact_topk(X, Q, int(cfg.get("k", 10)))
        for method in methods:
            step += 1
            progress(f"online {step}/{total}: dataset={path.stem} method={method.get('name', method.get('family'))}")
            try:
                frames.append(run_method_online_loaded(
                    path,
                    X,
                    Q,
                    truth_ids,
                    truth_scores,
                    method,
                    k=int(cfg.get("k", 10)),
                ))
            except Exception as exc:
                progress(f"online {step}/{total}: failed dataset={path.stem} method={method.get('name')} error={exc}")
                frames.append(pd.DataFrame([{"dataset": str(path), "method": method.get("name"), "config_id": method.get("config_id"), "online_or_batch": "online", "status": f"failed: {exc}"}]))
            _write_result_frames(frames, dst)
    if frames:
        build = _online_build_frame(pd.concat(frames, ignore_index=True))
        if not build.empty:
            build.to_csv(build_dst, index=False)
            progress(f"online build results written: {build_dst}")
    progress(f"online results written: {dst}")
    return dst


def run_batch(paths: list[Path], cfg: dict, out: Path) -> Path:
    frames = []
    dst = out / "results" / "batch_results.csv"
    dataset_items = cfg.get("datasets", {}).get("items", [])
    methods = [m for m in cfg.get("benchmarks", {}).get("methods", []) if "batch" in m.get("modes", [])]
    total = max(len(paths) * len(methods), 1)
    step = 0
    for path, item in zip(paths, dataset_items):
        limit_n = item.get("limit_n", cfg.get("limits", {}).get("n"))
        limit_q = item.get("limit_q", cfg.get("limits", {}).get("q"))
        progress(f"batch: loading dataset={path.stem} limit_n={limit_n} limit_q={limit_q}")
        X, Q = load_hdf5(path, limit_n, limit_q)
        progress(f"batch: computing exact top-{int(cfg.get('k', 10))} ground truth for dataset={path.stem} N={X.shape[0]} Q={Q.shape[0]}")
        truth_ids, truth_scores = exact_topk(X, Q, int(cfg.get("k", 10)))
        for method in methods:
            step += 1
            progress(f"batch {step}/{total}: dataset={path.stem} method={method.get('name', method.get('family'))}")
            try:
                frames.append(run_method_batch_loaded(
                    path,
                    X,
                    Q,
                    truth_ids,
                    truth_scores,
                    method,
                    k=int(cfg.get("k", 10)),
                ))
            except Exception as exc:
                progress(f"batch {step}/{total}: failed dataset={path.stem} method={method.get('name')} error={exc}")
                frames.append(pd.DataFrame([{"dataset": str(path), "method": method.get("name"), "config_id": method.get("config_id"), "online_or_batch": "batch", "status": f"failed: {exc}"}]))
            _write_result_frames(frames, dst)
    progress(f"batch results written: {dst}")
    return dst


def run_analysis(out: Path) -> Path:
    frames = []
    for name in ["online_results.csv", "batch_results.csv"]:
        path = out / "results" / name
        if path.exists():
            progress(f"analyze: reading {path}")
            frames.append(pd.read_csv(path))
    dst = out / "results" / "all_results.csv"
    pd.concat(frames, ignore_index=True).to_csv(dst, index=False)
    progress(f"analysis results written: {dst}")
    return dst


def run_config(config_path: str | Path, out_override: str | None = None, stage: str = "all") -> Path:
    set_single_thread_env()
    cfg_path = Path(config_path)
    progress(f"loading config: {cfg_path}")
    raw_cfg = load_config(cfg_path)
    cfg = normalize_config(raw_cfg)
    out = resolve_output_dir(cfg, out_override)
    setup_dirs(out)
    progress(f"output directory: {out}")
    (out / "raw_config.yaml").write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
    write_env_report(out, cfg_path, cfg)
    progress(f"environment report written: {out / 'logs' / 'environment.json'}")
    build_cpp_if_needed(out)
    selected = stage_list(stage, cfg)
    progress(f"selected stages: {', '.join(selected)}")
    paths: list[Path] = []
    if "data" in selected or any(s in selected for s in ["ground_truth", "online", "batch"]):
        progress("stage data: start" if "data" in selected else "stage data: loading manifest")
        paths = prepare_datasets(cfg, out) if "data" in selected else dataset_paths(cfg, out)
        progress("stage data: done")
    if "ground_truth" in selected:
        progress("stage ground_truth: start")
        compute_ground_truth(paths, cfg, out)
        progress("stage ground_truth: done")
    if "online" in selected or "ablations" in selected:
        progress("stage online: start")
        run_online(paths or dataset_paths(cfg, out), cfg, out)
        progress("stage online: done")
    if "batch" in selected:
        progress("stage batch: start")
        run_batch(paths or dataset_paths(cfg, out), cfg, out)
        progress("stage batch: done")
    if "dynamic" in selected:
        progress("stage dynamic: start")
        dynamic, dynamic_build = run_streaming_dynamic_with_build(cfg.get("dynamic"), progress_fn=progress)
        if dynamic.empty:
            progress("stage dynamic: no configured streaming workloads; running microbenchmark")
            dynamic = run_dynamic_baselines()
        dynamic.to_csv(out / "results" / "dynamic_results.csv", index=False)
        progress(f"stage dynamic: wrote {out / 'results' / 'dynamic_results.csv'}")
        if not dynamic_build.empty:
            dynamic_build.to_csv(out / "results" / "dynamic_build_results.csv", index=False)
            progress(f"stage dynamic: wrote {out / 'results' / 'dynamic_build_results.csv'}")
        progress("stage dynamic: done")
    if "analyze" in selected:
        progress("stage analyze: start")
        run_analysis(out)
        progress("stage analyze: done")
    if "tables" in selected:
        progress("stage tables: start")
        result = out / "results" / "all_results.csv"
        if not result.exists():
            result = run_analysis(out)
        make_tables(result, out / "tables")
        progress(f"stage tables: wrote outputs under {out / 'tables'}")
        progress("stage tables: done")
    progress(f"run complete: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--out")
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = p.parse_args()
    print(run_config(args.config, args.out, args.stage))


if __name__ == "__main__":
    main()
