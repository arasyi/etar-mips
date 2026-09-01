from __future__ import annotations

import argparse
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_FILES = {
    "smoke": Path("results/online_results.csv"),
    "static": Path("results/online_results.csv"),
    "dynamic": Path("results/dynamic_results.csv"),
    "compaction": Path("results/dynamic_results.csv"),
    "scaling": Path("results/online_results.csv"),
    "scaling-rss": Path("results/fresh_rss_summary.csv"),
}

BUILD_RESULT_FILE = Path("results/dynamic_build_results.csv")
ONLINE_BUILD_RESULT_FILE = Path("results/online_build_results.csv")
SCALING_RSS_BUILD_RESULT_FILE = Path("results/fresh_rss_build_results.csv")

ONLINE_KEYS = [
    "dataset_name",
    "dataset_short",
    "dataset_family",
    "method",
    "config_id",
    "family",
    "index",
    "rho",
    "R",
    "r",
    "h",
    "h_max",
    "storage_preset",
    "storage_backend",
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

DYNAMIC_KEYS = ["workload", "method", "row_type", "operation"]

SCALING_RSS_KEYS = ["N", "method"]

ONLINE_BUILD_META = [
    "dataset_name",
    "dataset_short",
    "dataset_family",
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

BUILD_VALUE_COLS = [
    "index_build_ms",
    "build_ms",
    "index_build_s",
    "build_s",
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


def _dataset_name(path: object) -> str:
    return os.path.basename(str(path)).replace(".hdf5", "")


def _dataset_short(name: str) -> str:
    if "dense_gaussian" in name:
        return "Dense"
    if "sparse10" in name:
        return "Sparse"
    if "mixed_dropout_heavytail" in name:
        return "Mixed"
    if "norm_heavy" in name:
        return "Norm-heavy"
    return name.replace("-euclidean", "").replace("-angular", "").replace("-dot", "")


def _dataset_family(name: str) -> str:
    return "Synthetic" if name.startswith(("conference_synthetic", "wfiot_synthetic")) else "ANN"


def _repeat_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _read_result_file(root: Path, target: str, rel: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    target_dir = root / target
    if not target_dir.exists():
        return pd.DataFrame()
    for repeat_idx, run_dir in enumerate(sorted(target_dir.glob("run_*"), key=_repeat_sort_key), start=1):
        result = run_dir / rel
        if not result.exists():
            continue
        df = pd.read_csv(result, low_memory=False)
        df["repeat_id"] = repeat_idx
        df["repeat_name"] = run_dir.name
        df["repeat_dir"] = str(run_dir)
        if "dataset" in df.columns:
            df["dataset_name"] = df["dataset"].map(_dataset_name)
            df["dataset_short"] = df["dataset_name"].map(_dataset_short)
            df["dataset_family"] = df["dataset_name"].map(_dataset_family)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _read_target(root: Path, target: str) -> pd.DataFrame:
    return _read_result_file(root, target, RESULT_FILES[target])


def _normalize_build_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "index_build_ms" in out.columns and "index_build_s" not in out.columns:
        out["index_build_s"] = pd.to_numeric(out["index_build_ms"], errors="coerce") / 1000.0
    if "index_build_s" in out.columns and "index_build_ms" not in out.columns:
        out["index_build_ms"] = pd.to_numeric(out["index_build_s"], errors="coerce") * 1000.0
    if "build_ms" in out.columns and "build_s" not in out.columns:
        out["build_s"] = pd.to_numeric(out["build_ms"], errors="coerce") / 1000.0
    if "build_s" in out.columns and "build_ms" not in out.columns:
        out["build_ms"] = pd.to_numeric(out["build_s"], errors="coerce") * 1000.0
    return out


def _online_build_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = _normalize_build_units(df)
    cols = [c for c in ONLINE_BUILD_META + BUILD_VALUE_COLS + ["repeat_id", "repeat_name", "repeat_dir"] if c in out.columns]
    if not cols:
        return pd.DataFrame()
    return out[cols].copy()


def _scaling_rss_build_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = _normalize_build_units(df)
    if "operation" not in out.columns:
        out["operation"] = "build"
    meta = ["N", "method", "operation", "repeat_id", "repeat_name", "repeat_dir"]
    cols = [c for c in meta + BUILD_VALUE_COLS if c in out.columns]
    if not cols:
        return pd.DataFrame()
    return out[cols].copy()


def _write_build_outputs(build_df: pd.DataFrame, target: str, out: Path, group_cols: list[str]) -> None:
    if build_df.empty:
        return
    build_df.to_csv(out / f"{target}_build_all_repeats.csv", index=False)
    summary = _summarize(build_df, group_cols)
    if not summary.empty:
        summary.to_csv(out / f"{target}_build_summary.csv", index=False)


def _numeric_columns(df: pd.DataFrame, group_cols: list[str]) -> list[str]:
    excluded = set(group_cols) | {"repeat_id"}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    group_cols = [col for col in group_cols if col in df.columns]
    numeric_cols = _numeric_columns(df, group_cols)
    if df.empty or not group_cols or not numeric_cols:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False)[numeric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().add_suffix("_n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        summary = pd.concat([mean, std, count], axis=1).reset_index()

    ci_cols = {}
    for col in numeric_cols:
        n_col = f"{col}_n"
        std_col = f"{col}_std"
        if n_col not in summary or std_col not in summary:
            continue
        sem = summary[std_col] / np.sqrt(summary[n_col].clip(lower=1))
        ci_cols[f"{col}_ci95"] = 1.96 * sem
    if ci_cols:
        summary = pd.concat([summary, pd.DataFrame(ci_cols, index=summary.index)], axis=1)
    return summary.copy()


def _write_static_headline(static: pd.DataFrame, out: Path) -> None:
    if static.empty:
        return
    required = {"repeat_id", "dataset_name", "method", "latency_ms_mean", "recall_at_k"}
    if not required.issubset(static.columns):
        return

    rows = []
    for repeat_id, part in static.groupby("repeat_id", dropna=False):
        exact = part[part["method"] == "exact_f32"][["dataset_name", "latency_ms_mean"]].rename(
            columns={"latency_ms_mean": "exact_ms"}
        )
        etar = part[
            (part["method"] == "etar_8p32_sweep")
            & np.isclose(pd.to_numeric(part.get("rho"), errors="coerce"), 0.90)
            & (pd.to_numeric(part.get("R"), errors="coerce") == 100)
        ][["dataset_name", "latency_ms_mean", "recall_at_k", "candidate_coverage"]].rename(
            columns={
                "latency_ms_mean": "etar_ms",
                "recall_at_k": "etar_recall",
                "candidate_coverage": "etar_coverage",
            }
        )
        merged = exact.merge(etar, on="dataset_name", how="inner")
        if merged.empty:
            continue
        merged["speedup"] = merged["exact_ms"] / merged["etar_ms"]
        rows.append(
            {
                "repeat_id": repeat_id,
                "datasets": int(merged["dataset_name"].nunique()),
                "exact_mean_ms": float(merged["exact_ms"].mean()),
                "etar_mean_ms": float(merged["etar_ms"].mean()),
                "ratio_of_mean_latencies": float(merged["exact_ms"].mean() / merged["etar_ms"].mean()),
                "mean_per_dataset_speedup": float(merged["speedup"].mean()),
                "etar_mean_recall": float(merged["etar_recall"].mean()),
                "etar_mean_coverage": float(merged["etar_coverage"].mean()) if "etar_coverage" in merged else math.nan,
            }
        )

    by_repeat = pd.DataFrame(rows)
    if by_repeat.empty:
        return
    by_repeat.to_csv(out / "static_representative_speedup_by_repeat.csv", index=False)
    summary = _summarize(
        by_repeat,
        ["datasets"],
    )
    summary.to_csv(out / "static_representative_speedup_summary.csv", index=False)


def aggregate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target in RESULT_FILES:
        df = _read_target(root, target)
        if df.empty:
            continue
        df.to_csv(out / f"{target}_all_repeats.csv", index=False)
        if target in {"smoke", "static", "scaling"}:
            summary = _summarize(df, ONLINE_KEYS)
            if target in {"static", "scaling"}:
                build_df = _read_result_file(root, target, ONLINE_BUILD_RESULT_FILE)
                if build_df.empty:
                    build_df = _online_build_frame(df)
                else:
                    build_df = _online_build_frame(build_df)
                _write_build_outputs(
                    build_df,
                    target,
                    out,
                    ONLINE_KEYS + ["method_impl", "backend", "N", "d", "Q", "K"],
                )
        elif target in {"dynamic", "compaction"}:
            summary = _summarize(df[df.get("row_type", "") == "summary"].copy(), DYNAMIC_KEYS)
            event_summary = _summarize(
                df[df.get("row_type", "") == "event"].copy(),
                ["workload", "method", "operation"],
            )
            if not event_summary.empty:
                event_summary.to_csv(out / f"{target}_event_summary.csv", index=False)
            build_df = _read_result_file(root, target, BUILD_RESULT_FILE)
            if not build_df.empty:
                build_df.to_csv(out / f"{target}_build_all_repeats.csv", index=False)
                build_summary = _summarize(build_df, ["workload", "method", "operation", "status"])
                if not build_summary.empty:
                    build_summary.to_csv(out / f"{target}_build_summary.csv", index=False)
        else:
            summary = _summarize(df, SCALING_RSS_KEYS)
            build_df = _read_result_file(root, target, SCALING_RSS_BUILD_RESULT_FILE)
            if build_df.empty:
                build_df = _scaling_rss_build_frame(df)
            else:
                build_df = _scaling_rss_build_frame(build_df)
            _write_build_outputs(build_df, target, out, ["N", "method", "operation"])
        if not summary.empty:
            summary.to_csv(out / f"{target}_summary.csv", index=False)
        if target == "static":
            _write_static_headline(df, out)
        manifest.append(
            {
                "target": target,
                "repeat_rows": int(len(df)),
                "repeat_count": int(df["repeat_id"].nunique()) if "repeat_id" in df else 0,
            }
        )
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated WF-IoT artifact runs.")
    parser.add_argument("--root", type=Path, default=Path("results/repeats"))
    parser.add_argument("--out", type=Path, default=Path("results/repeats/summary"))
    args = parser.parse_args()
    aggregate(args.root, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
