from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.write_text(df.to_markdown(index=False), encoding="utf-8")
    return path


def make_tables(results_csv: str | Path, out_dir: str | Path = "tables") -> list[Path]:
    df = pd.read_csv(results_csv)
    out = Path(out_dir)
    paths = []

    notation = pd.DataFrame([
        ("N", "Number of corpus vectors"),
        ("d", "Vector dimension"),
        ("K", "Final top-K result size"),
        ("R", "Fixed candidate rerank budget"),
        ("h", "Number of selected query-head coordinates"),
        ("rho", "Target retained query energy"),
        ("alpha_t", "Tail-risk coefficient"),
        ("alpha_q", "Optional quantization-risk coefficient"),
    ], columns=["symbol", "meaning"])
    paths.append(_write(notation, out / "tab01_notation.md"))

    if {"dataset", "N", "Q", "d", "mean_h"}.issubset(df.columns):
        dataset = df.groupby("dataset", dropna=False).agg(
            N=("N", "max"), Q=("Q", "max"), d=("d", "max"), mean_h_rho090=("mean_h", "mean")
        ).reset_index()
        paths.append(_write(dataset, out / "tab02_datasets.md"))

    configs = pd.DataFrame([
        ("ETAR 8+32", "etar", "rho,R,h_max,lambda_tail,int8 scan query", True, True, "append/delete-marker/update"),
        ("ETAR 8+8", "etar", "rho,R,h_max,lambda_tail,int8 scan query", True, True, "append/delete-marker/update"),
        ("full int8", "etar", "full-coordinate quantized scan", True, True, "append/delete-marker/update"),
        ("FAISS", "faiss", "index type,nlist,nprobe,efSearch", True, True, "varies"),
        ("HNSWlib", "hnswlib", "M,efConstruction,efSearch", True, True, "add/delete"),
        ("Annoy", "annoy", "trees,search_k", True, False, "rebuild"),
        ("ScaNN", "scann", "leaves,probes,reorder", True, True, "static/rebuild"),
    ], columns=["method", "implementation", "main_parameters", "online_timing", "batch_timing", "update_support"])
    paths.append(_write(configs, out / "tab03_baseline_configs.md"))

    if "online_or_batch" in df.columns:
        online = df[df["online_or_batch"] == "online"].copy()
    else:
        online = pd.DataFrame()
    if not online.empty:
        keep = [
            "dataset",
            "method",
            "recall_at_k",
            "latency_ms_mean",
            "latency_ms_p50",
            "latency_ms_p95",
            "storage_backend",
            "storage_preset",
            "storage_mb_estimate",
            "rss_after_build_mb",
            "rss_build_delta_mb",
            "hot_bytes_per_vector",
            "cold_bytes_per_query",
        ]
        keep = [c for c in keep if c in online.columns]
        main = online.sort_values(["dataset", "method", "recall_at_k"], ascending=[True, True, False])[keep]
        paths.append(_write(main, out / "tab04_online_results.md"))

        ablation_methods = ["exact_f32", "full_i8", "reduced_fixed", "reduced_energy", "etar", "etar_qrisk", "etar_iq", "etar_8p32", "etar_8p8"]
        ablation = online[online["method"].isin(ablation_methods)]
        keep = [c for c in ["dataset", "method", "recall_at_k", "candidate_coverage", "mean_h", "mean_retained_mass", "latency_ms_mean", "hot_bytes_per_vector"] if c in ablation.columns]
        paths.append(_write(ablation[keep], out / "tab05_ablation.md"))

        storage = online[[c for c in [
            "dataset",
            "method",
            "storage_backend",
            "storage_preset",
            "storage_capacity",
            "full_precision_bytes_estimate",
            "scan_bytes_estimate",
            "rerank_code_bytes_estimate",
            "metadata_bytes_estimate",
            "storage_mb_estimate",
            "rss_after_build_mb",
            "rss_build_delta_mb",
            "hot_bytes_per_vector",
            "cold_bytes_per_query",
        ] if c in online.columns]].drop_duplicates()
        paths.append(_write(storage, out / "tab06_storage_traffic.md"))

    dynamic_path = Path(results_csv).with_name("dynamic_results.csv")
    if dynamic_path.exists():
        dyn = pd.read_csv(dynamic_path)
        if "row_type" in dyn.columns:
            dyn = dyn[dyn["row_type"].isin(["summary", "checkpoint"])].copy()
        paths.append(_write(dyn, out / "tab07_dynamic_cost.md"))
    return paths
