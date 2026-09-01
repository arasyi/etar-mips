from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


SWEEP_SKIP_KEYS = {"name", "family", "method", "index", "modes", "source", "path", "download_url", "notes"}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def stable_config_id(prefix: str, cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    name = str(cfg.get("name") or prefix).replace(" ", "_")
    return f"{name}-{digest}"


def expand_sweep(item: dict[str, Any], skip_keys: set[str] | None = None) -> list[dict[str, Any]]:
    skip = SWEEP_SKIP_KEYS if skip_keys is None else skip_keys
    sweep_keys = [k for k, v in item.items() if k not in skip and isinstance(v, list)]
    if not sweep_keys:
        return [dict(item)]
    expanded = []
    for values in itertools.product(*[item[k] for k in sweep_keys]):
        cfg = dict(item)
        for key, value in zip(sweep_keys, values):
            cfg[key] = value
        expanded.append(cfg)
    return expanded


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    cfg["datasets"] = {"items": normalize_dataset_items(raw)}
    cfg["benchmarks"] = {"methods": normalize_method_items(raw)}
    for key in ("name", "output_dir", "seed", "k", "ground_truth_k", "stages", "limits", "dynamic"):
        if key in raw:
            cfg[key] = raw[key]
    return cfg


def normalize_dataset_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
    data = raw.get("datasets", {}) or {}
    if data.get("items"):
        items = [dict(x) for x in data["items"]]
    else:
        items = []
        seed = int(raw.get("seed", 123))
        for preset in data.get("generate", []):
            items.extend(_preset_items(str(preset), seed))
        for path in data.get("ann", []):
            items.append({"name": Path(path).stem, "source": "hdf5", "path": str(path)})
    out = []
    for item in items:
        source = item.get("source", "hdf5")
        if source == "generated_sweep":
            kinds = item.get("kinds", ["dense_gaussian"])
            n_values = item.get("n_values", [item.get("n", 5000)])
            d_values = item.get("d_values", [item.get("d", 128)])
            seeds = item.get("seeds", [item.get("seed", raw.get("seed", 123))])
            queries = item.get("queries", 100)
            for kind, n, d, seed in itertools.product(kinds, n_values, d_values, seeds):
                suffix = f"{kind}_N{n}_d{d}_q{queries}_s{seed}"
                name = f"{item['name']}_{suffix}" if item.get("name") else suffix
                out.append({
                    "name": name,
                    "source": "generated",
                    "kind": kind,
                    "n": int(n),
                    "d": int(d),
                    "queries": int(queries),
                    "seed": int(seed),
                })
        else:
            out.append(item)
    return out


def _preset_items(preset: str, seed: int) -> list[dict[str, Any]]:
    if preset == "smoke":
        specs = [
            ("dense_gaussian", 1500, 64, 50),
            ("sparse10", 1500, 64, 50),
            ("mixed_dropout_heavytail", 1500, 64, 50),
        ]
    elif preset == "synthetic":
        kinds = ["dense_gaussian", "sparse10", "mixed_dropout_heavytail", "norm_heavy", "low_margin", "adversarial_tail"]
        specs = [(kind, 20000, 128, 200) for kind in kinds]
    elif preset == "synthetic-large":
        specs = [("dense_gaussian", 500000, 64, 100)]
    else:
        raise ValueError(f"unknown generated preset {preset!r}")
    return [
        {"name": f"{kind}_N{n}_d{d}", "source": "generated", "kind": kind, "n": n, "d": d, "queries": q, "seed": seed + i}
        for i, (kind, n, d, q) in enumerate(specs)
    ]


def normalize_method_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
    bench = raw.get("benchmarks", {}) or {}
    if bench.get("methods"):
        methods = [dict(x) for x in bench["methods"]]
    else:
        backend = bench.get("backend", "auto")
        r_values = bench.get("r_values", [10, 50, 100])
        rho_values = bench.get("rho_values", [0.8, 0.9])
        h_values = bench.get("h_max_values", [16, 32, 64])
        methods = [
            {"name": "exact_f32", "family": "etar", "method": "exact_f32", "backend": backend, "modes": ["online"]},
            {"name": "full_i8", "family": "etar", "method": "full_i8", "backend": backend, "r": r_values, "h_max": None, "modes": ["online"]},
            {"name": "reduced_fixed", "family": "etar", "method": "reduced_fixed", "backend": backend, "r": r_values, "h_max": h_values, "modes": ["online"]},
            {"name": "reduced_energy", "family": "etar", "method": "reduced_energy", "backend": backend, "r": r_values, "rho": rho_values, "h_max": h_values, "modes": ["online"]},
            {"name": "etar_fq", "family": "etar", "method": "etar", "backend": backend, "r": r_values, "rho": rho_values, "h_max": h_values, "query_mode": "float", "modes": ["online"]},
            {"name": "etar_qrisk", "family": "etar", "method": "etar_qrisk", "backend": backend, "r": r_values, "rho": rho_values, "h_max": h_values, "alpha_q": 0.1, "modes": ["online"]},
            {"name": "etar_iq", "family": "etar", "method": "etar_iq", "backend": backend, "r": r_values, "rho": rho_values, "h_max": h_values, "query_mode": "int8", "modes": ["online"]},
        ]
        if bench.get("ann", True):
            methods.extend([
                {"name": "faiss_flat", "family": "faiss", "index": "flat", "modes": ["online"]},
                {"name": "faiss_hnsw", "family": "faiss", "index": "hnsw", "M": 32, "ef_search": 128, "modes": ["online"]},
                {"name": "faiss_ivf", "family": "faiss", "index": "ivf", "nlist": 128, "nprobe": 16, "modes": ["online"]},
                {"name": "faiss_ivfpq", "family": "faiss", "index": "ivfpq", "nlist": 128, "nprobe": 16, "m": 16, "modes": ["online"]},
                {"name": "hnswlib", "family": "hnswlib", "M": 32, "ef_construction": 200, "ef_search": 128, "modes": ["online"]},
                {"name": "annoy", "family": "annoy", "trees": 50, "search_k": -1, "modes": ["online"]},
                {"name": "scann", "family": "scann", "leaves": 200, "leaves_to_search": 20, "reorder": 50, "modes": ["online"]},
            ])
    expanded = []
    for method in methods:
        for cfg in expand_sweep(method):
            cfg.setdefault("modes", ["online"])
            cfg["config_id"] = stable_config_id(str(cfg.get("name", cfg.get("family", "method"))), cfg)
            expanded.append(cfg)
    return expanded
