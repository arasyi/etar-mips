from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_preset
from .experiments import run_ann_baselines, run_dynamic_baselines, run_index_free, run_suite


def generate_data_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "synthetic", "synthetic-large"], default="smoke")
    p.add_argument("--out-dir", default="data/generated")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    for path in generate_preset(args.preset, args.out_dir, args.seed):
        print(path)


def smoke_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["auto", "cpp", "numba"], default="auto")
    p.add_argument("--data-dir", default="data/generated/smoke")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--ann", action="store_true")
    args = p.parse_args()
    if args.generate or not list(Path(args.data_dir).glob("*.hdf5")):
        generate_preset("smoke", Path(args.data_dir).parents[0])
    paths = sorted(Path(args.data_dir).glob("*.hdf5"))
    for path in paths:
        print(f"== {path} ==")
        print(run_index_free(path, backend=args.backend, limit_q=10).to_string(index=False))
        if args.ann:
            print(run_ann_baselines(path, limit_q=10).to_string(index=False))
    print(run_dynamic_baselines().to_string(index=False))


def run_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="results/etar_results.csv")
    p.add_argument("--backend", choices=["auto", "cpp", "numba"], default="auto")
    p.add_argument("--no-ann", action="store_true")
    p.add_argument("--limit-n", type=int)
    p.add_argument("--limit-q", type=int, default=50)
    args = p.parse_args()
    print(run_suite(args.paths, args.out, backend=args.backend, ann=not args.no_ann, limit_n=args.limit_n, limit_q=args.limit_q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["generate-data", "smoke", "run", "run-config"])
    args, rest = p.parse_known_args()
    import sys

    sys.argv = [sys.argv[0], *rest]
    if args.command == "generate-data":
        generate_data_main()
    elif args.command == "smoke":
        smoke_main()
    elif args.command == "run":
        run_main()
    elif args.command == "run-config":
        from .config_runner import main as run_config_main

        run_config_main()


if __name__ == "__main__":
    main()
