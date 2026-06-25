#!/usr/bin/env python
"""Expand and run the configured synthetic GNN experiment grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from distensibility.simulation import load_yaml
from experiment import (
    create_run_directory,
    expand_experiment_grid,
    load_manifest_datasets,
    save_resolved_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gnn_experiments.yaml",
    )
    parser.add_argument("--datasets", type=Path, nargs="*", default=None)
    parser.add_argument("--model", choices=(
        "physics_informed_gnn", "vanilla_gcn", "edge_local_mlp"
    ), default=None)
    parser.add_argument("--K", type=int, nargs="*", default=None)
    parser.add_argument("--harmonic-mode", choices=(
        "dc_only", "dc_h1", "dc_h1_h2"
    ), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base = load_yaml(args.config.expanduser().resolve())
    experiments = expand_experiment_grid(base)
    if args.model:
        experiments = [x for x in experiments if x["model_name"] == args.model]
    if args.K is not None:
        experiments = [x for x in experiments if x["K"] in set(args.K)]
    if args.harmonic_mode:
        experiments = [
            x for x in experiments if x["harmonic_mode"] == args.harmonic_mode
        ]
    datasets = (
        [path.expanduser().resolve() for path in args.datasets]
        if args.datasets
        else load_manifest_datasets(PROJECT_ROOT, base["datasets"]["manifest"])
    )
    total = len(datasets) * len(experiments)
    index = 0
    for dataset in datasets:
        for config in experiments:
            index += 1
            if args.device:
                config["training"]["device"] = args.device
            if args.epochs is not None:
                config["training"]["epochs"] = int(args.epochs)
            run_dir = create_run_directory(PROJECT_ROOT, dataset, config)
            resolved = run_dir / "config.yaml"
            if args.skip_existing and (run_dir / "metrics.json").exists():
                print(f"[{index}/{total}] skip {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            save_resolved_config(resolved, config)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_gnn.py"),
                str(dataset),
                "--config",
                str(resolved),
                "--out-dir",
                str(run_dir),
            ]
            print(f"[{index}/{total}] {' '.join(command)}")
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
