#!/usr/bin/env python
"""Train one synthetic GNN experiment and save comparison-ready artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from experiment import (
    build_model,
    create_run_directory,
    read_resolved_config,
    resolve_device,
    save_outputs,
    set_random_seed,
)
from gnn_training import build_gnn_data, train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_resolved_config(args.config.expanduser().resolve())
    if args.device:
        config["training"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    dataset_path = args.dataset.expanduser().resolve()
    set_random_seed(config["training"]["seed"])
    device = resolve_device(config["training"]["device"])
    data = build_gnn_data(
        dataset_path, config["harmonic_mode"], config["data"]
    )
    model = build_model(data, config).to(device)
    run_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else create_run_directory(PROJECT_ROOT, dataset_path, config)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset={dataset_path.name}")
    print(
        f"model={config['model_name']} K={config['K']} "
        f"harmonics={config['harmonic_mode']} device={device}"
    )
    model, outputs, history, metrics = train_model(
        model, data, config, run_dir / "checkpoint.pt"
    )
    if not (run_dir / "checkpoint.pt").exists():
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    save_outputs(run_dir, data, outputs, metrics, history, config)
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
