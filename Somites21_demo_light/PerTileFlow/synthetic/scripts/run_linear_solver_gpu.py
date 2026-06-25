#!/usr/bin/env python
"""Run the deterministic solver with batched CUDA solves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from distensibility.experiment import run_solver_experiment
from distensibility.simulation import load_yaml
from models.gpu_tile import solve_tile_gpu


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "solver_base.yaml",
    )
    parser.add_argument(
        "--alpha-mode", choices=("prescribed", "solved"), default=None
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--harmonics", choices=("h1", "h1_h2"), default=None
    )
    parser.add_argument(
        "--spatial-mode",
        choices=("tile", "whole_mosaic"),
        default="tile",
    )
    parser.add_argument("--tiles", type=int, nargs="+", default=None)
    parser.add_argument("--num-D0", type=int, default=None)
    parser.add_argument("--num-alpha", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--pressure-field", type=Path, default=None)
    parser.add_argument(
        "--pressure-mode",
        choices=("off", "absolute", "scaled"),
        default="scaled",
    )
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--pressure-sigma-pa", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    method_name = (
        "linear_tile_gpu"
        if args.spatial_mode == "tile"
        else "linear_mosaic_gpu"
    )
    config["solver"]["method"] = method_name
    config["solver"]["spatial_mode"] = args.spatial_mode
    if args.alpha_mode:
        config["solver"]["alpha_mode"] = args.alpha_mode
    if args.alpha is not None:
        config["solver"]["prescribed_alpha"] = args.alpha
    if (
        config["solver"]["alpha_mode"] == "prescribed"
        and config["solver"]["prescribed_alpha"] is None
    ):
        raise SystemExit("--alpha is required when alpha is prescribed")
    if args.harmonics:
        config["solver"]["harmonics_used"] = (
            [1] if args.harmonics == "h1" else [1, 2]
        )
    if args.num_D0:
        config["parameter_grid"]["num_D0"] = args.num_D0
    if args.num_alpha:
        config["parameter_grid"]["num_alpha"] = args.num_alpha

    def solve(dataset, problem, effective_config, pressure_conditioning=None):
        return solve_tile_gpu(
            dataset,
            problem,
            effective_config,
            method="linear",
            device=args.device,
            chunk_size=args.chunk_size,
            pressure_conditioning=pressure_conditioning,
        )

    run_solver_experiment(
        PROJECT_ROOT,
        args.dataset,
        config,
        method_name,
        config_path,
        tile_ids=args.tiles,
        pressure_field_path=args.pressure_field,
        pressure_mode=args.pressure_mode,
        pressure_weight=args.pressure_weight,
        pressure_sigma_pa=args.pressure_sigma_pa,
        solver_function_override=solve,
        spatial_mode_override=args.spatial_mode,
    )


if __name__ == "__main__":
    main()
