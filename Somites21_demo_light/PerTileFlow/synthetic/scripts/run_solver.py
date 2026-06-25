#!/usr/bin/env python
"""Run a deterministic or Bayesian tile/mosaic distensibility solver."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from distensibility.experiment import METHODS, run_solver_experiment
from distensibility.simulation import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Synthetic .npz dataset")
    parser.add_argument(
        "--method", choices=sorted(METHODS), required=True
    )
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
        "--harmonics",
        choices=("h1", "h1_h2"),
        default=None,
    )
    parser.add_argument(
        "--tiles",
        type=int,
        nargs="+",
        default=None,
        help="Optional tile IDs for tile-specific methods.",
    )
    parser.add_argument("--num-D0", type=int, default=None)
    parser.add_argument("--num-alpha", type=int, default=None)
    parser.add_argument(
        "--pressure-field",
        type=Path,
        default=None,
        help=(
            "GNN run directory or pressure_field.npz. DC supplies the "
            "boundary-pressure prior; stored H1/H2 fields may be fixed."
        ),
    )
    parser.add_argument(
        "--pressure-mode",
        choices=("off", "absolute", "scaled"),
        default="scaled",
        help="How to condition unknown harmonic boundary pressures.",
    )
    parser.add_argument(
        "--pressure-weight",
        type=float,
        default=1.0,
        help="Strength of the DC pressure-shape regularization.",
    )
    parser.add_argument(
        "--pressure-sigma-pa",
        type=float,
        default=0.0,
        help="Pressure prior sigma in Pa; <=0 uses a robust automatic scale.",
    )
    parser.add_argument(
        "--do-not-fix-stored-harmonics",
        action="store_true",
        help="Use stored H1/H2 fields as priors instead of exact fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    config["solver"]["method"] = args.method
    config["solver"]["spatial_mode"] = (
        "tile" if args.method.endswith("_tile") else "whole_mosaic"
    )
    if args.alpha_mode:
        config["solver"]["alpha_mode"] = args.alpha_mode
    if args.alpha is not None:
        config["solver"]["prescribed_alpha"] = float(args.alpha)
    if config["solver"]["alpha_mode"] == "prescribed" and config["solver"][
        "prescribed_alpha"
    ] is None:
        raise SystemExit("--alpha is required when alpha is prescribed")
    if args.harmonics:
        config["solver"]["harmonics_used"] = (
            [1] if args.harmonics == "h1" else [1, 2]
        )
    if args.num_D0:
        config["parameter_grid"]["num_D0"] = int(args.num_D0)
    if args.num_alpha:
        config["parameter_grid"]["num_alpha"] = int(args.num_alpha)
    run_solver_experiment(
        PROJECT_ROOT,
        args.dataset,
        config,
        args.method,
        config_path,
        tile_ids=args.tiles,
        pressure_field_path=args.pressure_field,
        pressure_mode=args.pressure_mode,
        pressure_weight=args.pressure_weight,
        pressure_sigma_pa=args.pressure_sigma_pa,
        fix_available_harmonics=not args.do_not_fix_stored_harmonics,
    )


if __name__ == "__main__":
    main()
