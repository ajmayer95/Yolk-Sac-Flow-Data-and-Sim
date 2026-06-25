#!/usr/bin/env python
"""Generate the configured whole-mosaic synthetic distensibility grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from distensibility.simulation import generate_experiment_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help=(
            "Mosaic graph override. Defaults to the configured processed graph, "
            "then the known Somites21_demo source graph."
        ),
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=None,
        help="PerTileFlow root containing pertile.analysis.transmission_line.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing parameter-matched .npz datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_experiment_grid(
        PROJECT_ROOT,
        graph_path=args.graph,
        simulation_root=args.simulation_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
