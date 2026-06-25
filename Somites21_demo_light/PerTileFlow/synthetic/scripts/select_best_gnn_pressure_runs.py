#!/usr/bin/env python
"""Select the best pressure-producing GNN run per model/harmonic configuration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_run_dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["physics_informed_gnn", "edge_local_mlp"],
    )
    parser.add_argument(
        "--group-by",
        choices=("model_harmonic", "model"),
        default="model_harmonic",
    )
    parser.add_argument(
        "--metric",
        default="best_validation_loss",
        help="Dot-separated metrics.json field to minimize.",
    )
    return parser.parse_args()


def nested_value(payload: dict, dotted: str) -> float:
    value = payload
    for part in dotted.split("."):
        value = value[part]
    return float(value)


def produces_pressure(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            key = (
                "predicted_pressure_pa"
                if "predicted_pressure_pa" in archive
                else "pressure_field_pa"
            )
            return key in archive and bool(np.isfinite(archive[key]).any())
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    args = parse_args()
    root = args.dataset_run_dir.expanduser().resolve()
    allowed_models = set(args.models)
    best: dict[tuple[str, str], tuple[float, Path]] = {}

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        pressure_path = run_dir / "pressure_field.npz"
        if not (
            config_path.is_file()
            and metrics_path.is_file()
            and pressure_path.is_file()
        ):
            continue
        try:
            config = json.loads(config_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            model = str(config["model_name"])
            harmonic_mode = str(config["harmonic_mode"])
            score = nested_value(metrics, args.metric)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if (
            model not in allowed_models
            or not math.isfinite(score)
            or not produces_pressure(pressure_path)
        ):
            continue
        key = (
            (model, harmonic_mode)
            if args.group_by == "model_harmonic"
            else (model, "best_overall")
        )
        candidate = (score, run_dir)
        if key not in best or candidate < best[key]:
            best[key] = candidate

    if not best:
        raise SystemExit(f"No complete pressure-producing runs found under {root}")

    for (model, group), (score, run_dir) in sorted(best.items()):
        config = json.loads((run_dir / "config.yaml").read_text())
        harmonic_mode = str(config["harmonic_mode"])
        print(f"{run_dir}\t{score:.17g}\t{model}\t{harmonic_mode}")


if __name__ == "__main__":
    main()
