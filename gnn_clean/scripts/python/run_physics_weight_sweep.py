#!/usr/bin/env python
"""Run the Step 2 physics-constraint weight sweep."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physics_weight_sweep_lib import (
    ensure_unique_run_names,
    expected_run_files,
    generate_gnn_run_configs,
    generate_poiseuille_run_configs,
    launcher_metadata_path,
)
from utils import load_yaml, write_yaml


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
GNN_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "gnn_flow.py"
POISEUILLE_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "poiseuille_only_baseline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--viscosity-pa-s", type=float, default=3.5e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("both", "gnn-only", "poiseuille-only"),
        default="both",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_gnn_override(base_config: dict, run_cfg: dict[str, object], args: argparse.Namespace) -> dict:
    override = {
        "K": 2,
        "model": {
            "hidden_dim": 64,
            "correction_bound": 0.5,
            "correction_min": -0.5,
            "correction_max": 0.5,
            "correction_parameterization": "tanh",
            "initialize_decoder_near_zero": True,
        },
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
        },
        "physics": {
            "arterial_flow_mode": "dataset",
            "solver_kind": "constrained_dc_equal_A_equal_V",
            "pressure_constraints": ["equal-a-equal-v"],
            "pressure_detach": False,
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 1.0,
            "pressure_solver_lambda_kirchhoff": 1.0,
            "pressure_solver_lambda_pressure_constraints": 100.0,
        },
        "gnn_outer_losses": {
            "flow": float(run_cfg["lambda_q"]),
            "kirchhoff": float(run_cfg["lambda_k"]),
            "boundary": 100.0,
            "delta_l2": float(run_cfg["lambda_delta"]),
            "delta_smooth": 0.0,
            "pressure_shape": 0.0,
        },
    }

    def deep_update(base: dict, update: dict) -> dict:
        merged = dict(base)
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_update(merged[key], value)
            else:
                merged[key] = value
        return merged

    return deep_update(base_config, override)


def gnn_command(
    python_bin: str,
    graph: Path,
    output_dir: Path,
    config_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(python_bin),
        str(GNN_SCRIPT),
        str(graph),
        "--output-dir",
        str(output_dir.parent),
        "--run-name",
        output_dir.name,
        "--preset",
        "solver_QKB_outer_QKBdelta",
        "--device",
        str(args.device),
        "--require-cuda",
        "--epochs",
        str(int(args.epochs)),
        "--seed",
        str(int(args.seed)),
        "--viscosity-pa-s",
        str(float(args.viscosity_pa_s)),
        "--config",
        str(config_path),
        "--no-pressure-detach",
    ]


def poiseuille_command(
    python_bin: str,
    graph: Path,
    output_dir: Path,
    run_cfg: dict[str, object],
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(python_bin),
        str(POISEUILLE_SCRIPT),
        str(graph),
        "--output-dir",
        str(output_dir.parent),
        "--run-name",
        output_dir.name,
        "--device",
        "cpu",
        "--viscosity-pa-s",
        str(float(args.viscosity_pa_s)),
        "--dc-solve-mode",
        "reduced-soft-constrained-lstsq",
        "--arterial-flow-mode",
        "dataset",
        "--pressure-constraint",
        "equal-a-equal-v",
        "--lambda-kirchhoff",
        str(float(run_cfg["lambda_k"])),
        "--lambda-pressure-constraints",
        "100.0",
        "--lambda-flow-residual",
        str(float(run_cfg["lambda_q"])),
    ]


def run_is_complete(run_dir: Path, model_family: str) -> bool:
    return all((run_dir / name).exists() for name in expected_run_files(model_family))


def run_set(mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if mode in {"both", "gnn-only"}:
        rows.extend(generate_gnn_run_configs())
    if mode in {"both", "poiseuille-only"}:
        rows.extend(generate_poiseuille_run_configs())
    ensure_unique_run_names(rows)
    return rows


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    output_root = args.output_root.expanduser().resolve()
    graph = args.graph.expanduser().resolve()
    gnn_root = output_root / "gnn"
    poiseuille_root = output_root / "poiseuille"
    generated_config_root = output_root / "_generated_configs"
    generated_config_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        analyze_cmd = [
            str(args.python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "analyze_physics_weight_sweep.py"),
            "--input-root",
            str(output_root),
        ]
        print("Command:", " ".join(analyze_cmd))
        if not args.dry_run:
            subprocess.run(analyze_cmd, check=True)
        return

    base_config: dict = {}
    if args.base_config is not None:
        base_config = load_yaml(args.base_config.expanduser().resolve())

    requested_runs = run_set(args.mode)
    shard_runs = [
        row for idx, row in enumerate(requested_runs) if idx % args.num_shards == args.shard_index
    ]
    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: "
        f"{len(shard_runs)} of {len(requested_runs)} runs"
    )

    failures: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for run_cfg in shard_runs:
        model_family = str(run_cfg["model_family"])
        base_root = gnn_root if model_family == "gnn" else poiseuille_root
        run_dir = base_root / str(run_cfg["run_name"])
        run_dir.mkdir(parents=True, exist_ok=True)

        launcher_payload = {
            **run_cfg,
            "graph_path": str(graph),
            "device": args.device if model_family == "gnn" else "cpu",
            "require_cuda": bool(args.require_cuda) if model_family == "gnn" else False,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "viscosity_pa_s": float(args.viscosity_pa_s),
        }
        write_yaml(launcher_metadata_path(run_dir), launcher_payload)

        if run_is_complete(run_dir, model_family) and not args.overwrite:
            manifest_rows.append({**launcher_payload, "status": "skipped_complete", "run_dir": str(run_dir)})
            continue

        if model_family == "gnn":
            config_path = generated_config_root / f"{run_cfg['run_name']}.yaml"
            write_yaml(config_path, build_gnn_override(base_config, run_cfg, args))
            cmd = gnn_command(
                python_bin=args.python_bin,
                graph=graph,
                output_dir=run_dir,
                config_path=config_path,
                args=args,
            )
        else:
            cmd = poiseuille_command(
                python_bin=args.python_bin,
                graph=graph,
                output_dir=run_dir,
                run_cfg=run_cfg,
                args=args,
            )

        print("Command:", " ".join(cmd))
        if args.dry_run:
            manifest_rows.append({**launcher_payload, "status": "dry_run", "run_dir": str(run_dir)})
            continue

        try:
            subprocess.run(cmd, check=True)
            status = "completed" if run_is_complete(run_dir, model_family) else "missing_outputs"
            manifest_rows.append({**launcher_payload, "status": status, "run_dir": str(run_dir)})
            if status != "completed":
                failures.append({**launcher_payload, "failure_reason": status, "run_dir": str(run_dir)})
        except subprocess.CalledProcessError as exc:
            failures.append(
                {
                    **launcher_payload,
                    "failure_reason": f"exit_code_{exc.returncode}",
                    "run_dir": str(run_dir),
                }
            )
            manifest_rows.append({**launcher_payload, "status": "failed", "run_dir": str(run_dir)})

    write_rows(output_root / "launcher_manifest.csv", manifest_rows)
    if failures:
        write_rows(output_root / "launcher_failures.csv", failures)

    if args.aggregate_after:
        analyze_cmd = [
            str(args.python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "analyze_physics_weight_sweep.py"),
            "--input-root",
            str(output_root),
        ]
        print("Command:", " ".join(analyze_cmd))
        if not args.dry_run:
            subprocess.run(analyze_cmd, check=True)


if __name__ == "__main__":
    main()
