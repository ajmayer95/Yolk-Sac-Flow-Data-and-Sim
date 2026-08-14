#!/usr/bin/env python
"""Run Step 3 pressure-constraint sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pressure_constraint_sensitivity_lib import (
    CONSTRAINT_ORDER,
    CONSTRAINT_SPECS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPRESENTATIVE_CSV,
    DEFAULT_REPRESENTATIVE_LABELS_CSV,
    expected_run_files,
    gnn_constraint_run_name,
    launcher_metadata_path,
    poiseuille_constraint_run_name,
    representative_label,
)
from utils import load_yaml, write_yaml
from workflow_selection import resolve_dc_representative_row, resolve_dc_step2_row


GNN_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "gnn_flow.py"
POISEUILLE_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "poiseuille_only_baseline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative-csv", type=Path, default=DEFAULT_REPRESENTATIVE_CSV)
    parser.add_argument("--representative-labels-csv", type=Path, default=DEFAULT_REPRESENTATIVE_LABELS_CSV)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--viscosity-pa-s", type=float, default=3.5e-3)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("both", "gnn", "poiseuille"), default="both")
    parser.add_argument("--constraints", nargs="*", choices=CONSTRAINT_ORDER, default=None)
    parser.add_argument("--representative-labels", nargs="*", default=None)
    parser.add_argument("--lambda-q", type=float, default=None)
    parser.add_argument("--lambda-k", type=float, default=None)
    parser.add_argument("--lambda-delta", type=float, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def default_step2_root_for_output(output_root: Path) -> Path:
    return output_root.parent / "02_physics_weight_sweep"


def resolve_step2_inputs(
    output_root: Path,
    representative_csv: Path,
    representative_labels_csv: Path | None,
) -> tuple[Path, Path | None]:
    default_rep_csv = DEFAULT_REPRESENTATIVE_CSV.expanduser().resolve()
    resolved_rep_csv = representative_csv.expanduser().resolve()
    resolved_labels_csv = (
        representative_labels_csv.expanduser().resolve()
        if representative_labels_csv is not None
        else None
    )
    if resolved_rep_csv == default_rep_csv:
        inferred_step2_root = default_step2_root_for_output(output_root)
        inferred_rep_csv = inferred_step2_root / "representative_configurations.csv"
        if inferred_rep_csv.exists():
            resolved_rep_csv = inferred_rep_csv
            if resolved_labels_csv is None or resolved_labels_csv == DEFAULT_REPRESENTATIVE_LABELS_CSV.expanduser().resolve():
                inferred_labels_csv = inferred_step2_root / "figures" / "representative_plot_labels.csv"
                resolved_labels_csv = inferred_labels_csv
    return resolved_rep_csv, resolved_labels_csv


def read_representatives(rep_csv: Path, labels_csv: Path | None) -> pd.DataFrame:
    reps = pd.read_csv(rep_csv)
    required = {
        "run_name",
        "lambda_q",
        "lambda_k",
        "lambda_delta",
        "selection_category",
        "selection_rank_within_regime",
    }
    missing = sorted(required - set(reps.columns))
    if missing:
        raise ValueError(f"Representative CSV missing required columns: {missing}")
    if labels_csv is not None and labels_csv.exists():
        labels_df = pd.read_csv(labels_csv)
        if {"run_name", "plot_label"} <= set(labels_df.columns):
            reps = reps.merge(labels_df[["run_name", "plot_label"]], on="run_name", how="left")
    if "plot_label" not in reps.columns:
        reps["plot_label"] = [
            representative_label(row["selection_category"], row["selection_rank_within_regime"])
            for _, row in reps.iterrows()
        ]
    reps["plot_label"] = reps["plot_label"].fillna(
        pd.Series(
            [
                representative_label(row["selection_category"], row["selection_rank_within_regime"])
                for _, row in reps.iterrows()
            ]
        )
    )
    reps = reps.drop_duplicates(subset=["run_name"]).copy()
    return reps


def select_representatives(args: argparse.Namespace, reps: pd.DataFrame) -> pd.DataFrame:
    explicit_lambda = any(
        value is not None for value in (args.lambda_q, args.lambda_k, args.lambda_delta)
    )
    if explicit_lambda:
        step2_root = args.representative_csv.expanduser().resolve().parent
        row = resolve_dc_step2_row(
            step2_root,
            lambda_q=args.lambda_q,
            lambda_k=args.lambda_k,
            lambda_delta=args.lambda_delta,
        )
        selected = reps[reps["run_name"].astype(str) == str(row.get("run_name", ""))].copy()
        if selected.empty:
            raise ValueError(
                "Resolved Step 2 run is missing from the loaded representative table."
            )
        return selected

    if args.representative_labels:
        requested = set(args.representative_labels)
        selected = reps[reps["plot_label"].astype(str).isin(requested)].copy()
        missing = sorted(requested - set(selected["plot_label"].astype(str)))
        if missing:
            raise ValueError(
                "Representative labels not found in representative_configurations.csv: "
                + ", ".join(missing)
            )
        return selected

    default_row = resolve_dc_representative_row(args.representative_csv)
    selected = reps[reps["run_name"].astype(str) == str(default_row.get("run_name", ""))].copy()
    if selected.empty:
        raise ValueError(
            "Balanced default Step 2 representative is missing from the loaded representative table."
        )
    return selected


def build_gnn_override(base_config: dict, rep_row: pd.Series, constraint_type: str, args: argparse.Namespace) -> dict:
    spec = CONSTRAINT_SPECS[constraint_type]
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
            "pressure_constraints": list(spec["pressure_constraints"]),
            "alpha_pa": spec["alpha_pa"],
            "pressure_detach": False,
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 1.0,
            "pressure_solver_lambda_kirchhoff": 1.0,
            "pressure_solver_lambda_pressure_constraints": 100.0,
        },
        "gnn_outer_losses": {
            "flow": float(rep_row["lambda_q"]),
            "kirchhoff": float(rep_row["lambda_k"]),
            "boundary": 100.0,
            "delta_l2": float(rep_row["lambda_delta"]),
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
    lambda_q: float,
    lambda_k: float,
    constraint_type: str,
    args: argparse.Namespace,
) -> list[str]:
    spec = CONSTRAINT_SPECS[constraint_type]
    cmd = [
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
        "--lambda-kirchhoff",
        str(float(lambda_k)),
        "--lambda-pressure-constraints",
        "100.0",
        "--lambda-flow-residual",
        str(float(lambda_q)),
    ]
    for constraint in spec["pressure_constraints"]:
        if constraint == "gauge-only":
            cmd.extend(["--pressure-constraint", "gauge_only"])
        else:
            cmd.extend(["--pressure-constraint", str(constraint)])
    if spec["alpha_pa"] is not None:
        cmd.extend(["--alpha-pa", str(float(spec["alpha_pa"]))])
    return cmd


def run_is_complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in expected_run_files())


def build_run_manifest(args: argparse.Namespace, reps: pd.DataFrame) -> list[dict[str, object]]:
    constraints = list(args.constraints) if args.constraints else list(CONSTRAINT_ORDER)

    rows: list[dict[str, object]] = []
    if args.mode in {"both", "gnn"}:
        for _, rep_row in reps.iterrows():
            for constraint_type in constraints:
                rows.append(
                    {
                        "model_family": "gnn",
                        "run_name": gnn_constraint_run_name(str(rep_row["run_name"]), constraint_type),
                        "parent_step2_run_name": str(rep_row["run_name"]),
                        "representative_label": str(rep_row["plot_label"]),
                        "selection_category": str(rep_row["selection_category"]),
                        "selection_rank_within_regime": int(float(rep_row["selection_rank_within_regime"])),
                        "lambda_q": float(rep_row["lambda_q"]),
                        "lambda_k": float(rep_row["lambda_k"]),
                        "lambda_b": 100.0,
                        "lambda_delta": float(rep_row["lambda_delta"]),
                        "pressure_constraint_type": constraint_type,
                    }
                )
    if args.mode in {"both", "poiseuille"}:
        unique_pairs = (
            reps[["lambda_q", "lambda_k"]]
            .drop_duplicates()
            .sort_values(["lambda_q", "lambda_k"])
            .reset_index(drop=True)
        )
        for _, pair_row in unique_pairs.iterrows():
            for constraint_type in constraints:
                rows.append(
                    {
                        "model_family": "poiseuille_baseline",
                        "run_name": poiseuille_constraint_run_name(
                            float(pair_row["lambda_q"]),
                            float(pair_row["lambda_k"]),
                            constraint_type,
                        ),
                        "parent_step2_run_name": "",
                        "representative_label": "",
                        "selection_category": "",
                        "selection_rank_within_regime": float("nan"),
                        "lambda_q": float(pair_row["lambda_q"]),
                        "lambda_k": float(pair_row["lambda_k"]),
                        "lambda_b": 100.0,
                        "lambda_delta": float("nan"),
                        "pressure_constraint_type": constraint_type,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    output_root = args.output_root.expanduser().resolve()
    graph = args.graph.expanduser().resolve()
    representative_csv, representative_labels_csv = resolve_step2_inputs(
        output_root,
        args.representative_csv,
        args.representative_labels_csv,
    )
    if not representative_csv.exists():
        inferred_step2_root = default_step2_root_for_output(output_root)
        raise FileNotFoundError(
            "Missing Step 2 representative CSV. "
            f"Looked for {representative_csv}. "
            "Run Step 2 aggregation first, or pass --representative-csv explicitly. "
            f"For this output root, the expected sibling Step 2 directory is {inferred_step2_root}."
        )
    reps = read_representatives(
        representative_csv,
        representative_labels_csv,
    )
    reps = select_representatives(args, reps)

    if args.aggregate_only:
        cmd = [
            str(args.python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "analyze_pressure_constraint_sensitivity.py"),
            "--input-root",
            str(output_root),
        ]
        print("Command:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        return

    manifest = build_run_manifest(args, reps)
    shard_manifest = [
        row for idx, row in enumerate(manifest) if idx % args.num_shards == args.shard_index
    ]

    base_config: dict = {}
    if args.base_config is not None:
        base_config = load_yaml(args.base_config.expanduser().resolve())

    generated_config_root = output_root / "_generated_configs"
    generated_config_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []

    for row in shard_manifest:
        model_family = str(row["model_family"])
        run_dir = output_root / ("gnn" if model_family == "gnn" else "poiseuille") / str(row["run_name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = CONSTRAINT_SPECS[str(row["pressure_constraint_type"])]
        launcher_payload = {
            **row,
            "graph_path": str(graph),
            "device": args.device if model_family == "gnn" else "cpu",
            "require_cuda": bool(args.require_cuda) if model_family == "gnn" else False,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "viscosity_pa_s": float(args.viscosity_pa_s),
            "alpha_pa": spec["alpha_pa"],
            "pressure_constraints": list(spec["pressure_constraints"]),
            "constraint_description": spec["description"],
        }
        write_yaml(launcher_metadata_path(run_dir), launcher_payload)

        if run_is_complete(run_dir) and not args.overwrite:
            print(f"Skipping completed run: {run_dir}")
            continue

        if model_family == "gnn":
            rep_row = reps.loc[reps["run_name"].astype(str) == str(row["parent_step2_run_name"])].iloc[0]
            override = build_gnn_override(base_config, rep_row, str(row["pressure_constraint_type"]), args)
            config_path = generated_config_root / f"{row['run_name']}.yaml"
            write_yaml(config_path, override)
            cmd = gnn_command(args.python_bin, graph, run_dir, config_path, args)
        else:
            cmd = poiseuille_command(
                args.python_bin,
                graph,
                run_dir,
                float(row["lambda_q"]),
                float(row["lambda_k"]),
                str(row["pressure_constraint_type"]),
                args,
            )
        print("Command:", " ".join(str(part) for part in cmd))
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append({"run_name": row["run_name"], "returncode": int(exc.returncode)})

    if failures:
        failed_path = output_root / "failed_runs.csv"
        with failed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_name", "returncode"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"Wrote failures to {failed_path}")

    if args.aggregate_after and not args.dry_run:
        subprocess.run(
            [
                str(args.python_bin),
                str(PROJECT_ROOT / "scripts" / "python" / "analyze_pressure_constraint_sensitivity.py"),
                "--input-root",
                str(output_root),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
