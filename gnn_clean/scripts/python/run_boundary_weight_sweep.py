#!/usr/bin/env python
"""Run and aggregate the Poiseuille boundary-weight calibration sweep."""

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

from utils import write_yaml


DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "harmonized_scaled_dataset.gpickle"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "dc" / "01_boundary_parameter_calibration"
)
BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "poiseuille_only_baseline.py"
LAMBDA_B_VALUES = (1.0, 10.0, 100.0, 1000.0)
EXPECTED_RUN_FILES = (
    "summary.csv",
    "summary.yaml",
    "node_predictions.csv",
    "edge_predictions.csv",
)
COMBINED_COLUMNS = [
    "lambda_b",
    "flow_rmse_nl_s",
    "flow_mae_nl_s",
    "flow_nrmse_median",
    "kirchhoff_rms_per_internal_node_nl_s",
    "kirchhoff_mae_per_internal_node_nl_s",
    "kirchhoff_p95_abs_nl_s",
    "kirchhoff_max_abs_nl_s",
    "arterial_equality_residual_pa",
    "venous_equality_residual_pa",
    "boundary_residual_rms_pa",
    "boundary_residual_max_pa",
    "pressure_min_pa",
    "pressure_max_pa",
    "pressure_range_pa",
    "solver_success",
    "runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viscosity-pa-s", type=float, default=3.5e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def run_name_for_lambda(lambda_b: float) -> str:
    if float(lambda_b).is_integer():
        return f"lambda_b_{int(lambda_b)}"
    label = str(lambda_b).replace(".", "p")
    return f"lambda_b_{label}"


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows[0]


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


def parse_scalar(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and text not in {"nan", "inf", "-inf"}:
        return int(number)
    return number


def parse_summary_row(row: dict[str, str]) -> dict[str, object]:
    return {key: parse_scalar(value) for key, value in row.items()}


def verify_run_outputs(run_dir: Path) -> None:
    missing = [name for name in EXPECTED_RUN_FILES if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected outputs in {run_dir}: {', '.join(missing)}")


def build_command(args: argparse.Namespace, lambda_b: float, run_name: str) -> list[str]:
    command = [
        str(args.python_bin),
        str(BASELINE_SCRIPT),
        str(args.graph.expanduser().resolve()),
        "--output-dir",
        str(args.output_root.expanduser().resolve()),
        "--run-name",
        run_name,
        "--device",
        str(args.device),
        "--viscosity-pa-s",
        str(float(args.viscosity_pa_s)),
        "--dc-solve-mode",
        "reduced-soft-constrained-lstsq",
        "--arterial-flow-mode",
        "dataset",
        "--pressure-constraint",
        "equal-a-equal-v",
        "--lambda-kirchhoff",
        "1.0",
        "--lambda-pressure-constraints",
        str(float(lambda_b)),
        "--lambda-flow-residual",
        "1.0",
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config.expanduser().resolve())])
    return command


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for lambda_b in LAMBDA_B_VALUES:
        run_name = run_name_for_lambda(lambda_b)
        run_dir = output_root / run_name
        should_run = args.overwrite or not all(
            (run_dir / name).exists() for name in EXPECTED_RUN_FILES
        )
        if should_run and not args.aggregate_only:
            command = build_command(args, lambda_b=lambda_b, run_name=run_name)
            print(f"[run] {run_name}: {' '.join(command)}")
            subprocess.run(command, check=True)
        verify_run_outputs(run_dir)
        summary = parse_summary_row(read_single_row_csv(run_dir / "summary.csv"))
        summary["lambda_b"] = float(lambda_b)
        summary["run_name"] = run_name
        summary["run_dir"] = str(run_dir)
        run_rows.append(summary)

    run_rows.sort(key=lambda row: float(row["lambda_b"]))
    combined_csv = output_root / "boundary_weight_summary.csv"
    combined_yaml = output_root / "boundary_weight_summary.yaml"
    write_rows(combined_csv, run_rows)
    write_yaml(
        combined_yaml,
        {
            "graph_path": str(args.graph.expanduser().resolve()),
            "output_root": str(output_root),
            "lambda_q": 1.0,
            "lambda_k": 1.0,
            "pressure_constraints": ["equal-a-equal-v"],
            "runs": run_rows,
            "reported_columns": COMBINED_COLUMNS,
        },
    )

    print(f"[ok] Combined summary written to {combined_csv}")
    print("[ok] Key columns:")
    for column in COMBINED_COLUMNS:
        print(f"  - {column}")


if __name__ == "__main__":
    main()
