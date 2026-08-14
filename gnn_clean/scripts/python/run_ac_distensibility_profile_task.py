#!/usr/bin/env python
"""Run one AC Step 3 distensibility-profile task for a selected representative."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_F0_HZ = 2.749420195642236
DEFAULT_ALPHA_VALUES = (0, 1, 2)
DEFAULT_D0_VALUES = tuple(10.0 ** exponent for exponent in (-6, -5, -4, -3, -2, -1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--harmonic-number", type=int, choices=(1, 2), required=True)
    parser.add_argument("--representative-label", default="B1")
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--f0-hz", type=float, default=DEFAULT_F0_HZ)
    parser.add_argument(
        "--arterial-boundary-mode",
        choices=("all", "per_tip_highest_snr"),
        default="all",
    )
    parser.add_argument(
        "--venous-boundary-mode",
        choices=("observed", "rebalance_to_sources"),
        default="observed",
    )
    parser.add_argument("--alpha-values", type=int, nargs="*", default=None)
    parser.add_argument("--d0-values", type=float, nargs="*", default=None)
    parser.add_argument("--no-observed-flow-snr-weighting", action="store_true")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lstsq-backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def d0_values(custom_values: list[float] | None = None) -> list[float]:
    if custom_values:
        return [float(value) for value in custom_values]
    return [10.0 ** (-6.0 + step / 10.0) for step in range(51)]


def d0_token(value: float) -> str:
    return f"{value:.12g}".replace("+", "").replace(".", "p").replace("-", "m")


def select_representative(step2_root: Path, representative_label: str) -> dict[str, str]:
    reps_path = step2_root / "ac_physics_weight_representatives.csv"
    rows = read_rows(reps_path)
    matches = [
        row
        for row in rows
        if str(row.get("plot_label", "")).strip() == representative_label
        or (
            representative_label == "B1"
            and str(row.get("selection_category", "")).strip() == "balanced"
            and normalize_bool(row.get("selected_representative", ""))
        )
    ]
    if not matches:
        raise RuntimeError(f"No representative matching {representative_label!r} found in {reps_path}.")
    matches.sort(
        key=lambda row: (
            safe_float(row.get("selection_rank_within_regime")),
            safe_float(row.get("selection_score")),
            str(row.get("run_name", "")),
        )
    )
    return matches[0]


def build_jobs(alpha_values: list[int], d0_grid: list[float]) -> list[tuple[int, float]]:
    jobs: list[tuple[int, float]] = []
    for alpha in alpha_values:
        for d0 in d0_grid:
            jobs.append((alpha, d0))
    return jobs


def resolve_task_id(explicit_task_id: int | None) -> int:
    if explicit_task_id is not None:
        return int(explicit_task_id)
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        raise RuntimeError("Task id not provided and SLURM_ARRAY_TASK_ID is not set.")
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def command_prefix(conda_env: str | None, python_bin: str) -> list[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, python_bin]
    return [python_bin]


def main() -> None:
    args = parse_args()
    step2_root = args.step2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    graph_path = args.graph_path.expanduser().resolve()
    alpha_values = [int(value) for value in (args.alpha_values if args.alpha_values is not None else DEFAULT_ALPHA_VALUES)]
    d0_grid = d0_values(args.d0_values)

    representative = select_representative(step2_root, args.representative_label)
    task_id = resolve_task_id(args.task_id)
    jobs = build_jobs(alpha_values, d0_grid)
    if task_id < 0 or task_id >= len(jobs):
        raise ValueError(f"Task id {task_id} is out of range for {len(jobs)} jobs.")
    alpha, d0 = jobs[task_id]

    outdir = output_root / args.representative_label / f"alpha_{alpha}" / f"D0_{d0_token(d0)}"
    summary_path = outdir / "summary.csv"
    if summary_path.exists() and not args.overwrite:
        print(f"[skip] {summary_path} already exists")
        return

    outdir.parent.mkdir(parents=True, exist_ok=True)
    b1_run_dir = Path(str(representative["run_dir"])).expanduser().resolve()
    cmd = [
        *command_prefix(args.conda_env, args.python_bin),
        str(PROJECT_ROOT / "scripts" / "python" / "harmonic_stage1_admittance_model_comparison.py"),
        "--graph-path",
        str(graph_path),
        "--b1-run-dir",
        str(b1_run_dir),
        "--harmonic-number",
        str(int(args.harmonic_number)),
        "--f0-hz",
        f"{float(args.f0_hz):.15g}",
        "--pressure-solver-mode",
        "constrained_least_squares",
        "--lambda-q",
        str(representative["lambda_q"]),
        "--lambda-k",
        str(representative["lambda_k"]),
        "--lambda-b",
        str(representative["lambda_b"]),
        "--arterial-boundary-mode",
        str(args.arterial_boundary_mode),
        "--venous-boundary-mode",
        str(args.venous_boundary_mode),
        "--lstsq-backend",
        str(args.lstsq_backend),
        "--device",
        str(args.device),
        "--D0",
        f"{d0:.12g}",
        "--alpha",
        str(alpha),
        "--output-dir",
        str(outdir),
        "--overwrite",
    ]
    if bool(args.require_cuda):
        cmd.append("--require-cuda")
    if bool(args.no_observed_flow_snr_weighting):
        cmd.append("--no-observed-flow-snr-weighting")

    print(
        f"[run] harmonic=H{args.harmonic_number} representative={args.representative_label} "
        f"alpha={alpha} D0={d0:.12g}"
    )
    print(f"[run] step2_root={step2_root}")
    print(f"[run] b1_run_dir={b1_run_dir}")
    print(f"[run] output_dir={outdir}")
    print("[run] command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
