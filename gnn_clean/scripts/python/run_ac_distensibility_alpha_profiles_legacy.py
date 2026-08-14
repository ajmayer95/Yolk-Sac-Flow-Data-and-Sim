#!/usr/bin/env python
"""Run AC Step 3 distensibility-alpha profile sweeps for selected representatives."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP2_ROOT = PROJECT_ROOT / "outputs" / "ac" / "02_physics_weight_sweep"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "03_distensibility_alpha_profiles"
DEFAULT_LABELS = ("F1", "B1", "K1")
DEFAULT_ALPHA_VALUES = (0, 1, 2)
DEFAULT_D0_VALUES = tuple(10.0 ** exponent for exponent in (-6, -5, -4, -3, -2, -1))


def d0_values(custom_values: list[float] | None = None) -> list[float]:
    if custom_values:
        return [float(value) for value in custom_values]
    return [10.0 ** (-6.0 + step / 10.0) for step in range(51)]


def d0_token(value: float) -> str:
    return f"{value:.12g}".replace("+", "").replace(".", "p").replace("-", "m")


def task_lookup(task_id: int, alpha_values: list[int], d0_grid: list[float]) -> tuple[int, float]:
    jobs: list[tuple[int, float]] = []
    for alpha in alpha_values:
        for d0 in d0_grid:
            jobs.append((alpha, d0))
    return jobs[task_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-root", type=Path, default=DEFAULT_STEP2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lstsq-backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--harmonic-numbers", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--representative-labels", nargs="*", default=list(DEFAULT_LABELS))
    parser.add_argument("--alpha-values", type=int, nargs="*", default=None)
    parser.add_argument("--d0-values", type=float, nargs="*", default=None)
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
    parser.add_argument("--no-observed-flow-snr-weighting", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--plot-after", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def expected_summary_path(
    output_root: Path,
    label: str,
    task_id: int,
    alpha_values: list[int],
    d0_grid: list[float],
) -> Path:
    alpha, d0 = task_lookup(task_id, alpha_values, d0_grid)
    return output_root / label / f"alpha_{alpha}" / f"D0_{d0_token(d0)}" / "summary.csv"


def run_command(
    python_bin: str,
    step2_root: Path,
    output_root: Path,
    graph_path: Path,
    harmonic_number: int,
    label: str,
    task_id: int,
    arterial_boundary_mode: str,
    venous_boundary_mode: str,
    use_observed_flow_snr_weighting: bool,
    alpha_values: list[int],
    d0_grid: list[float],
    device: str,
    lstsq_backend: str,
    require_cuda: bool,
) -> list[str]:
    cmd = [
        str(python_bin),
        str(PROJECT_ROOT / "scripts" / "python" / "run_ac_distensibility_profile_task.py"),
        "--step2-root",
        str(step2_root),
        "--output-root",
        str(output_root),
        "--harmonic-number",
        str(int(harmonic_number)),
        "--representative-label",
        str(label),
        "--graph-path",
        str(graph_path),
        "--arterial-boundary-mode",
        str(arterial_boundary_mode),
        "--venous-boundary-mode",
        str(venous_boundary_mode),
        "--task-id",
        str(int(task_id)),
        "--alpha-values",
        *[str(int(value)) for value in alpha_values],
        "--d0-values",
        *[f"{float(value):.12g}" for value in d0_grid],
        "--python-bin",
        str(python_bin),
        "--device",
        str(device),
        "--lstsq-backend",
        str(lstsq_backend),
        "--overwrite",
    ]
    if require_cuda:
        cmd.append("--require-cuda")
    if not use_observed_flow_snr_weighting:
        cmd.append("--no-observed-flow-snr-weighting")
    return cmd


def post_commands(python_bin: str, harmonic_root: Path) -> list[list[str]]:
    return [
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "analyze_ac_distensibility_alpha_profiles.py"),
            "--input-root",
            str(harmonic_root),
        ],
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "plot_ac_distensibility_alpha_profiles.py"),
            "--input-root",
            str(harmonic_root),
        ],
    ]


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    step2_root = args.step2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    graph_path = args.graph_path.expanduser().resolve()
    harmonic_numbers = sorted({int(value) for value in args.harmonic_numbers})
    representative_labels = [str(value) for value in args.representative_labels]
    alpha_values = [int(value) for value in (args.alpha_values if args.alpha_values is not None else DEFAULT_ALPHA_VALUES)]
    d0_grid = d0_values(args.d0_values)
    jobs: list[tuple[int, str, int]] = []
    total_task_ids = len(d0_grid) * len(alpha_values)
    for harmonic_number in harmonic_numbers:
        for label in representative_labels:
            for task_id in range(total_task_ids):
                jobs.append((harmonic_number, label, task_id))

    if not args.aggregate_only:
        shard_jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]
        print(
            f"Shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(shard_jobs)} of {len(jobs)} AC distensibility-profile runs"
        )
        for harmonic_number, label, task_id in shard_jobs:
            harmonic_step2_root = step2_root / f"H{int(harmonic_number)}"
            harmonic_output_root = output_root / f"H{int(harmonic_number)}"
            summary_path = expected_summary_path(
                harmonic_output_root,
                label,
                task_id,
                alpha_values,
                d0_grid,
            )
            if summary_path.exists() and not args.overwrite:
                print(f"[skip] H{harmonic_number} {label} task={task_id}: {summary_path}")
                continue
            cmd = run_command(
                python_bin=args.python_bin,
                step2_root=harmonic_step2_root,
                output_root=harmonic_output_root,
                graph_path=graph_path,
                harmonic_number=harmonic_number,
                label=label,
                task_id=task_id,
                arterial_boundary_mode=args.arterial_boundary_mode,
                venous_boundary_mode=args.venous_boundary_mode,
                use_observed_flow_snr_weighting=not bool(args.no_observed_flow_snr_weighting),
                alpha_values=alpha_values,
                d0_grid=d0_grid,
                device=args.device,
                lstsq_backend=args.lstsq_backend,
                require_cuda=bool(args.require_cuda),
            )
            print("[run]", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    if args.aggregate_after or args.plot_after or args.aggregate_only:
        for harmonic_number in harmonic_numbers:
            harmonic_output_root = output_root / f"H{int(harmonic_number)}"
            commands = post_commands(args.python_bin, harmonic_output_root)
            for idx, cmd in enumerate(commands):
                if idx == 1 and not args.plot_after:
                    continue
                print("[post]", " ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
