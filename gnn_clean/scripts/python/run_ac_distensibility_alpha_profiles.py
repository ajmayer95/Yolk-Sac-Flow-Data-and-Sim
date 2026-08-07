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


def d0_values() -> list[float]:
    return [10.0 ** (-6.0 + step / 10.0) for step in range(51)]


def d0_token(value: float) -> str:
    return f"{value:.12g}".replace("+", "").replace(".", "p").replace("-", "m")


def task_lookup(task_id: int) -> tuple[int, float]:
    jobs: list[tuple[int, float]] = []
    for alpha in (0, 1, 2):
        for d0 in d0_values():
            jobs.append((alpha, d0))
    return jobs[task_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-root", type=Path, default=DEFAULT_STEP2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--harmonic-numbers", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--representative-labels", nargs="*", default=list(DEFAULT_LABELS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--plot-after", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def expected_summary_path(output_root: Path, label: str, task_id: int) -> Path:
    alpha, d0 = task_lookup(task_id)
    return output_root / label / f"alpha_{alpha}" / f"D0_{d0_token(d0)}" / "summary.csv"


def run_command(
    python_bin: str,
    step2_root: Path,
    output_root: Path,
    graph_path: Path,
    harmonic_number: int,
    label: str,
    task_id: int,
) -> list[str]:
    return [
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
        "--task-id",
        str(int(task_id)),
        "--python-bin",
        str(python_bin),
        "--overwrite",
    ]


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
    jobs: list[tuple[int, str, int]] = []
    total_task_ids = len(d0_values()) * 3
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
            summary_path = expected_summary_path(harmonic_output_root, label, task_id)
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
