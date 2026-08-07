#!/usr/bin/env python
"""Run AC Step 2 physics-weight sweeps for a compatible graph workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from workflow_selection import resolve_balanced_dc_run_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DC_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "02_physics_weight_sweep"
DEFAULT_LAMBDA_B_BY_HARMONIC = {1: 10.0, 2: 100.0}
WEIGHT_VALUES = (0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--dc-step2-root", type=Path, default=DEFAULT_DC_STEP2_ROOT)
    parser.add_argument("--b1-run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--harmonic-numbers", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--lambda-q-values", type=float, nargs="*", default=list(WEIGHT_VALUES))
    parser.add_argument("--lambda-k-values", type=float, nargs="*", default=list(WEIGHT_VALUES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--plot-after", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def build_run_dir(output_root: Path, harmonic_number: int, lambda_q: float, lambda_k: float) -> Path:
    return output_root / f"H{int(harmonic_number)}" / f"q_{token(lambda_q)}__k_{token(lambda_k)}"


def run_is_complete(run_dir: Path) -> bool:
    return (run_dir / "summary.csv").exists()


def run_command(
    python_bin: str,
    graph_path: Path,
    b1_run_dir: Path,
    output_dir: Path,
    harmonic_number: int,
    lambda_q: float,
    lambda_k: float,
    lambda_b: float,
) -> list[str]:
    return [
        str(python_bin),
        str(PROJECT_ROOT / "scripts" / "python" / "harmonic_stage1_admittance_model_comparison.py"),
        "--graph-path",
        str(graph_path),
        "--b1-run-dir",
        str(b1_run_dir),
        "--harmonic-number",
        str(int(harmonic_number)),
        "--pressure-solver-mode",
        "constrained_least_squares",
        "--lambda-q",
        f"{float(lambda_q):g}",
        "--lambda-k",
        f"{float(lambda_k):g}",
        "--lambda-b",
        f"{float(lambda_b):g}",
        "--lstsq-backend",
        "numpy",
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]


def post_commands(python_bin: str, harmonic_root: Path) -> list[list[str]]:
    return [
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "analyze_ac_physics_weight_sweep.py"),
            "--input-root",
            str(harmonic_root),
        ],
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "plot_ac_physics_weight_sweep.py"),
            "--input-root",
            str(harmonic_root),
            "--output-dir",
            str(harmonic_root / "figures"),
        ],
    ]


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    graph_path = args.graph_path.expanduser().resolve()
    b1_run_dir = resolve_balanced_dc_run_dir(args.dc_step2_root, args.b1_run_dir)
    output_root = args.output_root.expanduser().resolve()
    harmonic_numbers = sorted({int(value) for value in args.harmonic_numbers})
    lambda_q_values = [float(value) for value in args.lambda_q_values]
    lambda_k_values = [float(value) for value in args.lambda_k_values]

    jobs: list[tuple[int, float, float, float]] = []
    for harmonic_number in harmonic_numbers:
        if harmonic_number not in DEFAULT_LAMBDA_B_BY_HARMONIC:
            raise ValueError(f"Unsupported harmonic number: {harmonic_number}")
        lambda_b = DEFAULT_LAMBDA_B_BY_HARMONIC[harmonic_number]
        for lambda_q in lambda_q_values:
            for lambda_k in lambda_k_values:
                jobs.append((harmonic_number, lambda_q, lambda_k, lambda_b))

    if not args.aggregate_only:
        shard_jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]
        print(
            f"Shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(shard_jobs)} of {len(jobs)} AC physics-weight runs"
        )
        for harmonic_number, lambda_q, lambda_k, lambda_b in shard_jobs:
            run_dir = build_run_dir(output_root, harmonic_number, lambda_q, lambda_k)
            if run_is_complete(run_dir) and not args.overwrite:
                print(f"[skip] H{harmonic_number} q={lambda_q:g} k={lambda_k:g}: {run_dir}")
                continue
            cmd = run_command(
                python_bin=args.python_bin,
                graph_path=graph_path,
                b1_run_dir=b1_run_dir,
                output_dir=run_dir,
                harmonic_number=harmonic_number,
                lambda_q=lambda_q,
                lambda_k=lambda_k,
                lambda_b=lambda_b,
            )
            print("[run]", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    if args.aggregate_after or args.plot_after or args.aggregate_only:
        for harmonic_number in harmonic_numbers:
            harmonic_root = output_root / f"H{int(harmonic_number)}"
            commands = post_commands(args.python_bin, harmonic_root)
            for idx, cmd in enumerate(commands):
                if idx == 1 and not args.plot_after:
                    continue
                print("[post]", " ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
