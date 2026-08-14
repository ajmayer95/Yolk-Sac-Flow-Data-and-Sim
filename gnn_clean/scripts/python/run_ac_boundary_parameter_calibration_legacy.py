#!/usr/bin/env python
"""Run AC Step 1 boundary-parameter calibration for a compatible graph workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from workflow_selection import resolve_dc_run_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DC_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "01_boundary_parameter_calibration"
HARMONIC_NUMBERS = (1, 2)
LAMBDA_B_VALUES = (0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--dc-step2-root", type=Path, default=DEFAULT_DC_STEP2_ROOT)
    parser.add_argument("--b1-run-dir", type=Path, default=None)
    parser.add_argument("--lambda-q", type=float, default=None)
    parser.add_argument("--lambda-k", type=float, default=None)
    parser.add_argument("--lambda-delta", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lstsq-backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--harmonic-numbers", type=int, nargs="*", default=list(HARMONIC_NUMBERS))
    parser.add_argument("--lambda-b-values", type=float, nargs="*", default=list(LAMBDA_B_VALUES))
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


def lambda_b_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def build_run_dir(output_root: Path, harmonic_number: int, lambda_b: float) -> Path:
    return output_root / f"H{int(harmonic_number)}" / f"lambda_b_{lambda_b_label(lambda_b)}"


def run_is_complete(run_dir: Path) -> bool:
    return (run_dir / "summary.csv").exists()


def harmonic_stage1_command(
    python_bin: str,
    graph_path: Path,
    b1_run_dir: Path,
    output_dir: Path,
    harmonic_number: int,
    lambda_b: float,
    arterial_boundary_mode: str,
    venous_boundary_mode: str,
    use_observed_flow_snr_weighting: bool,
    device: str,
    lstsq_backend: str,
    require_cuda: bool,
) -> list[str]:
    cmd = [
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
        "1",
        "--lambda-k",
        "1",
        "--lambda-b",
        f"{float(lambda_b):g}",
        "--arterial-boundary-mode",
        str(arterial_boundary_mode),
        "--venous-boundary-mode",
        str(venous_boundary_mode),
        "--lstsq-backend",
        str(lstsq_backend),
        "--device",
        str(device),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    if require_cuda:
        cmd.append("--require-cuda")
    if not use_observed_flow_snr_weighting:
        cmd.append("--no-observed-flow-snr-weighting")
    return cmd


def aggregate_commands(python_bin: str, output_root: Path, harmonic_number: int) -> list[list[str]]:
    summary_csv = output_root / f"boundary_parameter_calibration_summary_H{int(harmonic_number)}.csv"
    return [
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "analyze_ac_boundary_parameter_calibration.py"),
            "--input-root",
            str(output_root),
            "--harmonic-number",
            str(int(harmonic_number)),
            "--output-csv",
            str(summary_csv),
        ],
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "plot_ac_boundary_parameter_calibration.py"),
            "--input-csv",
            str(summary_csv),
            "--harmonic-number",
            str(int(harmonic_number)),
            "--output-dir",
            str(output_root / "figures"),
        ],
    ]


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    graph_path = args.graph_path.expanduser().resolve()
    b1_run_dir = resolve_dc_run_dir(
        args.dc_step2_root,
        explicit_run_dir=args.b1_run_dir,
        lambda_q=args.lambda_q,
        lambda_k=args.lambda_k,
        lambda_delta=args.lambda_delta,
    )
    output_root = args.output_root.expanduser().resolve()
    harmonics = sorted({int(value) for value in args.harmonic_numbers})
    lambda_b_values = [float(value) for value in args.lambda_b_values]

    jobs: list[tuple[int, float]] = []
    for harmonic_number in harmonics:
        if harmonic_number not in HARMONIC_NUMBERS:
            raise ValueError(f"Unsupported harmonic number: {harmonic_number}")
        for lambda_b in lambda_b_values:
            jobs.append((harmonic_number, lambda_b))

    if not args.aggregate_only:
        shard_jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]
        print(
            f"Shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(shard_jobs)} of {len(jobs)} AC boundary-calibration runs"
        )
        for harmonic_number, lambda_b in shard_jobs:
            run_dir = build_run_dir(output_root, harmonic_number, lambda_b)
            if run_is_complete(run_dir) and not args.overwrite:
                print(f"[skip] H{harmonic_number} lambda_b={lambda_b:g}: {run_dir}")
                continue
            cmd = harmonic_stage1_command(
                python_bin=args.python_bin,
                graph_path=graph_path,
                b1_run_dir=b1_run_dir,
                output_dir=run_dir,
                harmonic_number=harmonic_number,
                lambda_b=lambda_b,
                arterial_boundary_mode=args.arterial_boundary_mode,
                venous_boundary_mode=args.venous_boundary_mode,
                use_observed_flow_snr_weighting=not bool(args.no_observed_flow_snr_weighting),
                device=args.device,
                lstsq_backend=args.lstsq_backend,
                require_cuda=bool(args.require_cuda),
            )
            print("[run]", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    if args.aggregate_after or args.plot_after or args.aggregate_only:
        for harmonic_number in harmonics:
            commands = aggregate_commands(args.python_bin, output_root, harmonic_number)
            for idx, cmd in enumerate(commands):
                if idx == 1 and not args.plot_after:
                    continue
                print("[post]", " ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
