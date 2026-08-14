#!/usr/bin/env python
"""Run the AC fixed-admittance D0/alpha distensibility sweep summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from workflow_selection import resolve_dc_run_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DC_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "00_ideal_models" / "distensibility_sweep"
DEFAULT_SCRATCH_ROOT = DEFAULT_OUTPUT_ROOT / "_raw_runs"


def d0_values() -> list[float]:
    return [10.0 ** (-6.0 + step / 10.0) for step in range(51)]


def d0_token(value: float) -> str:
    return f"{value:.12g}".replace("+", "").replace(".", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--dc-step2-root", type=Path, default=DEFAULT_DC_STEP2_ROOT)
    parser.add_argument("--b1-run-dir", type=Path, default=None)
    parser.add_argument("--lambda-q", type=float, default=None)
    parser.add_argument("--lambda-k", type=float, default=None)
    parser.add_argument("--lambda-delta", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--harmonic-number", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lstsq-backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--require-cuda", action="store_true")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-after", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def build_run_dir(scratch_root: Path, harmonic_number: int, alpha: int, d0: float) -> Path:
    return scratch_root / f"H{int(harmonic_number)}" / f"alpha_{int(alpha)}" / f"D0_{d0_token(d0)}"


def run_command(
    python_bin: str,
    graph_path: Path,
    b1_run_dir: Path,
    run_dir: Path,
    harmonic_number: int,
    alpha: int,
    d0: float,
    arterial_boundary_mode: str,
    venous_boundary_mode: str,
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
        "--D0",
        f"{float(d0):.12g}",
        "--alpha",
        str(int(alpha)),
        "--device",
        str(device),
        "--lstsq-backend",
        str(lstsq_backend),
        "--arterial-boundary-mode",
        str(arterial_boundary_mode),
        "--venous-boundary-mode",
        str(venous_boundary_mode),
        "--output-dir",
        str(run_dir),
        "--overwrite",
    ]
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def aggregate_metrics(scratch_root: Path, harmonic_number: int, output_csv: Path) -> None:
    rows: list[dict[str, object]] = []
    for summary_path in sorted((scratch_root / f"H{int(harmonic_number)}").glob("alpha_*/D0_*/summary.csv")):
        summary_df = pd.read_csv(summary_path)
        if summary_df.empty:
            continue
        for _, series in summary_df.iterrows():
            row = {
                "run_name": summary_path.parent.name,
                "source_summary_path": str(summary_path.resolve()),
                "model_name": series.get("model_name"),
                "model_label": series.get("model_label"),
                "harmonic_number": series.get("harmonic_number"),
                "D0": series.get("D0"),
                "alpha": series.get("alpha"),
                "complex_flow_rmse_nl_s": series.get("complex_flow_rmse_nl_s"),
                "kirchhoff_rms_per_internal_node_nl_s": series.get("kirchhoff_rms_per_internal_node_nl_s"),
                "arterial_pressure_phase_difference_deg": series.get("arterial_pressure_phase_difference_deg"),
            }
            rows.append(row)
    if not rows:
        raise FileNotFoundError(
            "No AC Step 0 distensibility summaries were found under "
            f"{scratch_root / f'H{int(harmonic_number)}'}. "
            "This usually means the raw runs were written to a non-shared temporary directory. "
            "Re-run with --scratch-root pointing to a persistent project directory."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


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
    scratch_root = args.scratch_root.expanduser().resolve()
    output_csv = output_root / "distensibility_sweep_metrics.csv"

    jobs: list[tuple[int, float]] = []
    for alpha in (0, 1, 2):
        for d0 in d0_values():
            jobs.append((alpha, d0))

    if not args.aggregate_only:
        shard_jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]
        print(
            f"Shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(shard_jobs)} of {len(jobs)} AC distensibility-sweep runs"
        )
        for alpha, d0 in shard_jobs:
            run_dir = build_run_dir(scratch_root, args.harmonic_number, alpha, d0)
            summary_path = run_dir / "summary.csv"
            if summary_path.exists() and not args.overwrite:
                print(f"[skip] alpha={alpha} D0={d0:.12g}: {summary_path}")
                continue
            cmd = run_command(
                python_bin=args.python_bin,
                graph_path=graph_path,
                b1_run_dir=b1_run_dir,
                run_dir=run_dir,
                harmonic_number=args.harmonic_number,
                alpha=alpha,
                d0=d0,
                arterial_boundary_mode=args.arterial_boundary_mode,
                venous_boundary_mode=args.venous_boundary_mode,
                device=args.device,
                lstsq_backend=args.lstsq_backend,
                require_cuda=bool(args.require_cuda),
            )
            print("[run]", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    print(f"[post] aggregating metrics -> {output_csv}")
    if not args.dry_run:
        aggregate_metrics(scratch_root, args.harmonic_number, output_csv)

    if args.plot_after:
        cmd = [
            str(args.python_bin),
            str(PROJECT_ROOT / "scripts" / "python" / "plot_distensibility_sweep.py"),
            "--input",
            str(output_csv),
            "--output-dir",
            str(output_root / "figures"),
        ]
        print("[post]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
