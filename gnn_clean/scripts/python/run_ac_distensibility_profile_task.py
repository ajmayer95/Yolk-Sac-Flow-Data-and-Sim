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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--harmonic-number", type=int, choices=(1, 2), required=True)
    parser.add_argument("--representative-label", default="B1")
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--f0-hz", type=float, default=DEFAULT_F0_HZ)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--conda-env", default=None)
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


def d0_values() -> list[float]:
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


def build_jobs() -> list[tuple[int, float]]:
    jobs: list[tuple[int, float]] = []
    for alpha in (0, 1, 2):
        for d0 in d0_values():
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

    representative = select_representative(step2_root, args.representative_label)
    task_id = resolve_task_id(args.task_id)
    jobs = build_jobs()
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
        "--lstsq-backend",
        "numpy",
        "--D0",
        f"{d0:.12g}",
        "--alpha",
        str(alpha),
        "--output-dir",
        str(outdir),
        "--overwrite",
    ]

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
