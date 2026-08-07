#!/usr/bin/env python
"""Extract selected metrics from a harmonic distensibility sweep run summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    summary_path = run_dir / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary file was not produced: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Summary file is empty: {summary_path}")

    keep = df.loc[
        :,
        [
            "model_name",
            "model_label",
            "harmonic_number",
            "D0",
            "alpha",
            "complex_flow_rmse_nl_s",
            "kirchhoff_rms_per_internal_node_nl_s",
            "arterial_pressure_phase_difference_deg",
        ],
    ].copy()
    keep.insert(0, "run_name", str(args.run_name))
    keep.insert(1, "source_summary_path", str(summary_path))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
    print(f"[ok] Appended sweep metrics to {output_csv}")


if __name__ == "__main__":
    main()
