#!/usr/bin/env python
"""Aggregate AC boundary-parameter calibration runs across harmonics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "01_boundary_parameter_calibration"
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_ROOT / "boundary_parameter_calibration_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--harmonic-number", type=int, choices=(1, 2), default=None)
    return parser.parse_args()


def harmonic_number_from_dir(path: Path) -> int | None:
    name = path.name
    if name.startswith("H") and name[1:].isdigit():
        return int(name[1:])
    return None


def lambda_b_from_dir(path: Path) -> float | None:
    name = path.name
    if not name.startswith("lambda_b_"):
        return None
    token = name[len("lambda_b_") :]
    try:
        return float(token.replace("p", "."))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    if not input_root.exists():
        parent = input_root.parent
        sibling_matches = []
        if parent.exists():
            sibling_matches = sorted(
                path.name
                for path in parent.glob(f"{input_root.name}*")
                if path.is_dir()
            )
        message = f"AC boundary calibration input root does not exist: {input_root}"
        if sibling_matches:
            message += ". Candidate directories: " + ", ".join(sibling_matches)
        raise FileNotFoundError(message)

    rows: list[dict[str, object]] = []
    for harmonic_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        harmonic_number = harmonic_number_from_dir(harmonic_dir)
        if harmonic_number is None:
            continue
        if args.harmonic_number is not None and harmonic_number != int(args.harmonic_number):
            continue
        for run_dir in sorted(path for path in harmonic_dir.iterdir() if path.is_dir()):
            lambda_b = lambda_b_from_dir(run_dir)
            if lambda_b is None:
                continue
            summary_path = run_dir / "summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path)
            if df.empty:
                continue
            if "harmonic_number_dir" in df.columns:
                df["harmonic_number_dir"] = harmonic_number
            else:
                df.insert(0, "harmonic_number_dir", harmonic_number)
            if "lambda_b_dir" in df.columns:
                df["lambda_b_dir"] = lambda_b
            else:
                df.insert(1, "lambda_b_dir", lambda_b)
            df["run_dir"] = str(run_dir)
            rows.extend(df.to_dict(orient="records"))

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError(f"No AC boundary calibration summaries found under {input_root}")

    summary_df["harmonic_number_dir"] = pd.to_numeric(summary_df["harmonic_number_dir"], errors="coerce")
    summary_df["lambda_b_dir"] = pd.to_numeric(summary_df["lambda_b_dir"], errors="coerce")
    summary_df = summary_df.sort_values(["harmonic_number_dir", "model_name", "lambda_b_dir"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    if args.harmonic_number is None:
        print(f"[ok] Wrote AC boundary calibration summary to {output_csv}")
    else:
        print(f"[ok] Wrote H{int(args.harmonic_number)} AC boundary calibration summary to {output_csv}")


if __name__ == "__main__":
    main()
