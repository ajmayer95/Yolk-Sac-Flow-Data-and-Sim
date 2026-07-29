#!/usr/bin/env python
"""Plot the Poiseuille boundary-weight calibration sweep."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "dc"
    / "01_boundary_parameter_calibration"
    / "boundary_weight_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "dc" / "01_boundary_parameter_calibration" / "figures"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    numeric_columns = [
        "lambda_b",
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
        "boundary_residual_rms_pa",
        "boundary_residual_max_pa",
        "pressure_range_pa",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("lambda_b").reset_index(drop=True)
    return df


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def base_axes(title: str, x_label: str, y_label: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def plot_single_metric(
    df: pd.DataFrame,
    output_dir: Path,
    y_column: str,
    stem: str,
    title: str,
    y_label: str,
) -> None:
    fig, ax = base_axes(title, r"$\lambda_B$", y_label)
    ax.plot(
        df["lambda_b"],
        df[y_column],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
    )
    save_figure(fig, output_dir, stem)


def plot_boundary_residuals(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = base_axes(
        "Boundary Residual vs Boundary Weight",
        r"$\lambda_B$",
        "Residual (Pa)",
    )
    ax.plot(
        df["lambda_b"],
        df["boundary_residual_rms_pa"],
        marker="o",
        linewidth=2.0,
        label="RMS",
        color="#1f77b4",
    )
    ax.plot(
        df["lambda_b"],
        df["boundary_residual_max_pa"],
        marker="s",
        linewidth=2.0,
        label="Max abs",
        color="#ff7f0e",
    )
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "boundary_residual_vs_lambda_b")


def main() -> None:
    args = parse_args()
    df = load_summary(args.input_csv.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()

    plot_boundary_residuals(df, output_dir)
    plot_single_metric(
        df,
        output_dir,
        y_column="flow_rmse_nl_s",
        stem="flow_rmse_vs_lambda_b",
        title="Flow RMSE vs Boundary Weight",
        y_label="Flow RMSE (nL/s)",
    )
    plot_single_metric(
        df,
        output_dir,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        stem="kirchhoff_rms_vs_lambda_b",
        title="Kirchhoff RMS vs Boundary Weight",
        y_label="Kirchhoff RMS per internal node (nL/s)",
    )
    plot_single_metric(
        df,
        output_dir,
        y_column="pressure_range_pa",
        stem="pressure_range_vs_lambda_b",
        title="Pressure Range vs Boundary Weight",
        y_label="Pressure range (Pa)",
    )


if __name__ == "__main__":
    main()
