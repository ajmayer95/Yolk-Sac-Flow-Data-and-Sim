#!/usr/bin/env python
"""Plot AC boundary-parameter calibration metrics versus lambda_B."""

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
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "ac" / "01_boundary_parameter_calibration" / "boundary_parameter_calibration_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ac" / "01_boundary_parameter_calibration" / "figures"
METRICS = (
    ("complex_flow_rmse_nl_s", "Complex Flow RMSE (nL/s)", "complex_flow_rmse_vs_lambda_b"),
    ("kirchhoff_rms_per_internal_node_nl_s", "Kirchhoff RMS (nL/s)", "kirchhoff_rms_vs_lambda_b"),
    ("arterial_pressure_phase_difference_deg", "A-node Phase Difference (deg)", "arterial_phase_difference_vs_lambda_b"),
)
MODEL_COLORS = {
    "full_ideal": "#1f77b4",
    "taylor_ideal": "#d62728",
    "taylor_dc_transferred": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--harmonic-number", type=int, choices=(1, 2), default=None)
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv.expanduser().resolve())
    if df.empty:
        raise ValueError(f"No rows found in {args.input_csv}")

    df["harmonic_number_dir"] = pd.to_numeric(df["harmonic_number_dir"], errors="coerce")
    df["lambda_b_dir"] = pd.to_numeric(df["lambda_b_dir"], errors="coerce")
    if args.harmonic_number is not None:
        df = df[df["harmonic_number_dir"] == int(args.harmonic_number)].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for H{int(args.harmonic_number)} in {args.input_csv}"
            )
    output_dir = args.output_dir.expanduser().resolve()

    harmonics = sorted(df["harmonic_number_dir"].dropna().unique())
    for metric_key, ylabel, stem in METRICS:
        fig, axes = plt.subplots(1, len(harmonics), figsize=(6.2 * len(harmonics), 4.8), constrained_layout=True, sharey=True)
        if len(harmonics) == 1:
            axes = [axes]
        for ax, harmonic in zip(axes, harmonics, strict=True):
            harmonic_df = df[df["harmonic_number_dir"] == harmonic].copy()
            for model_name in sorted(harmonic_df["model_name"].dropna().astype(str).unique()):
                model_df = harmonic_df[harmonic_df["model_name"] == model_name].sort_values("lambda_b_dir")
                color = MODEL_COLORS.get(model_name)
                ax.plot(
                    model_df["lambda_b_dir"],
                    model_df[metric_key],
                    marker="o",
                    linewidth=1.8,
                    label=model_name,
                    color=color,
                )
            ax.set_xscale("log")
            ax.set_xlabel(r"$\lambda_B$")
            ax.set_title(f"H{int(harmonic)}")
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False, loc="best")
        suffix = f"_H{int(args.harmonic_number)}" if args.harmonic_number is not None else ""
        save_figure(fig, output_dir / f"{stem}{suffix}.png")

    if args.harmonic_number is None:
        print(f"[ok] Wrote AC boundary calibration plots to {output_dir}")
    else:
        print(f"[ok] Wrote H{int(args.harmonic_number)} AC boundary calibration plots to {output_dir}")


if __name__ == "__main__":
    main()
