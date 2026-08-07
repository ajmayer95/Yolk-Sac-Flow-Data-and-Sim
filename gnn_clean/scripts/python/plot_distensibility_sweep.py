#!/usr/bin/env python
"""Plot D0 sweep metrics for the AC distensibility sweep."""

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
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "ac" / "00_ideal_models" / "distensibility_sweep" / "distensibility_sweep_metrics.csv"
METRICS = (
    ("complex_flow_rmse_nl_s", "Complex Flow RMSE (nL/s)"),
    ("kirchhoff_rms_per_internal_node_nl_s", "Kirchhoff RMS (nL/s)"),
    ("arterial_pressure_phase_difference_deg", "A-node Phase Difference (deg)"),
)
ALPHA_COLORS = {0.0: "#1f77b4", 1.0: "#d62728", 2.0: "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", action="append", default=None)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model(df: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    model_df = df[df["model_name"] == model_name].copy()
    if model_df.empty:
        return
    model_df["D0"] = pd.to_numeric(model_df["D0"], errors="coerce")
    model_df["alpha"] = pd.to_numeric(model_df["alpha"], errors="coerce")
    model_df = model_df.dropna(subset=["D0", "alpha"]).sort_values(["alpha", "D0"])

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(7.4, 10.0), constrained_layout=True, sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    alpha_values = sorted(model_df["alpha"].unique())
    for ax, (metric_key, metric_label) in zip(axes, METRICS, strict=True):
        for alpha in alpha_values:
            subset = model_df[model_df["alpha"] == alpha].sort_values("D0")
            if subset.empty:
                continue
            color = ALPHA_COLORS.get(float(alpha), None)
            ax.plot(
                subset["D0"],
                subset[metric_key],
                linewidth=1.6,
                color=color,
                label=rf"$\alpha = {alpha:g}$",
            )
        ax.set_xscale("log")
        ax.set_ylabel(metric_label)
        ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel(r"$D_0$")
    fig.suptitle(f"D0 Sweep Metrics: {model_name}", fontsize=13)
    save_figure(fig, output_dir / f"d0_sweep_{model_name}.png")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input metrics table not found: {input_path}. "
            "Generate distensibility_sweep_metrics.csv first."
        )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "figures"
    )

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input metrics table is empty: {input_path}")

    model_names = args.model_name if args.model_name else sorted(df["model_name"].dropna().astype(str).unique())
    for model_name in model_names:
        plot_model(df, str(model_name), output_dir)

    print(f"[ok] Wrote D0 sweep plots to {output_dir}")


if __name__ == "__main__":
    main()
