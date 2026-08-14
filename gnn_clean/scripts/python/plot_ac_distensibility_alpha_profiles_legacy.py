#!/usr/bin/env python
"""Plot AC Step 3 distensibility-alpha profile sweep results."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "03_distensibility_alpha_profiles" / "H1"
PREFIX_ORDER = {"F": 0, "B": 1, "K": 2}
PREFIX_DISPLAY = {
    "F": "Flow-Prioritized",
    "B": "Balanced",
    "K": "Kirchhoff-Prioritized",
}
MODEL_LABELS = {
    "full_ideal": "Full Ideal",
    "taylor_ideal": "Taylor Ideal",
    "taylor_dc_transferred": "Taylor DC Transferred",
}
ALPHA_ORDER = (0.0, 1.0, 2.0)
ALPHA_COLORS = {
    0.0: "#1f77b4",
    1.0: "#ff7f0e",
    2.0: "#2ca02c",
}
LINE_WIDTH = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def label_prefix(label: str) -> str:
    return str(label).strip()[:1].upper()


def label_rank(label: str) -> int:
    suffix = str(label).strip()[1:]
    try:
        return int(suffix)
    except ValueError:
        return 9999


def label_sort_key(label: str) -> tuple[int, int, str]:
    prefix = label_prefix(label)
    return (PREFIX_ORDER.get(prefix, 999), label_rank(label), str(label))


def display_label(label: str) -> str:
    prefix = label_prefix(label)
    rank = label_rank(label)
    base = PREFIX_DISPLAY.get(prefix, str(label))
    if rank != 9999:
        return f"{base} {rank}"
    return base


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run analyze_ac_distensibility_alpha_profiles.py first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}.")
    return df


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
        }
    )


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def alpha_label(alpha_value: float) -> str:
    if float(alpha_value).is_integer():
        return rf"$\alpha = {int(alpha_value)}$"
    return rf"$\alpha = {alpha_value:g}$"


def d0_math_label(value: float) -> str:
    if not math.isfinite(value) or value <= 0.0:
        return r"$D_0 = \mathrm{nan}$"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10 ** exponent)
    return rf"$D_0 = {mantissa:.3f} \times 10^{{{exponent}}}$"


def line_legend_handle(color: str) -> Line2D:
    return Line2D([0], [0], color=color, linewidth=LINE_WIDTH, marker="*", markersize=8)


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def apply_truncated_linear_axis(
    ax: plt.Axes,
    axis: str,
    values: np.ndarray,
    *,
    pad_fraction: float = 0.12,
    upper_pad_fraction: float = 0.06,
    nbins: int = 5,
) -> None:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return
    lower_data = float(np.min(finite))
    upper_data = float(np.max(finite))
    span = upper_data - lower_data
    if not math.isfinite(span) or span <= 0.0:
        span = max(abs(lower_data) * 0.05, 1.0e-6)
    lower = lower_data - pad_fraction * span
    upper = upper_data + upper_pad_fraction * span
    locator = MaxNLocator(nbins=nbins, min_n_ticks=max(4, nbins - 1))
    ticks = locator.tick_values(lower, upper)
    ticks = np.asarray([tick for tick in ticks if lower < tick <= upper], dtype=float)
    if axis == "x":
        ax.set_xlim(lower, upper)
        if ticks.size:
            ax.set_xticks(ticks)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.set_ylim(lower, upper)
        if ticks.size:
            ax.set_yticks(ticks)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def apply_metric_y_axis(ax: plt.Axes, values: np.ndarray) -> bool:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return False
    upper_data = float(np.max(finite))
    span = upper_data
    if not math.isfinite(span) or span <= 0.0:
        span = 1.0e-6
    ax.set_ylim(0.0, upper_data + 0.08 * span)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return False


def add_axis_break_marks(ax: plt.Axes, break_x: bool = False, break_y: bool = False) -> None:
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.1)
    if break_x:
        ax.plot((0.03, 0.05), (-0.02, 0.02), **kwargs)
        ax.plot((0.06, 0.08), (-0.02, 0.02), **kwargs)
    if break_y:
        ax.plot((-0.02, 0.02), (0.015, 0.035), **kwargs)
        ax.plot((-0.02, 0.02), (0.045, 0.065), **kwargs)


def add_alpha_legend(ax: plt.Axes) -> None:
    handles = [Line2D([0], [0], color=ALPHA_COLORS[alpha], linewidth=LINE_WIDTH) for alpha in ALPHA_ORDER]
    labels = [alpha_label(alpha) for alpha in ALPHA_ORDER]
    ax.legend(handles, labels, loc="best", frameon=True, facecolor="white", edgecolor="#d0d0d0")


def add_minima_legend(
    ax: plt.Axes,
    minima_df: pd.DataFrame,
) -> None:
    handles: list[Line2D] = []
    labels: list[str] = []
    for alpha in ALPHA_ORDER:
        row_df = minima_df[np.isclose(minima_df["alpha"], alpha, equal_nan=False)].copy()
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        handles.append(line_legend_handle(ALPHA_COLORS[alpha]))
        labels.append(f"{alpha_label(alpha)}, {d0_math_label(float(row['representative_D0']))}")
    if labels:
        ax.legend(handles, labels, loc="lower left", frameon=True, facecolor="white", edgecolor="#d0d0d0")


def add_pareto_legend(
    ax: plt.Axes,
    rep_df: pd.DataFrame,
) -> None:
    handles: list[Line2D] = []
    labels: list[str] = []
    for alpha in ALPHA_ORDER:
        row_df = rep_df[np.isclose(rep_df["alpha"], alpha, equal_nan=False)].copy()
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        handles.append(line_legend_handle(ALPHA_COLORS[alpha]))
        labels.append(f"{alpha_label(alpha)}, {d0_math_label(float(row['representative_D0']))}")
    if labels:
        ax.legend(handles, labels, loc="best", frameon=True, facecolor="white", edgecolor="#d0d0d0")


def add_direction_arrows(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, color: str) -> None:
    if len(xs) < 3:
        return
    arrow_indices = [max(1, len(xs) // 3), max(1, (2 * len(xs)) // 3)]
    used: set[int] = set()
    for idx in arrow_indices:
        idx = min(idx, len(xs) - 1)
        if idx in used or idx <= 0:
            continue
        used.add(idx)
        ax.annotate(
            "",
            xy=(xs[idx], ys[idx]),
            xytext=(xs[idx - 1], ys[idx - 1]),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "shrinkA": 0.0, "shrinkB": 0.0},
        )


def plot_metric_panel(
    ax: plt.Axes,
    label_df: pd.DataFrame,
    minima_df: pd.DataFrame,
    metric: str,
    ylabel: str | None,
    title: str | None,
) -> None:
    for alpha in ALPHA_ORDER:
        subset = label_df[np.isclose(label_df["alpha"], alpha, equal_nan=False)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("D0")
        ax.plot(
            subset["D0"],
            subset[metric],
            linewidth=LINE_WIDTH,
            color=ALPHA_COLORS[alpha],
        )
        chosen_df = minima_df[np.isclose(minima_df["alpha"], alpha, equal_nan=False)].copy()
        if not chosen_df.empty:
            chosen = chosen_df.iloc[0]
            ax.scatter(
                [float(chosen["representative_D0"])],
                [float(chosen["metric_value"])],
                marker="*",
                s=115,
                color=ALPHA_COLORS[alpha],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
    metric_values = pd.to_numeric(label_df[metric], errors="coerce").to_numpy(dtype=float)
    apply_metric_y_axis(ax, metric_values)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_0\ [\mathrm{Pa}^{-1}]$")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    add_minima_legend(ax, minima_df)


def plot_phase_panel(
    ax: plt.Axes,
    label_df: pd.DataFrame,
    ylabel: str | None,
    title: str | None,
) -> None:
    for alpha in ALPHA_ORDER:
        subset = label_df[np.isclose(label_df["alpha"], alpha, equal_nan=False)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("D0")
        ax.plot(
            subset["D0"],
            subset["folded_arterial_phase_difference_deg"],
            linewidth=LINE_WIDTH,
            color=ALPHA_COLORS[alpha],
        )
    phase_values = pd.to_numeric(label_df["folded_arterial_phase_difference_deg"], errors="coerce").to_numpy(dtype=float)
    finite_phase = phase_values[np.isfinite(phase_values)]
    if finite_phase.size:
        upper = float(np.max(finite_phase))
        span = max(upper, 1.0e-6)
        ax.set_ylim(0.0, upper + 0.06 * span)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_0\ [\mathrm{Pa}^{-1}]$")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    add_alpha_legend(ax)


def plot_pareto_panel(
    ax: plt.Axes,
    label_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    ylabel: str | None,
    title: str | None,
) -> None:
    for alpha in ALPHA_ORDER:
        subset = label_df[np.isclose(label_df["alpha"], alpha, equal_nan=False)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("D0")
        xs = subset["kirchhoff_rms_per_internal_node_nl_s"].to_numpy(dtype=float)
        ys = subset["complex_flow_rmse_nl_s"].to_numpy(dtype=float)
        ax.plot(xs, ys, linewidth=LINE_WIDTH, color=ALPHA_COLORS[alpha])
        add_direction_arrows(ax, xs, ys, ALPHA_COLORS[alpha])
        chosen_df = rep_df[np.isclose(rep_df["alpha"], alpha, equal_nan=False)].copy()
        if not chosen_df.empty:
            chosen = chosen_df.iloc[0]
            ax.scatter(
                [float(chosen["kirchhoff_rms_per_internal_node_nl_s"])],
                [float(chosen["complex_flow_rmse_nl_s"])],
                marker="*",
                s=125,
                color=ALPHA_COLORS[alpha],
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
    finite_x = pd.to_numeric(label_df["kirchhoff_rms_per_internal_node_nl_s"], errors="coerce").to_numpy(dtype=float)
    finite_y = pd.to_numeric(label_df["complex_flow_rmse_nl_s"], errors="coerce").to_numpy(dtype=float)
    x_finite = finite_x[np.isfinite(finite_x)]
    y_finite = finite_y[np.isfinite(finite_y)]
    if x_finite.size:
        x_upper = float(np.max(x_finite))
        x_span = max(x_upper, 1.0e-6)
        ax.set_xlim(0.0, x_upper + 0.06 * x_span)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if y_finite.size:
        y_upper = float(np.max(y_finite))
        y_span = max(y_upper, 1.0e-6)
        ax.set_ylim(0.0, y_upper + 0.06 * y_span)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if x_finite.size:
        x_span = float(np.max(x_finite) - np.min(x_finite))
        decimals = 4 if x_span < 0.01 else 3 if x_span < 0.1 else 2
        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax.set_xlabel("Kirchhoff RMS [nL/s]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    add_pareto_legend(ax, rep_df)


def plot_model_grid(
    combined_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    minima_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    model_df = combined_df[combined_df["model_name"].astype(str) == model_name].copy()
    model_rep_df = rep_df[rep_df["model_name"].astype(str) == model_name].copy()
    model_minima_df = minima_df[minima_df["model_name"].astype(str) == model_name].copy()
    if model_df.empty:
        return

    labels = sorted(
        model_df["representative_label"].dropna().astype(str).unique().tolist(),
        key=label_sort_key,
    )
    if not labels:
        return

    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18.0, max(3.4 * n_rows, 4.2)), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    column_titles = (
        "Complex Flow RMSE",
        "Kirchhoff RMS",
        "A-Node Phase Difference",
        "Pareto Tradeoff",
    )

    for row_idx, label in enumerate(labels):
        label_df = model_df[model_df["representative_label"].astype(str) == label].copy()
        label_rep_df = model_rep_df[model_rep_df["representative_label"].astype(str) == label].copy()
        label_minima_df = model_minima_df[model_minima_df["representative_label"].astype(str) == label].copy()
        label_text = display_label(label)

        flow_minima = label_minima_df[label_minima_df["metric_name"].astype(str) == "complex_flow_rmse_nl_s"].copy()
        kirchhoff_minima = label_minima_df[
            label_minima_df["metric_name"].astype(str) == "kirchhoff_rms_per_internal_node_nl_s"
        ].copy()

        plot_metric_panel(
            axes[row_idx, 0],
            label_df,
            flow_minima,
            metric="complex_flow_rmse_nl_s",
            ylabel=f"{label_text}\nComplex Flow RMSE [nL/s]",
            title=column_titles[0] if row_idx == 0 else None,
        )
        plot_metric_panel(
            axes[row_idx, 1],
            label_df,
            kirchhoff_minima,
            metric="kirchhoff_rms_per_internal_node_nl_s",
            ylabel="Kirchhoff RMS [nL/s]" if row_idx > 0 else f"{label_text}\nKirchhoff RMS [nL/s]",
            title=column_titles[1] if row_idx == 0 else None,
        )
        plot_phase_panel(
            axes[row_idx, 2],
            label_df,
            ylabel="Phase Difference [deg]" if row_idx > 0 else f"{label_text}\nPhase Difference [deg]",
            title=column_titles[2] if row_idx == 0 else None,
        )
        plot_pareto_panel(
            axes[row_idx, 3],
            label_df,
            label_rep_df,
            ylabel="Complex Flow RMSE [nL/s]" if row_idx > 0 else f"{label_text}\nComplex Flow RMSE [nL/s]",
            title=column_titles[3] if row_idx == 0 else None,
        )

    fig.suptitle(f"Distensibility Alpha Profiles: {model_label(model_name)}", fontsize=15)
    save(fig, output_dir / f"distensibility_alpha_profiles_{model_name}.png", dpi)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    configure_matplotlib()
    combined_df = load_csv(input_root / "combined_results.csv")
    rep_df = load_csv(input_root / "representative_configurations.csv")
    minima_df = load_csv(input_root / "metric_minima.csv")
    combined_df = numeric(
        combined_df,
        [
            "D0",
            "alpha",
            "complex_flow_rmse_nl_s",
            "kirchhoff_rms_per_internal_node_nl_s",
            "arterial_pressure_phase_difference_deg",
            "folded_arterial_phase_difference_deg",
            "selection_score",
        ],
    )
    rep_df = numeric(
        rep_df,
        [
            "representative_D0",
            "alpha",
            "complex_flow_rmse_nl_s",
            "kirchhoff_rms_per_internal_node_nl_s",
            "arterial_pressure_phase_difference_deg",
            "folded_arterial_phase_difference_deg",
            "selection_score",
        ],
    )
    minima_df = numeric(
        minima_df,
        [
            "representative_D0",
            "alpha",
            "metric_value",
            "arterial_pressure_phase_difference_deg",
            "folded_arterial_phase_difference_deg",
        ],
    )
    output_dir = input_root / "figures"
    for model_name in ("full_ideal", "taylor_ideal", "taylor_dc_transferred"):
        plot_model_grid(combined_df, rep_df, minima_df, model_name, output_dir, int(args.dpi))
    print(f"[ok] Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
