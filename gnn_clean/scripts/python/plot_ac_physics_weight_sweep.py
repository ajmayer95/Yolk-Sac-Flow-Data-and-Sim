#!/usr/bin/env python
"""Plot AC Step 2 physics-weight sweep summaries."""

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
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "02_physics_weight_sweep" / "H1"
REGIME_ORDER = ("flow_prioritized", "balanced", "conservation_prioritized")
REGIME_COLORS = {
    "flow_prioritized": "#1f77b4",
    "balanced": "#2ca02c",
    "conservation_prioritized": "#d62728",
}
REGIME_LABELS = {
    "flow_prioritized": "Flow-prioritized",
    "balanced": "Balanced",
    "conservation_prioritized": "Conservation-prioritized",
}
MODEL_ORDER = ("full_ideal", "taylor_ideal", "taylor_dc_transferred")
MODEL_LABELS = {
    "full_ideal": "Full Ideal",
    "taylor_ideal": "Taylor Ideal",
    "taylor_dc_transferred": "Taylor DC Transferred",
}
HEATMAP_METRICS = (
    ("complex_flow_rmse_nl_s", "Complex Flow RMSE [nL/s]"),
    ("kirchhoff_rms_per_internal_node_nl_s", "Kirchhoff RMS [nL/s]"),
    ("arterial_pressure_phase_difference_deg", "A-node phase difference [deg]"),
    ("selection_score", "Selection score"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--all-runs-csv", type=Path, default=None)
    parser.add_argument("--representatives-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    input_root = args.input_root.expanduser().resolve()
    return {
        "input_root": input_root,
        "all_runs_csv": (
            args.all_runs_csv.expanduser().resolve()
            if args.all_runs_csv is not None
            else input_root / "ac_physics_weight_all_runs.csv"
        ),
        "representatives_csv": (
            args.representatives_csv.expanduser().resolve()
            if args.representatives_csv is not None
            else input_root / "ac_physics_weight_representatives.csv"
        ),
        "output_dir": (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else input_root / "figures"
        ),
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.65,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def regime_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=REGIME_COLORS[regime],
            markeredgecolor="none",
            markersize=7,
            label=REGIME_LABELS[regime],
        )
        for regime in REGIME_ORDER
    ]


def filter_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return df[df["model_name"].astype(str) == model_name].copy()


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def add_rep_annotations(ax: plt.Axes, rep_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    if rep_df.empty:
        return
    reps = rep_df.copy()
    if "selected_representative" in reps.columns:
        reps = reps[reps["selected_representative"] == True]  # noqa: E712
    if "selection_rank_within_regime" in reps.columns:
        reps = reps.sort_values("selection_rank_within_regime", na_position="last")
    elif "plot_label" in reps.columns:
        reps = reps.sort_values("plot_label")
    label_texts = reps["plot_label"].astype(str).tolist() if "plot_label" in reps.columns else []
    if label_texts and all(label.startswith("B") for label in label_texts):
        directions = ((-10, -10), (10, -10), (-10, 10), (10, 10))
    else:
        directions = ((-6, -6), (6, -6), (-6, 6), (6, 6))
    for idx, (_, row) in enumerate(reps.iterrows()):
        x = row.get(x_col)
        y = row.get(y_col)
        if not (pd.notna(x) and pd.notna(y)):
            continue
        dx, dy = directions[idx % len(directions)]
        h_align = "left" if dx >= 0 else "right"
        v_align = "bottom" if dy >= 0 else "top"
        ax.annotate(
            str(row["plot_label"]),
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=h_align,
            va=v_align,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.9},
        )


def plot_pareto_by_model(all_df: pd.DataFrame, rep_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        model_df = filter_model(all_df, model_name)
        model_rep_df = filter_model(rep_df, model_name)
        for regime in REGIME_ORDER:
            subset = model_df[model_df["weighting_regime"].astype(str) == regime].copy()
            if subset.empty:
                continue
            ax.scatter(
                subset["complex_flow_rmse_nl_s"],
                subset["kirchhoff_rms_per_internal_node_nl_s"],
                s=50,
                c=REGIME_COLORS[regime],
                alpha=0.9,
                edgecolors="none",
            )
        pareto = model_df[model_df["is_pareto_front"] == True]  # noqa: E712
        if not pareto.empty:
            ax.scatter(
                pareto["complex_flow_rmse_nl_s"],
                pareto["kirchhoff_rms_per_internal_node_nl_s"],
                s=92,
                facecolors="none",
                edgecolors="#222222",
                linewidths=1.2,
            )
        add_rep_annotations(
            ax,
            model_rep_df,
            "complex_flow_rmse_nl_s",
            "kirchhoff_rms_per_internal_node_nl_s",
        )
        ax.set_title(model_label(model_name))
        ax.set_xlabel("Complex Flow RMSE [nL/s]")
        ax.set_ylabel("Kirchhoff RMS [nL/s]")
    fig.legend(
        handles=regime_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    save(fig, output_dir / "pareto_by_model.png", dpi)


def plot_phase_vs_flow_by_model(all_df: pd.DataFrame, rep_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        model_df = filter_model(all_df, model_name)
        model_rep_df = filter_model(rep_df, model_name)
        for regime in REGIME_ORDER:
            subset = model_df[model_df["weighting_regime"].astype(str) == regime].copy()
            if subset.empty:
                continue
            ax.scatter(
                subset["complex_flow_rmse_nl_s"],
                subset["arterial_pressure_phase_difference_deg"],
                s=50,
                c=REGIME_COLORS[regime],
                alpha=0.9,
                edgecolors="none",
            )
        add_rep_annotations(
            ax,
            model_rep_df,
            "complex_flow_rmse_nl_s",
            "arterial_pressure_phase_difference_deg",
        )
        ax.set_title(model_label(model_name))
        ax.set_xlabel("Complex Flow RMSE [nL/s]")
        ax.set_ylabel("A-node phase difference [deg]")
    fig.legend(
        handles=regime_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    save(fig, output_dir / "phase_vs_flow_by_model.png", dpi)


def heatmap_matrix(df: pd.DataFrame, value_column: str) -> tuple[np.ndarray, list[float], list[float]]:
    lambda_q_values = sorted(pd.to_numeric(df["lambda_q"], errors="coerce").dropna().unique().tolist())
    lambda_k_values = sorted(pd.to_numeric(df["lambda_k"], errors="coerce").dropna().unique().tolist())
    matrix = np.full((len(lambda_q_values), len(lambda_k_values)), np.nan, dtype=np.float64)
    q_index = {value: idx for idx, value in enumerate(lambda_q_values)}
    k_index = {value: idx for idx, value in enumerate(lambda_k_values)}
    for _, row in df.iterrows():
        q = float(row["lambda_q"])
        k = float(row["lambda_k"])
        value = pd.to_numeric(row.get(value_column), errors="coerce")
        if pd.isna(value):
            continue
        matrix[q_index[q], k_index[k]] = float(value)
    return matrix, lambda_q_values, lambda_k_values


def pretty_lambda(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:g}"


def plot_heatmaps_by_model(all_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    for model_name in MODEL_ORDER:
        model_df = filter_model(all_df, model_name)
        if model_df.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
        for ax, (metric, title) in zip(axes.flat, HEATMAP_METRICS):
            matrix, q_values, k_values = heatmap_matrix(model_df, metric)
            image = ax.imshow(matrix, cmap="viridis", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel(r"$\lambda_K$")
            ax.set_ylabel(r"$\lambda_Q$")
            ax.set_xticks(range(len(k_values)))
            ax.set_xticklabels([pretty_lambda(value) for value in k_values])
            ax.set_yticks(range(len(q_values)))
            ax.set_yticklabels([pretty_lambda(value) for value in q_values])
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if math.isfinite(float(value)):
                        ax.text(
                            j,
                            i,
                            f"{float(value):.3g}",
                            ha="center",
                            va="center",
                            color="white" if float(value) > np.nanmedian(matrix) else "#202020",
                            fontsize=8,
                        )
            fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
        fig.suptitle(f"AC Step 2 heatmaps: {model_label(model_name)}", fontsize=14)
        save(fig, output_dir / f"heatmaps_{model_name}.png", dpi)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    configure_matplotlib()
    all_df = load_csv(paths["all_runs_csv"])
    rep_df = load_csv(paths["representatives_csv"])
    numeric_cols = [
        "lambda_q",
        "lambda_k",
        "complex_flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
        "arterial_pressure_phase_difference_deg",
        "selection_score",
    ]
    all_df = numeric(all_df, numeric_cols)
    rep_df = numeric(rep_df, numeric_cols)
    plot_pareto_by_model(all_df, rep_df, paths["output_dir"], int(args.dpi))
    plot_phase_vs_flow_by_model(all_df, rep_df, paths["output_dir"], int(args.dpi))
    plot_heatmaps_by_model(all_df, paths["output_dir"], int(args.dpi))


if __name__ == "__main__":
    main()
