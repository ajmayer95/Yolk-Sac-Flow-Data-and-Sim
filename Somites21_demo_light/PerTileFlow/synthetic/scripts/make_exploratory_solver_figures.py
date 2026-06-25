#!/usr/bin/env python
"""Create ten exploratory figures from completed synthetic solver runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.stats import rankdata, spearmanr

from make_classical_solver_comparison import (
    deduplicate,
    load_row,
)


ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics"
RUNS = ROOT / "outputs" / "runs"
OUT = ROOT / "outputs" / "figures" / "exploratory_solver_figures"
COLORS = {
    "linear_tile": "#1f77b4",
    "linear_mosaic": "#17a2b8",
    "bayesian_tile": "#d95f02",
    "bayesian_mosaic": "#8c564b",
}
PRIOR_LABEL = {
    "none": "No prior",
    "k0": "K=0 prior",
    "physics_gnn": "Physics-GNN prior",
}


def strategy(row) -> str:
    return f"{row['method'].replace('_', ' ').title()} · {PRIOR_LABEL.get(row['pressure_prior'], row['pressure_prior'])}"


def composite(row) -> float:
    if not all(np.isfinite([row["D0_hat"], row["D0"], row["alpha_hat"], row["alpha"]])):
        return np.nan
    return abs(np.log10(row["D0_hat"] / row["D0"])) + abs(row["alpha_hat"] - row["alpha"])


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(METRICS.glob("*/*/*/summary_metrics.json")):
        try:
            row = load_row(path, METRICS, RUNS)
        except Exception:
            continue
        if row["method"] in COLORS:
            row["strategy"] = strategy(row)
            row["composite_error"] = composite(row)
            rows.append(row)
    return deduplicate(rows, include_engines=False)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def solved_h1h2(rows):
    return [
        r for r in rows
        if r["alpha_mode"] == "solved" and r["harmonics"] == "H1 + H2"
        and np.isfinite(r["composite_error"])
    ]


def method_win_rate(rows):
    use = solved_h1h2(rows)
    counts = defaultdict(set)
    for r in use:
        counts[r["strategy"]].add(r["dataset"])
    names = sorted(k for k, v in counts.items() if len(v) >= 8)
    lookup = {(r["strategy"], r["dataset"]): r["composite_error"] for r in use}
    matrix = np.full((len(names), len(names)), np.nan)
    annotations = [["" for _ in names] for _ in names]
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            common = counts[left] & counts[right]
            if i == j:
                matrix[i, j] = 0.5
                annotations[i][j] = "—"
            elif common:
                wins = sum(lookup[left, d] < lookup[right, d] for d in common)
                ties = sum(lookup[left, d] == lookup[right, d] for d in common)
                matrix[i, j] = (wins + 0.5 * ties) / len(common)
                annotations[i][j] = f"{100*matrix[i,j]:.0f}%\n(n={len(common)})"
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8)
    ax.set_title("Method win rate (row beats column)\nSolved alpha, H1+H2; composite recovery error")
    fig.colorbar(im, ax=ax, label="Pairwise win fraction")
    save(fig, "01_method_win_rate_matrix.png")


def error_decomposition(rows):
    use = [r for r in rows if r["alpha_mode"] == "solved" and np.isfinite(r["composite_error"])]
    frame = pd.DataFrame(use)
    frame["response"] = np.log1p(frame["composite_error"])
    factors = ["method", "pressure_prior", "harmonics", "noise", "D0", "alpha"]
    blocks = {}
    matrices = [np.ones((len(frame), 1))]
    start = 1
    for factor in factors:
        dummy = pd.get_dummies(frame[factor].astype(str), prefix=factor, drop_first=True).to_numpy(float)
        blocks[factor] = (start, start + dummy.shape[1])
        start += dummy.shape[1]
        matrices.append(dummy)
    X = np.column_stack(matrices)
    y = frame["response"].to_numpy()
    full_sse = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    effects = {}
    for factor, (a, b) in blocks.items():
        keep = np.r_[0:a, b:X.shape[1]]
        reduced = X[:, keep]
        sse = np.sum((y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]) ** 2)
        effects[factor] = max(0.0, (sse - full_sse) / total)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Solver", "Pressure prior", "Harmonics", "Noise", "True D0", "True alpha"]
    vals = [effects[f] for f in factors]
    ax.barh(labels[::-1], vals[::-1], color="#4c78a8")
    ax.set_xlabel("Drop-one partial variance explained")
    ax.set_title("Error decomposition of log(1 + composite recovery error)")
    ax.grid(axis="x", alpha=.25)
    save(fig, "02_error_decomposition.png")


def surface_statistics(row):
    path = RUNS / row["raw_method"] / row["dataset"] / row["configuration"] / "parameter_surfaces.npz"
    correlations, sharpness = [], []
    if not path.is_file():
        return np.nan, np.nan
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            if not key.endswith("__surface"):
                continue
            prefix = key[:-9]
            lk, ak = prefix + "__log10_D0_grid", prefix + "__alpha_grid"
            if lk not in z or ak not in z:
                continue
            surface, lg, ag = np.asarray(z[key], float), np.asarray(z[lk], float), np.asarray(z[ak], float)
            if len(ag) < 2:
                continue
            if row["method"].startswith("bayesian"):
                weights = np.maximum(surface, 0)
                prof = -2*np.log(np.maximum(np.nanmax(surface, axis=1), 1e-300) /
                                 max(np.nanmax(surface), 1e-300))
            else:
                weights = np.exp(-0.5 * np.clip(surface - np.nanmin(surface), 0, 100))
                prof = np.nanmin(surface, axis=1)
                prof -= np.nanmin(prof)
            weights = np.where(np.isfinite(weights), weights, 0)
            if weights.sum() > 0:
                x, y = np.meshgrid(ag, lg)
                mx, my = np.sum(weights*x)/weights.sum(), np.sum(weights*y)/weights.sum()
                cov = np.sum(weights*(x-mx)*(y-my))/weights.sum()
                vx = np.sum(weights*(x-mx)**2)/weights.sum()
                vy = np.sum(weights*(y-my)**2)/weights.sum()
                if vx > 0 and vy > 0:
                    correlations.append(cov/np.sqrt(vx*vy))
            i = int(np.nanargmin(prof))
            if 0 < i < len(lg)-1:
                dx = (lg[i+1]-lg[i-1])/2
                sharpness.append((prof[i+1]-2*prof[i]+prof[i-1])/(dx*dx))
    return (float(np.nanmedian(correlations)) if correlations else np.nan,
            float(np.nanmedian(sharpness)) if sharpness else np.nan)


def correlation_and_sharpness(rows):
    use = solved_h1h2(rows)
    for r in use:
        r["parameter_correlation"], r["profile_sharpness"] = surface_statistics(r)
    groups = defaultdict(list)
    for r in use:
        if np.isfinite(r["parameter_correlation"]):
            groups[r["strategy"]].append(r["parameter_correlation"])
    names = sorted(groups)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([groups[n] for n in names], tick_labels=names, showfliers=False)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Weighted correlation: log10(D0) vs alpha")
    ax.set_title("Parameter correlation within joint likelihood/posterior surfaces")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "03_parameter_correlation.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in use:
        if np.isfinite(r["profile_sharpness"]) and r["profile_sharpness"] > 0:
            ax.scatter(r["noise"]*100, r["profile_sharpness"], color=COLORS[r["method"]],
                       alpha=.45, s=24)
    ax.set_yscale("log")
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Median D0-profile curvature")
    ax.set_title("Profile sharpness versus observation noise")
    for method, color in COLORS.items():
        ax.scatter([], [], color=color, label=method.replace("_", " ").title())
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    save(fig, "04_profile_sharpness.png")


def robustness_waterfall(rows):
    use = [
        r for r in rows
        if r["alpha_mode"] == "solved"
        and np.isfinite(r["median_velocity_relative_rmse"])
    ]
    stages = [
        ("No prior\nH1", "none", "H1"),
        ("No prior\nH1+H2", "none", "H1 + H2"),
        ("K=0 prior\nH1+H2", "k0", "H1 + H2"),
        ("Physics-GNN\nH1+H2", "physics_gnn", "H1 + H2"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in COLORS:
        medians = []
        for _, prior, harmonics in stages:
            vals = [r["median_velocity_relative_rmse"] for r in use if r["method"] == method and
                    r["pressure_prior"] == prior and r["harmonics"] == harmonics]
            medians.append(np.nanmedian(vals) if vals else np.nan)
        ax.plot(range(len(stages)), medians, marker="o", lw=2, color=COLORS[method],
                label=method.replace("_", " ").title())
    ax.set_xticks(range(len(stages)), [s[0] for s in stages])
    ax.set_ylabel("Median held-out velocity relative RMSE")
    ax.set_title("Robustness waterfall across harmonics and pressure priors")
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, "05_robustness_waterfall.png")


def rank_stability(rows):
    use = [
        r for r in solved_h1h2(rows)
        if np.isfinite(r["median_velocity_relative_rmse"])
    ]
    by_condition = defaultdict(dict)
    for r in use:
        key = (r["D0"], r["alpha"], r["noise"])
        by_condition[key][r["strategy"]] = r["median_velocity_relative_rmse"]
    noise_levels = [0.10, 0.25, 0.50]
    means, spreads = [], []
    for noise in noise_levels:
        correlations = []
        for D0 in sorted({r["D0"] for r in use}):
            for alpha in sorted({r["alpha"] for r in use}):
                base, other = by_condition.get((D0, alpha, 0.0), {}), by_condition.get((D0, alpha, noise), {})
                names = sorted(set(base) & set(other))
                if len(names) >= 3:
                    value = spearmanr(
                        rankdata([base[n] for n in names]),
                        rankdata([other[n] for n in names]),
                    ).statistic
                    if np.isfinite(value):
                        correlations.append(value)
        means.append(np.mean(correlations) if correlations else np.nan)
        spreads.append(np.std(correlations) if correlations else np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(np.array(noise_levels)*100, means, yerr=spreads, marker="o", capsize=5)
    ax.set_ylim(-1, 1.05)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Spearman correlation with zero-noise ranking")
    ax.set_title("Rank stability by held-out velocity reconstruction error")
    ax.grid(alpha=.25)
    save(fig, "06_rank_stability.png")


def wall_law_curves(rows):
    use = solved_h1h2(rows)
    radius = np.geomspace(8, 70, 160)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, alpha in zip(axes, [0.0, 1.0, 2.0]):
        subset = [r for r in use if r["D0"] == 1e-3 and r["alpha"] == alpha and r["noise"] == .25]
        truth = 1e-3*(radius/25)**alpha
        ax.plot(radius, truth, "k--", lw=3, label="Truth")
        grouped = defaultdict(list)
        for r in subset:
            grouped[r["strategy"]].append((r["D0_hat"], r["alpha_hat"]))
        for name, estimates in grouped.items():
            D0 = np.nanmedian([x[0] for x in estimates])
            a = np.nanmedian([x[1] for x in estimates])
            method = next(r["method"] for r in subset if r["strategy"] == name)
            ax.plot(radius, D0*(radius/25)**a, color=COLORS[method], alpha=.7, lw=1.6)
        ax.set_title(f"True alpha = {alpha:g}")
        ax.set_xlabel("Radius (µm)")
        ax.set_yscale("log")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Distensibility (1/Pa)")
    axes[0].legend()
    fig.suptitle("Predicted wall-law curves at D0=1e-3, 25% noise")
    save(fig, "07_predicted_wall_law_curves.png")


def radius_stratified(rows):
    use = [
        r for r in rows
        if r["harmonics"] == "H1 + H2"
        and r["pressure_prior"] == "none"
        and np.isfinite(r["D0_hat"])
        and np.isfinite(r["alpha_hat"])
    ]
    radii = np.load(ROOT / "data/synthetic/pl_d1e-03_a1_n00_s42.npz")["edge_radius_m"]*1e6
    centers = np.quantile(radii, [0.1, .3, .5, .7, .9])
    bins = ["small", "mid-small", "middle", "mid-large", "large"]
    methods = ["linear_tile", "linear_mosaic", "bayesian_tile", "bayesian_mosaic"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        errors = []
        subset = [r for r in use if r["method"] == method]
        for R in centers:
            vals = [abs(np.log10((r["D0_hat"]*(R/25)**r["alpha_hat"]) /
                                 (r["D0"]*(R/25)**r["alpha"]))) for r in subset]
            errors.append(np.nanmedian(vals))
        ax.plot(bins, errors, marker="o", color=COLORS[method],
                label=method.replace("_", " ").title())
    ax.set_ylabel("Median |log10 inferred D(R) / true D(R)|")
    ax.set_title("Radius-stratified wall-law recovery\nNo pressure prior; solved and prescribed-alpha runs")
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, "08_radius_stratified_recovery.png")


def spatial_uncertainty():
    dataset = "pl_d1e-03_a1_n25_s42"
    raw_method = "linear_tile_gpu"
    config = "alpha_solved__h1_h2"
    csv_path = RUNS / raw_method / dataset / config / "spatial_summary.csv"
    data_path = ROOT / "data/synthetic" / f"{dataset}.npz"
    frame = pd.read_csv(csv_path)
    widths = {int(name.split("_")[-1]): np.log10(hi/lo) if lo > 0 and hi > 0 else np.nan
              for name, lo, hi in zip(frame.problem_name, frame.D0_interval_low, frame.D0_interval_high)}
    with np.load(data_path, allow_pickle=False) as d:
        xy, src, dst = d["node_xy_px"], d["edge_source_index"], d["edge_target_index"]
        offsets, ids = d["edge_tile_offsets"], d["edge_tile_ids"]
    segments, values = [], []
    for i, (u, v) in enumerate(zip(src, dst)):
        tile_ids = ids[offsets[i]:offsets[i+1]]
        vals = [widths.get(int(t), np.nan) for t in tile_ids]
        vals = [x for x in vals if np.isfinite(x)]
        if vals:
            segments.append([xy[u], xy[v]])
            values.append(np.mean(vals))
    fig, ax = plt.subplots(figsize=(10, 8))
    collection = LineCollection(segments, array=np.asarray(values), cmap="magma",
                                linewidths=.8)
    ax.add_collection(collection)
    ax.autoscale()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Spatial uncertainty map: D0 interval width\n{dataset}, Linear Tile GPU, solved alpha, H1+H2")
    ax.set_xlabel("Mosaic x (px)")
    ax.set_ylabel("Mosaic y (px)")
    fig.colorbar(collection, ax=ax, label="95% D0 interval width (decades)")
    save(fig, "09_spatial_uncertainty_map.png")


def calibration(rows):
    use = solved_h1h2(rows)
    groups = defaultdict(lambda: [[], []])
    for r in use:
        groups[r["strategy"]][0].append(r["D0_coverage_rate"])
        groups[r["strategy"]][1].append(r["alpha_coverage_rate"])
    names = sorted(k for k, v in groups.items() if len(v[0]) >= 8)
    d0 = [np.nanmean(groups[n][0]) for n in names]
    alpha = [np.nanmean(groups[n][1]) for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, max(5, .55*len(names))))
    ax.scatter(d0, y-.13, label="D0 interval", s=55)
    ax.scatter(alpha, y+.13, label="Alpha interval", s=55)
    ax.axvline(.95, color="black", ls="--", label="Nominal 95%")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Empirical coverage")
    ax.set_title("95% interval calibration across datasets")
    ax.legend()
    ax.grid(axis="x", alpha=.25)
    save(fig, "10_calibration.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    method_win_rate(rows)
    error_decomposition(rows)
    correlation_and_sharpness(rows)
    robustness_waterfall(rows)
    rank_stability(rows)
    wall_law_curves(rows)
    radius_stratified(rows)
    spatial_uncertainty()
    calibration(rows)
    files = sorted(path.name for path in OUT.glob("*.png"))
    manifest = {
        "completed_rows_used": len(rows),
        "figures": files,
        "ranking_metric": "|log10(D0_hat/D0)| + |alpha_hat-alpha|",
        "notes": [
            "Parameter plots use solved-alpha configurations.",
            "Tile profiles and estimates are summarized by their median.",
            "Calibration evaluates the saved nominal 95% intervals.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(files)} figures to {OUT}")
    for name in files:
        print(name)


if __name__ == "__main__":
    main()
