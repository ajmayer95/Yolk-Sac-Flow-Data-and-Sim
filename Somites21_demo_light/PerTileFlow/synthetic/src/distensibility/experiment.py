"""High-level orchestration for classical inverse-solver experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .io import VascularDataset, load_dataset, write_json
from .physics import tile_problems, whole_mosaic_problem
from .plotting import plot_flow_error_map, plot_pressure_map, write_dashboard
from .pressure import PressureConditioning, load_pressure_conditioning
from models._shared import SolverResult
from models.bayesian_mosaic import solve as solve_bayesian_mosaic
from models.bayesian_tile import solve as solve_bayesian_tile
from models.linear_mosaic import solve as solve_linear_mosaic
from models.linear_tile import solve as solve_linear_tile


METHODS = {
    "linear_tile": (solve_linear_tile, "tile"),
    "linear_mosaic": (solve_linear_mosaic, "whole_mosaic"),
    "bayesian_tile": (solve_bayesian_tile, "tile"),
    "bayesian_mosaic": (solve_bayesian_mosaic, "whole_mosaic"),
}


def solver_configuration_name(
    config: dict,
    pressure_conditioning: PressureConditioning | None = None,
) -> str:
    """Return a filesystem-safe label for one solver configuration."""
    solver = config["solver"]
    if solver["alpha_mode"] == "prescribed":
        alpha = f"{float(solver['prescribed_alpha']):g}"
        alpha = alpha.replace("-", "m").replace(".", "p")
        alpha_part = f"alpha_prescribed_{alpha}"
    else:
        alpha_part = "alpha_solved"
    harmonic_part = (
        "h1_h2"
        if tuple(int(h) for h in solver["harmonics_used"]) == (1, 2)
        else "h1"
    )
    name = f"{alpha_part}__{harmonic_part}"
    if pressure_conditioning is not None:
        source_name = pressure_conditioning.source.parent.name
        safe_source = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in source_name
        )
        name += (
            f"__pressure_{pressure_conditioning.mode}"
            f"__{safe_source}"
        )
    return name


def _aggregate_predictions(
    dataset: VascularDataset, results: Sequence[SolverResult]
) -> tuple[np.ndarray, np.ndarray]:
    velocity_sum = np.zeros((dataset.n_edges, 3), dtype=np.complex128)
    velocity_count = np.zeros((dataset.n_edges, 3), dtype=np.int32)
    pressure_sum = np.zeros((dataset.n_nodes, 3), dtype=np.complex128)
    pressure_count = np.zeros((dataset.n_nodes, 3), dtype=np.int32)
    for result in results:
        valid_v = np.isfinite(result.predicted_velocity)
        velocity_sum[valid_v] += result.predicted_velocity[valid_v]
        velocity_count[valid_v] += 1
        valid_p = np.isfinite(result.predicted_pressure)
        pressure_sum[valid_p] += result.predicted_pressure[valid_p]
        pressure_count[valid_p] += 1
    velocity = np.full(
        (dataset.n_edges, 3), np.nan + 1j * np.nan, dtype=np.complex128
    )
    pressure = np.full(
        (dataset.n_nodes, 3), np.nan + 1j * np.nan, dtype=np.complex128
    )
    np.divide(
        velocity_sum,
        velocity_count,
        out=velocity,
        where=velocity_count > 0,
    )
    np.divide(
        pressure_sum,
        pressure_count,
        out=pressure,
        where=pressure_count > 0,
    )
    return velocity, pressure


def _summary(
    method: str,
    configuration_name: str,
    dataset: VascularDataset,
    results,
    pressure_conditioning: PressureConditioning | None = None,
):
    rows = [result.metrics for result in results]
    D_errors = np.asarray([row["relative_D0_error"] for row in rows])
    alpha_errors = np.asarray([row["alpha_absolute_error"] for row in rows])
    return {
        "method": method,
        "configuration": configuration_name,
        "dataset": dataset.path.name,
        "n_spatial_problems": len(results),
        "ground_truth": {
            "D0_per_pa": dataset.D0_true,
            "alpha": dataset.alpha_true,
        },
        "pressure_conditioning": (
            {
                "source": pressure_conditioning.source,
                "mode": pressure_conditioning.mode,
                "weight": pressure_conditioning.weight,
                "sigma_pa": pressure_conditioning.sigma_pa,
                "available_harmonics": pressure_conditioning.harmonic_index,
                "fix_available_harmonics": (
                    pressure_conditioning.fix_available_harmonics
                ),
            }
            if pressure_conditioning is not None
            else None
        ),
        "aggregate_metrics": {
            "median_relative_D0_error": float(np.nanmedian(D_errors)),
            "mean_relative_D0_error": float(np.nanmean(D_errors)),
            "median_alpha_absolute_error": float(np.nanmedian(alpha_errors)),
            "mean_alpha_absolute_error": float(np.nanmean(alpha_errors)),
            "D0_interval_coverage_rate": float(
                np.mean([row["D0_interval_covers_true"] for row in rows])
            ),
            "alpha_interval_coverage_rate": float(
                np.mean([row["alpha_interval_covers_true"] for row in rows])
            ),
            "boundary_hit_rate": float(
                np.mean([row["boundary_hit"] for row in rows])
            ),
            "median_D0_interval_width_decades": float(
                np.nanmedian(
                    [row["D0_interval_width_decades"] for row in rows]
                )
            ),
            "median_alpha_interval_width": float(
                np.nanmedian([row["alpha_interval_width"] for row in rows])
            ),
            "median_held_out_velocity_relative_rmse": float(
                np.nanmedian(
                    [row["held_out_velocity_relative_rmse"] for row in rows]
                )
            ),
        },
        "spatial_results": rows,
    }


def run_solver_experiment(
    project_root: Path,
    dataset_path: Path,
    config: dict,
    method: str,
    config_path: Path,
    tile_ids: Sequence[int] | None = None,
    pressure_field_path: Path | None = None,
    pressure_mode: str = "scaled",
    pressure_weight: float = 1.0,
    pressure_sigma_pa: float = 0.0,
    fix_available_harmonics: bool = True,
    solver_function_override=None,
    spatial_mode_override: str | None = None,
) -> dict:
    """Run one configured solver and write all required artifacts."""
    if method not in METHODS and solver_function_override is None:
        raise ValueError(f"Unknown method: {method}")
    if solver_function_override is None:
        solver_function, spatial_mode = METHODS[method]
    else:
        solver_function = solver_function_override
        spatial_mode = spatial_mode_override or "tile"
    dataset = load_dataset(dataset_path)
    pressure_conditioning = (
        load_pressure_conditioning(
            pressure_field_path,
            dataset,
            mode=pressure_mode,
            weight=pressure_weight,
            sigma_pa=pressure_sigma_pa,
            fix_available_harmonics=fix_available_harmonics,
        )
        if pressure_field_path is not None
        else None
    )
    dataset_name = dataset.path.stem
    configuration_name = solver_configuration_name(
        config, pressure_conditioning
    )
    run_dir = (
        project_root
        / "outputs"
        / "runs"
        / method
        / dataset_name
        / configuration_name
    )
    metric_dir = (
        project_root
        / "outputs"
        / "metrics"
        / method
        / dataset_name
        / configuration_name
    )
    figure_dir = (
        project_root
        / "outputs"
        / "figures"
        / method
        / dataset_name
        / configuration_name
    )
    for directory in (run_dir, metric_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and records command-line overrides in the effective
    # configuration rather than merely copying the base file.
    write_json(run_dir / "solver_config.yaml", config)

    if spatial_mode == "tile":
        problems = tile_problems(
            dataset,
            config["solver"]["fit_edges"],
            tile_ids=tile_ids,
        )
    else:
        problems = [
            whole_mosaic_problem(
                dataset, config["solver"]["fit_edges"]
            )
        ]
    if not problems:
        raise RuntimeError("No valid spatial problems were constructed")

    results = []
    for index, problem in enumerate(problems, start=1):
        print(
            f"[{method}] {problem.name} "
            f"({index}/{len(problems)}, {len(problem.observed_edge_indices)} fit edges)"
        )
        result = solver_function(
            dataset,
            problem,
            config,
            pressure_conditioning=pressure_conditioning,
        )
        results.append(result)

    predicted_velocity, predicted_pressure = _aggregate_predictions(
        dataset, results
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        predicted_velocity_m_s=predicted_velocity,
        predicted_pressure_pa=predicted_pressure,
    )
    np.savez_compressed(
        run_dir / "parameter_surfaces.npz",
        **{
            key: value
            for result in results
            for key, value in (
                (f"{result.problem_name}__surface", result.surface),
                (
                    f"{result.problem_name}__log10_D0_grid",
                    result.log10_D0_grid,
                ),
                (
                    f"{result.problem_name}__alpha_grid",
                    result.alpha_grid,
                ),
                *[
                    (
                        f"{result.problem_name}__boundary_pressure_H{harmonic}",
                        pressure,
                    )
                    for harmonic, pressure in result.boundary_pressure.items()
                ],
            )
        },
    )
    fields = [
        "problem_name",
        "D0_hat",
        "alpha_hat",
        "D0_interval_low",
        "D0_interval_high",
        "alpha_interval_low",
        "alpha_interval_high",
        "relative_D0_error",
        "alpha_absolute_error",
        "held_out_velocity_relative_rmse",
        "boundary_hit",
    ]
    with (run_dir / "spatial_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"problem_name": result.problem_name}
            row.update({field: result.metrics.get(field) for field in fields[1:]})
            writer.writerow(row)

    summary = _summary(
        method,
        configuration_name,
        dataset,
        results,
        pressure_conditioning,
    )
    write_json(metric_dir / "summary_metrics.json", summary)
    harmonics = tuple(int(h) for h in config["solver"]["harmonics_used"])
    map_harmonic = harmonics[0]
    plot_pressure_map(
        figure_dir / "pressure_map.png",
        dataset,
        predicted_pressure,
        map_harmonic,
        method,
    )
    plot_flow_error_map(
        figure_dir / "flow_error_map.png",
        dataset,
        predicted_velocity,
        harmonics,
        method,
    )
    write_dashboard(
        figure_dir / "distensibility_dashboard.html",
        dataset,
        method,
        results,
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "method": method,
            "configuration": configuration_name,
            "dataset": dataset.path,
            "config": config,
            "pressure_conditioning": (
                {
                    "source": pressure_conditioning.source,
                    "mode": pressure_conditioning.mode,
                    "weight": pressure_conditioning.weight,
                    "sigma_pa": pressure_conditioning.sigma_pa,
                    "available_harmonics": (
                        pressure_conditioning.harmonic_index
                    ),
                    "fix_available_harmonics": (
                        pressure_conditioning.fix_available_harmonics
                    ),
                }
                if pressure_conditioning is not None
                else None
            ),
            "metrics": metric_dir / "summary_metrics.json",
            "figures": {
                "pressure_map": figure_dir / "pressure_map.png",
                "flow_error_map": figure_dir / "flow_error_map.png",
                "dashboard": figure_dir / "distensibility_dashboard.html",
            },
        },
    )
    return summary
