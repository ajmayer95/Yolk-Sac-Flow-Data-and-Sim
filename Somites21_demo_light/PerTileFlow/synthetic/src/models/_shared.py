"""Shared numerical engine for the four classical solver models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from distensibility.io import VascularDataset
from distensibility.physics import (
    SpatialProblem,
    build_transfer_operator,
    complex_to_real,
    edge_admittances,
)
from distensibility.pressure import PressureConditioning, pressure_sigma


@dataclass
class SolverResult:
    problem_name: str
    method: str
    D0_hat: float
    alpha_hat: float
    D0_interval: tuple[float, float]
    alpha_interval: tuple[float, float]
    surface: np.ndarray
    log10_D0_grid: np.ndarray
    alpha_grid: np.ndarray
    predicted_velocity: np.ndarray
    predicted_pressure: np.ndarray
    boundary_pressure: dict[int, np.ndarray]
    metrics: dict


def parameter_grids(config: dict) -> tuple[np.ndarray, np.ndarray]:
    grid = config["parameter_grid"]
    logD = np.linspace(
        float(grid["log10_D0_min"]),
        float(grid["log10_D0_max"]),
        int(grid["num_D0"]),
    )
    solver = config["solver"]
    if solver["alpha_mode"] == "prescribed":
        alpha = np.asarray([float(solver["prescribed_alpha"])])
    else:
        alpha = np.linspace(
            float(grid["alpha_min"]),
            float(grid["alpha_max"]),
            int(grid["num_alpha"]),
        )
    return logD, alpha


def _noise_sigma(
    dataset: VascularDataset, edges: np.ndarray, harmonic: int
) -> np.ndarray:
    sigma = np.asarray(
        dataset.velocity_noise_sigma_m_s[edges, harmonic], dtype=float
    )
    values = np.abs(dataset.velocity_observed_m_s[edges, harmonic])
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    signal_scale = float(np.nanmedian(values[values > 0])) if np.any(values > 0) else 1.0
    floor = max(
        float(np.nanmedian(positive)) * 1.0e-3 if positive.size else 0.0,
        signal_scale * 1.0e-6,
        1.0e-15,
    )
    return np.where(np.isfinite(sigma) & (sigma > floor), sigma, floor)


def _weighted_complex_fit(
    operator: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    regularization: float,
    pressure_reference: np.ndarray | None = None,
    pressure_mode: str = "off",
    pressure_weight: float = 0.0,
    pressure_sigma_pa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    weighted_operator = operator / sigma[:, None]
    weighted_observed = observed / sigma
    if (
        pressure_reference is not None
        and pressure_mode != "off"
        and pressure_weight > 0
    ):
        reference = np.asarray(pressure_reference, dtype=float)
        reference = reference - float(np.mean(reference))
        prior_sigma = pressure_sigma(reference, pressure_sigma_pa)
        prior_weight = math.sqrt(pressure_weight / max(len(reference), 1))
        prior_weight /= prior_sigma
        if pressure_mode == "scaled":
            denom = float(np.dot(reference, reference))
            if denom > 1.0e-30:
                penalty = np.eye(len(reference)) - np.outer(
                    reference, reference
                ) / denom
                weighted_operator = np.vstack(
                    [weighted_operator, prior_weight * penalty]
                )
                weighted_observed = np.concatenate(
                    [
                        weighted_observed,
                        np.zeros(len(reference), dtype=np.complex128),
                    ]
                )
        else:
            weighted_operator = np.vstack(
                [
                    weighted_operator,
                    prior_weight
                    * np.eye(len(reference), dtype=np.complex128),
                ]
            )
            weighted_observed = np.concatenate(
                [
                    weighted_observed,
                    prior_weight * reference.astype(np.complex128),
                ]
            )
    normal = weighted_operator.conj().T @ weighted_operator
    scale = max(float(np.median(np.abs(np.diag(normal)))), 1.0e-30)
    normal += float(regularization) * scale * np.eye(normal.shape[0])
    rhs = weighted_operator.conj().T @ weighted_observed
    try:
        pressure = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        pressure = np.linalg.pinv(normal, rcond=1.0e-12) @ rhs
    predicted = operator @ pressure
    weighted_residual = weighted_observed - weighted_operator @ pressure
    chi2 = float(np.sum(np.abs(weighted_residual) ** 2))
    return pressure, predicted, chi2


def _marginal_log_likelihood(
    operator: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    pressure_scale: float,
    pressure_reference: np.ndarray | None = None,
    pressure_mode: str = "off",
    pressure_weight: float = 0.0,
    pressure_sigma_pa: float = 0.0,
) -> tuple[float, np.ndarray]:
    real_operator, real_observed = complex_to_real(operator, observed)
    variance = np.concatenate([0.5 * sigma**2, 0.5 * sigma**2])
    inv_variance = 1.0 / np.maximum(variance, 1.0e-30)
    beta = max(0.5 * float(pressure_scale) ** 2, 1.0e-30)
    n_boundary = operator.shape[1]
    prior_precision = (1.0 / beta) * np.eye(real_operator.shape[1])
    prior_mean = np.zeros(real_operator.shape[1], dtype=float)
    if (
        pressure_reference is not None
        and pressure_mode != "off"
        and pressure_weight > 0
    ):
        reference = np.asarray(pressure_reference, dtype=float)
        reference = reference - float(np.mean(reference))
        prior_sigma = pressure_sigma(reference, pressure_sigma_pa)
        strength = pressure_weight / (
            max(n_boundary, 1) * prior_sigma**2
        )
        if pressure_mode == "scaled":
            denom = float(np.dot(reference, reference))
            if denom > 1.0e-30:
                penalty = np.eye(n_boundary) - np.outer(
                    reference, reference
                ) / denom
                prior_precision[:n_boundary, :n_boundary] += (
                    strength * penalty
                )
                prior_precision[n_boundary:, n_boundary:] += (
                    strength * penalty
                )
        else:
            prior_precision += strength * np.eye(2 * n_boundary)
            prior_mean[:n_boundary] = reference
    precision = (
        (real_operator.T * inv_variance[None, :]) @ real_operator
        + prior_precision
    )
    precision = 0.5 * (precision + precision.T)
    rhs = (
        real_operator.T @ (inv_variance * real_observed)
        + prior_precision @ prior_mean
    )
    jitter = max(float(np.median(np.diag(precision))) * 1.0e-12, 1.0e-18)
    cf = cho_factor(
        precision + jitter * np.eye(precision.shape[0]),
        lower=True,
        check_finite=False,
    )
    posterior_mean = cho_solve(cf, rhs, check_finite=False)
    logdet_precision = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    prior_cf = cho_factor(
        prior_precision,
        lower=True,
        check_finite=False,
    )
    logdet_prior_precision = 2.0 * float(
        np.sum(np.log(np.diag(prior_cf[0])))
    )
    logdet = (
        float(np.sum(np.log(variance)))
        - logdet_prior_precision
        + logdet_precision
    )
    quadratic = float(
        np.sum(real_observed**2 * inv_variance)
        + prior_mean @ prior_precision @ prior_mean
        - rhs @ posterior_mean
    )
    quadratic = max(quadratic, 0.0)
    log_likelihood = -0.5 * (
        logdet
        + quadratic
        + real_observed.size * math.log(2.0 * math.pi)
    )
    complex_mean = (
        posterior_mean[:n_boundary]
        + 1j * posterior_mean[n_boundary:]
    )
    return float(log_likelihood), complex_mean


def _problem_pressure_reference(
    conditioning: PressureConditioning | None,
    problem: SpatialProblem,
) -> np.ndarray | None:
    if conditioning is None or conditioning.mode == "off":
        return None
    dc = conditioning.field(0)
    if dc is None:
        return None
    reference = np.asarray(dc[problem.boundary_node_indices].real, dtype=float)
    return reference if np.isfinite(reference).all() else None


def _fixed_pressure_prediction(
    dataset: VascularDataset,
    problem: SpatialProblem,
    pressure_field: np.ndarray,
    D0: float,
    alpha: float,
    harmonic: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict modeled edge velocity from a fully known nodal pressure field."""
    nodes = problem.node_indices
    local_pressure = np.asarray(pressure_field[nodes], dtype=np.complex128)
    if not np.isfinite(local_pressure).all():
        raise ValueError("Fixed pressure field is incomplete on this problem")
    edges = problem.model_edge_indices
    ys, yt = edge_admittances(
        dataset.edge_radius_m[edges],
        dataset.edge_length_m[edges],
        D0,
        alpha,
        dataset.R0_m,
        harmonic,
        dataset.frequency_hz,
        dataset.viscosity_pa_s,
        dataset.density_kg_m3,
    )
    node_lookup = np.full(dataset.n_nodes, -1, dtype=np.int32)
    node_lookup[nodes] = np.arange(len(nodes), dtype=np.int32)
    u = node_lookup[dataset.edge_source_index[edges]]
    v = node_lookup[dataset.edge_target_index[edges]]
    velocity = (
        ys * local_pressure[u] - yt * local_pressure[v]
    ) / dataset.edge_area_m2[edges]
    return local_pressure, velocity


def _credible_interval(grid: np.ndarray, mass: np.ndarray):
    marginal = np.sum(mass, axis=1)
    marginal = marginal / max(float(np.sum(marginal)), 1.0e-300)
    cdf = np.cumsum(marginal)
    low_index = min(int(np.searchsorted(cdf, 0.025)), len(grid) - 1)
    high_index = min(int(np.searchsorted(cdf, 0.975)), len(grid) - 1)
    return (
        float(grid[low_index]),
        float(grid[high_index]),
    )


def _profile_interval(
    grid: np.ndarray, objective: np.ndarray, axis: int, threshold: float
):
    profile = np.nanmin(objective, axis=axis)
    profile = profile - np.nanmin(profile)
    accepted = np.isfinite(profile) & (profile <= threshold)
    if not accepted.any():
        return (float("nan"), float("nan"))
    return float(np.min(grid[accepted])), float(np.max(grid[accepted]))


def solve_problem(
    dataset: VascularDataset,
    problem: SpatialProblem,
    config: dict,
    method: str,
    pressure_conditioning: PressureConditioning | None = None,
    progress: Callable[[str], None] | None = None,
) -> SolverResult:
    """Solve one tile or the whole mosaic on the configured parameter grid."""
    logD_grid, alpha_grid = parameter_grids(config)
    harmonics = tuple(int(h) for h in config["solver"]["harmonics_used"])
    regularization = float(config["linear_solver"]["regularization"])
    pressure_scale = float(config["bayesian_solver"]["boundary_prior_scale"])
    objective = np.full(
        (len(logD_grid), len(alpha_grid)), np.inf, dtype=float
    )
    cache: dict[tuple[int, int], tuple[dict, dict]] = {}
    observed_edges = problem.observed_edge_indices
    pressure_reference = _problem_pressure_reference(
        pressure_conditioning, problem
    )

    for i, logD in enumerate(logD_grid):
        D0 = 10.0 ** float(logD)
        if progress:
            progress(
                f"{problem.name}: D0 row {i + 1}/{len(logD_grid)}"
            )
        for j, alpha in enumerate(alpha_grid):
            boundary_pressures = {}
            transfers = {}
            total = 0.0
            valid_point = True
            for harmonic in harmonics:
                valid = dataset.observation_valid[
                    observed_edges, harmonic
                ].astype(bool)
                edges_h = observed_edges[valid]
                if not valid.any():
                    valid_point = False
                    break
                observed = dataset.velocity_observed_m_s[
                    edges_h, harmonic
                ]
                sigma = _noise_sigma(dataset, edges_h, harmonic)
                fixed_field = (
                    pressure_conditioning.field(harmonic)
                    if pressure_conditioning is not None
                    and pressure_conditioning.fix_available_harmonics
                    else None
                )
                if fixed_field is not None and np.isfinite(
                    fixed_field[problem.node_indices]
                ).all():
                    local_pressure, model_velocity = _fixed_pressure_prediction(
                        dataset,
                        problem,
                        fixed_field,
                        D0,
                        float(alpha),
                        harmonic,
                    )
                    model_lookup = {
                        int(edge): row
                        for row, edge in enumerate(problem.model_edge_indices)
                    }
                    rows = np.asarray(
                        [model_lookup[int(edge)] for edge in edges_h],
                        dtype=np.int32,
                    )
                    residual = (observed - model_velocity[rows]) / sigma
                    chi2 = float(np.sum(np.abs(residual) ** 2))
                    score = chi2 if method == "linear" else -0.5 * chi2
                    pressure = fixed_field[problem.boundary_node_indices]
                    transfer = None
                    transfers[harmonic] = (
                        local_pressure,
                        model_velocity,
                    )
                else:
                    transfer = build_transfer_operator(
                        dataset, problem, D0, float(alpha), harmonic
                    )
                    operator = transfer.operator[valid]
                    prior_kwargs = {
                        "pressure_reference": pressure_reference,
                        "pressure_mode": (
                            pressure_conditioning.mode
                            if pressure_conditioning is not None
                            else "off"
                        ),
                        "pressure_weight": (
                            pressure_conditioning.weight
                            if pressure_conditioning is not None
                            else 0.0
                        ),
                        "pressure_sigma_pa": (
                            pressure_conditioning.sigma_pa
                            if pressure_conditioning is not None
                            else 0.0
                        ),
                    }
                    if method == "linear":
                        pressure, _, score = _weighted_complex_fit(
                            operator,
                            observed,
                            sigma,
                            regularization,
                            **prior_kwargs,
                        )
                    elif method == "bayesian":
                        score, pressure = _marginal_log_likelihood(
                            operator,
                            observed,
                            sigma,
                            pressure_scale,
                            **prior_kwargs,
                        )
                    else:
                        raise ValueError(method)
                    transfers[harmonic] = transfer
                boundary_pressures[harmonic] = pressure
                total += score
            if valid_point:
                if method == "bayesian":
                    prior_D = -0.5 * (
                        (
                            float(logD)
                            - float(
                                config["bayesian_solver"][
                                    "log_D0_prior_mean"
                                ]
                            )
                        )
                        / float(
                            config["bayesian_solver"]["log_D0_prior_sd"]
                        )
                    ) ** 2
                    prior_alpha = -0.5 * (
                        (
                            float(alpha)
                            - float(
                                config["bayesian_solver"][
                                    "alpha_prior_mean"
                                ]
                            )
                        )
                        / float(
                            config["bayesian_solver"]["alpha_prior_sd"]
                        )
                    ) ** 2
                    total += prior_D + prior_alpha
                objective[i, j] = total
                cache[(i, j)] = (boundary_pressures, transfers)

    if method == "linear":
        best_flat = int(np.nanargmin(objective))
    else:
        best_flat = int(np.nanargmax(objective))
    best_i, best_j = np.unravel_index(best_flat, objective.shape)
    D0_hat = 10.0 ** float(logD_grid[best_i])
    alpha_hat = float(alpha_grid[best_j])
    best_pressures, best_transfers = cache[(best_i, best_j)]

    predicted_velocity = np.full(
        (dataset.n_edges, 3), np.nan + 1j * np.nan, dtype=np.complex128
    )
    predicted_pressure = np.full(
        (dataset.n_nodes, 3), np.nan + 1j * np.nan, dtype=np.complex128
    )
    for harmonic in harmonics:
        transfer = best_transfers[harmonic]
        pressure_boundary = best_pressures[harmonic]
        if isinstance(transfer, tuple):
            local_pressure, model_velocity = transfer
            local_node_indices = problem.node_indices
        else:
            local_pressure = transfer.pressure_basis @ pressure_boundary
            model_velocity = (
                transfer.model_velocity_operator @ pressure_boundary
            )
            local_node_indices = transfer.local_node_indices
        predicted_pressure[
            local_node_indices, harmonic
        ] = local_pressure
        predicted_velocity[
            problem.model_edge_indices, harmonic
        ] = model_velocity

    if method == "linear":
        D_log_interval = _profile_interval(
            logD_grid, objective, axis=1, threshold=3.84
        )
        if len(alpha_grid) == 1:
            alpha_interval = (alpha_hat, alpha_hat)
        else:
            alpha_interval = _profile_interval(
                alpha_grid, objective, axis=0, threshold=3.84
            )
        D_interval = (
            10.0 ** D_log_interval[0],
            10.0 ** D_log_interval[1],
        )
        surface_for_output = objective - np.nanmin(objective)
    else:
        log_mass = objective - logsumexp(objective[np.isfinite(objective)])
        mass = np.exp(log_mass)
        D_log_interval = _credible_interval(logD_grid, mass)
        D_interval = (
            10.0 ** D_log_interval[0],
            10.0 ** D_log_interval[1],
        )
        if len(alpha_grid) == 1:
            alpha_interval = (alpha_hat, alpha_hat)
        else:
            alpha_interval = _credible_interval(
                alpha_grid, mass.T
            )
        surface_for_output = mass

    evaluation_code = {"test": 2, "train": 0}.get(
        config["solver"].get("evaluate_edges", "test"), 2
    )
    evaluation = (
        dataset.edge_split_code == evaluation_code
    ) & np.isfinite(predicted_velocity[:, harmonics[0]])
    errors = []
    truths = []
    for harmonic in harmonics:
        valid = (
            evaluation
            & dataset.observation_valid[:, harmonic]
            & np.isfinite(predicted_velocity[:, harmonic])
        )
        if valid.any():
            errors.append(
                predicted_velocity[valid, harmonic]
                - dataset.velocity_true_m_s[valid, harmonic]
            )
            truths.append(dataset.velocity_true_m_s[valid, harmonic])
    if errors:
        error = np.concatenate(errors)
        truth = np.concatenate(truths)
        relative_rmse = float(
            np.sqrt(np.mean(np.abs(error) ** 2))
            / max(np.sqrt(np.mean(np.abs(truth) ** 2)), 1.0e-30)
        )
    else:
        relative_rmse = float("nan")
    boundary_hit = bool(
        best_i in {0, len(logD_grid) - 1}
        or (
            len(alpha_grid) > 1
            and best_j in {0, len(alpha_grid) - 1}
        )
    )
    metrics = {
        "D0_true": dataset.D0_true,
        "alpha_true": dataset.alpha_true,
        "D0_hat": D0_hat,
        "alpha_hat": alpha_hat,
        "log10_D0_error": abs(
            math.log10(D0_hat) - math.log10(dataset.D0_true)
        ),
        "relative_D0_error": abs(D0_hat - dataset.D0_true)
        / dataset.D0_true,
        "alpha_absolute_error": abs(alpha_hat - dataset.alpha_true),
        "D0_interval_low": D_interval[0],
        "D0_interval_high": D_interval[1],
        "alpha_interval_low": alpha_interval[0],
        "alpha_interval_high": alpha_interval[1],
        "D0_interval_covers_true": bool(
            D_interval[0] <= dataset.D0_true <= D_interval[1]
        ),
        "alpha_interval_covers_true": bool(
            alpha_interval[0]
            <= dataset.alpha_true
            <= alpha_interval[1]
        ),
        "D0_interval_width_decades": math.log10(
            D_interval[1] / D_interval[0]
        )
        if D_interval[0] > 0
        else float("nan"),
        "alpha_interval_width": alpha_interval[1] - alpha_interval[0],
        "boundary_hit": boundary_hit,
        "held_out_velocity_relative_rmse": relative_rmse,
        "n_fit_edges": int(len(problem.observed_edge_indices)),
        "n_model_edges": int(len(problem.model_edge_indices)),
        "n_boundary_nodes": int(len(problem.boundary_node_indices)),
        "harmonics_used": list(harmonics),
        "pressure_conditioning_mode": (
            pressure_conditioning.mode
            if pressure_conditioning is not None
            else "off"
        ),
        "pressure_conditioning_source": (
            str(pressure_conditioning.source)
            if pressure_conditioning is not None
            else None
        ),
        "fixed_pressure_harmonics": (
            [
                int(h)
                for h in harmonics
                if pressure_conditioning is not None
                and pressure_conditioning.fix_available_harmonics
                and pressure_conditioning.field(h) is not None
            ]
        ),
    }
    return SolverResult(
        problem_name=problem.name,
        method=method,
        D0_hat=D0_hat,
        alpha_hat=alpha_hat,
        D0_interval=D_interval,
        alpha_interval=alpha_interval,
        surface=surface_for_output,
        log10_D0_grid=logD_grid,
        alpha_grid=alpha_grid,
        predicted_velocity=predicted_velocity,
        predicted_pressure=predicted_pressure,
        boundary_pressure=best_pressures,
        metrics=metrics,
    )
