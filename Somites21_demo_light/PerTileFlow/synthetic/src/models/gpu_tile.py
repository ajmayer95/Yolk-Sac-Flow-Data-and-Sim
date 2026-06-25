"""Batched CUDA parameter-grid engine for tile-specific classical solvers."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from scipy.special import logsumexp

from distensibility.io import VascularDataset
from distensibility.physics import SpatialProblem
from distensibility.pressure import PressureConditioning, pressure_sigma
from models._shared import (
    SolverResult,
    _credible_interval,
    _noise_sigma,
    _profile_interval,
    parameter_grids,
    solve_problem,
)


def _edge_admittances_torch(
    radius,
    length,
    D0,
    alpha,
    R0,
    harmonic,
    frequency,
    viscosity,
    density,
):
    distensibility = D0[:, None] * (radius[None, :] / R0) ** alpha[:, None]
    omega = 2.0 * math.pi * float(harmonic) * frequency
    resistance = 8.0 * viscosity / (math.pi * radius**4)
    inertance = density / (math.pi * radius**2)
    compliance = math.pi * radius[None, :] ** 2 * distensibility
    series = resistance[None, :] + 1j * omega * inertance[None, :]
    shunt = 1j * omega * compliance
    gamma = torch.sqrt(series * shunt)
    impedance = torch.sqrt(series / shunt)
    xi = gamma * length[None, :]
    inv_impedance = 1.0 / impedance
    small = torch.abs(xi) < 1.0e-5
    heavy = (~small) & (torch.real(xi) > 500.0)
    regular = ~(small | heavy)
    self_admittance = torch.empty_like(xi)
    trans_admittance = torch.empty_like(xi)
    self_admittance[small] = inv_impedance[small] * (
        1.0 / xi[small] + xi[small] / 3.0
    )
    trans_admittance[small] = inv_impedance[small] * (
        1.0 / xi[small] - xi[small] / 6.0
    )
    self_admittance[heavy] = inv_impedance[heavy]
    trans_admittance[heavy] = 0.0
    self_admittance[regular] = inv_impedance[regular] / torch.tanh(
        xi[regular]
    )
    trans_admittance[regular] = inv_impedance[regular] / torch.sinh(
        xi[regular]
    )
    return self_admittance, trans_admittance


def _operators(
    dataset: VascularDataset,
    problem: SpatialProblem,
    D0: torch.Tensor,
    alpha: torch.Tensor,
    harmonic: int,
    device: torch.device,
):
    boundary = np.asarray(problem.boundary_node_indices, dtype=np.int64)
    interior = np.asarray(problem.interior_node_indices, dtype=np.int64)
    nodes = np.concatenate([boundary, interior])
    lookup = np.full(dataset.n_nodes, -1, dtype=np.int64)
    lookup[nodes] = np.arange(len(nodes))
    model_edges = problem.model_edge_indices
    u_np = lookup[dataset.edge_source_index[model_edges]]
    v_np = lookup[dataset.edge_target_index[model_edges]]
    u = torch.as_tensor(u_np, device=device)
    v = torch.as_tensor(v_np, device=device)
    radius = torch.as_tensor(
        dataset.edge_radius_m[model_edges],
        dtype=torch.float64,
        device=device,
    )
    length = torch.as_tensor(
        dataset.edge_length_m[model_edges],
        dtype=torch.float64,
        device=device,
    )
    area = torch.as_tensor(
        dataset.edge_area_m2[model_edges],
        dtype=torch.float64,
        device=device,
    )
    ys, yt = _edge_admittances_torch(
        radius,
        length,
        D0,
        alpha,
        dataset.R0_m,
        harmonic,
        dataset.frequency_hz,
        dataset.viscosity_pa_s,
        dataset.density_kg_m3,
    )
    batch = len(D0)
    n_nodes = len(nodes)
    flat = torch.zeros(
        (batch, n_nodes * n_nodes),
        dtype=torch.complex128,
        device=device,
    )
    indices = (
        u * n_nodes + u,
        v * n_nodes + v,
        u * n_nodes + v,
        v * n_nodes + u,
    )
    for index, values in zip(indices, (ys, ys, -yt, -yt)):
        flat.scatter_add_(
            1, index[None, :].expand(batch, -1), values
        )
    nodal = flat.reshape(batch, n_nodes, n_nodes)
    n_boundary = len(boundary)
    pressure_basis = torch.zeros(
        (batch, n_nodes, n_boundary),
        dtype=torch.complex128,
        device=device,
    )
    pressure_basis[:, :n_boundary, :] = torch.eye(
        n_boundary, dtype=torch.complex128, device=device
    )
    if len(interior):
        pressure_basis[:, n_boundary:, :] = torch.linalg.solve(
            nodal[:, n_boundary:, n_boundary:],
            -nodal[:, n_boundary:, :n_boundary],
        )
    model_operator = (
        ys[:, :, None] * pressure_basis[:, u, :]
        - yt[:, :, None] * pressure_basis[:, v, :]
    ) / area[None, :, None]
    model_lookup = {
        int(edge): row for row, edge in enumerate(model_edges)
    }
    observed_rows = torch.as_tensor(
        [model_lookup[int(edge)] for edge in problem.observed_edge_indices],
        dtype=torch.long,
        device=device,
    )
    return model_operator[:, observed_rows, :]


def _linear_scores(
    operator,
    observed,
    sigma,
    regularization,
    pressure_reference=None,
    pressure_mode="off",
    pressure_weight=0.0,
    pressure_sigma_pa=0.0,
):
    weighted = operator / sigma[None, :, None]
    target = (observed[None, :] / sigma[None, :]).expand(
        operator.shape[0], -1
    )
    if (
        pressure_reference is not None
        and pressure_mode != "off"
        and pressure_weight > 0
    ):
        reference = np.asarray(pressure_reference, dtype=float)
        reference = reference - float(np.mean(reference))
        prior_sigma = pressure_sigma(reference, pressure_sigma_pa)
        weight = math.sqrt(pressure_weight / max(len(reference), 1))
        weight /= prior_sigma
        reference_t = torch.as_tensor(
            reference, dtype=torch.float64, device=operator.device
        )
        if pressure_mode == "scaled":
            denominator = torch.dot(reference_t, reference_t)
            if float(denominator) > 1.0e-30:
                penalty = torch.eye(
                    len(reference),
                    dtype=torch.float64,
                    device=operator.device,
                ) - torch.outer(reference_t, reference_t) / denominator
                prior_operator = (
                    weight * penalty.to(torch.complex128)
                )[None, :, :].expand(operator.shape[0], -1, -1)
                prior_target = torch.zeros(
                    (operator.shape[0], len(reference)),
                    dtype=torch.complex128,
                    device=operator.device,
                )
                weighted = torch.cat([weighted, prior_operator], dim=1)
                target = torch.cat([target, prior_target], dim=1)
        else:
            prior_operator = (
                weight
                * torch.eye(
                    len(reference),
                    dtype=torch.complex128,
                    device=operator.device,
                )
            )[None, :, :].expand(operator.shape[0], -1, -1)
            prior_target = (
                weight * reference_t.to(torch.complex128)
            )[None, :].expand(operator.shape[0], -1)
            weighted = torch.cat([weighted, prior_operator], dim=1)
            target = torch.cat([target, prior_target], dim=1)
    normal = weighted.mH @ weighted
    diagonal = torch.abs(torch.diagonal(normal, dim1=-2, dim2=-1))
    scale = torch.median(diagonal, dim=-1).values.clamp_min(1.0e-30)
    identity = torch.eye(
        normal.shape[-1], dtype=normal.dtype, device=normal.device
    )
    normal = normal + regularization * scale[:, None, None] * identity
    rhs = weighted.mH @ target[:, :, None]
    pressure = torch.linalg.solve(normal, rhs).squeeze(-1)
    residual = target - torch.einsum("bmn,bn->bm", weighted, pressure)
    return torch.sum(torch.abs(residual) ** 2, dim=-1)


def _bayesian_scores(
    operator,
    observed,
    sigma,
    pressure_scale,
    pressure_reference=None,
    pressure_mode="off",
    pressure_weight=0.0,
    pressure_sigma_pa=0.0,
):
    real_operator = torch.cat(
        [
            torch.cat([operator.real, -operator.imag], dim=-1),
            torch.cat([operator.imag, operator.real], dim=-1),
        ],
        dim=1,
    )
    real_observed = torch.cat([observed.real, observed.imag])
    variance = torch.cat([0.5 * sigma**2, 0.5 * sigma**2])
    inverse = 1.0 / variance.clamp_min(1.0e-30)
    beta = max(0.5 * pressure_scale**2, 1.0e-30)
    identity = torch.eye(
        real_operator.shape[-1],
        dtype=torch.float64,
        device=operator.device,
    )
    prior_precision = (1.0 / beta) * identity
    prior_mean = torch.zeros(
        real_operator.shape[-1],
        dtype=torch.float64,
        device=operator.device,
    )
    n_boundary = operator.shape[-1]
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
        reference_t = torch.as_tensor(
            reference, dtype=torch.float64, device=operator.device
        )
        if pressure_mode == "scaled":
            denominator = torch.dot(reference_t, reference_t)
            if float(denominator) > 1.0e-30:
                penalty = torch.eye(
                    n_boundary,
                    dtype=torch.float64,
                    device=operator.device,
                ) - torch.outer(reference_t, reference_t) / denominator
                prior_precision[:n_boundary, :n_boundary] += (
                    strength * penalty
                )
                prior_precision[n_boundary:, n_boundary:] += (
                    strength * penalty
                )
        else:
            prior_precision += strength * identity
            prior_mean[:n_boundary] = reference_t
    precision = (
        real_operator.transpose(-2, -1)
        @ (real_operator * inverse[None, :, None])
        + prior_precision[None, :, :]
    )
    rhs = (
        real_operator.transpose(-2, -1)
        @ (inverse * real_observed)[None, :, None]
        + (prior_precision @ prior_mean)[None, :, None]
    )
    jitter = (
        torch.median(
            torch.diagonal(precision, dim1=-2, dim2=-1), dim=-1
        ).values
        * 1.0e-12
    ).clamp_min(1.0e-18)
    precision = precision + jitter[:, None, None] * identity
    posterior = torch.linalg.solve(precision, rhs).squeeze(-1)
    sign, logdet_precision = torch.linalg.slogdet(precision)
    if not bool(torch.all(sign > 0)):
        raise RuntimeError("Non-positive Bayesian precision determinant")
    prior_sign, logdet_prior_precision = torch.linalg.slogdet(
        prior_precision
    )
    if float(prior_sign) <= 0:
        raise RuntimeError("Non-positive Bayesian prior precision determinant")
    logdet = (
        torch.sum(torch.log(variance))
        - logdet_prior_precision
        + logdet_precision
    )
    quadratic = (
        torch.sum(real_observed**2 * inverse)
        + prior_mean @ prior_precision @ prior_mean
        - torch.sum(rhs.squeeze(-1) * posterior, dim=-1)
    ).clamp_min(0.0)
    return -0.5 * (
        logdet
        + quadratic
        + real_observed.numel() * math.log(2.0 * math.pi)
    )


def _finish(
    dataset,
    problem,
    config,
    method,
    objective,
    logD_grid,
    alpha_grid,
    pressure_conditioning=None,
):
    best_flat = (
        int(np.nanargmin(objective))
        if method == "linear"
        else int(np.nanargmax(objective))
    )
    best_i, best_j = np.unravel_index(best_flat, objective.shape)
    single = copy.deepcopy(config)
    single["parameter_grid"]["log10_D0_min"] = float(logD_grid[best_i])
    single["parameter_grid"]["log10_D0_max"] = float(logD_grid[best_i])
    single["parameter_grid"]["num_D0"] = 1
    single["solver"]["alpha_mode"] = "prescribed"
    single["solver"]["prescribed_alpha"] = float(alpha_grid[best_j])
    result = solve_problem(
        dataset,
        problem,
        single,
        method,
        pressure_conditioning=pressure_conditioning,
    )
    result.log10_D0_grid = logD_grid
    result.alpha_grid = alpha_grid
    result.D0_hat = 10.0 ** float(logD_grid[best_i])
    result.alpha_hat = float(alpha_grid[best_j])
    if method == "linear":
        d_log_interval = _profile_interval(
            logD_grid, objective, axis=1, threshold=3.84
        )
        alpha_interval = (
            (result.alpha_hat, result.alpha_hat)
            if len(alpha_grid) == 1
            else _profile_interval(
                alpha_grid, objective, axis=0, threshold=3.84
            )
        )
        surface = objective - np.nanmin(objective)
    else:
        mass = np.exp(
            objective - logsumexp(objective[np.isfinite(objective)])
        )
        d_log_interval = _credible_interval(logD_grid, mass)
        alpha_interval = (
            (result.alpha_hat, result.alpha_hat)
            if len(alpha_grid) == 1
            else _credible_interval(alpha_grid, mass.T)
        )
        surface = mass
    result.D0_interval = (
        10.0 ** d_log_interval[0],
        10.0 ** d_log_interval[1],
    )
    result.alpha_interval = alpha_interval
    result.surface = surface
    metrics = result.metrics
    metrics.update(
        {
            "D0_hat": result.D0_hat,
            "alpha_hat": result.alpha_hat,
            "log10_D0_error": abs(
                math.log10(result.D0_hat)
                - math.log10(dataset.D0_true)
            ),
            "relative_D0_error": abs(
                result.D0_hat - dataset.D0_true
            )
            / dataset.D0_true,
            "alpha_absolute_error": abs(
                result.alpha_hat - dataset.alpha_true
            ),
            "D0_interval_low": result.D0_interval[0],
            "D0_interval_high": result.D0_interval[1],
            "alpha_interval_low": alpha_interval[0],
            "alpha_interval_high": alpha_interval[1],
            "D0_interval_covers_true": bool(
                result.D0_interval[0]
                <= dataset.D0_true
                <= result.D0_interval[1]
            ),
            "alpha_interval_covers_true": bool(
                alpha_interval[0]
                <= dataset.alpha_true
                <= alpha_interval[1]
            ),
            "D0_interval_width_decades": math.log10(
                result.D0_interval[1] / result.D0_interval[0]
            ),
            "alpha_interval_width": alpha_interval[1]
            - alpha_interval[0],
            "boundary_hit": bool(
                best_i in {0, len(logD_grid) - 1}
                or (
                    len(alpha_grid) > 1
                    and best_j in {0, len(alpha_grid) - 1}
                )
            ),
            "compute_backend": "torch_cuda_batched_dense",
        }
    )
    return result


def solve_tile_gpu(
    dataset: VascularDataset,
    problem: SpatialProblem,
    config: dict,
    method: str,
    device: str = "cuda",
    chunk_size: int = 64,
    pressure_conditioning: PressureConditioning | None = None,
) -> SolverResult:
    """Evaluate a tile parameter grid using batched dense CUDA solves."""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch")
    if method not in {"linear", "bayesian"}:
        raise ValueError(method)
    logD_grid, alpha_grid = parameter_grids(config)
    logD_mesh, alpha_mesh = np.meshgrid(
        logD_grid, alpha_grid, indexing="ij"
    )
    points_logD = logD_mesh.reshape(-1)
    points_alpha = alpha_mesh.reshape(-1)
    scores = np.zeros(len(points_logD), dtype=float)
    torch_device = torch.device(device)
    observed_edges = problem.observed_edge_indices
    harmonics = tuple(int(h) for h in config["solver"]["harmonics_used"])
    pressure_reference = None
    if pressure_conditioning is not None:
        for harmonic in harmonics:
            if (
                pressure_conditioning.fix_available_harmonics
                and pressure_conditioning.field(harmonic) is not None
            ):
                raise ValueError(
                    "GPU conditioning currently supports a DC spatial prior, "
                    "not fixed H1/H2 pressure fields"
                )
        dc = pressure_conditioning.field(0)
        if dc is not None:
            pressure_reference = np.asarray(
                dc[problem.boundary_node_indices].real, dtype=float
            )
            if not np.isfinite(pressure_reference).all():
                pressure_reference = None
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

    for start in range(0, len(points_logD), int(chunk_size)):
        stop = min(start + int(chunk_size), len(points_logD))
        D0 = torch.as_tensor(
            10.0 ** points_logD[start:stop],
            dtype=torch.float64,
            device=torch_device,
        )
        alpha = torch.as_tensor(
            points_alpha[start:stop],
            dtype=torch.float64,
            device=torch_device,
        )
        total = torch.zeros(stop - start, dtype=torch.float64, device=torch_device)
        for harmonic in harmonics:
            valid = dataset.observation_valid[
                observed_edges, harmonic
            ].astype(bool)
            operator = _operators(
                dataset,
                problem,
                D0,
                alpha,
                harmonic,
                torch_device,
            )[:, torch.as_tensor(valid, device=torch_device), :]
            edges = observed_edges[valid]
            observed = torch.as_tensor(
                dataset.velocity_observed_m_s[edges, harmonic],
                dtype=torch.complex128,
                device=torch_device,
            )
            sigma = torch.as_tensor(
                _noise_sigma(dataset, edges, harmonic),
                dtype=torch.float64,
                device=torch_device,
            )
            if method == "linear":
                total += _linear_scores(
                    operator,
                    observed,
                    sigma,
                    float(config["linear_solver"]["regularization"]),
                    **prior_kwargs,
                )
            else:
                total += _bayesian_scores(
                    operator,
                    observed,
                    sigma,
                    float(
                        config["bayesian_solver"]["boundary_prior_scale"]
                    ),
                    **prior_kwargs,
                )
        if method == "bayesian":
            total += -0.5 * (
                (
                    torch.log10(D0)
                    - float(
                        config["bayesian_solver"]["log_D0_prior_mean"]
                    )
                )
                / float(config["bayesian_solver"]["log_D0_prior_sd"])
            ) ** 2
            total += -0.5 * (
                (
                    alpha
                    - float(config["bayesian_solver"]["alpha_prior_mean"])
                )
                / float(config["bayesian_solver"]["alpha_prior_sd"])
            ) ** 2
        scores[start:stop] = total.detach().cpu().numpy()
    objective = scores.reshape(len(logD_grid), len(alpha_grid))
    return _finish(
        dataset,
        problem,
        config,
        method,
        objective,
        logD_grid,
        alpha_grid,
        pressure_conditioning,
    )
