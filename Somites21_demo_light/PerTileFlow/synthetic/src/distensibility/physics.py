"""Harmonic compliant-network operators used by every inverse solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

from .io import VascularDataset


@dataclass
class SpatialProblem:
    """One whole-mosaic or tile-local pressure/velocity inverse problem."""

    name: str
    node_indices: np.ndarray
    model_edge_indices: np.ndarray
    observed_edge_indices: np.ndarray
    boundary_node_indices: np.ndarray
    interior_node_indices: np.ndarray


@dataclass
class TransferResult:
    """Boundary-pressure transfer operator and reconstruction ingredients."""

    operator: np.ndarray
    pressure_basis: np.ndarray
    local_node_indices: np.ndarray
    model_velocity_operator: np.ndarray


def split_mask(dataset: VascularDataset, selection: str) -> np.ndarray:
    codes = {"train": {0}, "train_val": {0, 1}, "all": {0, 1, 2}}
    if selection not in codes:
        raise ValueError(f"Unknown edge selection: {selection}")
    return np.isin(dataset.edge_split_code, list(codes[selection]))


def whole_mosaic_problem(
    dataset: VascularDataset, fit_edges: str
) -> SpatialProblem:
    observed = np.flatnonzero(split_mask(dataset, fit_edges))
    all_nodes = np.arange(dataset.n_nodes, dtype=np.int32)
    boundary = np.asarray(dataset.boundary_node_index, dtype=np.int32)
    interior = np.setdiff1d(all_nodes, boundary, assume_unique=False)
    return SpatialProblem(
        name="whole_mosaic",
        node_indices=all_nodes,
        model_edge_indices=np.arange(dataset.n_edges, dtype=np.int32),
        observed_edge_indices=observed.astype(np.int32),
        boundary_node_indices=boundary,
        interior_node_indices=interior.astype(np.int32),
    )


def tile_problems(
    dataset: VascularDataset,
    fit_edges: str,
    tile_ids: Sequence[int] | None = None,
) -> list[SpatialProblem]:
    """Build spatial-bounding-box tile carves from stored tile membership."""
    tile_edges = dataset.tile_edge_indices()
    requested = set(int(tile) for tile in tile_ids) if tile_ids else None
    fit = split_mask(dataset, fit_edges)
    problems = []
    source = dataset.edge_source_index
    target = dataset.edge_target_index
    xy = dataset.node_xy_px

    for tile_id, measured_edges in tile_edges.items():
        if requested is not None and tile_id not in requested:
            continue
        tile_nodes = np.unique(
            np.concatenate([source[measured_edges], target[measured_edges]])
        )
        finite_xy = np.isfinite(xy[tile_nodes]).all(axis=1)
        if finite_xy.any():
            coords = xy[tile_nodes[finite_xy]]
            xmin, ymin = np.min(coords, axis=0)
            xmax, ymax = np.max(coords, axis=0)
            inside = (
                np.isfinite(xy).all(axis=1)
                & (xy[:, 0] >= xmin)
                & (xy[:, 0] <= xmax)
                & (xy[:, 1] >= ymin)
                & (xy[:, 1] <= ymax)
            )
            model_edges = np.flatnonzero(inside[source] & inside[target])
        else:
            model_edges = measured_edges
        model_nodes = np.unique(
            np.concatenate([source[model_edges], target[model_edges]])
        )
        node_in_model = np.zeros(dataset.n_nodes, dtype=bool)
        node_in_model[model_nodes] = True
        edge_in_model = np.zeros(dataset.n_edges, dtype=bool)
        edge_in_model[model_edges] = True
        boundary_set = set()
        for edge_index in np.flatnonzero(~edge_in_model):
            u = int(source[edge_index])
            v = int(target[edge_index])
            if node_in_model[u]:
                boundary_set.add(u)
            if node_in_model[v]:
                boundary_set.add(v)
        boundary_set.update(
            int(node)
            for node in dataset.boundary_node_index
            if node_in_model[int(node)]
        )
        boundary = np.asarray(sorted(boundary_set), dtype=np.int32)
        if len(boundary) < 2:
            degree = np.zeros(dataset.n_nodes, dtype=np.int32)
            np.add.at(degree, source[model_edges], 1)
            np.add.at(degree, target[model_edges], 1)
            fallback = model_nodes[np.argsort(degree[model_nodes])[-2:]]
            boundary = np.unique(np.concatenate([boundary, fallback])).astype(
                np.int32
            )
        interior = np.setdiff1d(model_nodes, boundary, assume_unique=False)
        observed = measured_edges[fit[measured_edges]]
        if len(observed) < max(8, len(boundary)):
            continue
        problems.append(
            SpatialProblem(
                name=f"tile_{tile_id:03d}",
                node_indices=model_nodes.astype(np.int32),
                model_edge_indices=model_edges.astype(np.int32),
                observed_edge_indices=observed.astype(np.int32),
                boundary_node_indices=boundary,
                interior_node_indices=interior.astype(np.int32),
            )
        )
    return problems


def edge_admittances(
    radius_m: np.ndarray,
    length_m: np.ndarray,
    D0: float,
    alpha: float,
    R0_m: float,
    harmonic: int,
    frequency_hz: float,
    viscosity_pa_s: float,
    density_kg_m3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distributed compliant-tube self and trans admittances."""
    D_edge = float(D0) * (radius_m / float(R0_m)) ** float(alpha)
    omega = 2.0 * np.pi * float(harmonic) * float(frequency_hz)
    resistance_per_length = (
        8.0 * float(viscosity_pa_s) / (np.pi * radius_m**4)
    )
    inertance_per_length = float(density_kg_m3) / (np.pi * radius_m**2)
    compliance_per_length = np.pi * radius_m**2 * D_edge
    series_impedance = resistance_per_length + 1j * omega * inertance_per_length
    shunt_admittance = 1j * omega * compliance_per_length
    gamma = np.sqrt(series_impedance * shunt_admittance)
    characteristic_impedance = np.sqrt(
        series_impedance / shunt_admittance
    )
    xi = gamma * length_m
    inv_characteristic = 1.0 / characteristic_impedance
    self_admittance = np.empty_like(xi, dtype=np.complex128)
    trans_admittance = np.empty_like(xi, dtype=np.complex128)
    small = np.abs(xi) < 1.0e-5
    heavy = (~small) & (np.real(xi) > 500.0)
    regular = ~(small | heavy)
    self_admittance[small] = inv_characteristic[small] * (
        1.0 / xi[small] + xi[small] / 3.0
    )
    trans_admittance[small] = inv_characteristic[small] * (
        1.0 / xi[small] - xi[small] / 6.0
    )
    self_admittance[heavy] = inv_characteristic[heavy]
    trans_admittance[heavy] = 0.0
    self_admittance[regular] = inv_characteristic[regular] / np.tanh(
        xi[regular]
    )
    trans_admittance[regular] = inv_characteristic[regular] / np.sinh(
        xi[regular]
    )
    return self_admittance, trans_admittance


def build_transfer_operator(
    dataset: VascularDataset,
    problem: SpatialProblem,
    D0: float,
    alpha: float,
    harmonic: int,
) -> TransferResult:
    """Map boundary pressure phasors to observed and modeled velocities."""
    boundary = np.asarray(problem.boundary_node_indices, dtype=np.int32)
    interior = np.asarray(problem.interior_node_indices, dtype=np.int32)
    local_nodes = np.concatenate([boundary, interior])
    local_index = np.full(dataset.n_nodes, -1, dtype=np.int32)
    local_index[local_nodes] = np.arange(len(local_nodes), dtype=np.int32)
    model_edges = problem.model_edge_indices
    u_global = dataset.edge_source_index[model_edges]
    v_global = dataset.edge_target_index[model_edges]
    u = local_index[u_global]
    v = local_index[v_global]
    ys, yt = edge_admittances(
        dataset.edge_radius_m[model_edges],
        dataset.edge_length_m[model_edges],
        D0,
        alpha,
        dataset.R0_m,
        harmonic,
        dataset.frequency_hz,
        dataset.viscosity_pa_s,
        dataset.density_kg_m3,
    )

    rows = np.concatenate([u, v, u, v])
    cols = np.concatenate([u, v, v, u])
    values = np.concatenate([ys, ys, -yt, -yt])
    Y = coo_matrix(
        (values, (rows, cols)), shape=(len(local_nodes), len(local_nodes))
    ).tocsc()
    n_boundary = len(boundary)
    n_interior = len(interior)
    pressure_basis = np.zeros(
        (len(local_nodes), n_boundary), dtype=np.complex128
    )
    pressure_basis[:n_boundary] = np.eye(n_boundary, dtype=np.complex128)
    if n_interior:
        Y_ii = csc_matrix(Y[n_boundary:, n_boundary:])
        Y_ib = np.asarray(Y[n_boundary:, :n_boundary].todense())
        try:
            pressure_basis[n_boundary:] = splu(Y_ii).solve(-Y_ib)
        except RuntimeError:
            dense = np.asarray(Y_ii.todense())
            pressure_basis[n_boundary:] = np.linalg.lstsq(
                dense, -Y_ib, rcond=1.0e-12
            )[0]

    area = dataset.edge_area_m2[model_edges]
    model_operator = (
        ys[:, None] * pressure_basis[u]
        - yt[:, None] * pressure_basis[v]
    ) / area[:, None]
    model_lookup = {
        int(edge_index): row for row, edge_index in enumerate(model_edges)
    }
    observed_rows = np.asarray(
        [model_lookup[int(edge)] for edge in problem.observed_edge_indices],
        dtype=np.int32,
    )
    return TransferResult(
        operator=model_operator[observed_rows],
        pressure_basis=pressure_basis,
        local_node_indices=local_nodes,
        model_velocity_operator=model_operator,
    )


def complex_to_real(operator: np.ndarray, values: np.ndarray):
    """Convert a complex linear system y=Tz to a real-stacked system."""
    real_operator = np.block(
        [
            [operator.real, -operator.imag],
            [operator.imag, operator.real],
        ]
    )
    real_values = np.concatenate([values.real, values.imag])
    return real_operator, real_values
