"""Physics-layer abstractions for conductance-only GNN training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ConstrainedSolveResult:
    pressure_pa: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    edge_pressure_drop_pa: torch.Tensor | None = None
    edge_flow_m3_s: torch.Tensor | None = None
    nodal_residual_m3_s: torch.Tensor | None = None


def _node_label(data, index: int) -> str:
    return str(data.node_id[int(index)])


def dataset_source_vector(
    data,
    arterial_flow_mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_nodes = int(len(data.node_id))
    if arterial_flow_mode == "dataset":
        return data.boundary_injection_m3_s.to(device=device, dtype=dtype).flatten()
    if arterial_flow_mode == "none":
        return torch.zeros(num_nodes, dtype=dtype, device=device)
    raise ValueError(f"Unsupported arterial-flow mode: {arterial_flow_mode}")


def gauge_node_index(data) -> int:
    venous_nodes = data.venous_node_indices.detach().cpu().numpy().astype(np.int64)
    if venous_nodes.size:
        return int(venous_nodes[0])
    return int(data.reference_node)


def laplacian_scale_value(matrix: torch.Tensor) -> torch.Tensor:
    nonzero = torch.abs(matrix[torch.abs(matrix) > 0.0])
    if nonzero.numel() == 0:
        return matrix.new_tensor(1.0)
    return torch.median(nonzero).clamp_min(1.0e-30)


def build_pressure_constraint_rows(
    data,
    num_nodes: int,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    arterial = np.sort(data.arterial_node_indices.detach().cpu().numpy().astype(np.int64))
    venous = np.sort(data.venous_node_indices.detach().cpu().numpy().astype(np.int64))
    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []
    labels: list[str] = []

    gauge_node = int(venous[0]) if venous.size else int(data.reference_node)
    row = torch.zeros(num_nodes, dtype=dtype, device=device)
    row[gauge_node] = 1.0
    rows.append(row)
    rhs.append(torch.zeros((), dtype=dtype, device=device))
    labels.append(f"gauge:{_node_label(data, gauge_node)}=0")

    for constraint in pressure_constraints:
        if constraint in {"gauge-only", "gauge_only"}:
            continue
        if constraint == "equal-a-equal-v":
            if arterial.size >= 2:
                row = torch.zeros(num_nodes, dtype=dtype, device=device)
                row[int(arterial[0])] = 1.0
                row[int(arterial[1])] = -1.0
                rows.append(row)
                rhs.append(torch.zeros((), dtype=dtype, device=device))
                labels.append("equal_a")
            if venous.size >= 2:
                row = torch.zeros(num_nodes, dtype=dtype, device=device)
                row[int(venous[0])] = 1.0
                row[int(venous[1])] = -1.0
                rows.append(row)
                rhs.append(torch.zeros((), dtype=dtype, device=device))
                labels.append("equal_v")
        elif constraint == "equal-av-pressure-drop":
            if arterial.size < 2 or venous.size < 2:
                raise ValueError(
                    "equal-av-pressure-drop requires two arterial and two venous nodes."
                )
            row = torch.zeros(num_nodes, dtype=dtype, device=device)
            row[int(arterial[0])] = 1.0
            row[int(venous[0])] = -1.0
            row[int(arterial[1])] = -1.0
            row[int(venous[1])] = 1.0
            rows.append(row)
            rhs.append(torch.zeros((), dtype=dtype, device=device))
            labels.append("equal_av_pressure_drop")
        elif constraint == "mean-a-minus-v-alpha-equal-v":
            if arterial.size < 2 or venous.size < 1:
                raise ValueError(
                    "mean-a-minus-v-alpha-equal-v requires at least two arterial nodes and one venous node."
                )
            if alpha_pa is None or not np.isfinite(float(alpha_pa)):
                raise ValueError(
                    "A finite alpha_pa is required for mean-a-minus-v-alpha-equal-v."
                )
            if venous.size >= 2:
                row = torch.zeros(num_nodes, dtype=dtype, device=device)
                row[int(venous[0])] = 1.0
                row[int(venous[1])] = -1.0
                rows.append(row)
                rhs.append(torch.zeros((), dtype=dtype, device=device))
                labels.append("equal_v")
            row = torch.zeros(num_nodes, dtype=dtype, device=device)
            row[int(arterial[0])] = 0.5
            row[int(arterial[1])] = 0.5
            row[int(venous[0])] = -1.0
            rows.append(row)
            rhs.append(torch.tensor(float(alpha_pa), dtype=dtype, device=device))
            labels.append("mean_a_minus_v_alpha")
        else:
            raise ValueError(f"Unsupported pressure constraint: {constraint}")
    return rows, rhs, labels


def _reduced_constraint_system(
    data,
    num_nodes: int,
    unknown_nodes: torch.Tensor,
    gauge_node: int,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rows, rhs, labels = build_pressure_constraint_rows(
        data=data,
        num_nodes=num_nodes,
        pressure_constraints=pressure_constraints,
        alpha_pa=alpha_pa,
        dtype=dtype,
        device=device,
    )
    kept_rows: list[torch.Tensor] = []
    kept_rhs: list[torch.Tensor] = []
    kept_labels: list[str] = []
    for row, row_rhs, label in zip(rows, rhs, labels):
        is_gauge = bool(
            torch.count_nonzero(row).item() == 1
            and abs(float(row[gauge_node].detach().cpu()) - 1.0) < 1.0e-12
            and abs(float(row_rhs.detach().cpu())) < 1.0e-12
        )
        if is_gauge:
            continue
        kept_rows.append(row)
        kept_rhs.append(row_rhs)
        kept_labels.append(label)
    if not kept_rows:
        return (
            torch.zeros((0, int(unknown_nodes.numel())), dtype=dtype, device=device),
            torch.zeros((0,), dtype=dtype, device=device),
            [],
        )
    matrix_full = torch.stack(kept_rows, dim=0)
    rhs_full = torch.stack(kept_rhs, dim=0)
    return matrix_full.index_select(1, unknown_nodes), rhs_full, kept_labels


def _observed_flow_block(
    data,
    conductance: torch.Tensor,
    unknown_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    q_obs = (
        data.velocity_observed_m_s[:, 0, 0].to(device=device, dtype=dtype)
        * data.area_m2.to(device=device, dtype=dtype)
    )
    valid_mask = (
        (data.train_mask | data.val_mask | data.test_mask).to(device=device)
        & torch.isfinite(q_obs)
    )
    valid_edges = torch.nonzero(valid_mask, as_tuple=False).flatten()
    n_rows = int(valid_edges.numel())
    n_unknowns = int(unknown_nodes.numel())
    if n_rows == 0:
        return (
            torch.zeros((0, n_unknowns), dtype=dtype, device=device),
            torch.zeros((0,), dtype=dtype, device=device),
            0,
        )
    node_to_col = torch.full((int(len(data.node_id)),), -1, dtype=torch.long, device=device)
    node_to_col[unknown_nodes] = torch.arange(n_unknowns, dtype=torch.long, device=device)
    src = edge_index[0].index_select(0, valid_edges)
    dst = edge_index[1].index_select(0, valid_edges)
    cols_src = node_to_col[src]
    cols_dst = node_to_col[dst]
    row_idx = torch.arange(n_rows, dtype=torch.long, device=device)
    matrix = torch.zeros((n_rows, n_unknowns), dtype=dtype, device=device)
    edge_g = conductance.index_select(0, valid_edges)
    src_keep = cols_src >= 0
    dst_keep = cols_dst >= 0
    if bool(torch.any(src_keep)):
        matrix[row_idx[src_keep], cols_src[src_keep]] += edge_g[src_keep]
    if bool(torch.any(dst_keep)):
        matrix[row_idx[dst_keep], cols_dst[dst_keep]] -= edge_g[dst_keep]
    rhs = q_obs.index_select(0, valid_edges)
    return matrix, rhs, n_rows


def solve_reduced_pressure(
    data,
    conductance: torch.Tensor,
    arterial_flow_mode: str,
    pressure_solver_mode: str,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    lambda_kirchhoff: float,
    lambda_pressure_constraints: float,
    lambda_flow_residual: float,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    conductance = conductance.to(device=device, dtype=dtype)
    edge_index = data.edge_index.to(device=device)
    num_nodes = int(len(data.node_id))
    laplacian = build_weighted_laplacian(
        edge_index=edge_index,
        conductance=conductance,
        n_nodes=num_nodes,
    ).to(device=device, dtype=dtype)
    source_vector = dataset_source_vector(
        data=data,
        arterial_flow_mode=arterial_flow_mode,
        dtype=dtype,
        device=device,
    )
    gauge_node = gauge_node_index(data)
    unknown_nodes = torch.nonzero(
        torch.arange(num_nodes, device=device) != gauge_node,
        as_tuple=False,
    ).flatten()
    reduced_matrix = laplacian.index_select(0, unknown_nodes).index_select(1, unknown_nodes)
    reduced_rhs = source_vector.index_select(0, unknown_nodes)
    formulation_warning = ""
    net_injection = float(source_vector.sum().detach().cpu())
    if abs(net_injection) > 1.0e-10:
        formulation_warning = (
            "WARNING: net injection is not balanced; pure flow-driven Laplacian solve may be inconsistent."
        )

    venous_nodes = data.venous_node_indices.to(device=device, dtype=torch.long).flatten()
    arterial_nodes = data.arterial_node_indices.to(device=device, dtype=torch.long).flatten()
    reduced_constraints, reduced_constraint_rhs, constraint_labels = _reduced_constraint_system(
        data=data,
        num_nodes=num_nodes,
        unknown_nodes=unknown_nodes,
        gauge_node=gauge_node,
        pressure_constraints=pressure_constraints,
        alpha_pa=alpha_pa,
        dtype=dtype,
        device=device,
    )
    reduced_flow_matrix, reduced_flow_rhs, flow_edge_count = _observed_flow_block(
        data=data,
        conductance=conductance,
        unknown_nodes=unknown_nodes,
        edge_index=edge_index,
        dtype=dtype,
        device=device,
    )

    laplacian_scale = laplacian_scale_value(reduced_matrix)
    reduced_matrix_scaled = reduced_matrix / laplacian_scale
    reduced_rhs_scaled = reduced_rhs / laplacian_scale
    flow_row_scale = laplacian_scale_value(reduced_flow_matrix)
    reduced_flow_matrix_scaled = reduced_flow_matrix / flow_row_scale
    reduced_flow_rhs_scaled = reduced_flow_rhs / flow_row_scale

    if pressure_solver_mode == "partitioned-flow-gauge":
        used_lstsq = False
        try:
            reduced_pressure = torch.linalg.solve(reduced_matrix, reduced_rhs)
        except RuntimeError:
            used_lstsq = True
            reduced_pressure = torch.linalg.lstsq(reduced_matrix, reduced_rhs).solution
        pressure_constraints_used = "gauge_only"
    elif pressure_solver_mode == "reduced-soft-constrained-lstsq":
        if not np.isfinite(lambda_kirchhoff) or float(lambda_kirchhoff) < 0.0:
            raise ValueError("lambda_kirchhoff must be finite and non-negative.")
        if not np.isfinite(lambda_pressure_constraints) or float(lambda_pressure_constraints) < 0.0:
            raise ValueError("lambda_pressure_constraints must be finite and non-negative.")
        if not np.isfinite(lambda_flow_residual) or float(lambda_flow_residual) < 0.0:
            raise ValueError("lambda_flow_residual must be finite and non-negative.")
        block_matrices: list[torch.Tensor] = []
        block_rhs: list[torch.Tensor] = []
        if float(lambda_kirchhoff) > 0.0:
            block_matrices.append(np.sqrt(float(lambda_kirchhoff)) * reduced_matrix_scaled)
            block_rhs.append(np.sqrt(float(lambda_kirchhoff)) * reduced_rhs_scaled)
        if float(lambda_pressure_constraints) > 0.0 and int(reduced_constraints.shape[0]) > 0:
            block_matrices.append(
                np.sqrt(float(lambda_pressure_constraints)) * reduced_constraints
            )
            block_rhs.append(
                np.sqrt(float(lambda_pressure_constraints)) * reduced_constraint_rhs
            )
        if float(lambda_flow_residual) > 0.0 and int(reduced_flow_matrix.shape[0]) > 0:
            block_matrices.append(np.sqrt(float(lambda_flow_residual)) * reduced_flow_matrix_scaled)
            block_rhs.append(np.sqrt(float(lambda_flow_residual)) * reduced_flow_rhs_scaled)
        if not block_matrices:
            raise ValueError(
                "At least one reduced-soft-constrained-lstsq block must be active."
            )
        stacked_matrix = torch.cat(block_matrices, dim=0)
        stacked_rhs = torch.cat(block_rhs, dim=0)
        reduced_pressure = torch.linalg.lstsq(stacked_matrix, stacked_rhs).solution
        used_lstsq = True
        pressure_constraints_used = "reduced_soft"
    else:
        raise ValueError(f"Unsupported pressure_solver_mode: {pressure_solver_mode}")

    pressure = torch.zeros(num_nodes, dtype=dtype, device=device)
    pressure.index_copy_(0, unknown_nodes, reduced_pressure)
    pressure[gauge_node] = 0.0

    edge_pressure_drop = pressure[edge_index[0]] - pressure[edge_index[1]]
    edge_flow = conductance * edge_pressure_drop
    nodal_residual = laplacian @ pressure - source_vector
    kirchhoff_residual = reduced_matrix @ reduced_pressure - reduced_rhs
    constraint_residual = (
        reduced_constraints @ reduced_pressure - reduced_constraint_rhs
        if reduced_constraints.numel()
        else pressure.new_zeros((0,))
    )
    flow_residual = (
        reduced_flow_matrix @ reduced_pressure - reduced_flow_rhs
        if reduced_flow_matrix.numel()
        else pressure.new_zeros((0,))
    )

    diagnostics = {
        "pressure_solver_mode": pressure_solver_mode,
        "pressure_solver_used_lstsq": pressure.new_tensor(1.0 if used_lstsq else 0.0),
        "pressure_solver_relative_residual": (
            torch.linalg.vector_norm(kirchhoff_residual)
            / torch.linalg.vector_norm(reduced_rhs).clamp_min(1.0e-30)
        ),
        "pressure_solver_kirchhoff_residual_l2": torch.linalg.vector_norm(kirchhoff_residual),
        "pressure_solver_constraint_residual_l2": (
            torch.linalg.vector_norm(constraint_residual)
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "pressure_solver_constraint_residual_max": (
            torch.max(torch.abs(constraint_residual))
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "pressure_solver_flow_residual_l2": (
            torch.linalg.vector_norm(flow_residual)
            if flow_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "pressure_solver_flow_residual_rmse": (
            torch.sqrt(torch.mean(flow_residual**2))
            if flow_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "pressure_solver_flow_edge_count": pressure.new_tensor(float(flow_edge_count)),
        "pressure_solver_pressure_range_pa": torch.max(pressure) - torch.min(pressure),
        "pressure_solver_flow_row_scale": flow_row_scale,
        "pressure_solver_laplacian_scale": laplacian_scale,
        "pressure_solver_lambda_kirchhoff": pressure.new_tensor(float(lambda_kirchhoff)),
        "pressure_solver_lambda_pressure_constraints": pressure.new_tensor(
            float(lambda_pressure_constraints)
        ),
        "pressure_solver_lambda_flow_residual": pressure.new_tensor(
            float(lambda_flow_residual)
        ),
        "arterial_dataset_flow_total_m3_s": (
            source_vector[arterial_nodes].sum()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_dataset_flow_total_m3_s": (
            source_vector[venous_nodes].sum()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_predicted_flow_total_m3_s": (
            (laplacian @ pressure)[arterial_nodes].sum()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_predicted_flow_total_m3_s": (
            (laplacian @ pressure)[venous_nodes].sum()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "gauge_node_index": pressure.new_tensor(float(gauge_node)),
        "unknown_pressure_node_count": pressure.new_tensor(float(unknown_nodes.numel())),
        "reduced_constraint_count": pressure.new_tensor(float(reduced_constraints.shape[0])),
        "constraint_labels": constraint_labels if constraint_labels else ["gauge_only"],
        "pressure_constraints_used_in_solve": pressure_constraints_used,
    }
    return {
        "pressure_pa": pressure.to(dtype=torch.float32),
        "edge_pressure_drop_pa": edge_pressure_drop.to(dtype=torch.float32),
        "edge_flow_m3_s": edge_flow.to(dtype=torch.float32),
        "nodal_residual_m3_s": nodal_residual.to(dtype=torch.float32),
        "source_vector_m3_s": source_vector.to(dtype=torch.float32),
        "gauge_node_index": gauge_node,
        "gauge_node_id": _node_label(data, gauge_node),
        "pressure_prescribed_node_ids": _node_label(data, gauge_node),
        "formulation_warning": formulation_warning,
        "diagnostics": diagnostics,
    }


def apply_bounded_delta(
    raw_delta: torch.Tensor,
    delta_min: float,
    delta_max: float,
    parameterization: str = "tanh",
) -> torch.Tensor:
    if parameterization == "identity":
        return raw_delta.clamp(min=float(delta_min), max=float(delta_max))
    if parameterization != "tanh":
        raise ValueError(f"Unsupported delta parameterization: {parameterization}")
    lo = float(delta_min)
    hi = float(delta_max)
    center = 0.5 * (hi + lo)
    radius = 0.5 * (hi - lo)
    if radius <= 0.0:
        raise ValueError("delta_max must be greater than delta_min.")
    target_zero = (0.0 - center) / radius
    target_zero = min(max(target_zero, -1.0 + 1.0e-6), 1.0 - 1.0e-6)
    offset = np.arctanh(target_zero)
    return raw_delta.new_tensor(center) + raw_delta.new_tensor(radius) * torch.tanh(
        raw_delta + raw_delta.new_tensor(offset)
    )


def conductance_from_delta(
    base_conductance: torch.Tensor,
    delta_e: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    conductance_ratio = torch.exp(delta_e)
    return base_conductance * conductance_ratio, conductance_ratio


def build_weighted_laplacian(
    edge_index: torch.Tensor,
    conductance: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    source, target = edge_index
    laplacian = conductance.new_zeros((n_nodes, n_nodes))
    diagonal = conductance.new_zeros(n_nodes)
    diagonal.index_add_(0, source, conductance)
    diagonal.index_add_(0, target, conductance)
    laplacian.index_put_((source, target), -conductance, accumulate=True)
    laplacian.index_put_((target, source), -conductance, accumulate=True)
    node_index = torch.arange(n_nodes, device=conductance.device)
    laplacian[node_index, node_index] = diagonal
    return laplacian


def build_incidence_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    source, target = edge_index
    n_edges = int(source.numel())
    incidence = torch.zeros(
        (n_edges, int(num_nodes)),
        dtype=dtype or torch.float32,
        device=device or source.device,
    )
    edge_ids = torch.arange(n_edges, device=incidence.device)
    non_self = source != target
    incidence[edge_ids[non_self], source[non_self]] = 1.0
    incidence[edge_ids[non_self], target[non_self]] = -1.0
    return incidence


def build_weighted_laplacian_from_incidence(
    incidence: torch.Tensor,
    g_edge: torch.Tensor,
) -> torch.Tensor:
    weighted_incidence = g_edge[:, None] * incidence
    return incidence.transpose(0, 1) @ weighted_incidence


def solve_least_squares_with_hard_linear_constraints(
    system_matrix: torch.Tensor,
    rhs: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_rhs: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Solve min ||A x - b|| subject to C x = d."""

    n_unknowns = int(system_matrix.shape[1]) if system_matrix.ndim == 2 else 0
    if n_unknowns == 0:
        return rhs.new_zeros((0,)), False
    if constraint_matrix.numel() == 0:
        return torch.linalg.lstsq(system_matrix, rhs).solution, True

    normal_matrix = system_matrix.transpose(0, 1) @ system_matrix
    normal_rhs = system_matrix.transpose(0, 1) @ rhs
    n_constraints = int(constraint_matrix.shape[0])
    kkt_matrix = system_matrix.new_zeros(
        (n_unknowns + n_constraints, n_unknowns + n_constraints)
    )
    kkt_rhs = rhs.new_zeros((n_unknowns + n_constraints,))
    kkt_matrix[:n_unknowns, :n_unknowns] = normal_matrix
    kkt_matrix[:n_unknowns, n_unknowns:] = constraint_matrix.transpose(0, 1)
    kkt_matrix[n_unknowns:, :n_unknowns] = constraint_matrix
    kkt_rhs[:n_unknowns] = normal_rhs
    kkt_rhs[n_unknowns:] = constraint_rhs
    used_lstsq = False
    try:
        solution = torch.linalg.solve(kkt_matrix, kkt_rhs)
    except RuntimeError:
        used_lstsq = True
        solution = torch.linalg.lstsq(kkt_matrix, kkt_rhs).solution
    return solution[:n_unknowns], used_lstsq


def constrained_dc_solve_torch(
    edge_index: torch.Tensor,
    g_edge: torch.Tensor,
    num_nodes: int,
    arterial_nodes: torch.Tensor,
    arterial_flows: torch.Tensor,
    venous_nodes: torch.Tensor,
    device=None,
) -> ConstrainedSolveResult:
    """Differentiable reduced DC solve with grounded venous nodes.

    Hard constraints:
    - prescribed inflows at arterial nodes,
    - zero source/sink at all non-arterial, non-venous nodes,
    - equal venous pressure by grounding all venous nodes to 0.
    """

    if device is None:
        device = g_edge.device
    device = torch.device(device)
    dtype = g_edge.dtype
    edge_index = edge_index.to(device=device)
    g_edge = g_edge.to(device=device, dtype=dtype)
    arterial_nodes = arterial_nodes.to(device=device, dtype=torch.long).flatten()
    arterial_flows = arterial_flows.to(device=device, dtype=dtype).flatten()
    venous_nodes = venous_nodes.to(device=device, dtype=torch.long).flatten()
    if arterial_nodes.numel() != arterial_flows.numel():
        raise ValueError("arterial_nodes and arterial_flows must have the same length.")
    if venous_nodes.numel() == 0:
        raise ValueError("At least one venous node is required for grounded constrained solve.")

    incidence = build_incidence_matrix(
        edge_index,
        int(num_nodes),
        dtype=dtype,
        device=device,
    )
    laplacian = build_weighted_laplacian_from_incidence(incidence, g_edge)

    source_vector = torch.zeros(int(num_nodes), dtype=dtype, device=device)
    if arterial_nodes.numel():
        source_vector.index_add_(0, arterial_nodes, arterial_flows)

    grounded_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    grounded_mask[venous_nodes] = True
    free_mask = ~grounded_mask
    free_nodes = torch.nonzero(free_mask, as_tuple=False).flatten()
    grounded_nodes = torch.nonzero(grounded_mask, as_tuple=False).flatten()

    reduced_laplacian = laplacian.index_select(0, free_nodes).index_select(1, free_nodes)
    reduced_rhs = source_vector.index_select(0, free_nodes)
    if grounded_nodes.numel():
        grounded_pressure = torch.zeros(grounded_nodes.numel(), dtype=dtype, device=device)
        coupling = laplacian.index_select(0, free_nodes).index_select(1, grounded_nodes)
        reduced_rhs = reduced_rhs - coupling @ grounded_pressure
    pressure_free = torch.linalg.solve(reduced_laplacian, reduced_rhs)
    pressure = torch.zeros(int(num_nodes), dtype=dtype, device=device)
    pressure[free_nodes] = pressure_free
    pressure[grounded_nodes] = 0.0

    edge_pressure_drop = incidence @ pressure
    edge_flow = g_edge * edge_pressure_drop
    nodal_residual = laplacian @ pressure - source_vector

    arterial_residual = nodal_residual[arterial_nodes] if arterial_nodes.numel() else nodal_residual[:0]
    internal_mask = free_mask.clone()
    if arterial_nodes.numel():
        internal_mask[arterial_nodes] = False
    internal_nodes = torch.nonzero(internal_mask, as_tuple=False).flatten()
    internal_residual = nodal_residual[internal_nodes] if internal_nodes.numel() else nodal_residual[:0]
    diagnostics = {
        "pressure_solver_iterations_used": pressure.new_tensor(1.0),
        "pressure_solver_final_relative_residual": (
            torch.linalg.vector_norm(
                torch.cat([arterial_residual, internal_residual])
                if arterial_residual.numel() or internal_residual.numel()
                else nodal_residual[:1] * 0.0
            )
            / torch.linalg.vector_norm(source_vector).clamp_min(1.0e-30)
        ),
        "pressure_solver_reached_max_iterations": pressure.new_tensor(0.0),
        "hard_boundary_constraint_residual_l2": (
            torch.linalg.vector_norm(
                torch.cat([arterial_residual, internal_residual])
                if arterial_residual.numel() or internal_residual.numel()
                else nodal_residual[:1] * 0.0
            )
        ),
        "hard_boundary_constraint_residual_max": (
            torch.max(
                torch.abs(
                    torch.cat([arterial_residual, internal_residual])
                    if arterial_residual.numel() or internal_residual.numel()
                    else nodal_residual[:1] * 0.0
                )
            )
        ),
        "venous_pressure_spread_pa": (
            pressure[venous_nodes].max() - pressure[venous_nodes].min()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_ground_abs_max_pa": (
            torch.max(torch.abs(pressure[venous_nodes]))
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_inflow_total_m3_s": arterial_flows.sum() if arterial_flows.numel() else pressure.new_tensor(0.0),
        "venous_outflow_total_m3_s": nodal_residual[venous_nodes].sum() if venous_nodes.numel() else pressure.new_tensor(0.0),
    }
    return ConstrainedSolveResult(
        pressure_pa=pressure,
        diagnostics=diagnostics,
        edge_pressure_drop_pa=edge_pressure_drop,
        edge_flow_m3_s=edge_flow,
        nodal_residual_m3_s=nodal_residual,
    )


def constrained_dc_solve_equal_A_equal_V_torch(
    edge_index: torch.Tensor,
    g_edge: torch.Tensor,
    num_nodes: int,
    arterial_nodes: torch.Tensor,
    arterial_flows: torch.Tensor,
    venous_nodes: torch.Tensor,
    device=None,
) -> ConstrainedSolveResult:
    """Differentiable DC solve with equal arterial/venous boundary pressures.

    The two arterial nodes share one pressure variable, the venous nodes are
    grounded to the same pressure, and arterial nodal flows are prescribed.
    A hard linear flow constraint additionally enforces that total venous
    outflow equals total prescribed arterial inflow. The remaining reduced
    Kirchhoff equations are fit in least-squares form subject to those hard
    constraints.
    """

    if device is None:
        device = g_edge.device
    device = torch.device(device)
    dtype = g_edge.dtype
    edge_index = edge_index.to(device=device)
    g_edge = g_edge.to(device=device, dtype=dtype)
    arterial_nodes = torch.unique(arterial_nodes.to(device=device, dtype=torch.long).flatten())
    arterial_flows = arterial_flows.to(device=device, dtype=dtype).flatten()
    venous_nodes = torch.unique(venous_nodes.to(device=device, dtype=torch.long).flatten())
    if arterial_nodes.numel() != arterial_flows.numel():
        raise ValueError("arterial_nodes and arterial_flows must have the same length.")
    if arterial_nodes.numel() == 0:
        raise ValueError("At least one arterial node is required for constrained solve.")
    if venous_nodes.numel() == 0:
        raise ValueError("At least one venous node is required for grounded constrained solve.")

    incidence = build_incidence_matrix(
        edge_index,
        int(num_nodes),
        dtype=dtype,
        device=device,
    )
    laplacian = build_weighted_laplacian_from_incidence(incidence, g_edge)

    source_vector = torch.zeros(int(num_nodes), dtype=dtype, device=device)
    source_vector.index_add_(0, arterial_nodes, arterial_flows)

    venous_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    venous_mask[venous_nodes] = True
    free_mask = ~venous_mask
    free_nodes = torch.nonzero(free_mask, as_tuple=False).flatten()

    arterial_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    arterial_mask[arterial_nodes] = True
    free_non_arterial_nodes = torch.nonzero(free_mask & ~arterial_mask, as_tuple=False).flatten()

    n_unknowns = int(free_non_arterial_nodes.numel()) + 1
    transform = torch.zeros((int(num_nodes), n_unknowns), dtype=dtype, device=device)
    transform[arterial_nodes, 0] = 1.0
    for col, node in enumerate(free_non_arterial_nodes.tolist(), start=1):
        transform[int(node), col] = 1.0

    reduced_rows = torch.nonzero(free_mask, as_tuple=False).flatten()
    reduced_system = (laplacian @ transform).index_select(0, reduced_rows)
    reduced_rhs = source_vector.index_select(0, reduced_rows)
    total_flow_constraint = (laplacian @ transform).index_select(0, venous_nodes).sum(
        dim=0, keepdim=True
    )
    total_flow_rhs = arterial_flows.sum().reshape(1)
    reduced_pressure, used_lstsq = solve_least_squares_with_hard_linear_constraints(
        system_matrix=reduced_system,
        rhs=reduced_rhs,
        constraint_matrix=total_flow_constraint,
        constraint_rhs=total_flow_rhs,
    )

    pressure = transform @ reduced_pressure
    pressure[venous_nodes] = 0.0

    edge_pressure_drop = incidence @ pressure
    edge_flow = g_edge * edge_pressure_drop
    nodal_residual = laplacian @ pressure - source_vector

    internal_mask = free_mask.clone()
    internal_mask[arterial_nodes] = False
    internal_nodes = torch.nonzero(internal_mask, as_tuple=False).flatten()
    arterial_residual = nodal_residual[arterial_nodes] if arterial_nodes.numel() else nodal_residual[:0]
    internal_residual = nodal_residual[internal_nodes] if internal_nodes.numel() else nodal_residual[:0]
    constrained_residual = nodal_residual[reduced_rows] if reduced_rows.numel() else nodal_residual[:0]
    total_flow_balance_residual = (
        total_flow_constraint @ reduced_pressure - total_flow_rhs
    )
    diagnostics = {
        "pressure_solver_iterations_used": pressure.new_tensor(1.0),
        "pressure_solver_final_relative_residual": (
            torch.linalg.vector_norm(constrained_residual)
            / torch.linalg.vector_norm(source_vector).clamp_min(1.0e-30)
        ),
        "pressure_solver_used_lstsq": pressure.new_tensor(1.0 if used_lstsq else 0.0),
        "pressure_solver_reached_max_iterations": pressure.new_tensor(0.0),
        "hard_boundary_constraint_residual_l2": torch.linalg.vector_norm(constrained_residual),
        "hard_boundary_constraint_residual_max": torch.max(torch.abs(constrained_residual))
        if constrained_residual.numel()
        else pressure.new_tensor(0.0),
        "hard_total_flow_balance_residual_l2": torch.linalg.vector_norm(
            total_flow_balance_residual
        ),
        "hard_total_flow_balance_residual_max": torch.max(
            torch.abs(total_flow_balance_residual)
        )
        if total_flow_balance_residual.numel()
        else pressure.new_tensor(0.0),
        "internal_kirchhoff_residual_l2": torch.linalg.vector_norm(internal_residual)
        if internal_residual.numel()
        else pressure.new_tensor(0.0),
        "internal_kirchhoff_residual_max": torch.max(torch.abs(internal_residual))
        if internal_residual.numel()
        else pressure.new_tensor(0.0),
        "arterial_pressure_spread_pa": (
            pressure[arterial_nodes].max() - pressure[arterial_nodes].min()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_pressure_spread_pa": (
            pressure[venous_nodes].max() - pressure[venous_nodes].min()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_ground_abs_max_pa": (
            torch.max(torch.abs(pressure[venous_nodes]))
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_inflow_total_m3_s": arterial_flows.sum(),
        "venous_outflow_total_m3_s": nodal_residual[venous_nodes].sum()
        if venous_nodes.numel()
        else pressure.new_tensor(0.0),
    }
    return ConstrainedSolveResult(
        pressure_pa=pressure,
        diagnostics=diagnostics,
        edge_pressure_drop_pa=edge_pressure_drop,
        edge_flow_m3_s=edge_flow,
        nodal_residual_m3_s=nodal_residual,
    )


def constrained_dc_solve(
    laplacian: torch.Tensor,
    boundary_conditions: torch.Tensor,
    reference_node: int,
    solver_config: dict,
) -> ConstrainedSolveResult:
    """Placeholder constrained solver interface.

    For now this uses the same gauge-pinned Kirchhoff solve as the previous
    workflow. A future step can replace this body with a differentiable
    constrained solver while keeping the call signature unchanged.
    """

    del solver_config
    rhs = boundary_conditions.clone()
    q_mask = torch.abs(rhs) > 1.0e-12
    if bool(q_mask.any()):
        q_rhs = rhs[q_mask]
        q_total = torch.sum(q_rhs)
        if float(torch.abs(q_total).detach().cpu()) > 1.0e-10:
            q_abs = torch.abs(q_rhs)
            q_abs_total = torch.sum(q_abs).clamp_min(1.0e-30)
            q_new = q_rhs - q_abs / q_abs_total * q_total
            nonzero = torch.abs(q_rhs) > 1.0e-10
            q_scale = torch.ones_like(q_rhs)
            q_scale[nonzero] = q_new[nonzero] / q_rhs[nonzero]
            rhs[q_mask] = rhs[q_mask] * q_scale
    system = laplacian.clone()
    system[reference_node, :] = 0.0
    system[:, reference_node] = 0.0
    system[reference_node, reference_node] = 1.0
    rhs[reference_node] = 0.0
    pressure = torch.linalg.solve(system, rhs)
    residual = laplacian @ pressure - boundary_conditions
    boundary_residual = residual[q_mask] if bool(q_mask.any()) else residual[:0]
    diagnostics = {
        "pressure_solver_iterations_used": pressure.new_tensor(0.0),
        "pressure_solver_final_relative_residual": (
            torch.linalg.vector_norm(residual)
            / torch.linalg.vector_norm(boundary_conditions).clamp_min(1.0e-30)
        ),
        "pressure_solver_reached_max_iterations": pressure.new_tensor(0.0),
        "hard_boundary_constraint_residual_l2": (
            torch.linalg.vector_norm(boundary_residual)
            if boundary_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "hard_boundary_constraint_residual_max": (
            torch.max(torch.abs(boundary_residual))
            if boundary_residual.numel()
            else pressure.new_tensor(0.0)
        ),
    }
    return ConstrainedSolveResult(pressure_pa=pressure, diagnostics=diagnostics)


class PhysicsLayer:
    """Map conductance corrections to pressure, flow, and velocity predictions."""

    def __init__(self, config: dict):
        self.config = config
        self.parameterization = str(
            config.get("model", {}).get("correction_parameterization", "tanh")
        )
        self.delta_min = float(
            config.get("model", {}).get(
                "correction_min",
                config.get("model", {}).get(
                    "delta_min",
                    -float(config.get("model", {}).get("correction_bound", 0.25)),
                ),
            )
        )
        self.delta_max = float(
            config.get("model", {}).get(
                "correction_max",
                config.get("model", {}).get(
                    "delta_max",
                    float(config.get("model", {}).get("correction_bound", 0.25)),
                ),
            )
        )

    def forward(self, data, delta_e: torch.Tensor) -> dict[str, torch.Tensor]:
        conductance_star, conductance_ratio = conductance_from_delta(
            data.base_conductance,
            delta_e,
        )
        laplacian_star = build_weighted_laplacian(
            data.edge_index,
            conductance_star,
            int(len(data.node_id)),
        )
        solver_kind = str(self.config.get("physics", {}).get("solver_kind", "kirchhoff_placeholder"))
        if solver_kind == "constrained_dc_equal_A_equal_V":
            solve_result = constrained_dc_solve_equal_A_equal_V_torch(
                edge_index=data.edge_index,
                g_edge=conductance_star,
                num_nodes=int(len(data.node_id)),
                arterial_nodes=data.arterial_node_indices,
                arterial_flows=data.boundary_injection_m3_s[data.arterial_node_indices],
                venous_nodes=data.venous_node_indices,
                device=conductance_star.device,
            )
        else:
            solve_result = constrained_dc_solve(
                laplacian_star,
                data.boundary_injection_m3_s,
                int(data.reference_node),
                solver_config=self.config.get("physics", {}),
            )
        source, target = data.edge_index
        flow_pred = conductance_star * (
            solve_result.pressure_pa[source] - solve_result.pressure_pa[target]
        )
        velocity_dc = flow_pred / data.area_m2.clamp_min(1.0e-30)
        velocity_pred = data.velocity_observed_m_s.new_zeros(
            (data.n_edges, data.n_channels, 2)
        )
        velocity_pred[:, 0, 0] = velocity_dc
        velocity_normalized = (
            velocity_pred - data.velocity_center_m_s[None, :, :]
        ) / data.velocity_scale_m_s[None, :, :]
        return {
            "delta_e": delta_e,
            "delta_dc": delta_e,
            "conductance_ratio": conductance_ratio,
            "conductance_m3_pa_s": conductance_star,
            "laplacian_star": laplacian_star,
            "pressure_pa": solve_result.pressure_pa,
            "flow_m3_s": flow_pred,
            "velocity_m_s": velocity_pred,
            "velocity_normalized": velocity_normalized,
            **solve_result.diagnostics,
        }
