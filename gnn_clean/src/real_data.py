"""Real-data graph loading and GNN tensor construction."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from edge_geometry import edge_geometry
from tilewise_flow_normalization import tilewise_flow_normalization
from utils import install_numpy_pickle_compat, safe_float


MU = 3.5e-3
PX_SIZE_M = 1.7e-6
nL_per_m3 = 1.0e12


@dataclass
class RealGNNData:
    graph_path: Path
    node_id: np.ndarray
    edge_ids: list[tuple[object, object]]
    edge_index: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    radius_m: torch.Tensor
    length_m: torch.Tensor
    area_m2: torch.Tensor
    base_conductance: torch.Tensor
    boundary_injection_m3_s: torch.Tensor
    reference_boundary_injection_m3_s: torch.Tensor
    velocity_observed_raw_m_s: torch.Tensor
    velocity_observed_m_s: torch.Tensor
    velocity_reference_m_s: torch.Tensor
    velocity_normalized: torch.Tensor
    velocity_center_m_s: torch.Tensor
    velocity_scale_m_s: torch.Tensor
    reference_pressure_pa: torch.Tensor
    delta_zero_reference_pressure_pa: torch.Tensor
    delta_zero_reference_velocity_m_s: torch.Tensor
    dc_loss_weight: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    source_node_mask: torch.Tensor
    sink_node_mask: torch.Tensor
    arterial_node_indices: torch.Tensor
    venous_node_indices: torch.Tensor
    edge_neighbor_index: torch.Tensor
    reference_node: int
    n_harmonics: int
    n_channels: int
    n_edges: int
    node_xy_px: np.ndarray
    edge_tile_id: np.ndarray
    edge_tile_offsets: np.ndarray
    edge_tile_ids: np.ndarray
    flow_normalization: dict | None

    def to(self, device):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if torch.is_tensor(value) else value
        return RealGNNData(**values)


def load_graph(path: Path):
    install_numpy_pickle_compat()
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dc_flow_m3_s(edge_data: dict, u, v) -> tuple[float, bool]:
    value = (
        edge_data.get("Q_DC")
        or edge_data.get("mean_Q_piv")
        or edge_data.get("mean_Q")
        or edge_data.get("mean_Q_nL_s")
    )
    q_nls = safe_float(value)
    if not math.isfinite(q_nls):
        return 0.0, False
    flow_from = edge_data.get("flow_from")
    flow_to = edge_data.get("flow_to")
    if flow_from is None or flow_to is None:
        signed_nls = q_nls
    else:
        signed_nls = q_nls if (flow_from == u and flow_to == v) else -q_nls
    return signed_nls / nL_per_m3, True


def _edge_tile_id(edge_data: dict) -> int:
    for key in ("tile_id", "tile_index"):
        value = edge_data.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for key in ("tile_ids", "tiles"):
        value = edge_data.get(key)
        if isinstance(value, (list, tuple, np.ndarray)) and len(value):
            try:
                return int(value[0])
            except (TypeError, ValueError):
                pass
    return -1


def _edge_tile_ids(edge_data: dict) -> list[int]:
    tile_ids: set[int] = set()
    for key in ("tile_id", "tile_index"):
        value = edge_data.get(key)
        if value is not None:
            try:
                tile_ids.add(int(value))
            except (TypeError, ValueError):
                pass
    for key in ("tile_ids", "tiles"):
        values = edge_data.get(key)
        if isinstance(values, (list, tuple, np.ndarray)):
            for value in values:
                try:
                    tile_ids.add(int(value))
                except (TypeError, ValueError):
                    pass
    for key in ("measurements_piv", "measurements"):
        for row in edge_data.get(key, []) or []:
            value = row.get("tile_id")
            if value is not None:
                try:
                    tile_ids.add(int(value))
                except (TypeError, ValueError):
                    pass
    return sorted(tile_ids) if tile_ids else [-1]


def _best_measurement(edge_data: dict, tile_id: int | None):
    piv = edge_data.get("measurements_piv", []) or []
    if tile_id is not None:
        matching = [row for row in piv if row.get("tile_id") == tile_id]
        if matching:
            piv = matching
    if piv:
        return max(
            piv,
            key=lambda row: safe_float(
                row.get("snr_pulse", row.get("snr_db", -np.inf)), -np.inf
            ),
        )
    for key in ("measurements",):
        rows = edge_data.get(key, []) or []
        if tile_id is not None:
            matching = [row for row in rows if row.get("tile_id") == tile_id]
            if matching:
                rows = matching
        if rows:
            return max(
                rows,
                key=lambda row: safe_float(
                    row.get("snr_pulse", row.get("snr_db", -np.inf)), -np.inf
                ),
            )
    return None


def _dc_weight(edge_data: dict, q_m3_s: float, valid: bool) -> float:
    if not valid:
        return 0.0
    for key in ("Q_DC_snr_db", "mean_Q_snr_db", "snr_pulse", "snr_db"):
        snr_db = safe_float(edge_data.get(key))
        if math.isfinite(snr_db):
            snr_linear = 10.0 ** (snr_db / 20.0)
            sigma = max(abs(q_m3_s) / max(snr_linear, 1.0e-6), 1.0e-18)
            return 1.0 / sigma
    return 1.0


def _measurement_snr(measurement: dict | None, edge_data: dict) -> float:
    if measurement is not None:
        for key in ("snr_pulse", "snr_db", "best_hr_snr"):
            snr = safe_float(measurement.get(key))
            if math.isfinite(snr):
                if key.endswith("_db") or key == "best_hr_snr":
                    return max(10.0 ** (snr / 20.0), 1.0e-6)
                return max(snr, 1.0e-6)
    for key in ("Q_DC_snr_db", "mean_Q_snr_db", "snr_pulse", "snr_db"):
        snr = safe_float(edge_data.get(key))
        if math.isfinite(snr):
            if key.endswith("_db"):
                return max(10.0 ** (snr / 20.0), 1.0e-6)
            return max(snr, 1.0e-6)
    return float("nan")


def _measurement_velocity_m_s(
    edge_data: dict,
    u,
    v,
    area_m2: float,
    tile_id: int | None = None,
) -> tuple[float, bool, float]:
    measurement = _best_measurement(edge_data, tile_id)
    if measurement is not None:
        q_nls = safe_float(
            measurement.get("mean_Q", measurement.get("mean_Q_nL_s"))
        )
        if math.isfinite(q_nls):
            flow_from = edge_data.get("flow_from")
            flow_to = edge_data.get("flow_to")
            if flow_from == u and flow_to == v:
                signed_nls = q_nls
            elif flow_from == v and flow_to == u:
                signed_nls = -q_nls
            else:
                signed_nls = q_nls
            q_m3_s = signed_nls / nL_per_m3
            return (
                q_m3_s / max(area_m2, 1.0e-30),
                True,
                _measurement_snr(measurement, edge_data),
            )
    q_m3_s, is_valid = _dc_flow_m3_s(edge_data, u, v)
    return (
        q_m3_s / max(area_m2, 1.0e-30),
        is_valid,
        _measurement_snr(None, edge_data),
    )


def _boundary_injections(graph, node_ids: list[object], node_index: dict[object, int]):
    injections = np.zeros(len(node_ids), dtype=np.float32)
    boundary_kind = [""] * len(node_ids)
    for boundary_node, node_data in graph.nodes(data=True):
        if node_data.get("boundary_type") not in ("source", "sink"):
            continue
        if boundary_node in node_index:
            modeled_node = boundary_node
            neighbors = [neighbor for neighbor in graph.neighbors(boundary_node)]
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]
        else:
            neighbors = [
                neighbor for neighbor in graph.neighbors(boundary_node) if neighbor in node_index
            ]
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]
            modeled_node = neighbor
        edge_data = graph.edges[boundary_node, neighbor]
        q_m3_s, valid = _dc_flow_m3_s(edge_data, boundary_node, neighbor)
        if not valid:
            continue
        flow_from = edge_data.get("flow_from")
        flow_to = edge_data.get("flow_to")
        if flow_from == boundary_node:
            sign = 1.0
        elif flow_to == boundary_node:
            sign = -1.0
        else:
            sign = 1.0 if node_data.get("boundary_type") == "source" else -1.0
        idx = node_index[modeled_node]
        injections[idx] += float(q_m3_s * sign)
        boundary_kind[idx] = str(node_data.get("boundary_type"))
    return injections, boundary_kind


def _reference_pressure_node(node_ids: list[object], boundary_kind: list[str]) -> int:
    for idx, kind in enumerate(boundary_kind):
        if not kind:
            return int(idx)
    for idx, kind in enumerate(boundary_kind):
        if kind == "sink":
            return int(idx)
    return 0


def _resolve_node_group_indices(
    requested_nodes,
    node_ids: list[object],
    default_mask: np.ndarray,
) -> np.ndarray:
    if requested_nodes:
        lookup = {str(node): idx for idx, node in enumerate(node_ids)}
        indices = []
        for node in requested_nodes:
            key = str(node)
            if key not in lookup:
                raise KeyError(f"Configured node {node!r} was not found in the modeled graph.")
            indices.append(int(lookup[key]))
        return np.asarray(sorted(set(indices)), dtype=np.int64)
    return np.flatnonzero(default_mask).astype(np.int64)


def _edge_neighbor_pairs(edge_ids: list[tuple[object, object]]) -> np.ndarray:
    incident_edges: dict[object, list[int]] = {}
    for edge_idx, (u, v) in enumerate(edge_ids):
        incident_edges.setdefault(u, []).append(edge_idx)
        incident_edges.setdefault(v, []).append(edge_idx)
    pairs: set[tuple[int, int]] = set()
    for edge_list in incident_edges.values():
        if len(edge_list) < 2:
            continue
        for i, edge_a in enumerate(edge_list):
            for edge_b in edge_list[i + 1 :]:
                lo, hi = sorted((int(edge_a), int(edge_b)))
                pairs.add((lo, hi))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    ordered = np.asarray(sorted(pairs), dtype=np.int64)
    return ordered.T


def _split_masks(valid_mask: np.ndarray, fractions: dict, seed: int):
    valid_indices = np.flatnonzero(valid_mask)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(valid_indices)
    total = len(valid_indices)
    n_train = int(round(float(fractions["train"]) * total))
    n_val = int(round(float(fractions["validation"]) * total))
    n_train = min(max(n_train, 1 if total >= 3 else total), total)
    n_val = min(max(n_val, 1 if total >= 3 else 0), max(total - n_train, 0))
    n_test = max(total - n_train - n_val, 0)
    if total >= 3 and n_test == 0:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
    train_idx = valid_indices[:n_train]
    val_idx = valid_indices[n_train : n_train + n_val]
    test_idx = valid_indices[n_train + n_val :]
    train = np.zeros_like(valid_mask, dtype=bool)
    val = np.zeros_like(valid_mask, dtype=bool)
    test = np.zeros_like(valid_mask, dtype=bool)
    train[train_idx] = True
    val[val_idx] = True
    test[test_idx] = True
    return train, val, test


def _unit_flux_boundary_injections(
    injections_m3_s: np.ndarray,
    reference_flux_nL_per_s: float,
) -> np.ndarray:
    reference_flux_m3_s = float(reference_flux_nL_per_s) / nL_per_m3
    positive_flux = float(np.sum(injections_m3_s[injections_m3_s > 0.0]))
    if positive_flux <= 0.0 or not math.isfinite(positive_flux):
        return np.array(injections_m3_s, copy=True)
    return np.asarray(injections_m3_s, dtype=np.float32) * (
        reference_flux_m3_s / positive_flux
    )


def _solve_reference_pressure(
    conductance: np.ndarray,
    edge_index: np.ndarray,
    injections: np.ndarray,
    reference_node: int,
    iterations: int,
    tolerance: float,
) -> np.ndarray:
    conductance_t = torch.tensor(conductance, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    rhs = torch.tensor(injections, dtype=torch.float32)
    scale = conductance_t.detach().median().clamp_min(1.0e-30)
    normalized_conductance = conductance_t / scale
    rhs = rhs / scale
    rhs = rhs.clone()
    rhs[reference_node] = 0.0
    source, target = edge_index_t
    diagonal = torch.zeros_like(rhs)
    diagonal.index_add_(0, source, normalized_conductance)
    diagonal.index_add_(0, target, normalized_conductance)
    diagonal = diagonal.clamp_min(1.0e-8)
    diagonal = diagonal.clone()
    diagonal[reference_node] = 1.0
    pressure = torch.zeros_like(rhs)
    damping = 0.65
    initial_norm = torch.linalg.vector_norm(rhs).detach().clamp_min(1.0e-30)
    for _ in range(int(iterations)):
        drop = pressure[source] - pressure[target]
        flow = normalized_conductance * drop
        residual = rhs - torch.zeros_like(rhs).index_add_(0, source, flow).index_add_(
            0, target, -flow
        )
        residual[reference_node] = 0.0
        pressure = pressure + damping * residual / diagonal
        pressure = pressure.clone()
        pressure[reference_node] = 0.0
        if float(torch.linalg.vector_norm(residual).cpu()) <= float(tolerance) * float(
            initial_norm.cpu()
        ):
            break
    return pressure.cpu().numpy().astype(np.float32)


def build_real_gnn_data(graph_path: Path, config: dict) -> RealGNNData:
    graph = load_graph(graph_path)
    include_boundary_nodes = bool(
        config.get("data", {}).get("include_boundary_nodes_in_pressure_solve", False)
    )
    boundary_nodes = {
        node
        for node, node_data in graph.nodes(data=True)
        if node_data.get("boundary_type") in ("source", "sink")
    }
    if include_boundary_nodes:
        node_ids = list(graph.nodes())
    else:
        node_ids = [node for node in graph.nodes() if node not in boundary_nodes]
    node_index = {node: index for index, node in enumerate(node_ids)}

    edge_ids = []
    source = []
    target = []
    radii = []
    lengths = []
    areas = []
    conductance = []
    valid = []
    observed_velocity = []
    observed_velocity_raw = []
    dc_weights = []
    tile_ids = []
    tile_offsets = [0]
    tile_memberships = []
    membership_observed_velocity = []
    membership_snr = []
    edge_snr = []
    raw_edge_features = []

    degree = dict(graph.degree())
    for u, v, edge_data in graph.edges(data=True):
        if u not in node_index or v not in node_index:
            continue
        radius_m, length_m = edge_geometry(edge_data, PX_SIZE_M)
        radius_m = max(safe_float(radius_m), 1.0e-12)
        length_m = max(safe_float(length_m), 1.0e-12)
        area_m2 = math.pi * radius_m**2
        g_pois = math.pi * radius_m**4 / (8.0 * MU * length_m)
        q_m3_s, is_valid = _dc_flow_m3_s(edge_data, u, v)
        velocity_m_s = q_m3_s / max(area_m2, 1.0e-30)
        edge_ids.append((u, v))
        source.append(node_index[u])
        target.append(node_index[v])
        radii.append(radius_m)
        lengths.append(length_m)
        areas.append(area_m2)
        conductance.append(g_pois)
        valid.append(is_valid)
        observed_velocity.append(
            [
                [float(velocity_m_s if is_valid else 0.0), 0.0],
            ]
        )
        observed_velocity_raw.append(
            [
                [float(velocity_m_s if is_valid else 0.0), 0.0],
            ]
        )
        dc_weights.append(_dc_weight(edge_data, q_m3_s, is_valid))
        edge_snr.append(_measurement_snr(None, edge_data))
        tile_ids.append(_edge_tile_id(edge_data))
        memberships = _edge_tile_ids(edge_data)
        for tile_id in memberships:
            membership_velocity, membership_valid, membership_snr_value = (
                _measurement_velocity_m_s(edge_data, u, v, area_m2, tile_id)
            )
            tile_memberships.append(int(tile_id))
            membership_observed_velocity.append(
                float(membership_velocity if membership_valid else velocity_m_s)
            )
            membership_snr.append(float(membership_snr_value))
        tile_offsets.append(len(tile_memberships))
        raw_edge_features.append(
            [
                math.log(radius_m),
                math.log(length_m),
                math.log(max(g_pois, 1.0e-60)),
                float(degree.get(u, 0)),
                float(degree.get(v, 0)),
            ]
        )

    injections, boundary_kind = _boundary_injections(graph, node_ids, node_index)
    reference_node = _reference_pressure_node(node_ids, boundary_kind)
    source_node_mask = np.asarray([kind == "source" for kind in boundary_kind], dtype=bool)
    sink_node_mask = np.asarray([kind == "sink" for kind in boundary_kind], dtype=bool)
    physics_config = config.get("physics", {})
    arterial_node_indices = _resolve_node_group_indices(
        physics_config.get("arterial_node_ids"),
        node_ids,
        source_node_mask,
    )
    venous_node_indices = _resolve_node_group_indices(
        physics_config.get("venous_node_ids"),
        node_ids,
        sink_node_mask,
    )
    reference_flux_nL_per_s = float(
        config.get("data", {}).get("flow_normalization_reference_flux_nL_per_s", 1.0)
    )
    reference_injections = _unit_flux_boundary_injections(
        injections, reference_flux_nL_per_s
    )
    node_rows = []
    node_xy = []
    for node in node_ids:
        node_data = graph.nodes[node]
        idx = node_index[node]
        x = safe_float(node_data.get("x", node_data.get("graph_x")), 0.0)
        y = safe_float(node_data.get("y", node_data.get("graph_y")), 0.0)
        node_rows.append(
            [
                float(degree.get(node, 0)),
                float(injections[idx] * nL_per_m3),
                1.0 if boundary_kind[idx] == "source" else 0.0,
                1.0 if boundary_kind[idx] == "sink" else 0.0,
                x,
                y,
            ]
        )
        node_xy.append([x, y])

    x_node = np.asarray(node_rows, dtype=np.float32)
    x_edge = np.asarray(raw_edge_features, dtype=np.float32)
    node_mean = x_node.mean(axis=0)
    node_std = np.maximum(x_node.std(axis=0), 1.0e-12)
    edge_mean = x_edge.mean(axis=0)
    edge_std = np.maximum(x_edge.std(axis=0), 1.0e-12)
    x_node = (x_node - node_mean) / node_std
    x_edge = (x_edge - edge_mean) / edge_std

    velocity_raw = np.asarray(observed_velocity_raw, dtype=np.float32)
    velocity = np.asarray(observed_velocity, dtype=np.float32)
    edge_index_np = np.asarray([source, target], dtype=np.int64)
    radii_np = np.asarray(radii, dtype=np.float32)
    conductance_np = np.asarray(conductance, dtype=np.float32)
    areas_np = np.asarray(areas, dtype=np.float32)
    delta_zero_reference_pressure = _solve_reference_pressure(
        conductance=conductance_np,
        edge_index=edge_index_np,
        injections=injections,
        reference_node=reference_node,
        iterations=int(config.get("data", {}).get("reference_pressure_iterations", 5000)),
        tolerance=float(config.get("data", {}).get("reference_pressure_tolerance", 1.0e-8)),
    )
    reference_flow_m3_s = conductance_np * (
        delta_zero_reference_pressure[edge_index_np[0]]
        - delta_zero_reference_pressure[edge_index_np[1]]
    )
    delta_zero_reference_velocity = np.zeros((len(edge_ids), 1, 2), dtype=np.float32)
    delta_zero_reference_velocity[:, 0, 0] = (
        reference_flow_m3_s / np.maximum(areas_np, 1.0e-30)
    ).astype(np.float32)
    reference_pressure = np.array(delta_zero_reference_pressure, copy=True)
    reference_velocity = np.array(delta_zero_reference_velocity, copy=True)
    valid_mask = np.asarray(valid, dtype=bool)
    valid_weights = np.asarray(dc_weights, dtype=np.float32)
    finite_weights = valid_weights[valid_mask & np.isfinite(valid_weights) & (valid_weights > 0)]
    if len(finite_weights):
        valid_weights = valid_weights / max(float(np.median(finite_weights)), 1.0e-12)
    fractions = config.get("data", {}).get(
        "split_fractions",
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    )
    split_seed = int(config.get("data", {}).get("split_seed", config["training"]["seed"]))
    train_mask, val_mask, test_mask = _split_masks(valid_mask, fractions, split_seed)

    flow_normalization = None
    if bool(config.get("data", {}).get("use_tilewise_flow_normalization", False)):
        flow_normalization = tilewise_flow_normalization(
            reference_velocity_dc_m_s=reference_velocity[:, 0, 0],
            observed_velocity_dc_m_s=velocity_raw[:, 0, 0],
            edge_tile_offsets=np.asarray(tile_offsets, dtype=np.int32),
            edge_tile_ids=np.asarray(tile_memberships, dtype=np.int32),
            snr_edge=np.asarray(edge_snr, dtype=np.float32),
            membership_observed_velocity_dc_m_s=np.asarray(
                membership_observed_velocity, dtype=np.float32
            ),
            membership_snr=np.asarray(membership_snr, dtype=np.float32),
            valid_edge_mask=valid_mask,
            weight_mode=str(
                config.get("data", {}).get("tile_flux_weight", "snr_squared")
            ),
            reference_flux_nL_per_s=reference_flux_nL_per_s,
            min_tile_flux_scale=float(
                config.get("data", {}).get("min_tile_flux_scale", 0.1)
            ),
            max_tile_flux_scale=float(
                config.get("data", {}).get("max_tile_flux_scale", 10.0)
            ),
        )
        velocity = np.array(velocity_raw, copy=True)
        velocity[:, 0, 0] = flow_normalization["normalized_velocity_dc_m_s"].astype(
            np.float32
        )

    train_velocity = velocity[train_mask]
    center = np.nanmean(train_velocity, axis=0) if len(train_velocity) else np.zeros((1, 2), dtype=np.float32)
    scale = np.nanstd(train_velocity, axis=0) if len(train_velocity) else np.ones((1, 2), dtype=np.float32)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    velocity_normalized = (velocity - center[None, :, :]) / scale[None, :, :]
    velocity_normalized = np.where(np.isfinite(velocity_normalized), velocity_normalized, 0.0)
    edge_neighbor_index = _edge_neighbor_pairs(edge_ids)

    return RealGNNData(
        graph_path=graph_path,
        node_id=np.asarray(node_ids, dtype=object),
        edge_ids=edge_ids,
        edge_index=torch.tensor([source, target], dtype=torch.long),
        node_features=torch.tensor(x_node, dtype=torch.float32),
        edge_features=torch.tensor(x_edge, dtype=torch.float32),
        radius_m=torch.tensor(np.asarray(radii), dtype=torch.float32),
        length_m=torch.tensor(np.asarray(lengths), dtype=torch.float32),
        area_m2=torch.tensor(np.asarray(areas), dtype=torch.float32),
        base_conductance=torch.tensor(np.asarray(conductance), dtype=torch.float32),
        boundary_injection_m3_s=torch.tensor(injections, dtype=torch.float32),
        reference_boundary_injection_m3_s=torch.tensor(
            reference_injections, dtype=torch.float32
        ),
        velocity_observed_raw_m_s=torch.tensor(velocity_raw, dtype=torch.float32),
        velocity_observed_m_s=torch.tensor(velocity, dtype=torch.float32),
        velocity_reference_m_s=torch.tensor(reference_velocity, dtype=torch.float32),
        velocity_normalized=torch.tensor(velocity_normalized, dtype=torch.float32),
        velocity_center_m_s=torch.tensor(center, dtype=torch.float32),
        velocity_scale_m_s=torch.tensor(scale, dtype=torch.float32),
        reference_pressure_pa=torch.tensor(reference_pressure, dtype=torch.float32),
        delta_zero_reference_pressure_pa=torch.tensor(
            delta_zero_reference_pressure, dtype=torch.float32
        ),
        delta_zero_reference_velocity_m_s=torch.tensor(
            delta_zero_reference_velocity, dtype=torch.float32
        ),
        dc_loss_weight=torch.tensor(valid_weights, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        source_node_mask=torch.tensor(source_node_mask, dtype=torch.bool),
        sink_node_mask=torch.tensor(sink_node_mask, dtype=torch.bool),
        arterial_node_indices=torch.tensor(arterial_node_indices, dtype=torch.long),
        venous_node_indices=torch.tensor(venous_node_indices, dtype=torch.long),
        edge_neighbor_index=torch.tensor(edge_neighbor_index, dtype=torch.long),
        reference_node=reference_node,
        n_harmonics=0,
        n_channels=1,
        n_edges=len(edge_ids),
        node_xy_px=np.asarray(node_xy, dtype=np.float32),
        edge_tile_id=np.asarray(tile_ids, dtype=np.int32),
        edge_tile_offsets=np.asarray(tile_offsets, dtype=np.int32),
        edge_tile_ids=np.asarray(tile_memberships, dtype=np.int32),
        flow_normalization=flow_normalization,
    )
