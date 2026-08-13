#!/usr/bin/env python
"""Create a repo-level cut-ready Somite21 graph for the main DC/AC workflows.

This converter applies the same structural preprocessing used in
`preprocessing_cut/demo.ipynb`:

1. excise the `radial_uncertain` arterial regions and their feeding trunks,
2. rotate/shift the geometry into the display frame used by that notebook,
3. replace radii with the cleaned `r_meas` values from `CleanNetwork`,
4. expose the arterial ring as solver-friendly one-hop synthetic source nodes.

The output remains a NetworkX pickle that preserves the promoted DC/AC edge
aliases expected by the repository's existing DC/AC workflows.

Input:
- raw Somite21 mosaic graph, including the quail-share source bundle variant

Output:
- new cut-ready graph pickle, typically `datasets/somite21_mosaic_cut_pipeline_ready.gpickle`
"""

from __future__ import annotations

import argparse
import copy
import importlib
import math
import os
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_BUNDLE_ROOT = PROJECT_ROOT / "datasets" / "quail-flow-share"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "conversion"))
sys.path.insert(0, str(REFERENCE_BUNDLE_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from scripts.clean_network import CleanNetwork  # noqa: E402
from utils import install_numpy_pickle_compat  # noqa: E402


DEFAULT_INPUT_GRAPH = PROJECT_ROOT / "datasets" / "somite21_mosaic.gpickle"
DEFAULT_OUTPUT_GRAPH = PROJECT_ROOT / "datasets" / "somite21_mosaic_cut_pipeline_ready.gpickle"
RESCALE_M_PER_PX = float(CleanNetwork.rescale)
ROT_RAD = float(CleanNetwork.rot)
SOURCE_PREFIX = "synthetic_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-graph", type=Path, default=DEFAULT_INPUT_GRAPH)
    parser.add_argument("--output-graph", type=Path, default=DEFAULT_OUTPUT_GRAPH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_graph(path: Path):
    install_numpy_pickle_compat()
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_graph(graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _import_optional_conversion_helper():
    try:
        return importlib.import_module("prepare_somite21_mosaic_for_pipeline")
    except Exception:
        return None


OPTIONAL_CONVERT_MODULE = _import_optional_conversion_helper()


def _safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _phase_to_radians(value: float) -> float:
    value = float(value)
    if abs(value) > (math.pi + 1.0e-6):
        return math.radians(value)
    return value


def _harmonic_phasor_from_raw(edge_data: dict, harmonic_idx: int) -> complex | None:
    amp_key = "_h_amp_H1" if harmonic_idx == 1 else f"_h_amp_H{harmonic_idx}"
    phase_key = "_h_phase_H1" if harmonic_idx == 1 else f"_h_phase_H{harmonic_idx}"
    amp = _safe_float(edge_data.get(amp_key))
    phase = _safe_float(edge_data.get(phase_key))
    if math.isfinite(amp) and amp >= 0.0 and math.isfinite(phase):
        return complex(float(amp) * np.exp(1j * float(phase)))
    amp_fallback = _safe_float(
        edge_data.get("amp_Q_piv" if harmonic_idx == 1 else f"amp_Q_h{harmonic_idx}_piv")
    )
    phase_fallback = _safe_float(
        edge_data.get("phase_piv" if harmonic_idx == 1 else f"phase_h{harmonic_idx}_piv")
    )
    if math.isfinite(amp_fallback) and amp_fallback >= 0.0 and math.isfinite(phase_fallback):
        return complex(float(amp_fallback) * np.exp(1j * _phase_to_radians(phase_fallback)))
    return None


def promote_raw_graph_aliases(graph, input_graph_path: Path) -> None:
    if OPTIONAL_CONVERT_MODULE is not None and hasattr(OPTIONAL_CONVERT_MODULE, "convert_graph"):
        OPTIONAL_CONVERT_MODULE.convert_graph(graph, input_graph_path)
        return

    graph.graph["pipeline_conversion_source"] = str(input_graph_path)
    tile_f0s = graph.graph.get("tile_f0_piv")
    if isinstance(tile_f0s, dict):
        graph.graph.setdefault("tile_f0s", dict(tile_f0s))
        tile_values = [
            _safe_float(value)
            for value in tile_f0s.values()
            if math.isfinite(_safe_float(value)) and _safe_float(value) > 0.0
        ]
        if tile_values:
            graph.graph["f0_hz"] = float(np.median(np.asarray(tile_values, dtype=np.float64)))
    else:
        graph.graph.setdefault("f0_hz", _safe_float(tile_f0s))
    graph.graph.setdefault("tile_f0s", {})
    graph.graph.setdefault("bc_harmonics_convention", "raw_per_edge")

    for node, node_data in graph.nodes(data=True):
        bc_harmonics = node_data.get("bc_harmonics")
        if bc_harmonics is not None:
            bc_array = np.asarray(bc_harmonics, dtype=np.complex128).reshape(-1)
            node_data["bc_harmonics"] = bc_array
            if bc_array.size:
                node_data["Q_DC"] = float(np.real(bc_array[0]))

    for u, v, edge_data in graph.edges(data=True):
        flow_from = edge_data.get("flow_from")
        if flow_from is None:
            flow_from = edge_data.get("flow_from_piv")
        flow_to = edge_data.get("flow_to")
        if flow_to is None:
            flow_to = edge_data.get("flow_to_piv")
        if flow_from is not None:
            edge_data["flow_from"] = flow_from
        if flow_to is not None:
            edge_data["flow_to"] = flow_to

        mean_q = _safe_float(edge_data.get("mean_Q_piv"))
        if math.isfinite(mean_q):
            edge_data["mean_Q"] = float(mean_q)
            edge_data["Q_DC"] = float(abs(mean_q))
            sign = 1.0
            if flow_from == v and flow_to == u:
                sign = -1.0
            edge_data["Q_DC_signed_nl_s"] = float(sign * abs(mean_q))
            edge_data["Q_DC_complex_original_nl_s"] = complex(edge_data["Q_DC_signed_nl_s"], 0.0)

        f0_hz = _safe_float(edge_data.get("f0_hz"))
        if not math.isfinite(f0_hz) or f0_hz <= 0.0:
            f0_hz = _safe_float(edge_data.get("f0_hz_piv"))
        if math.isfinite(f0_hz) and f0_hz > 0.0:
            edge_data["f0_hz"] = float(f0_hz)

        bc_terms = [complex(mean_q if math.isfinite(mean_q) else 0.0, 0.0)]
        max_harmonic = 0
        for harmonic_idx in range(1, 4):
            phasor = _harmonic_phasor_from_raw(edge_data, harmonic_idx)
            if phasor is None:
                break
            max_harmonic = harmonic_idx
            amp = float(abs(phasor))
            phase = float(np.angle(phasor))
            bc_terms.append(phasor)
            edge_data[f"Q_H{harmonic_idx}"] = phasor
            edge_data[f"amp_Q_h{harmonic_idx}"] = amp
            edge_data[f"phase_h{harmonic_idx}"] = phase
            edge_data[f"amp_Q_h{harmonic_idx}_piv"] = amp
            edge_data[f"phase_h{harmonic_idx}_piv"] = phase
            if harmonic_idx == 1:
                edge_data["amp_Q_piv"] = amp
                edge_data["phase_piv"] = phase
        if max_harmonic:
            edge_data["bc_harmonics"] = np.asarray(bc_terms, dtype=np.complex128)
            edge_data["harmonic_conversion_ready"] = True
        elif "phase_piv" in edge_data:
            phase_piv = _safe_float(edge_data.get("phase_piv"))
            if math.isfinite(phase_piv):
                edge_data["phase_piv"] = _phase_to_radians(phase_piv)

        measurement_tile_ids = []
        for row in edge_data.get("measurements_piv", []) or []:
            tile_id = row.get("tile_id")
            try:
                measurement_tile_ids.append(int(tile_id))
            except (TypeError, ValueError):
                continue
        if measurement_tile_ids:
            edge_data["tiles"] = sorted(set(measurement_tile_ids))


def rotated_shifted_geometry_px(raw_graph) -> tuple[dict[object, np.ndarray], np.ndarray]:
    cos_theta = math.cos(ROT_RAD)
    sin_theta = math.sin(ROT_RAD)

    def rotate_point_px(x_px: float, y_px: float) -> np.ndarray:
        x_m = float(x_px) * RESCALE_M_PER_PX
        y_m = float(y_px) * RESCALE_M_PER_PX
        xr_m = x_m * cos_theta - y_m * sin_theta
        yr_m = x_m * sin_theta + y_m * cos_theta
        return np.asarray([xr_m, yr_m], dtype=np.float64)

    rotated_m = {
        node: rotate_point_px(node_data["x"], node_data["y"])
        for node, node_data in raw_graph.nodes(data=True)
    }
    origin_m = np.min(np.asarray(list(rotated_m.values()), dtype=np.float64), axis=0)
    shifted_px = {
        node: (point_m - origin_m) / RESCALE_M_PER_PX
        for node, point_m in rotated_m.items()
    }
    return shifted_px, origin_m


def transform_path_to_display_px(path, origin_m: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    cos_theta = math.cos(ROT_RAD)
    sin_theta = math.sin(ROT_RAD)
    x_m = arr[:, 0] * RESCALE_M_PER_PX
    y_m = arr[:, 1] * RESCALE_M_PER_PX
    xr_m = x_m * cos_theta - y_m * sin_theta
    yr_m = x_m * sin_theta + y_m * cos_theta
    shifted_m = np.column_stack((xr_m - origin_m[0], yr_m - origin_m[1]))
    return shifted_m / RESCALE_M_PER_PX


def ring_weights_from_clean_flux(clean_graph: CleanNetwork) -> tuple[dict[int, float], dict[int, float]]:
    boundary_flux = clean_graph.build_f(anchor="arterial")
    ring_totals = {
        int(tip): float(sum(boundary_flux[node] for node in clean_graph.rings[tip]))
        for tip in clean_graph.AORTAE
    }
    weights: dict[int, float] = {}
    for tip in clean_graph.AORTAE:
        ring_nodes = sorted(clean_graph.rings[tip])
        magnitudes = np.asarray([abs(boundary_flux[node]) for node in ring_nodes], dtype=np.float64)
        total = float(np.sum(magnitudes))
        if not math.isfinite(total) or total <= 0.0:
            magnitudes = np.ones(len(ring_nodes), dtype=np.float64)
            total = float(len(ring_nodes))
        normalized = magnitudes / total
        for node, weight in zip(ring_nodes, normalized):
            weights[int(node)] = float(weight)
    return weights, ring_totals


def scaled_boundary_harmonics(raw_graph, clean_graph: CleanNetwork) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float], dict[int, float]]:
    ring_weights, ring_totals = ring_weights_from_clean_flux(clean_graph)

    source_arrays: dict[int, np.ndarray] = {}
    for tip in clean_graph.AORTAE:
        raw_bc = np.asarray(raw_graph.nodes[tip].get("bc_harmonics"), dtype=np.complex128).reshape(-1)
        if raw_bc.size == 0:
            raise ValueError(f"Aortic node {tip!r} is missing bc_harmonics.")
        raw_dc = float(np.real(raw_bc[0]))
        if not math.isfinite(raw_dc) or abs(raw_dc) <= 1.0e-12:
            raise ValueError(f"Aortic node {tip!r} has invalid DC boundary harmonic {raw_bc[0]!r}.")
        scale = float(ring_totals[int(tip)]) / raw_dc
        source_arrays[int(tip)] = raw_bc * scale

    sink_raw = {
        int(vein): np.asarray(raw_graph.nodes[vein].get("bc_harmonics"), dtype=np.complex128).reshape(-1)
        for vein in clean_graph.VEINS
    }
    n_harmonics = min(len(values) for values in list(source_arrays.values()) + list(sink_raw.values()))
    if n_harmonics <= 0:
        raise ValueError("No boundary harmonics were available to distribute.")
    source_arrays = {node: values[:n_harmonics] for node, values in source_arrays.items()}
    sink_raw = {node: values[:n_harmonics] for node, values in sink_raw.items()}

    total_source = np.zeros(n_harmonics, dtype=np.complex128)
    for tip in clean_graph.AORTAE:
        total_source += source_arrays[int(tip)]

    total_sink_raw = np.zeros(n_harmonics, dtype=np.complex128)
    for vein in clean_graph.VEINS:
        total_sink_raw += sink_raw[int(vein)]

    sink_scales = np.ones(n_harmonics, dtype=np.complex128)
    for idx in range(n_harmonics):
        denom = total_sink_raw[idx]
        numer = -total_source[idx]
        if abs(denom) <= 1.0e-18:
            continue
        sink_scales[idx] = numer / denom

    sink_arrays = {
        int(vein): sink_raw[int(vein)] * sink_scales
        for vein in clean_graph.VEINS
    }
    return source_arrays, sink_arrays, ring_weights, ring_totals


def synthetic_source_node_id(tip: int, modeled_node: int) -> str:
    return f"{SOURCE_PREFIX}__{tip}__{modeled_node}"


def meters_to_px_length(value_m: float) -> float:
    value_m = float(value_m)
    if not math.isfinite(value_m) or value_m <= 0.0:
        return 1.0e-6 / RESCALE_M_PER_PX
    return value_m / RESCALE_M_PER_PX


def synthetic_boundary_geometry_from_clean_graph(
    clean_graph: CleanNetwork,
) -> dict[int, tuple[float, float, tuple[int, int]]]:
    """Infer synthetic source-stub geometry from the strongest local real edge.

    The synthetic one-hop arterial source edge is only a solver/device construct,
    so it should not become an artificial bottleneck. We therefore reuse the
    geometry of the most conductive incident real vessel edge at each modeled
    attachment node, using the cleaned radius and length already available in the
    valid cut subgraph.
    """

    by_node: dict[int, tuple[float, float, tuple[int, int], float]] = {}
    for edge_key, cleaned_edge in clean_graph.E.items():
        u, v = int(edge_key[0]), int(edge_key[1])
        try:
            radius_m = float(cleaned_edge["r_meas"])
            length_m = float(cleaned_edge["length"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (
            math.isfinite(radius_m)
            and radius_m > 0.0
            and math.isfinite(length_m)
            and length_m > 0.0
        ):
            continue
        score = radius_m**4 / length_m
        for node in (u, v):
            current = by_node.get(node)
            if current is None or score > current[3]:
                by_node[node] = (radius_m, length_m, (u, v), score)
    return {
        node: (radius_m, length_m, edge_key)
        for node, (radius_m, length_m, edge_key, _score) in by_node.items()
    }


def build_cut_pipeline_graph(raw_graph, input_graph_path: Path) -> nx.Graph:
    promoted_graph = copy.deepcopy(raw_graph)
    promote_raw_graph_aliases(promoted_graph, input_graph_path)

    clean_graph = CleanNetwork(raw_graph)
    display_coords_px, origin_m = rotated_shifted_geometry_px(raw_graph)
    source_bc, sink_bc, ring_weights, ring_totals = scaled_boundary_harmonics(raw_graph, clean_graph)
    synthetic_geometry = synthetic_boundary_geometry_from_clean_graph(clean_graph)

    cut_graph = nx.Graph()
    cut_graph.graph.update(copy.deepcopy(getattr(promoted_graph, "graph", {})))
    cut_graph.graph.update(
        {
            "pipeline_conversion_ready": True,
            "pipeline_conversion_kind": "somite21_clean_network_cut_pipeline_ready",
            "pipeline_conversion_note": (
                "Cut graph derived from preprocessing_cut CleanNetwork. "
                "Invalid arterial regions/trunks were excised and arterial rings "
                "were exposed as synthetic one-hop source boundaries."
            ),
            "pipeline_conversion_source": str(input_graph_path),
            "pipeline_phase_reference": "raw_per_edge",
            "pipeline_frequency_reference": "median_tile_f0_piv",
            "bc_harmonics_convention": "solver_ready",
            "cut_graph_removed_node_count": int(len(clean_graph.V_removed)),
            "cut_graph_removed_edge_count": int(len(clean_graph.E_removed)),
            "cut_graph_valid_node_count": int(len(clean_graph.V)),
            "cut_graph_valid_edge_count": int(len(clean_graph.E)),
            "cut_graph_arterial_ring_nodes": {
                str(int(tip)): [str(int(node)) for node in sorted(clean_graph.rings[tip])]
                for tip in clean_graph.AORTAE
            },
            "cut_graph_ring_total_dc_nl_s": {
                str(int(tip)): float(ring_totals[int(tip)])
                for tip in clean_graph.AORTAE
            },
            "cut_graph_source_mode": "synthetic_per_ring_node",
            "cut_graph_geometry_frame": "preprocessing_cut_display_px",
        }
    )

    for node, cleaned in clean_graph.V.items():
        node_data = copy.deepcopy(promoted_graph.nodes[node])
        pos_px = display_coords_px[node]
        node_data["x"] = float(pos_px[0])
        node_data["y"] = float(pos_px[1])
        node_data["graph_x"] = float(cleaned["x"])
        node_data["graph_y"] = float(cleaned["y"])
        node_data["x_m"] = float(cleaned["x"])
        node_data["y_m"] = float(cleaned["y"])
        node_data["cut_graph_member"] = True
        if node in clean_graph.VEINS:
            node_data["boundary_type"] = "sink"
            node_data["bc_harmonics"] = np.asarray(sink_bc[int(node)], dtype=np.complex128)
            node_data["cut_boundary_role"] = "venous"
        else:
            node_data.pop("boundary_type", None)
        cut_graph.add_node(node, **node_data)

    for u, v in clean_graph.E:
        edge_data = copy.deepcopy(promoted_graph.edges[u, v])
        cleaned_edge = clean_graph.E[(u, v)]
        radius_m = float(cleaned_edge["r_meas"])
        length_m = float(cleaned_edge["length"])
        edge_data["radius"] = radius_m
        edge_data["length"] = length_m
        edge_data["radius_m"] = radius_m
        edge_data["length_m"] = length_m
        # Preserve explicit pixel-scale geometry so legacy readers that
        # prioritize *_px_true / length_true do not accidentally rescale
        # meter-valued fields a second time.
        edge_data["radius_px_true"] = meters_to_px_length(radius_m)
        edge_data["length_true"] = meters_to_px_length(length_m)
        edge_data["radius_source"] = "clean_network_r_meas"
        edge_data["length_source"] = "clean_network_valid_subgraph"
        edge_data["cut_graph_member"] = True
        transformed_path = transform_path_to_display_px(edge_data.get("path"), origin_m)
        if transformed_path is not None:
            edge_data["path"] = transformed_path
        cut_graph.add_edge(u, v, **edge_data)

    for tip in clean_graph.AORTAE:
        tip_bc = np.asarray(source_bc[int(tip)], dtype=np.complex128)
        for modeled_node in sorted(clean_graph.rings[tip]):
            weight = float(ring_weights[int(modeled_node)])
            if weight <= 0.0:
                continue
            boundary_node = synthetic_source_node_id(int(tip), int(modeled_node))
            weighted_bc = np.asarray(tip_bc * weight, dtype=np.complex128)
            modeled_pos_px = display_coords_px[modeled_node]
            modeled_pos_m = clean_graph.V[modeled_node]
            node_data = {
                "x": float(modeled_pos_px[0]),
                "y": float(modeled_pos_px[1]),
                "graph_x": float(modeled_pos_m["x"]),
                "graph_y": float(modeled_pos_m["y"]),
                "x_m": float(modeled_pos_m["x"]),
                "y_m": float(modeled_pos_m["y"]),
                "boundary_type": "source",
                "bc_harmonics": weighted_bc,
                "cut_boundary_role": "arterial_ring_source",
                "cut_boundary_origin_tip": int(tip),
                "cut_boundary_target_node": int(modeled_node),
                "cut_boundary_weight": float(weight),
                "cut_graph_member": False,
            }
            cut_graph.add_node(boundary_node, **node_data)
            geom = synthetic_geometry.get(int(modeled_node))
            if geom is not None:
                radius_m, length_m, source_edge = geom
                radius_source = "incident_real_edge_max_conductance"
                length_source = "incident_real_edge_max_conductance"
            else:
                radius_m = 1.0e-6
                length_m = 1.0e-6
                source_edge = None
                radius_source = "fallback_no_valid_incident_edge"
                length_source = "fallback_no_valid_incident_edge"
            edge_data = {
                "radius": radius_m,
                "length": length_m,
                "radius_m": radius_m,
                "length_m": length_m,
                "radius_px_true": meters_to_px_length(radius_m),
                "length_true": meters_to_px_length(length_m),
                "radius_source": radius_source,
                "length_source": length_source,
                "synthetic_geometry_proxy_edge": source_edge,
                "synthetic_boundary_edge": True,
                "cut_graph_member": False,
                "flow_from": boundary_node,
                "flow_to": modeled_node,
                "Q_DC": float(np.real(weighted_bc[0])),
                "mean_Q": float(np.real(weighted_bc[0])),
                "mean_Q_piv": float(np.real(weighted_bc[0])),
                "f0_hz": float(cut_graph.graph.get("f0_hz", 0.0) or 0.0),
                "bc_harmonics": weighted_bc,
            }
            for harmonic_idx in range(1, len(weighted_bc)):
                phasor = complex(weighted_bc[harmonic_idx])
                edge_data[f"amp_Q_h{harmonic_idx}_piv"] = float(abs(phasor))
                edge_data[f"phase_h{harmonic_idx}_piv"] = float(np.angle(phasor))
                edge_data[f"amp_Q_h{harmonic_idx}"] = float(abs(phasor))
                edge_data[f"phase_h{harmonic_idx}"] = float(np.angle(phasor))
                if harmonic_idx == 1:
                    edge_data["amp_Q_piv"] = float(abs(phasor))
                    edge_data["phase_piv"] = float(np.angle(phasor))
            cut_graph.add_edge(
                boundary_node,
                modeled_node,
                **edge_data,
            )

    return cut_graph


def main() -> None:
    args = parse_args()
    input_graph = args.input_graph.expanduser().resolve()
    output_graph = args.output_graph.expanduser().resolve()
    if output_graph.exists() and not args.overwrite:
        raise FileExistsError(f"{output_graph} already exists. Re-run with --overwrite to replace it.")

    raw_graph = load_graph(input_graph)
    cut_graph = build_cut_pipeline_graph(raw_graph, input_graph)
    save_graph(cut_graph, output_graph)

    source_nodes = [
        node for node, data in cut_graph.nodes(data=True)
        if data.get("boundary_type") == "source"
    ]
    sink_nodes = [
        node for node, data in cut_graph.nodes(data=True)
        if data.get("boundary_type") == "sink"
    ]
    print(f"[ok] wrote {output_graph}")
    print(f"  nodes: {cut_graph.number_of_nodes()}")
    print(f"  edges: {cut_graph.number_of_edges()}")
    print(f"  source boundary nodes: {len(source_nodes)}")
    print(f"  sink boundary nodes: {len(sink_nodes)}")
    print(f"  synthetic arterial sources: {sum(str(node).startswith(SOURCE_PREFIX) for node in source_nodes)}")


if __name__ == "__main__":
    main()
