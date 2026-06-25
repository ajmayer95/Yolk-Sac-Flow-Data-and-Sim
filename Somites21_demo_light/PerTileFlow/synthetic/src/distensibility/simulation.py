"""Whole-mosaic synthetic distensibility data generation.

This module deliberately reuses the established transmission-line solver in
``Somites21_demo_light/PerTileFlow``. It adds the experiment-specific wall
law, observation noise, train/validation/test splits, and a compact on-disk
schema shared by synthetic and future real-data adapters.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "vascular_distensibility_npz_v1"
NL_PER_M3 = 1.0e12
DEFAULT_GRAPH_SOURCE = Path(
    "/mnt/home/sswee/ceph/Somites21_demo/emb1/analyzed/"
    "mosaic_graph_analyzed.gpickle"
)


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_yaml(path: Path) -> dict:
    """Load the small config subset used by this project.

    PyYAML is used when available. The standard-library fallback supports the
    mappings, scalar values, and inline lists used by the checked-in configs,
    so generation does not require adding a package to the current environment.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        with path.open() as handle:
            loaded = yaml.safe_load(handle)
        return loaded or {}

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        content = raw.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML at {path}:{line_number}")
        key, value = stripped.split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip()
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_solver(simulation_root: Path):
    root = str(simulation_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("pertile.analysis.transmission_line")
    return (
        module.solve_transmission_line,
        module._classify_boundary_nodes,
        module._get_edge_geometry,
    )


def _viewer_default_boundary_forcing(
    graph,
    classify_boundary_nodes,
    target_flux_nl_s: float,
    n_harmonics: int,
) -> Dict[object, np.ndarray]:
    """Mirror the default viewer's measured/equal-split boundary forcing."""
    boundary_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("boundary_type") is not None
    ]
    source_nodes, sink_nodes = classify_boundary_nodes(graph, boundary_nodes)

    signed: Dict[object, np.ndarray] = {}
    target_len = n_harmonics + 1
    for node in list(source_nodes) + list(sink_nodes):
        harmonics = graph.nodes[node].get("bc_harmonics")
        if harmonics is None:
            continue
        values = np.asarray(harmonics, dtype=np.complex128)
        padded = np.zeros(target_len, dtype=np.complex128)
        padded[: min(len(values), target_len)] = values[:target_len]
        sign = -1.0 if node in sink_nodes else 1.0
        signed[node] = sign * padded

    source_values = [signed[node] for node in source_nodes if node in signed]
    sink_values = [signed[node] for node in sink_nodes if node in signed]
    if not source_values or not sink_values:
        raise ValueError("Boundary nodes do not contain usable bc_harmonics")

    source_average = np.mean(np.stack(source_values), axis=0)
    sink_average = np.mean(np.stack(sink_values), axis=0)
    source_average[0] = float(target_flux_nl_s) / len(source_values)
    sink_average[0] = -float(target_flux_nl_s) / len(sink_values)

    forcing: Dict[object, np.ndarray] = {}
    for node in source_nodes:
        if node in signed:
            forcing[node] = source_average.copy()
    for node in sink_nodes:
        if node in signed:
            forcing[node] = sink_average.copy()
    return forcing


def _edge_geometry_arrays(graph, get_edge_geometry):
    edge_ids = []
    source_indices = []
    target_indices = []
    radius_m = []
    length_m = []
    tile_offsets = [0]
    tile_ids_flat = []
    node_ids = list(graph.nodes())
    node_index = {node: index for index, node in enumerate(node_ids)}

    for u, v, data in graph.edges(data=True):
        radius, length = get_edge_geometry(graph, u, v)
        if radius is None or length is None:
            continue
        edge_ids.append((u, v))
        source_indices.append(node_index[u])
        target_indices.append(node_index[v])
        radius_m.append(float(radius))
        length_m.append(float(length))
        tile_ids = sorted(
            {
                int(record["tile_id"])
                for record in (data.get("measurements_piv") or [])
                if record.get("tile_id") is not None
            }
        )
        tile_ids_flat.extend(tile_ids)
        tile_offsets.append(len(tile_ids_flat))

    return {
        "node_ids": node_ids,
        "node_index": node_index,
        "edge_ids": edge_ids,
        "edge_source_index": np.asarray(source_indices, dtype=np.int32),
        "edge_target_index": np.asarray(target_indices, dtype=np.int32),
        "edge_radius_m": np.asarray(radius_m, dtype=np.float64),
        "edge_length_m": np.asarray(length_m, dtype=np.float64),
        "edge_tile_offsets": np.asarray(tile_offsets, dtype=np.int32),
        "edge_tile_ids": np.asarray(tile_ids_flat, dtype=np.int16),
    }


def _as_id_array(values: Sequence[object]) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        width = max((len(str(value)) for value in values), default=1)
        return np.asarray([str(value) for value in values], dtype=f"<U{width}")


def _node_xy(graph, node_ids: Sequence[object]) -> np.ndarray:
    coordinates = np.full((len(node_ids), 2), np.nan, dtype=np.float64)
    for index, node in enumerate(node_ids):
        data = graph.nodes[node]
        x = data.get("x", data.get("x_px"))
        y = data.get("y", data.get("y_px"))
        if (x is None or y is None) and data.get("pos") is not None:
            try:
                x, y = data["pos"][:2]
            except (TypeError, ValueError):
                pass
        try:
            coordinates[index] = (float(x), float(y))
        except (TypeError, ValueError):
            continue
    return coordinates


def _result_edge_harmonics(result, edge_ids: Sequence[tuple]) -> np.ndarray:
    n_harmonics = result.n_harmonics
    values = np.full(
        (len(edge_ids), n_harmonics + 1),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    for index, (u, v) in enumerate(edge_ids):
        if (u, v) in result.edge_flows:
            coeffs = np.asarray(result.edge_flows[(u, v)], dtype=np.complex128)
        elif (v, u) in result.edge_flows:
            coeffs = -np.asarray(result.edge_flows[(v, u)], dtype=np.complex128)
        else:
            continue
        values[index, : min(len(coeffs), n_harmonics + 1)] = coeffs[
            : n_harmonics + 1
        ]
    return values / NL_PER_M3


def _result_node_pressures(
    result, node_ids: Sequence[object], n_harmonics: int
) -> np.ndarray:
    values = np.full(
        (len(node_ids), n_harmonics + 1),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    for index, node in enumerate(node_ids):
        pressure = result.node_pressures.get(node)
        if pressure is None:
            continue
        pressure = np.asarray(pressure, dtype=np.complex128)
        values[index, : min(len(pressure), n_harmonics + 1)] = pressure[
            : n_harmonics + 1
        ]
    return values


def _edge_splits(
    n_edges: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_edges)
    n_train = int(round(train_fraction * n_edges))
    n_val = int(round(val_fraction * n_edges))
    n_train = min(max(n_train, 0), n_edges)
    n_val = min(max(n_val, 0), n_edges - n_train)
    splits = np.full(n_edges, 2, dtype=np.uint8)
    splits[order[:n_train]] = 0
    splits[order[n_train : n_train + n_val]] = 1
    return splits


def _add_relative_complex_gaussian_noise(
    truth: np.ndarray,
    level: float,
    harmonics: Iterable[int],
    apply_to_dc: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    observed = truth.copy()
    sigma = np.zeros(truth.shape, dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = set(int(h) for h in harmonics)
    if apply_to_dc:
        selected.add(0)
    for harmonic in sorted(selected):
        magnitude = np.abs(truth[:, harmonic])
        sigma[:, harmonic] = float(level) * magnitude
        component_sd = sigma[:, harmonic] / math.sqrt(2.0)
        noise = component_sd * (
            rng.standard_normal(len(truth))
            + 1j * rng.standard_normal(len(truth))
        )
        finite = np.isfinite(truth[:, harmonic])
        observed[finite, harmonic] += noise[finite]
    if apply_to_dc:
        observed[:, 0] = observed[:, 0].real.astype(np.complex128)
    return observed, sigma


def _format_float(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "")


def dataset_filename(D0: float, alpha: float, noise: float, seed: int) -> str:
    alpha_text = f"{float(alpha):g}".replace("-", "m").replace(".", "p")
    noise_percent = int(round(float(noise) * 100))
    return (
        f"pl_d{_format_float(D0)}_a{alpha_text}_n{noise_percent:02d}"
        f"_s{int(seed)}.npz"
    )


def _metadata(
    *,
    graph_path: Path,
    graph_sha256: str,
    D0: float,
    alpha: float,
    R0_m: float,
    median_radius_m: float,
    frequency_hz: float,
    noise_level: float,
    seed: int,
    n_nodes: int,
    n_edges: int,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "data_kind": "synthetic",
        "graph_reference": str(graph_path),
        "graph_sha256": graph_sha256,
        "distensibility_model": "power_law",
        "D0_per_pa": float(D0),
        "alpha": float(alpha),
        "R0_m": float(R0_m),
        "graph_median_radius_m": float(median_radius_m),
        "wall_law": "D_e = D0 * (R_e / R0)^alpha",
        "harmonic_indices": [0, 1, 2],
        "heart_frequency_hz": float(frequency_hz),
        "fluid_density_kg_m3": 1060.0,
        "fluid_viscosity_pa_s": 0.0035,
        "boundary_condition_mode": "measured_equal_split_flux",
        "target_total_flux_nl_s": 1.0,
        "noise_model": "relative_complex_gaussian",
        "noise_level": float(noise_level),
        "noise_applied_to_harmonics": [1, 2],
        "noise_applied_to_dc": False,
        "seed": int(seed),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "units": {
            "radius": "m",
            "length": "m",
            "distensibility": "1/Pa",
            "pressure": "Pa",
            "flow": "m^3/s",
            "velocity": "m/s",
            "boundary_flow": "m^3/s",
        },
        "split_codes": {"train": 0, "val": 1, "test": 2},
        "orientation": (
            "Flow and velocity signs follow edge_source_node_id -> "
            "edge_target_node_id."
        ),
    }


def _save_dataset(
    path: Path,
    *,
    graph,
    geometry: Mapping[str, Any],
    result,
    boundary_forcing: Mapping[object, np.ndarray],
    D0: float,
    alpha: float,
    R0_m: float,
    median_radius_m: float,
    graph_path: Path,
    graph_sha256: str,
    frequency_hz: float,
    noise_level: float,
    noise_harmonics: Sequence[int],
    apply_noise_to_dc: bool,
    split_config: Mapping[str, Any],
    seed: int,
) -> dict:
    node_ids = geometry["node_ids"]
    edge_ids = geometry["edge_ids"]
    radius_m = geometry["edge_radius_m"]
    area_m2 = np.pi * radius_m**2
    flow_true = _result_edge_harmonics(result, edge_ids)
    velocity_true = flow_true / area_m2[:, None]
    velocity_observed, velocity_noise_sigma = (
        _add_relative_complex_gaussian_noise(
            velocity_true,
            noise_level,
            noise_harmonics,
            apply_noise_to_dc,
            seed,
        )
    )
    distensibility_true = D0 * (radius_m / R0_m) ** alpha
    pressure_true = _result_node_pressures(result, node_ids, 2)
    split_code = _edge_splits(
        len(edge_ids),
        float(split_config["train_fraction"]),
        float(split_config["val_fraction"]),
        int(split_config["seed"]),
    )

    boundary_nodes = list(boundary_forcing)
    boundary_indices = np.asarray(
        [geometry["node_index"][node] for node in boundary_nodes],
        dtype=np.int32,
    )
    boundary_types = np.asarray(
        [str(graph.nodes[node].get("boundary_type", "")) for node in boundary_nodes],
        dtype="<U8",
    )
    boundary_flow = (
        np.stack([boundary_forcing[node] for node in boundary_nodes], axis=0)
        / NL_PER_M3
    )
    metadata = _metadata(
        graph_path=graph_path,
        graph_sha256=graph_sha256,
        D0=D0,
        alpha=alpha,
        R0_m=R0_m,
        median_radius_m=median_radius_m,
        frequency_hz=frequency_hz,
        noise_level=noise_level,
        seed=seed,
        n_nodes=len(node_ids),
        n_edges=len(edge_ids),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        harmonic_index=np.asarray([0, 1, 2], dtype=np.int8),
        node_id=_as_id_array(node_ids),
        node_xy_px=_node_xy(graph, node_ids),
        edge_source_node_id=_as_id_array([u for u, _ in edge_ids]),
        edge_target_node_id=_as_id_array([v for _, v in edge_ids]),
        edge_source_index=geometry["edge_source_index"],
        edge_target_index=geometry["edge_target_index"],
        edge_radius_m=radius_m,
        edge_length_m=geometry["edge_length_m"],
        edge_area_m2=area_m2,
        edge_distensibility_true_per_pa=distensibility_true,
        edge_tile_offsets=geometry["edge_tile_offsets"],
        edge_tile_ids=geometry["edge_tile_ids"],
        edge_split_code=split_code,
        pressure_true_pa=pressure_true,
        flow_true_m3_s=flow_true,
        velocity_true_m_s=velocity_true,
        velocity_observed_m_s=velocity_observed,
        velocity_noise_sigma_m_s=velocity_noise_sigma,
        observation_valid=np.isfinite(velocity_observed),
        boundary_node_id=_as_id_array(boundary_nodes),
        boundary_node_index=boundary_indices,
        boundary_type=boundary_types,
        boundary_flow_m3_s=boundary_flow,
    )
    return {
        "file": path.name,
        "D0_per_pa": float(D0),
        "alpha": float(alpha),
        "R0_m": float(R0_m),
        "noise_level": float(noise_level),
        "seed": int(seed),
        "n_nodes": len(node_ids),
        "n_edges": len(edge_ids),
        "heart_frequency_hz": float(frequency_hz),
    }


def generate_experiment_grid(
    project_root: Path,
    graph_path: Path | None = None,
    simulation_root: Path | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Generate the configured power-law synthetic data grid."""
    project_root = project_root.resolve()
    base = load_yaml(project_root / "configs" / "synthetic_base.yaml")
    grid = load_yaml(project_root / "configs" / "experiment_grid.yaml")
    simulation = base["simulation"]
    noise = base["noise"]
    splits = base["splits"]

    configured_graph = project_root / base["graph"]["path"]
    if graph_path is None:
        graph_path = configured_graph
        if not graph_path.exists() and DEFAULT_GRAPH_SOURCE.exists():
            graph_path = DEFAULT_GRAPH_SOURCE
    graph_path = graph_path.expanduser().resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    if simulation_root is None:
        workspace_root = project_root.parents[2]
        simulation_root = workspace_root / "Somites21_demo_light" / "PerTileFlow"
    simulation_root = simulation_root.expanduser().resolve()
    solve, classify, get_geometry = _load_solver(simulation_root)

    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    geometry = _edge_geometry_arrays(graph, get_geometry)
    radii = geometry["edge_radius_m"]
    median_radius_m = float(np.median(radii))
    R0_m = float(simulation["R0_value"])
    n_harmonics = max(int(h) for h in simulation["harmonics_generated"])
    frequency = simulation.get("heart_frequency_hz")
    if frequency is None:
        frequency = float(np.median(list(graph.graph["tile_f0s"].values())))

    boundary_forcing = _viewer_default_boundary_forcing(
        graph,
        classify,
        target_flux_nl_s=1.0,
        n_harmonics=n_harmonics,
    )
    graph_digest = _sha256(graph_path)
    output_dir = project_root / base["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for D0 in grid["D0_values"]:
        for alpha in grid["alpha_values"]:
            expected_paths = [
                output_dir / dataset_filename(D0, alpha, noise_level, seed)
                for seed in grid["seeds"]
                for noise_level in grid["noise_levels"]
            ]
            if not overwrite and all(path.exists() for path in expected_paths):
                print(
                    f"Keeping existing D0={float(D0):.1e}, "
                    f"alpha={float(alpha):g} datasets"
                )
                for output_path in expected_paths:
                    with np.load(output_path, allow_pickle=False) as saved:
                        metadata = json.loads(str(saved["metadata_json"]))
                    manifest_rows.append(
                        {
                            "file": output_path.name,
                            "D0_per_pa": metadata["D0_per_pa"],
                            "alpha": metadata["alpha"],
                            "R0_m": metadata["R0_m"],
                            "noise_level": metadata["noise_level"],
                            "seed": metadata["seed"],
                            "n_nodes": metadata["n_nodes"],
                            "n_edges": metadata["n_edges"],
                            "heart_frequency_hz": metadata[
                                "heart_frequency_hz"
                            ],
                        }
                    )
                continue

            wall_law = (
                lambda radius, _D0=float(D0), _alpha=float(alpha), _R0=R0_m:
                _D0 * (radius / _R0) ** _alpha
            )
            print(
                f"Solving D0={float(D0):.1e}, alpha={float(alpha):g}, "
                f"R0={R0_m * 1e6:g} um"
            )
            result = solve(
                graph,
                D=wall_law,
                n_harmonics=n_harmonics,
                f0_hz=frequency,
                mu=float(simulation["fluid_viscosity"]),
                rho=float(simulation["fluid_density"]),
                bc_harmonics_override=boundary_forcing,
                verbose=False,
            )
            for seed in grid["seeds"]:
                for noise_level in grid["noise_levels"]:
                    filename = dataset_filename(D0, alpha, noise_level, seed)
                    output_path = output_dir / filename
                    row = _save_dataset(
                        output_path,
                        graph=graph,
                        geometry=geometry,
                        result=result,
                        boundary_forcing=boundary_forcing,
                        D0=float(D0),
                        alpha=float(alpha),
                        R0_m=R0_m,
                        median_radius_m=median_radius_m,
                        graph_path=graph_path,
                        graph_sha256=graph_digest,
                        frequency_hz=float(frequency),
                        noise_level=float(noise_level),
                        noise_harmonics=noise["apply_to_harmonics"],
                        apply_noise_to_dc=bool(noise["apply_to_dc"]),
                        split_config=splits,
                        seed=int(seed),
                    )
                    manifest_rows.append(row)
                    print(f"  wrote {filename}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {manifest_path} ({len(manifest_rows)} datasets)")
    return manifest_rows
