"""Dataset and result I/O for distensibility experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class VascularDataset:
    """In-memory representation of one synthetic or real-compatible mosaic."""

    path: Path
    metadata: dict
    node_id: np.ndarray
    node_xy_px: np.ndarray
    edge_source_index: np.ndarray
    edge_target_index: np.ndarray
    edge_radius_m: np.ndarray
    edge_length_m: np.ndarray
    edge_area_m2: np.ndarray
    edge_split_code: np.ndarray
    edge_tile_offsets: np.ndarray
    edge_tile_ids: np.ndarray
    velocity_observed_m_s: np.ndarray
    velocity_true_m_s: np.ndarray
    velocity_noise_sigma_m_s: np.ndarray
    flow_true_m3_s: np.ndarray
    pressure_true_pa: np.ndarray
    observation_valid: np.ndarray
    boundary_node_index: np.ndarray
    boundary_type: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(len(self.node_id))

    @property
    def n_edges(self) -> int:
        return int(len(self.edge_source_index))

    @property
    def D0_true(self) -> float:
        return float(self.metadata["D0_per_pa"])

    @property
    def alpha_true(self) -> float:
        return float(self.metadata["alpha"])

    @property
    def R0_m(self) -> float:
        return float(self.metadata["R0_m"])

    @property
    def frequency_hz(self) -> float:
        return float(self.metadata["heart_frequency_hz"])

    @property
    def viscosity_pa_s(self) -> float:
        return float(self.metadata["fluid_viscosity_pa_s"])

    @property
    def density_kg_m3(self) -> float:
        return float(self.metadata["fluid_density_kg_m3"])

    def tile_edge_indices(self) -> Dict[int, np.ndarray]:
        """Return edge indices grouped by tile ID."""
        grouped: dict[int, list[int]] = {}
        for edge_index in range(self.n_edges):
            start = int(self.edge_tile_offsets[edge_index])
            stop = int(self.edge_tile_offsets[edge_index + 1])
            for tile_id in self.edge_tile_ids[start:stop]:
                grouped.setdefault(int(tile_id), []).append(edge_index)
        return {
            tile_id: np.asarray(indices, dtype=np.int32)
            for tile_id, indices in sorted(grouped.items())
        }


def load_dataset(path: Path) -> VascularDataset:
    """Load a versioned compressed vascular dataset."""
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        kwargs = {
            name: data[name].copy()
            for name in (
                "node_id",
                "node_xy_px",
                "edge_source_index",
                "edge_target_index",
                "edge_radius_m",
                "edge_length_m",
                "edge_area_m2",
                "edge_split_code",
                "edge_tile_offsets",
                "edge_tile_ids",
                "velocity_observed_m_s",
                "velocity_true_m_s",
                "velocity_noise_sigma_m_s",
                "flow_true_m3_s",
                "pressure_true_pa",
                "observation_valid",
                "boundary_node_index",
                "boundary_type",
            )
        }
    return VascularDataset(path=path, metadata=metadata, **kwargs)


def write_json(path: Path, payload: dict) -> None:
    """Write JSON with NumPy values converted to ordinary Python values."""

    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, (np.complexfloating, complex)):
            return {"real": float(np.real(value)), "imag": float(np.imag(value))}
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, Path):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n")
