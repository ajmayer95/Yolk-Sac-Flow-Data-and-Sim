"""Load neural pressure artifacts for downstream classical inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import VascularDataset


@dataclass(frozen=True)
class PressureConditioning:
    """Node-aligned pressure fields and their downstream conditioning policy."""

    source: Path
    pressure_pa: np.ndarray
    harmonic_index: np.ndarray
    mode: str
    weight: float
    sigma_pa: float
    fix_available_harmonics: bool

    def field(self, harmonic: int) -> np.ndarray | None:
        matches = np.flatnonzero(self.harmonic_index == int(harmonic))
        if not len(matches):
            return None
        values = self.pressure_pa[:, int(matches[0])]
        return values if np.isfinite(values).any() else None


def _as_pressure_matrix(values: np.ndarray, n_harmonics: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(
            f"Pressure field must be one- or two-dimensional, got {values.shape}"
        )
    if values.shape[1] != n_harmonics and values.shape[0] == n_harmonics:
        values = values.T
    if values.shape[1] != n_harmonics:
        raise ValueError(
            "Pressure field column count does not match harmonic_index: "
            f"{values.shape[1]} != {n_harmonics}"
        )
    return values.astype(np.complex128, copy=False)


def load_pressure_conditioning(
    path: Path,
    dataset: VascularDataset,
    *,
    mode: str = "scaled",
    weight: float = 1.0,
    sigma_pa: float = 0.0,
    fix_available_harmonics: bool = True,
) -> PressureConditioning:
    """Load and align a GNN ``pressure_field.npz`` to dataset node order."""
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "pressure_field.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Pressure artifact not found: {path}")
    if mode not in {"off", "absolute", "scaled"}:
        raise ValueError(f"Unknown pressure conditioning mode: {mode}")

    with np.load(path, allow_pickle=False) as archive:
        key = (
            "predicted_pressure_pa"
            if "predicted_pressure_pa" in archive
            else "pressure_field_pa"
        )
        if key not in archive:
            raise KeyError(
                f"{path} has neither predicted_pressure_pa nor pressure_field_pa"
            )
        harmonics = np.asarray(
            archive["harmonic_index"]
            if "harmonic_index" in archive
            else [0],
            dtype=np.int16,
        ).reshape(-1)
        pressure = _as_pressure_matrix(archive[key], len(harmonics))
        node_id = np.asarray(
            archive["node_id"] if "node_id" in archive else dataset.node_id
        ).reshape(-1)

    if pressure.shape[0] != len(node_id):
        raise ValueError(
            f"Pressure rows ({pressure.shape[0]}) do not match node IDs "
            f"({len(node_id)}) in {path}"
        )
    lookup = {str(value): index for index, value in enumerate(node_id)}
    aligned = np.full(
        (dataset.n_nodes, len(harmonics)),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    missing = []
    for row, value in enumerate(dataset.node_id):
        source_row = lookup.get(str(value))
        if source_row is None:
            missing.append(value)
        else:
            aligned[row] = pressure[source_row]
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} dataset nodes; first={missing[0]}"
        )
    if not np.isfinite(aligned).any():
        raise ValueError(f"Pressure artifact contains no finite pressures: {path}")

    return PressureConditioning(
        source=path,
        pressure_pa=aligned,
        harmonic_index=harmonics,
        mode=mode,
        weight=max(float(weight), 0.0),
        sigma_pa=max(float(sigma_pa), 0.0),
        fix_available_harmonics=bool(fix_available_harmonics),
    )


def pressure_sigma(values: np.ndarray, configured_sigma_pa: float) -> float:
    """Return an explicit sigma or a robust 1 Pa-floored pressure scale."""
    if configured_sigma_pa > 0:
        return float(configured_sigma_pa)
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not finite.any():
        return 1.0
    centered = values[finite] - np.mean(values[finite])
    return max(float(np.sqrt(np.mean(np.abs(centered) ** 2))), 1.0)
