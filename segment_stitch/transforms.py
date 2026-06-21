"""Manual tile transform parsing and coordinate conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileTransform:
    tile_id: int
    matrix_yx: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    @property
    def translate_y(self) -> float:
        return self.matrix_yx[0][2]

    @property
    def translate_x(self) -> float:
        return self.matrix_yx[1][2]

    @property
    def scale_y(self) -> float:
        return self.matrix_yx[0][0]

    @property
    def scale_x(self) -> float:
        return self.matrix_yx[1][1]

    def local_to_global(self, y: float, x: float) -> tuple[float, float]:
        m = self.matrix_yx
        gy = m[0][0] * y + m[0][1] * x + m[0][2]
        gx = m[1][0] * y + m[1][1] * x + m[1][2]
        return gy, gx

    def global_to_local(self, gy: float, gx: float) -> tuple[float, float]:
        np = __import__("numpy")
        inv = np.linalg.inv(np.asarray(self.matrix_yx, dtype=float))
        ly, lx, _ = inv @ np.asarray([gy, gx, 1.0], dtype=float)
        return float(ly), float(lx)


def parse_transform(tile_id: int, entry: dict[str, Any]) -> TileTransform | None:
    matrix = entry.get("affine_matrix")
    if matrix is None:
        if {"scale_y", "scale_x", "translate_y", "translate_x"} <= set(entry):
            matrix = [
                [entry["scale_y"], 0.0, entry["translate_y"]],
                [0.0, entry["scale_x"], entry["translate_x"]],
                [0.0, 0.0, 1.0],
            ]
        else:
            LOG.warning("Tile %s has no affine matrix or scale/translation fields", tile_id)
            return None
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        LOG.warning("Tile %s has malformed affine_matrix", tile_id)
        return None
    return TileTransform(tile_id, tuple(tuple(float(v) for v in row) for row in matrix))


def parse_transforms(entries: dict[int, dict[str, Any]]) -> dict[int, TileTransform]:
    out = {}
    for tile_id, entry in entries.items():
        transform = parse_transform(tile_id, entry)
        if transform is not None:
            out[tile_id] = transform
    return out


def transformed_bounds(transforms: dict[int, TileTransform], tile_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = tile_shape
    ys: list[float] = []
    xs: list[float] = []
    for t in transforms.values():
        for y, x in [(0, 0), (h, 0), (0, w), (h, w)]:
            gy, gx = t.local_to_global(y, x)
            ys.append(gy)
            xs.append(gx)
    return min(ys), min(xs), max(ys), max(xs)
