"""Tile-wise DC flow normalization using a 1 nL/s Poiseuille reference."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean((x[finite] - y[finite]) ** 2)))


def _compute_weights(snr_values: np.ndarray, weight_mode: str) -> np.ndarray:
    mode = str(weight_mode or "snr_squared").lower()
    snr_values = np.asarray(snr_values, dtype=float)
    finite_positive = np.isfinite(snr_values) & (snr_values > 0.0)
    weights = np.ones_like(snr_values, dtype=float)
    if mode == "uniform":
        return weights
    if mode != "snr_squared":
        raise ValueError(f"Unsupported tile_flux_weight mode: {weight_mode}")
    weights[finite_positive] = snr_values[finite_positive] ** 2
    weights[~np.isfinite(weights)] = 1.0
    return np.clip(weights, 1.0e-12, np.inf)


def _empty_result(
    reference_velocity_dc_m_s: np.ndarray,
    observed_velocity_dc_m_s: np.ndarray,
    weight_mode: str,
    min_tile_flux_scale: float,
    max_tile_flux_scale: float,
    reference_flux_nL_per_s: float,
    membership_edge_index: np.ndarray,
    membership_tile_id: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(reference_velocity_dc_m_s, dtype=float)
    observed = np.asarray(observed_velocity_dc_m_s, dtype=float)
    diagnostics = {
        "enabled": True,
        "method": "snr_weighted_tilewise_flow_normalization",
        "reference_flux_nL_per_s": float(reference_flux_nL_per_s),
        "tile_flux_weight": str(weight_mode),
        "min_tile_flux_scale": float(min_tile_flux_scale),
        "max_tile_flux_scale": float(max_tile_flux_scale),
        "n_tiles": 0,
        "n_memberships": int(len(membership_edge_index)),
        "n_edges_with_stitched_weight": 0,
        "pre_normalization_rmse_m_s": _safe_rmse(reference, observed),
        "post_normalization_rmse_m_s": _safe_rmse(reference, observed),
        "pre_normalization_correlation": _safe_corr(reference, observed),
        "post_normalization_correlation": _safe_corr(reference, observed),
        "tile_flux_scale_summary": {
            "mean": 1.0,
            "median": 1.0,
            "min": 1.0,
            "max": 1.0,
        },
        "n_tiles_clipped": 0,
        "n_tiles_nonpositive_raw": 0,
        "tiles": [],
        "warning": "No tile memberships found; observed DC velocity left unchanged.",
    }
    return {
        "normalized_velocity_dc_m_s": np.array(observed, copy=True),
        "reference_velocity_dc_m_s": reference,
        "observed_velocity_dc_m_s": observed,
        "membership_edge_index": membership_edge_index,
        "membership_tile_id": membership_tile_id,
        "membership_weight": np.zeros(0, dtype=float),
        "membership_observed_velocity_dc_m_s": np.zeros(0, dtype=float),
        "membership_normalized_velocity_dc_m_s": np.zeros(0, dtype=float),
        "edge_weight_sum": np.zeros(len(reference), dtype=float),
        "tile_ids": np.zeros(0, dtype=np.int64),
        "tile_flux_scale": np.zeros(0, dtype=float),
        "tile_flux_scale_raw": np.zeros(0, dtype=float),
        "tile_valid_edge_count": np.zeros(0, dtype=np.int64),
        "tile_weight_sum": np.zeros(0, dtype=float),
        "diagnostics": diagnostics,
    }


def tilewise_flow_normalization(
    *,
    reference_velocity_dc_m_s: np.ndarray,
    observed_velocity_dc_m_s: np.ndarray,
    edge_tile_offsets: np.ndarray,
    edge_tile_ids: np.ndarray,
    snr_edge: np.ndarray | None = None,
    membership_observed_velocity_dc_m_s: np.ndarray | None = None,
    membership_snr: np.ndarray | None = None,
    valid_edge_mask: np.ndarray | None = None,
    weight_mode: str = "snr_squared",
    reference_flux_nL_per_s: float = 1.0,
    min_tile_flux_scale: float = 0.1,
    max_tile_flux_scale: float = 10.0,
) -> dict[str, Any]:
    """Fit one tile-level flux scale and divide observed velocities by it."""

    reference = np.asarray(reference_velocity_dc_m_s, dtype=float)
    observed_edge = np.asarray(observed_velocity_dc_m_s, dtype=float)
    offsets = np.asarray(edge_tile_offsets, dtype=np.int64)
    tile_ids_flat = np.asarray(edge_tile_ids, dtype=np.int64)
    n_edges = int(len(reference))
    if snr_edge is None:
        snr_edge = np.full(n_edges, np.nan, dtype=float)
    snr_edge = np.asarray(snr_edge, dtype=float)
    if valid_edge_mask is None:
        valid_edge_mask = np.ones(n_edges, dtype=bool)
    valid_edge_mask = np.asarray(valid_edge_mask, dtype=bool)

    membership_edge_index = np.repeat(np.arange(n_edges, dtype=np.int64), np.diff(offsets))
    if len(membership_edge_index) == 0:
        return _empty_result(
            reference,
            observed_edge,
            weight_mode,
            min_tile_flux_scale,
            max_tile_flux_scale,
            reference_flux_nL_per_s,
            membership_edge_index,
            tile_ids_flat,
        )

    if membership_observed_velocity_dc_m_s is None:
        membership_observed = observed_edge[membership_edge_index]
    else:
        membership_observed = np.asarray(membership_observed_velocity_dc_m_s, dtype=float)
    if membership_snr is None:
        membership_snr_values = snr_edge[membership_edge_index]
    else:
        membership_snr_values = np.asarray(membership_snr, dtype=float)
    membership_weights = _compute_weights(membership_snr_values, weight_mode)
    membership_reference = reference[membership_edge_index]
    membership_valid = (
        valid_edge_mask[membership_edge_index]
        & np.isfinite(membership_reference)
        & np.isfinite(membership_observed)
        & np.isfinite(membership_weights)
        & (membership_weights > 0.0)
    )

    unique_tile_ids = np.unique(tile_ids_flat.astype(np.int64))
    unique_tile_ids.sort()
    tile_flux_scale = np.ones(len(unique_tile_ids), dtype=float)
    tile_flux_scale_raw = np.ones(len(unique_tile_ids), dtype=float)
    tile_valid_count = np.zeros(len(unique_tile_ids), dtype=np.int64)
    tile_weight_sum = np.zeros(len(unique_tile_ids), dtype=float)
    tile_rows: list[dict[str, Any]] = []
    tile_lookup = {int(tile_id): idx for idx, tile_id in enumerate(unique_tile_ids)}
    membership_normalized = np.full(len(membership_edge_index), np.nan, dtype=float)
    clipped_count = 0
    nonpositive_count = 0

    for tile_id in unique_tile_ids:
        tile_index = tile_lookup[int(tile_id)]
        tile_mask = (tile_ids_flat == tile_id) & membership_valid
        tile_valid_count[tile_index] = int(np.sum(tile_mask))
        raw_scale = 1.0
        if np.any(tile_mask):
            weights = membership_weights[tile_mask]
            ur = membership_reference[tile_mask]
            uo = membership_observed[tile_mask]
            numerator = float(np.sum(weights * ur * uo))
            denominator = float(np.sum(weights * ur * ur))
            raw_scale = numerator / denominator if denominator > 0.0 else 1.0
        if not np.isfinite(raw_scale):
            raw_scale = 1.0
        tile_flux_scale_raw[tile_index] = raw_scale
        if raw_scale <= 0.0:
            nonpositive_count += 1
        clipped_scale = float(np.clip(raw_scale, min_tile_flux_scale, max_tile_flux_scale))
        if abs(clipped_scale - raw_scale) > 1.0e-12:
            clipped_count += 1
        tile_flux_scale[tile_index] = clipped_scale
        tile_membership_mask = tile_ids_flat == tile_id
        membership_normalized[tile_membership_mask] = (
            membership_observed[tile_membership_mask] / clipped_scale
        )
        tile_weight_sum[tile_index] = float(np.sum(membership_weights[tile_membership_mask]))
        tile_rows.append(
            {
                "tile_id": int(tile_id),
                "flux_scale_nL_per_s": clipped_scale,
                "flux_scale_raw_nL_per_s": float(raw_scale),
                "n_valid_edges": int(tile_valid_count[tile_index]),
                "pre_rmse_m_s": _safe_rmse(
                    membership_reference[tile_membership_mask],
                    membership_observed[tile_membership_mask],
                ),
                "post_rmse_m_s": _safe_rmse(
                    membership_reference[tile_membership_mask],
                    membership_normalized[tile_membership_mask],
                ),
                "pre_correlation": _safe_corr(
                    membership_reference[tile_membership_mask],
                    membership_observed[tile_membership_mask],
                ),
                "post_correlation": _safe_corr(
                    membership_reference[tile_membership_mask],
                    membership_normalized[tile_membership_mask],
                ),
                "was_clipped": bool(abs(clipped_scale - raw_scale) > 1.0e-12),
                "raw_scale_nonpositive": bool(raw_scale <= 0.0),
            }
        )

    weighted_sum = np.zeros(n_edges, dtype=float)
    edge_weight_sum = np.zeros(n_edges, dtype=float)
    valid_normalized = np.isfinite(membership_normalized) & np.isfinite(membership_weights)
    np.add.at(
        weighted_sum,
        membership_edge_index[valid_normalized],
        membership_weights[valid_normalized] * membership_normalized[valid_normalized],
    )
    np.add.at(
        edge_weight_sum,
        membership_edge_index[valid_normalized],
        membership_weights[valid_normalized],
    )
    stitched = np.array(observed_edge, copy=True)
    stitched_valid = edge_weight_sum > 0.0
    stitched[stitched_valid] = weighted_sum[stitched_valid] / edge_weight_sum[stitched_valid]

    diagnostics = {
        "enabled": True,
        "method": "snr_weighted_tilewise_flow_normalization",
        "reference_flux_nL_per_s": float(reference_flux_nL_per_s),
        "tile_flux_weight": str(weight_mode),
        "min_tile_flux_scale": float(min_tile_flux_scale),
        "max_tile_flux_scale": float(max_tile_flux_scale),
        "n_tiles": int(len(unique_tile_ids)),
        "n_memberships": int(len(membership_edge_index)),
        "n_edges_with_stitched_weight": int(np.sum(stitched_valid)),
        "pre_normalization_rmse_m_s": _safe_rmse(reference, observed_edge),
        "post_normalization_rmse_m_s": _safe_rmse(reference, stitched),
        "pre_normalization_correlation": _safe_corr(reference, observed_edge),
        "post_normalization_correlation": _safe_corr(reference, stitched),
        "tile_flux_scale_summary": {
            "mean": float(np.nanmean(tile_flux_scale)),
            "median": float(np.nanmedian(tile_flux_scale)),
            "min": float(np.nanmin(tile_flux_scale)),
            "max": float(np.nanmax(tile_flux_scale)),
        },
        "n_tiles_clipped": int(clipped_count),
        "n_tiles_nonpositive_raw": int(nonpositive_count),
        "tiles": tile_rows,
        "log_message": (
            "Tile-wise cardiac-output / inlet-flux normalization using a "
            f"{float(reference_flux_nL_per_s):g} nL/s Poiseuille reference."
        ),
    }
    return {
        "normalized_velocity_dc_m_s": stitched,
        "reference_velocity_dc_m_s": reference,
        "observed_velocity_dc_m_s": observed_edge,
        "membership_edge_index": membership_edge_index,
        "membership_tile_id": tile_ids_flat,
        "membership_weight": membership_weights,
        "membership_observed_velocity_dc_m_s": membership_observed,
        "membership_normalized_velocity_dc_m_s": membership_normalized,
        "edge_weight_sum": edge_weight_sum,
        "tile_ids": unique_tile_ids,
        "tile_flux_scale": tile_flux_scale,
        "tile_flux_scale_raw": tile_flux_scale_raw,
        "tile_valid_edge_count": tile_valid_count,
        "tile_weight_sum": tile_weight_sum,
        "diagnostics": diagnostics,
    }
