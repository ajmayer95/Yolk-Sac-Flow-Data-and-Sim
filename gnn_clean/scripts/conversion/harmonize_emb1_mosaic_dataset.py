#!/usr/bin/env python
"""Build a new harmonized emb1 mosaic dataset from an analyzed source graph.

This is a self-contained end-to-end converter that:

1. loads ``emb1_mosaic_graph_analyzed.gpickle``,
2. estimates one flow scale factor per tile from overlap consistency,
3. applies those scales back onto the graph measurements, and
4. writes ``harmonized_scaled_dataset.gpickle``.

The fitted scale convention matches the packaging step used downstream here:

    Q_harmonized = Q_observed / scale_tile

so larger scales reduce the apparent flow assigned to a tile.

Input:
- `datasets/emb1_mosaic_graph_analyzed.gpickle`

Output:
- `datasets/harmonized_scaled_dataset.gpickle`
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


NL_PER_M3 = 1.0e12
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_GRAPH = PROJECT_ROOT / "datasets" / "emb1_mosaic_graph_analyzed.gpickle"
DEFAULT_OUTPUT_GRAPH = PROJECT_ROOT / "datasets" / "harmonized_scaled_dataset.gpickle"


def install_numpy_pickle_compat() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    for new_name, old_name in aliases.items():
        if new_name in sys.modules:
            continue
        try:
            sys.modules[new_name] = importlib.import_module(old_name)
        except Exception:
            pass


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def write_json_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-graph", type=Path, default=DEFAULT_INPUT_GRAPH)
    parser.add_argument("--output-graph", type=Path, default=DEFAULT_OUTPUT_GRAPH)
    parser.add_argument("--reference-tile-id", type=int, default=14)
    parser.add_argument("--q-min-nl-s", type=float, default=0.01)
    parser.add_argument("--require-same-sign", action=argparse.BooleanOptionalAction, default=False)
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


def canonical_edge_key(u: Any, v: Any) -> str:
    a, b = (u, v) if str(u) <= str(v) else (v, u)
    return f"{a}__{b}"


def canonical_flow_direction(
    u: Any,
    v: Any,
    edge_data: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> tuple[Any, Any]:
    valid_orientations = {(u, v), (v, u)}

    flow_from = edge_data.get("flow_from")
    flow_to = edge_data.get("flow_to")
    if (flow_from, flow_to) in valid_orientations:
        return flow_from, flow_to

    orientation_weight: dict[tuple[Any, Any], float] = defaultdict(float)
    for measurement in measurements:
        m_from = measurement.get("flow_from")
        m_to = measurement.get("flow_to")
        if (m_from, m_to) not in valid_orientations:
            continue
        q_val = safe_float(measurement.get("mean_Q", measurement.get("mean_Q_nL_s")))
        weight = abs(q_val) if np.isfinite(q_val) else 1.0
        orientation_weight[(m_from, m_to)] += max(float(weight), 1.0)

    if orientation_weight:
        return max(orientation_weight.items(), key=lambda item: item[1])[0]

    return (u, v) if str(u) <= str(v) else (v, u)


def measurement_mean_flow_nl_s(measurement: dict[str, Any] | None, edge_data: dict[str, Any]) -> float:
    if measurement is not None:
        for key in ("mean_Q", "mean_Q_nL_s", "Q_DC"):
            value = safe_float(measurement.get(key))
            if np.isfinite(value):
                return float(value)
        q_t = measurement.get("Q_t")
        if q_t is not None:
            try:
                q_t_np = np.asarray(q_t, dtype=np.float64)
            except Exception:
                q_t_np = np.asarray([], dtype=np.float64)
            q_t_np = q_t_np[np.isfinite(q_t_np)]
            if q_t_np.size:
                return float(np.mean(q_t_np))

    for key in ("mean_Q", "mean_Q_nL_s", "Q_DC"):
        value = safe_float(edge_data.get(key))
        if np.isfinite(value):
            return float(value)

    return float("nan")


def extract_measurement_confidence(measurement: dict[str, Any] | None, edge_data: dict[str, Any]) -> float:
    def candidate(keys: tuple[str, ...], payload: dict[str, Any] | None) -> float:
        if not payload:
            return float("nan")
        for key in keys:
            value = safe_float(payload.get(key))
            if not np.isfinite(value):
                continue
            if key.endswith("_db") or key == "best_hr_snr":
                return float(10.0 ** (value / 20.0))
            if value > 0.0:
                return float(value)
        return float("nan")

    conf = candidate(("snr_pulse", "snr_db", "best_hr_snr", "snr_f0"), measurement)
    if np.isfinite(conf) and conf > 0.0:
        return conf
    conf = candidate(("Q_DC_snr_db", "mean_Q_snr_db", "snr_pulse", "snr_db"), edge_data)
    if np.isfinite(conf) and conf > 0.0:
        return conf
    return 1.0


def extract_measurement_rows(graph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    measurement_idx = 0

    for u, v, edge_data in graph.edges(data=True):
        measurements = edge_data.get("measurements_piv") or edge_data.get("measurements") or []
        if not measurements:
            continue

        canonical_from, canonical_to = canonical_flow_direction(u, v, edge_data, measurements)
        edge_key = canonical_edge_key(u, v)
        fallback_flow_from = edge_data.get("flow_from")
        fallback_flow_to = edge_data.get("flow_to")

        for measurement in measurements:
            try:
                tile_id = int(measurement.get("tile_id"))
            except (TypeError, ValueError):
                continue

            q_nl_s = measurement_mean_flow_nl_s(measurement, edge_data)
            if not np.isfinite(q_nl_s):
                continue

            flow_from = measurement.get("flow_from", fallback_flow_from)
            flow_to = measurement.get("flow_to", fallback_flow_to)
            if flow_from is not None and flow_to is not None:
                if flow_from == canonical_from and flow_to == canonical_to:
                    sign = 1.0
                elif flow_from == canonical_to and flow_to == canonical_from:
                    sign = -1.0
                else:
                    sign = 1.0
            else:
                sign = 1.0

            rows.append(
                {
                    "measurement_idx": measurement_idx,
                    "edge_id": edge_key,
                    "u": u,
                    "v": v,
                    "tile_id": tile_id,
                    "canonical_flow_from": canonical_from,
                    "canonical_flow_to": canonical_to,
                    "q_obs_nl_s": float(sign * q_nl_s),
                    "q_obs_m3_s": float(sign * q_nl_s / NL_PER_M3),
                    "q_obs_abs_nl_s": float(abs(q_nl_s)),
                    "q_obs_abs_m3_s": float(abs(q_nl_s) / NL_PER_M3),
                    "confidence": float(extract_measurement_confidence(measurement, edge_data)),
                }
            )
            measurement_idx += 1

    return rows


def build_overlap_rows(
    measurement_rows: list[dict[str, Any]],
    q_min_nl_s: float,
    require_same_sign: bool,
) -> list[dict[str, Any]]:
    by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measurement_rows:
        by_edge[str(row["edge_id"])].append(row)

    overlap_rows: list[dict[str, Any]] = []
    for edge_id, rows in by_edge.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda row: (int(row["tile_id"]), int(row["measurement_idx"])))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                row_i = rows[i]
                row_j = rows[j]
                tile_i = int(row_i["tile_id"])
                tile_j = int(row_j["tile_id"])
                if tile_i == tile_j:
                    continue

                q_i = float(row_i["q_obs_nl_s"])
                q_j = float(row_j["q_obs_nl_s"])
                if not (np.isfinite(q_i) and np.isfinite(q_j)):
                    continue
                if abs(q_i) < q_min_nl_s or abs(q_j) < q_min_nl_s:
                    continue
                if require_same_sign and np.sign(q_i) != np.sign(q_j):
                    continue

                overlap_rows.append(
                    {
                        "edge_id": edge_id,
                        "tile_i": tile_i,
                        "tile_j": tile_j,
                        "q_i_nl_s": q_i,
                        "q_j_nl_s": q_j,
                        "q_i_abs_nl_s": float(abs(q_i)),
                        "q_j_abs_nl_s": float(abs(q_j)),
                        "sign_i": int(np.sign(q_i)),
                        "sign_j": int(np.sign(q_j)),
                        "used_amplitude_only": True,
                        "confidence_i": float(row_i["confidence"]),
                        "confidence_j": float(row_j["confidence"]),
                        "measurement_idx_i": int(row_i["measurement_idx"]),
                        "measurement_idx_j": int(row_j["measurement_idx"]),
                        # Q_harmonized = Q_observed / scale
                        "target_log_scale_diff": float(math.log(abs(q_i)) - math.log(abs(q_j))),
                    }
                )
    return overlap_rows


def pair_weight(conf_i: float, conf_j: float, eps: float = 1.0e-12) -> float:
    denom = conf_i + conf_j + eps
    if denom <= 0.0:
        return 0.5
    weight = conf_i * conf_j / denom
    return weight if np.isfinite(weight) and weight > 0.0 else 0.5


def fit_tile_scales(
    overlap_rows: list[dict[str, Any]],
    reference_tile_id: int,
) -> tuple[dict[int, float], dict[int, float], list[dict[str, Any]]]:
    tile_ids = sorted(
        {
            int(row["tile_i"])
            for row in overlap_rows
        }
        | {
            int(row["tile_j"])
            for row in overlap_rows
        }
    )
    if reference_tile_id not in tile_ids:
        raise RuntimeError(
            f"Reference tile {reference_tile_id} was not found in valid overlap rows. "
            f"Available tiles: {tile_ids}"
        )

    free_tile_ids = [tile_id for tile_id in tile_ids if tile_id != int(reference_tile_id)]
    free_to_pos = {tile_id: idx for idx, tile_id in enumerate(free_tile_ids)}
    n_free = len(free_tile_ids)

    if not overlap_rows:
        raise RuntimeError("No valid overlap rows were found for scale fitting.")

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    equation_rows: list[dict[str, Any]] = []

    for row in overlap_rows:
        tile_i = int(row["tile_i"])
        tile_j = int(row["tile_j"])
        weight = math.sqrt(pair_weight(float(row["confidence_i"]), float(row["confidence_j"])))
        target = float(row["target_log_scale_diff"])
        vec = np.zeros(n_free, dtype=np.float64)
        if tile_i != reference_tile_id:
            vec[free_to_pos[tile_i]] += weight
        if tile_j != reference_tile_id:
            vec[free_to_pos[tile_j]] -= weight
        A_rows.append(vec)
        b_rows.append(weight * target)
        equation_rows.append(
            {
                "edge_id": row["edge_id"],
                "tile_i": tile_i,
                "tile_j": tile_j,
                "q_i_nl_s": row["q_i_nl_s"],
                "q_j_nl_s": row["q_j_nl_s"],
                "q_i_abs_nl_s": row["q_i_abs_nl_s"],
                "q_j_abs_nl_s": row["q_j_abs_nl_s"],
                "sign_i": row["sign_i"],
                "sign_j": row["sign_j"],
                "used_amplitude_only": bool(row["used_amplitude_only"]),
                "pair_weight": float(weight * weight),
                "target_log_scale_diff": target,
            }
        )

    A = np.vstack(A_rows)
    b = np.asarray(b_rows, dtype=np.float64)
    if n_free:
        ridge = 1.0e-8 * np.eye(n_free, dtype=np.float64)
        A_aug = np.vstack((A, ridge))
        b_aug = np.concatenate((b, np.zeros(n_free, dtype=np.float64)))
        solution, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        solution = np.zeros(0, dtype=np.float64)

    log_scales = {int(reference_tile_id): 0.0}
    scales = {int(reference_tile_id): 1.0}
    for tile_id in free_tile_ids:
        value = float(solution[free_to_pos[tile_id]])
        log_scales[int(tile_id)] = value
        scales[int(tile_id)] = float(math.exp(value))

    return scales, log_scales, equation_rows


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")
    values = values[finite]
    weights = weights[finite]
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def build_harmonized_graph(
    graph,
    measurement_rows: list[dict[str, Any]],
    scales: dict[int, float],
    reference_tile_id: int,
) -> tuple[object, list[dict[str, Any]]]:
    harmonized = graph.copy()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measurement_rows:
        grouped[str(row["edge_id"])].append(row)

    edge_rows: list[dict[str, Any]] = []
    for u, v, edge_data in harmonized.edges(data=True):
        edge_id = canonical_edge_key(u, v)
        rows = grouped.get(edge_id, [])
        if not rows:
            continue

        scaled_q_nl_s: list[float] = []
        scaled_q_m3_s: list[float] = []
        weights: list[float] = []
        tile_ids: list[int] = []
        canonical_from = rows[0]["canonical_flow_from"]
        canonical_to = rows[0]["canonical_flow_to"]
        preserved_flow_from = edge_data.get("flow_from")
        preserved_flow_to = edge_data.get("flow_to")
        if preserved_flow_from is None or preserved_flow_to is None:
            preserved_flow_from = canonical_from
            preserved_flow_to = canonical_to

        for row in rows:
            tile_id = int(row["tile_id"])
            scale = float(scales.get(tile_id, 1.0))
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            scaled_q_nl_s.append(float(row["q_obs_nl_s"]) / scale)
            scaled_q_m3_s.append(float(row["q_obs_m3_s"]) / scale)
            weights.append(max(float(row["confidence"]), 1.0e-12))
            tile_ids.append(tile_id)

        if not scaled_q_nl_s:
            continue

        q_nl_s = weighted_mean(np.asarray(scaled_q_nl_s, dtype=np.float64), np.asarray(weights, dtype=np.float64))
        q_m3_s = weighted_mean(np.asarray(scaled_q_m3_s, dtype=np.float64), np.asarray(weights, dtype=np.float64))
        if not np.isfinite(q_nl_s):
            q_nl_s = float(np.nanmean(np.asarray(scaled_q_nl_s, dtype=np.float64)))
        if not np.isfinite(q_m3_s):
            q_m3_s = float(np.nanmean(np.asarray(scaled_q_m3_s, dtype=np.float64)))

        q_nl_s_signed = float(q_nl_s)
        q_nl_s_mag = abs(q_nl_s_signed)

        measurement_record = {
            "tile_id": int(reference_tile_id),
            "flow_from": preserved_flow_from,
            "flow_to": preserved_flow_to,
            "mean_Q": q_nl_s_mag,
            "mean_Q_nL_s": q_nl_s_mag,
            "mean_Q_signed_nL_s": q_nl_s_signed,
            "fit_success": True,
            "quality_tier": "A",
            "snr_pulse": float(max(np.median(np.asarray(weights, dtype=np.float64)), 1.0)),
            "snr_db": float(20.0 * np.log10(max(np.median(np.asarray(weights, dtype=np.float64)), 1.0))),
            "harmonized_by": "harmonize_emb1_mosaic_dataset",
        }

        edge_data["mean_Q"] = q_nl_s_mag
        edge_data["mean_Q_piv"] = q_nl_s_mag
        edge_data["Q_DC"] = q_nl_s_mag
        edge_data["mean_Q_linear"] = q_nl_s_mag
        edge_data["Q_DC_signed_nl_s"] = q_nl_s_signed
        edge_data["Q_DC_complex_original_nl_s"] = complex(q_nl_s_signed, 0.0)
        edge_data["flow_from"] = preserved_flow_from
        edge_data["flow_to"] = preserved_flow_to
        edge_data["measurements_piv"] = [measurement_record]
        edge_data["measurements"] = [measurement_record]

        edge_rows.append(
            {
                "canonical_edge_id": edge_id,
                "edge_u": u,
                "edge_v": v,
                "n_measurements": len(rows),
                "harmonized_flow_nL_s": float(q_nl_s),
                "harmonized_flow_m3_s": float(q_m3_s),
                "tile_ids": ";".join(str(tile_id) for tile_id in sorted(set(tile_ids))),
                "scale_min": float(np.min([scales.get(tile_id, 1.0) for tile_id in tile_ids])),
                "scale_median": float(np.median([scales.get(tile_id, 1.0) for tile_id in tile_ids])),
                "scale_max": float(np.max([scales.get(tile_id, 1.0) for tile_id in tile_ids])),
            }
        )

    harmonized.graph.update(
        {
            "harmonized_dataset_ready": True,
            "harmonized_dataset_kind": "emb1_overlap_harmonized_scaled_dataset",
            "harmonized_scale_convention": "Q_harmonized = Q_observed / scale_tile",
            "harmonized_scale_fit_uses_flow_magnitude_only": True,
            "harmonized_dc_storage_convention": "Q_DC/mean_Q fields store magnitudes; direction is stored in flow_from/flow_to; signed scalar is preserved in Q_DC_signed_nl_s",
            "harmonized_reference_tile_id": int(reference_tile_id),
        }
    )
    return harmonized, edge_rows


def tile_snr_summary(measurement_rows: list[dict[str, Any]]) -> dict[int, float]:
    by_tile: dict[int, list[float]] = defaultdict(list)
    for row in measurement_rows:
        conf = float(row["confidence"])
        if np.isfinite(conf) and conf > 0.0:
            by_tile[int(row["tile_id"])].append(conf)
    result: dict[int, float] = {}
    for tile_id, values in by_tile.items():
        result[int(tile_id)] = float(np.median(np.asarray(values, dtype=np.float64)))
    return result


def main() -> None:
    args = parse_args()
    input_graph = args.input_graph.expanduser().resolve()
    output_graph = args.output_graph.expanduser().resolve()
    output_dir = output_graph.parent

    if output_graph.exists() and not args.overwrite:
        raise FileExistsError(f"{output_graph} already exists. Re-run with --overwrite to replace it.")

    graph = load_graph(input_graph)
    measurement_rows = extract_measurement_rows(graph)
    overlap_rows = build_overlap_rows(
        measurement_rows=measurement_rows,
        q_min_nl_s=float(args.q_min_nl_s),
        require_same_sign=bool(args.require_same_sign),
    )
    scales, log_scales, equation_rows = fit_tile_scales(
        overlap_rows=overlap_rows,
        reference_tile_id=int(args.reference_tile_id),
    )
    harmonized_graph, edge_rows = build_harmonized_graph(
        graph=graph,
        measurement_rows=measurement_rows,
        scales=scales,
        reference_tile_id=int(args.reference_tile_id),
    )

    save_graph(harmonized_graph, output_graph)

    snr_by_tile = tile_snr_summary(measurement_rows)
    tile_rows = [
        {
            "tile_id": int(tile_id),
            "snr": float(snr_by_tile.get(int(tile_id), 1.0)),
            "log_scale": float(log_scales[int(tile_id)]),
            "scale": float(scales[int(tile_id)]),
        }
        for tile_id in sorted(scales)
    ]

    write_csv(output_dir / "tile_scales.csv", tile_rows)
    write_csv(output_dir / "overlap_records.csv", equation_rows)
    write_csv(output_dir / "harmonized_scaled_edges.csv", edge_rows)
    write_json_yaml(
        output_dir / "harmonized_scaled_dataset_summary.yaml",
        {
            "source_graph": str(input_graph),
            "output_graph": str(output_graph),
            "reference_tile_id": int(args.reference_tile_id),
            "scale_fit_uses_flow_magnitude_only": True,
            "require_same_sign": bool(args.require_same_sign),
            "scale_convention": "Q_harmonized = Q_observed / scale_tile",
            "measurement_row_count": int(len(measurement_rows)),
            "overlap_row_count": int(len(overlap_rows)),
            "tile_scale_count": int(len(tile_rows)),
            "scaled_edge_count": int(len(edge_rows)),
            "tile_scales_csv": str(output_dir / "tile_scales.csv"),
            "overlap_records_csv": str(output_dir / "overlap_records.csv"),
            "harmonized_edges_csv": str(output_dir / "harmonized_scaled_edges.csv"),
        },
    )

    print(f"[ok] wrote {output_graph}")
    print(f"  measurements extracted: {len(measurement_rows)}")
    print(f"  overlap equations used: {len(overlap_rows)}")
    print(f"  fitted tile scales: {len(tile_rows)}")
    print(f"  harmonized edges: {len(edge_rows)}")


if __name__ == "__main__":
    main()
