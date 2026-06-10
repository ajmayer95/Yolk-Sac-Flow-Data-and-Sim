"""Shared helpers for whole-mosaic-to-tile simulation workflows.

The scripts in this folder historically grew as standalone analysis entry
points.  This module keeps the new mosaic-derived tile experiments small:

* resolve bundle config paths
* select tiles with available PIV measurements
* convert whole-mosaic transmission-line results into tile-local Q obs
* write compact CSV summaries for downstream plotting/notebooks
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_TILES = [4, 8, 10, 12, 15, 22, 23, 26, 32, 37, 38, 39, 48]


def resolve_path(cli_val: Optional[str], cfg: dict, cfg_dir: Optional[Path],
                 key: str) -> Optional[Path]:
    val = cli_val if cli_val is not None else cfg.get(key)
    if val is None:
        return None
    p = Path(val)
    if p.is_absolute() or cfg_dir is None:
        return p
    return (cfg_dir / p).resolve()


def load_config(config_path: Optional[str]) -> Tuple[dict, Optional[Path]]:
    if not config_path:
        return {}, None
    cfg_path = Path(config_path).resolve()
    with open(cfg_path) as f:
        return json.load(f), cfg_path.parent


def load_graph_from_args(args) -> Tuple[object, Path]:
    cfg, cfg_dir = load_config(getattr(args, "config", None))
    graph_path = resolve_path(getattr(args, "graph", None), cfg, cfg_dir,
                              "mosaic_graph")
    if graph_path is None:
        raise SystemExit("Provide --graph or --config with mosaic_graph.")
    with open(graph_path, "rb") as f:
        return pickle.load(f), graph_path


def available_tiles(graph) -> List[int]:
    seen = set()
    for _, _, d in graph.edges(data=True):
        for m in d.get("measurements_piv") or []:
            tid = m.get("tile_id")
            try:
                seen.add(int(tid))
            except (TypeError, ValueError):
                pass
    return sorted(seen)


def choose_tiles(graph, tiles: Optional[Sequence[int]],
                 all_tiles: bool = False) -> List[int]:
    avail = available_tiles(graph)
    if all_tiles:
        return avail
    if tiles:
        return [int(t) for t in tiles]
    avail_set = set(avail)
    return [t for t in DEFAULT_TILES if t in avail_set] or avail


def result_edge_harmonics(result, u: int, v: int) -> Optional[np.ndarray]:
    """Return simulated Q harmonics in the requested u->v orientation.

    ``solve_transmission_line`` stores directed edge currents for the graph's
    edge-list orientation.  Tile carves may request the reverse orientation,
    in which case every harmonic changes sign.
    """
    if (u, v) in result.edge_flows:
        return np.asarray(result.edge_flows[(u, v)], dtype=complex)
    if (v, u) in result.edge_flows:
        return -np.asarray(result.edge_flows[(v, u)], dtype=complex)
    return None


def observations_from_mosaic_result(prob: dict, result, harmonics:
                                    Sequence[int], nL_per_m3: float) -> dict:
    """Build tile-local observation arrays from a whole-mosaic solve.

    Returned units match the local inference scripts: SI m^3/s for Q.
    """
    n_edges = len(prob["edges_in"])
    q_dc = np.zeros(n_edges, dtype=float)
    valid = {"dc": np.zeros(n_edges, dtype=bool)}
    q_h: Dict[int, np.ndarray] = {
        int(h): np.zeros(n_edges, dtype=complex) for h in harmonics
    }
    for h in harmonics:
        valid[int(h)] = np.zeros(n_edges, dtype=bool)

    for i, (u, v) in enumerate(prob["edges_in"]):
        coeffs = result_edge_harmonics(result, u, v)
        if coeffs is None or len(coeffs) == 0:
            continue
        q_dc[i] = float(np.real(coeffs[0])) / nL_per_m3
        valid["dc"][i] = np.isfinite(q_dc[i])
        for h in harmonics:
            h = int(h)
            if h < len(coeffs):
                q_h[h][i] = complex(coeffs[h]) / nL_per_m3
                valid[h][i] = np.isfinite(q_h[h][i].real) and np.isfinite(
                    q_h[h][i].imag)

    return {"q_dc": q_dc, "q_h": q_h, "valid": valid}


def write_edge_flow_csv(path: Path, graph, result, n_harmonics: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["u", "v", "Q_dc_nL_s"]
    for h in range(1, n_harmonics + 1):
        fields.extend([
            f"Q_h{h}_real_nL_s", f"Q_h{h}_imag_nL_s",
            f"amp_h{h}_nL_s", f"phase_h{h}_rad",
        ])
    fields.extend(["measured_mean_Q_nL_s", "measured_amp_h1_nL_s",
                   "measured_phase_h1_rad"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for u, v in graph.edges():
            coeffs = result_edge_harmonics(result, u, v)
            if coeffs is None:
                continue
            row = {"u": u, "v": v, "Q_dc_nL_s": float(np.real(coeffs[0]))}
            for h in range(1, n_harmonics + 1):
                qh = coeffs[h] if h < len(coeffs) else 0.0j
                row[f"Q_h{h}_real_nL_s"] = float(np.real(qh))
                row[f"Q_h{h}_imag_nL_s"] = float(np.imag(qh))
                row[f"amp_h{h}_nL_s"] = float(abs(qh))
                row[f"phase_h{h}_rad"] = float(np.angle(qh))
            ed = graph.edges[u, v]
            row["measured_mean_Q_nL_s"] = (
                ed.get("Q_DC") or ed.get("mean_Q_piv")
                or ed.get("mean_Q") or ed.get("mean_Q_nL_s"))
            amp = ed.get("Q_H1_amp")
            phase = ed.get("Q_H1_phi")
            if amp is None or phase is None:
                amp = ed.get("amp_Q_h1_piv")
                phase = ed.get("phase_h1_piv")
            if amp is None or phase is None:
                amp = ed.get("amp_Q")
                phase = ed.get("phase")
            row["measured_amp_h1_nL_s"] = amp
            row["measured_phase_h1_rad"] = phase
            writer.writerow(row)


def write_tile_boundary_pressure_csv(path: Path, tile_probs: Dict[int, dict],
                                     result, n_harmonics: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tile_id", "boundary_node", "is_pin", "P_dc_Pa"]
    for h in range(1, n_harmonics + 1):
        fields.extend([f"P_h{h}_real_Pa", f"P_h{h}_imag_Pa",
                       f"amp_P_h{h}_Pa", f"phase_P_h{h}_rad"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tile_id, prob in tile_probs.items():
            for node in prob["boundary_nodes"]:
                p = np.asarray(result.node_pressures.get(node, []),
                               dtype=complex)
                row = {
                    "tile_id": int(tile_id),
                    "boundary_node": node,
                    "is_pin": node == prob.get("pin_node"),
                    "P_dc_Pa": float(np.real(p[0])) if len(p) else np.nan,
                }
                for h in range(1, n_harmonics + 1):
                    ph = p[h] if h < len(p) else np.nan + 1j * np.nan
                    row[f"P_h{h}_real_Pa"] = float(np.real(ph))
                    row[f"P_h{h}_imag_Pa"] = float(np.imag(ph))
                    row[f"amp_P_h{h}_Pa"] = float(abs(ph))
                    row[f"phase_P_h{h}_rad"] = float(np.angle(ph))
                writer.writerow(row)
