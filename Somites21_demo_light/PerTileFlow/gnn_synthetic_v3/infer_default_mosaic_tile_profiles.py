"""Synthetic-data tile distensibility profile-likelihood dashboard.

This workflow-local copy profiles distensibility against the synthetic velocity
observations written by ``default_mosaic_tile_profiles.py`` and, by default,
regularizes tile boundary pressures toward the validated GNN output.

Default assumptions:
  * synthetic graph observations from ``synthetic_mosaic_graph.gpickle``
  * validated GNN pressure prior from
    ``gnn_edge_velocity_dc/masked_edge_validation_15pct``
  * H1+H2 in tile profile scans
  * free tile-boundary pressures are refit independently at each D

Outputs:
  * gnn_synthetic_global_profile_constant_D[...].csv
  * gnn_synthetic_tile_profiles[...].csv
  * gnn_synthetic_tile_profile_summary[...].csv
  * infer_default_mosaic_tile_profiles[...].html
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "d0p001_results"
DEFAULT_CONFIG = PROJECT_ROOT.parent / "emb1" / "config.json"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from distensibility_ablation import (
    _metric_row,
    _n_params,
    _observations,
    _profile_free,
    _sigma_vectors,
    _transfer,
)
from inspect_tile import build_tile_problem
from pertile.analysis.local_pressure_inference import _edge_geometry
from tile_mosaic_simulation import choose_tiles, load_graph_from_args
from synthetic_validation_neumann_bc import PX_SIZE_M, nL_per_m3


DEFAULT_TILE_PROFILE_HARMONICS = (1, 2)
DEFAULT_SYNTHETIC_GRAPH = RESULTS_DIR / "synthetic_mosaic_graph.gpickle"
DEFAULT_GNN_PRESSURE_PRIOR_DIR = (
    RESULTS_DIR / "gnn_edge_velocity_dc" / "masked_edge_validation_15pct"
)
DEFAULT_OBSERVATION_SOURCE = "gnn_synthetic_graph"
DEFAULT_OUT_DIR = RESULTS_DIR / "infer_validated_gnn_tile_profiles"
PERIPHERY_TILES = {
    1, 2, 3, 4, 16, 17, 18, 31, 32, 43, 44, 53, 52, 51, 50, 49, 48,
    38, 37, 25, 24, 11, 10, 8,
}


def _install_numpy_pickle_compat() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    for new_name, old_name in aliases.items():
        if new_name not in sys.modules:
            try:
                sys.modules[new_name] = importlib.import_module(old_name)
            except Exception:
                pass


def _safe_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _resolve_pressure_prior_dir(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str).expanduser().resolve()
    if (path / "node_pressures_physics_gnn.csv").exists():
        return path
    sweep_csv = path / "sweep_summary.csv"
    if sweep_csv.exists():
        best_dir = None
        best_val = float("inf")
        with open(sweep_csv, newline="") as f:
            for row in csv.DictReader(f):
                val = _safe_float(row.get("val_NRMSE_v", row.get("val_NRMSE")))
                out_dir = row.get("out_dir")
                if math.isfinite(val) and out_dir and val < best_val:
                    best_val = val
                    best_dir = out_dir
        if best_dir:
            best_path = Path(best_dir).expanduser().resolve()
            if (best_path / "node_pressures_physics_gnn.csv").exists():
                return best_path
    raise SystemExit(
        f"Could not find node_pressures_physics_gnn.csv under {path_str}"
    )


def _load_pressure_prior(path_str: Optional[str]) -> Tuple[Dict[str, float], Optional[str]]:
    prior_dir = _resolve_pressure_prior_dir(path_str)
    if prior_dir is None:
        return {}, None
    out: Dict[str, float] = {}
    with open(prior_dir / "node_pressures_physics_gnn.csv", newline="") as f:
        for row in csv.DictReader(f):
            node_id = str(row.get("node_id", "")).strip()
            p = _safe_float(row.get("pressure_Pa"))
            if node_id and math.isfinite(p):
                out[node_id] = p
    return out, str(prior_dir)


def _tile_pressure_prior(prob: dict, pressure_prior: Dict[str, float]) -> Optional[np.ndarray]:
    vals = []
    for node in prob["boundary_nodes"]:
        p = pressure_prior.get(str(node))
        vals.append(float(p) if p is not None and math.isfinite(float(p)) else np.nan)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _edge_areas(prob: dict) -> np.ndarray:
    areas = []
    for u, v in prob["edges_in"]:
        r_m, _l_m = _edge_geometry(prob["sub"].edges[u, v], PX_SIZE_M)
        r_m = max(float(r_m), 1e-12)
        areas.append(math.pi * r_m ** 2)
    return np.asarray(areas, dtype=float)


def _first_present(edge_data: dict, names: Sequence[str]):
    for name in names:
        if name in edge_data and edge_data.get(name) is not None:
            return edge_data.get(name)
    return None


def _oriented_scalar(edge_data: dict, u, v, value: float) -> float:
    ff = edge_data.get("flow_from")
    ft = edge_data.get("flow_to")
    if ff is None or ft is None:
        return value
    return value if (ff == u and ft == v) else -value


def _direct_velocity_dc(edge_data: dict, u, v) -> float:
    value = _first_present(edge_data, (
        "V_DC_m_s", "mean_v_m_s", "mean_velocity_m_s", "velocity_m_s",
        "V_DC", "mean_v", "mean_velocity", "mean_v_piv",
    ))
    value = _safe_float(value)
    if not math.isfinite(value):
        return float("nan")
    return _oriented_scalar(edge_data, u, v, value)


def _direct_velocity_h(edge_data: dict, u, v, h: int) -> complex:
    real = _safe_float(edge_data.get(f"V_H{h}_real_m_s"))
    imag = _safe_float(edge_data.get(f"V_H{h}_imag_m_s"))
    if math.isfinite(real) and math.isfinite(imag):
        value = complex(real, imag)
    else:
        amp = _safe_float(edge_data.get(f"V_H{h}_amp_m_s"))
        phi = _safe_float(edge_data.get(f"V_H{h}_phi"))
        if not (math.isfinite(amp) and math.isfinite(phi)):
            return complex(float("nan"), float("nan"))
        value = amp * np.exp(1j * phi)
    sign = _oriented_scalar(edge_data, u, v, 1.0)
    return complex(sign) * value


def _observation_data(graph, prob: dict, harmonics: Sequence[int], args) -> dict:
    obs_q = _observations(graph, prob, harmonics)
    if args.observation_variable == "flow":
        obs_q["variable"] = "flow"
        obs_q["target_source"] = "flow_metadata"
        return obs_q

    area = _edge_areas(prob)
    direct_dc = np.asarray([
        _direct_velocity_dc(prob["sub"].edges[u, v], u, v)
        for u, v in prob["edges_in"]
    ], dtype=float)
    direct_valid_dc = np.isfinite(direct_dc)
    target_sources = ["direct_velocity_metadata" if ok else "derived_from_flow"
                      for ok in direct_valid_dc]
    obs_v = {
        "q_dc": np.where(direct_valid_dc, direct_dc,
                         np.asarray(obs_q["q_dc"], dtype=float) / area),
        "q_h": {},
        "valid": {k: np.asarray(v, dtype=bool).copy()
                  for k, v in obs_q["valid"].items()},
        "variable": "velocity",
        "target_source": (
            "direct_velocity_metadata" if all(s == "direct_velocity_metadata"
                                             for s in target_sources)
            else "derived_from_flow" if all(s == "derived_from_flow"
                                            for s in target_sources)
            else "mixed_velocity_metadata_and_flow"
        ),
        "area_m2": area,
        "flow_obs": obs_q,
    }
    obs_v["valid"]["dc"] = direct_valid_dc | np.asarray(obs_q["valid"]["dc"], dtype=bool)
    for h, q in obs_q["q_h"].items():
        direct_h = np.asarray([
            _direct_velocity_h(prob["sub"].edges[u, v], u, v, int(h))
            for u, v in prob["edges_in"]
        ], dtype=complex)
        direct_valid = np.isfinite(direct_h.real) & np.isfinite(direct_h.imag)
        obs_v["q_h"][int(h)] = np.where(
            direct_valid, direct_h, np.asarray(q, dtype=complex) / area)
        obs_v["valid"][int(h)] = (
            direct_valid | np.asarray(obs_q["valid"].get(int(h), []), dtype=bool)
        )
    return obs_v


def _sigma_vectors_for_observation(obs: dict, args) -> tuple[np.ndarray, Dict[int, np.ndarray]]:
    if obs.get("variable") != "velocity":
        return _sigma_vectors(obs, args)
    flow_obs = obs.get("flow_obs")
    area = np.asarray(obs.get("area_m2"), dtype=float)
    if flow_obs is None or area.size == 0:
        raise ValueError("Velocity observations require flow_obs and area_m2 for sigma conversion.")
    sig_dc_q, sig_h_q = _sigma_vectors(flow_obs, args)
    sig_dc = np.asarray(sig_dc_q, dtype=float) / area
    sig_h = {
        int(h): np.asarray(sig, dtype=float) / area
        for h, sig in sig_h_q.items()
    }
    return sig_dc, sig_h


def _transfer_for_observation(prob: dict, D: float, harmonics: Sequence[int],
                              args):
    T = _transfer(prob, D, harmonics)
    if args.observation_variable != "velocity":
        return T
    area = _edge_areas(prob)
    return {int(h): np.asarray(mat) / area[:, None] for h, mat in T.items()}


def _pressure_sigma(p_ref: np.ndarray, args) -> float:
    if float(args.pressure_prior_sigma_pa) > 0:
        return float(args.pressure_prior_sigma_pa)
    centered = np.asarray(p_ref, dtype=float) - float(np.nanmean(p_ref))
    return max(float(np.sqrt(np.nanmean(centered ** 2))), 1.0)


def _pressure_prior_terms(P: np.ndarray, p_ref: np.ndarray, args) -> Tuple[float, complex]:
    mode = str(args.pressure_prior_mode)
    if mode == "off" or p_ref is None:
        return 0.0, complex(np.nan)
    sigma = _pressure_sigma(p_ref, args)
    p = np.asarray(p_ref, dtype=float)
    P = np.asarray(P, dtype=complex)
    if mode == "scaled":
        denom = float(np.vdot(p, p).real)
        if denom <= 1e-30:
            return 0.0, complex(np.nan)
        scale = np.vdot(p, P) / denom
        resid = P - scale * p
    else:
        scale = 1.0 + 0.0j
        resid = P - p
    chi = float(args.lambda_pressure_prior) * float(
        np.mean((np.abs(resid) / sigma) ** 2))
    return chi, complex(scale)


def _fit_dc_with_pressure_prior(T0, q_dc, valid_dc, sig_dc, pin_idx,
                                p_ref: Optional[np.ndarray], args):
    n_bnd = T0.shape[1]
    keep = np.array([i for i in range(n_bnd) if i != pin_idx])
    P = np.zeros(n_bnd, dtype=complex)
    if valid_dc.any() and keep.size:
        A = T0[valid_dc][:, keep].real / sig_dc[valid_dc, None]
        b = q_dc[valid_dc] / sig_dc[valid_dc]
        if p_ref is not None and args.pressure_prior_mode != "off":
            sigma = _pressure_sigma(p_ref, args)
            lam = max(float(args.lambda_pressure_prior), 0.0)
            w = math.sqrt(lam / max(keep.size, 1)) / sigma
            p_keep = (np.asarray(p_ref, dtype=float) - float(p_ref[pin_idx]))[keep]
            if args.pressure_prior_mode == "scaled":
                denom = float(np.dot(p_keep, p_keep))
                if denom > 1e-30:
                    M = np.eye(keep.size) - np.outer(p_keep, p_keep) / denom
                    A = np.vstack([A, w * M])
                    b = np.concatenate([b, np.zeros(keep.size)])
            else:
                A = np.vstack([A, w * np.eye(keep.size)])
                b = np.concatenate([b, w * p_keep])
        sol, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
        P[keep] = sol
    pred = (T0 @ P).real
    r = (q_dc[valid_dc] - pred[valid_dc]) / sig_dc[valid_dc]
    data_chi = float(np.sum(r * r))
    prior_chi, scale = _pressure_prior_terms(P.real, (
        np.asarray(p_ref, dtype=float) - float(p_ref[pin_idx])
        if p_ref is not None else p_ref), args)
    return P, data_chi + prior_chi, r, data_chi, prior_chi, scale


def _fit_complex_with_pressure_prior(T, q, valid, sig,
                                     p_ref: Optional[np.ndarray], args):
    if not valid.any():
        P = np.zeros(T.shape[1], dtype=complex)
        prior_chi, scale = _pressure_prior_terms(P, p_ref, args)
        return P, prior_chi, np.array([]), 0.0, prior_chi, scale
    A = T[valid] / sig[valid, None]
    b = q[valid] / sig[valid]
    if p_ref is not None and args.pressure_prior_mode != "off":
        sigma = _pressure_sigma(p_ref, args)
        lam = max(float(args.lambda_pressure_prior), 0.0)
        w = math.sqrt(lam / max(T.shape[1], 1)) / sigma
        p = np.asarray(p_ref, dtype=float) - float(np.mean(p_ref))
        if args.pressure_prior_mode == "scaled":
            denom = float(np.dot(p, p))
            if denom > 1e-30:
                M = np.eye(T.shape[1]) - np.outer(p, p) / denom
                A = np.vstack([A, w * M.astype(complex)])
                b = np.concatenate([b, np.zeros(T.shape[1], dtype=complex)])
        else:
            A = np.vstack([A, w * np.eye(T.shape[1], dtype=complex)])
            b = np.concatenate([b, w * p.astype(complex)])
    P, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
    resid = (q[valid] - (T @ P)[valid]) / sig[valid]
    data_chi = float(np.sum(resid.real ** 2 + resid.imag ** 2))
    p_for_penalty = (np.asarray(p_ref, dtype=float) - float(np.mean(p_ref))
                     if p_ref is not None else p_ref)
    prior_chi, scale = _pressure_prior_terms(P, p_for_penalty, args)
    return P.astype(complex), data_chi + prior_chi, resid, data_chi, prior_chi, scale


def _profile_with_pressure_prior(prob, obs, sig_dc, sig_h, D_grid,
                                 harmonics, p_ref: Optional[np.ndarray],
                                 args):
    if p_ref is None or args.pressure_prior_mode == "off":
        return _profile_free(prob, obs, sig_dc, sig_h, D_grid, harmonics)
    rows = []
    best = None
    for D in D_grid:
        T = _transfer_for_observation(prob, D, harmonics, args)
        P_dc, chi_dc, r_dc, data_dc, prior_dc, scale_dc = (
            _fit_dc_with_pressure_prior(
                T[0], obs["q_dc"], obs["valid"]["dc"], sig_dc,
                prob["pin_idx"], p_ref, args))
        P_h = {}
        r_h = {}
        data_chi = data_dc
        prior_chi = prior_dc
        chi = chi_dc
        scales = {"dc": scale_dc}
        for h in harmonics:
            P_h[h], chi_h, r_h[h], data_h, prior_h, scale_h = (
                _fit_complex_with_pressure_prior(
                    T[h], obs["q_h"][h], obs["valid"][h], sig_h[h],
                    p_ref, args))
            chi += chi_h
            data_chi += data_h
            prior_chi += prior_h
            scales[f"h{h}"] = scale_h
        item = dict(D=float(D), chi2=float(chi), data_chi2=float(data_chi),
                    pressure_prior_chi2=float(prior_chi), P_dc=P_dc,
                    P_h=P_h, r_dc=r_dc, r_h=r_h,
                    pressure_prior_scales=scales)
        rows.append(item)
        if best is None or item["chi2"] < best["chi2"]:
            best = item
    return rows, best


def _profile_tile_from_observations(graph, tile_id: int, D_grid: np.ndarray,
                                    args, pressure_prior: Dict[str, float]):
    harmonics = tuple(int(h) for h in args.tile_harmonics)
    prob = build_tile_problem(graph, int(tile_id))
    obs = _observation_data(graph, prob, harmonics, args)
    weighted = _is_weighted_objective(args)
    if weighted:
        sig_dc, sig_h = _sigma_vectors_for_observation(obs, args)
        weight_mode = "sigma"
    else:
        sig_dc, sig_h = _constant_average_sigma_vectors(obs, args)
        weight_mode = "constant_average_sigma"
    objective = _objective_name(args)
    p_ref = _tile_pressure_prior(prob, pressure_prior)
    profile, best = _profile_with_pressure_prior(
        prob, obs, sig_dc, sig_h, D_grid, harmonics, p_ref, args)
    metrics = _metric_row(
        int(tile_id), f"{args.observation_source}_free_boundary",
        harmonics, profile, best,
        prob, obs, _n_params(prob, harmonics, "free"))
    metrics["objective"] = objective
    metrics["observation_source"] = args.observation_source
    metrics["observation_variable"] = obs.get("variable", args.observation_variable)
    metrics["target_source"] = obs.get("target_source", "")
    metrics["pressure_prior_mode"] = args.pressure_prior_mode
    metrics["lambda_pressure_prior"] = float(args.lambda_pressure_prior)
    metrics["pressure_prior_available"] = bool(p_ref is not None)
    metrics["data_chi2_at_D_hat"] = float(best.get("data_chi2", best["chi2"]))
    metrics["pressure_prior_chi2_at_D_hat"] = float(
        best.get("pressure_prior_chi2", 0.0))
    metrics.update(_weight_diagnostics(obs, sig_dc, sig_h, weight_mode))
    return profile, metrics


def _constant_average_sigma_vectors(
        obs: dict, args) -> tuple[np.ndarray, Dict[int, np.ndarray]]:
    weighted_dc, weighted_h = _sigma_vectors_for_observation(obs, args)
    vals = []
    valid_dc = np.asarray(obs["valid"]["dc"], dtype=bool)
    if valid_dc.any():
        vals.append(np.asarray(weighted_dc, dtype=float)[valid_dc])
    for h, sig in weighted_h.items():
        valid = np.asarray(obs["valid"].get(int(h), []), dtype=bool)
        if valid.any():
            # Complex harmonic residuals contribute real and imaginary parts.
            vals.extend([np.asarray(sig, dtype=float)[valid]] * 2)
    sigma0 = float(np.nanmean(np.concatenate(vals))) if vals else 1.0
    sigma0 = max(sigma0, 1e-30)

    sig_dc = np.full_like(np.asarray(obs["q_dc"], dtype=float), sigma0,
                          dtype=float)
    sig_h = {}
    for h, q in obs["q_h"].items():
        sig_h[int(h)] = np.full_like(np.asarray(q, dtype=complex), sigma0,
                                     dtype=float)
    return sig_dc, sig_h


def _objective_name(args) -> str:
    return ("ordinary_least_squares" if args.ordinary_least_squares
            else "weighted_least_squares")


def _is_weighted_objective(args) -> bool:
    return _objective_name(args) == "weighted_least_squares"


def _weight_diagnostics(obs: dict, sig_dc: np.ndarray,
                        sig_h: Dict[int, np.ndarray],
                        weight_mode: str) -> dict:
    rows = {
        "weight_mode": weight_mode,
    }
    variable = str(obs.get("variable", "flow"))
    unit_suffix = "m_s" if variable == "velocity" else "nL_s"

    def _range(prefix: str, sig: np.ndarray, valid: np.ndarray) -> None:
        vals = np.asarray(sig)[np.asarray(valid, dtype=bool)]
        if vals.size:
            rows[f"{prefix}_sigma_min"] = float(np.nanmin(vals))
            rows[f"{prefix}_sigma_max"] = float(np.nanmax(vals))
            rows[f"{prefix}_sigma_min_{unit_suffix}"] = float(np.nanmin(vals))
            rows[f"{prefix}_sigma_max_{unit_suffix}"] = float(np.nanmax(vals))
            if variable == "flow":
                rows[f"{prefix}_sigma_min_nL_s"] = float(np.nanmin(vals)
                                                         * nL_per_m3)
                rows[f"{prefix}_sigma_max_nL_s"] = float(np.nanmax(vals)
                                                         * nL_per_m3)
        else:
            rows[f"{prefix}_sigma_min"] = float("nan")
            rows[f"{prefix}_sigma_max"] = float("nan")
            rows[f"{prefix}_sigma_min_{unit_suffix}"] = float("nan")
            rows[f"{prefix}_sigma_max_{unit_suffix}"] = float("nan")

    _range("dc", sig_dc, obs["valid"]["dc"])
    for h in (1, 2):
        if h in sig_h and h in obs["valid"]:
            _range(f"h{h}", sig_h[h], obs["valid"][h])
    return rows


def _run_suffix(args) -> str:
    if args.ordinary_least_squares:
        return "_ordinary"
    return "_weighted" if args.weighted_least_squares else ""


def _out_name(base: str, args) -> str:
    path = Path(base)
    return f"{path.stem}{_run_suffix(args)}{path.suffix}"


def _summed_profile(tile_profiles: Dict[int, List[dict]]) -> List[dict]:
    by_d = {}
    for tile_id, profile in tile_profiles.items():
        for row in profile:
            d = float(row["D"])
            by_d.setdefault(d, 0.0)
            by_d[d] += float(row["chi2"])
    rows = [{"D": d, "chi2": chi2} for d, chi2 in sorted(by_d.items())]
    if not rows:
        return rows
    chi_min = min(r["chi2"] for r in rows)
    for r in rows:
        r["delta_chi2"] = r["chi2"] - chi_min
    return rows


def _best_D(profile: List[dict]):
    if not profile:
        return None
    best = min(profile, key=lambda r: float(r["chi2"]))
    return float(best["D"])


def _tile_location(tile_id: int) -> str:
    return "periphery" if int(tile_id) in PERIPHERY_TILES else "interior"


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clean_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return [_clean_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    return value


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Synthetic Tile Distensibility Profile Likelihoods</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #18222d;
  --muted: #647486;
  --line: #d8e0e8;
  --blue: #2868b7;
  --red: #c74e45;
  --green: #2c8a68;
  --orange: #c27a22;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  padding: 14px 18px;
  background: #202b36;
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  position: sticky;
  top: 0;
  z-index: 5;
}
h1 { font-size: 17px; margin: 0; }
main { padding: 14px; display: grid; gap: 14px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  min-width: 0;
}
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 0.55fr); gap: 14px; }
.gridTop { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); gap: 14px; }
h2 { font-size: 15px; margin: 0 0 10px; }
canvas {
  width: 100%;
  height: 380px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
}
.controls {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
label { color: var(--muted); font-size: 12px; display: grid; gap: 3px; }
select {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  color: var(--ink);
  padding: 0 8px;
}
.inlineControls {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: -2px 0 10px;
}
.inlineControls label {
  display: flex;
  align-items: center;
  gap: 6px;
}
.inlineControls select { min-width: 130px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th, td {
  text-align: right;
  border-bottom: 1px solid var(--line);
  padding: 5px 6px;
  white-space: nowrap;
}
th:first-child, td:first-child { text-align: left; }
thead th { position: sticky; top: 0; background: #edf2f6; }
.scroll { max-height: 380px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
.meta { color: #dce6ef; font-size: 12px; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.swatches { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin-top: 8px; }
.swatches span::before { content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 2px; background: var(--c); }
@media (max-width: 1050px) {
  .grid2, .gridTop, .controls { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Synthetic Tile Distensibility Profile Likelihoods</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="gridTop">
    <section>
      <h2>Constant Distensibility Across All Tiles</h2>
      <canvas id="globalCanvas"></canvas>
      <div class="note">The blue curve sums tile chi2 values at each D. Best D values are the minima of summed profiles: all tiles, interior-only tiles, and periphery-only tiles.</div>
    </section>
    <section>
      <h2>All Tile Profiles Overlayed</h2>
      <div class="inlineControls">
        <label>Show
          <select id="allFilter">
            <option value="all">All tiles</option>
            <option value="interior">Interior</option>
            <option value="periphery">Periphery</option>
          </select>
        </label>
      </div>
      <canvas id="allCanvas"></canvas>
      <div class="note">Each line is one tile's observation profile likelihood, normalized to that tile's own minimum. Reference lines show summed-profile best D values for all, interior, and periphery tiles.</div>
    </section>
  </div>
  <div class="grid2">
    <section>
      <h2>Selected Tile Profile Overlay</h2>
      <div class="controls">
        <label>Tile 1<select id="sel0"></select></label>
        <label>Tile 2<select id="sel1"></select></label>
        <label>Tile 3<select id="sel2"></select></label>
        <label>Tile 4<select id="sel3"></select></label>
      </div>
      <canvas id="selectedCanvas"></canvas>
      <div class="swatches">
        <span style="--c:#2868b7">Tile 1</span>
        <span style="--c:#c74e45">Tile 2</span>
        <span style="--c:#2c8a68">Tile 3</span>
        <span style="--c:#c27a22">Tile 4</span>
      </div>
    </section>
    <section>
      <h2>Tile Summary</h2>
      <div class="scroll"><table id="summaryTable"></table></div>
    </section>
  </div>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("payload").textContent);
const colors = ["#2868b7", "#c74e45", "#2c8a68", "#c27a22"];
const allColors = ["#8aa7c8", "#c8a18a", "#8bc0a9", "#b6a2ce", "#d8ba74", "#9cb5ba"];
const tileIds = Object.keys(data.tileProfiles).map(Number).sort((a,b)=>a-b).map(String);

function num(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const x = Number(v);
  if (x === 0) return "0";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.01) return x.toExponential(2);
  return x.toFixed(3).replace(/\.?0+$/, "");
}
function byId(id) { return document.getElementById(id); }
function setupSelectors() {
  for (let i = 0; i < 4; i++) {
    const sel = byId(`sel${i}`);
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "None";
    sel.appendChild(none);
    for (const tid of tileIds) {
      const opt = document.createElement("option");
      opt.value = tid;
      opt.textContent = tid.padStart(3, "0");
      sel.appendChild(opt);
    }
    if (tileIds[i]) sel.value = tileIds[i];
    sel.addEventListener("change", render);
  }
  byId("allFilter").addEventListener("change", render);
}
function canvasCtx(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(360, rect.width * dpr);
  canvas.height = Math.max(260, rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: canvas.width / dpr, h: canvas.height / dpr};
}
function clear(ctx,w,h) {
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "white";
  ctx.fillRect(0,0,w,h);
}
function axes(ctx,w,h,xlab,ylab) {
  ctx.strokeStyle = "#d8e0e8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(58, 18);
  ctx.lineTo(58, h - 44);
  ctx.lineTo(w - 18, h - 44);
  ctx.stroke();
  ctx.fillStyle = "#647486";
  ctx.font = "12px sans-serif";
  ctx.fillText(xlab, Math.max(64, w / 2 - 45), h - 15);
  ctx.save();
  ctx.translate(16, h / 2 + 52);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylab, 0, 0);
  ctx.restore();
}
function logTickValues(xmin, xmax) {
  const ticks = [];
  const lo = Math.ceil(xmin);
  const hi = Math.floor(xmax);
  for (let e = lo; e <= hi; e++) ticks.push(Math.pow(10, e));
  if (!ticks.length) {
    ticks.push(Math.pow(10, xmin), Math.pow(10, xmax));
  }
  return ticks;
}
function linearTickValues(ymax) {
  const rawStep = ymax / 4;
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
  const candidates = [1, 2, 5, 10].map(v => v * pow);
  let step = candidates[candidates.length - 1];
  for (const c of candidates) {
    if (rawStep <= c) { step = c; break; }
  }
  const ticks = [];
  for (let y = 0; y <= ymax + step * 0.5; y += step) {
    ticks.push(y);
  }
  return ticks;
}
function drawTicks(ctx,w,h,xTicks,yTicks,sx,sy) {
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.strokeStyle = "#9aa8b5";
  ctx.fillStyle = "#647486";
  for (const xVal of xTicks) {
    const x = sx(xVal);
    if (x < 58 || x > w - 18) continue;
    ctx.beginPath();
    ctx.moveTo(x, h - 44);
    ctx.lineTo(x, h - 38);
    ctx.stroke();
    ctx.fillText(num(xVal), x, h - 34);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const yVal of yTicks) {
    const y = sy(yVal);
    if (y < 18 || y > h - 44) continue;
    ctx.beginPath();
    ctx.moveTo(52, y);
    ctx.lineTo(58, y);
    ctx.stroke();
    ctx.fillText(num(yVal), 48, y);
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}
function drawProfiles(canvas, series, opts={}) {
  const {ctx,w,h} = canvasCtx(canvas);
  clear(ctx,w,h);
  axes(ctx,w,h,"D (1/Pa, log scale)","Delta chi2");
  const points = [];
  for (const s of series) {
    for (const p of s.rows) {
      if (p.D > 0 && p.delta_chi2 !== null && p.delta_chi2 !== undefined) {
        points.push([Math.log10(p.D), Number(p.delta_chi2)]);
      }
    }
  }
  if (!points.length) return;
  const xmin = Math.min(...points.map(p => p[0]));
  const xmax = Math.max(...points.map(p => p[0]));
  const rawYmax = Math.max(...points.map(p => p[1]));
  const cap = opts.yCap === undefined ? 12 : opts.yCap;
  const ymax = Math.max(4, Math.min(cap, rawYmax));
  const sx = D => 58 + (Math.log10(D) - xmin) / ((xmax - xmin) || 1) * (w - 82);
  const sy = y => h - 44 - Math.min(y, ymax) / ymax * (h - 68);
  drawTicks(ctx, w, h, logTickValues(xmin, xmax), linearTickValues(ymax),
            sx, sy);
  for (const y of [1, 3.84]) {
    ctx.strokeStyle = y === 1 ? "#18222d" : "#647486";
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    ctx.moveTo(58, sy(y));
    ctx.lineTo(w - 18, sy(y));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#647486";
    ctx.fillText(String(y), 62, sy(y) - 4);
  }
  const refs = opts.referenceLines || [];
  for (const ref of refs) {
    if (!ref || !ref.D) continue;
    ctx.strokeStyle = ref.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx(ref.D), 18);
    ctx.lineTo(sx(ref.D), h - 44);
    ctx.stroke();
    if (ref.annotate) {
      ctx.fillStyle = ref.color;
      ctx.font = "12px sans-serif";
      ctx.fillText(`${ref.label} = ${num(ref.D)}`,
                   Math.min(w - 170, sx(ref.D) + 6), ref.y || 31);
    }
  }
  for (const s of series) {
    if (!s.rows.length) continue;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.globalAlpha = s.alpha || 1;
    ctx.lineWidth = s.width || 1.4;
    ctx.beginPath();
    s.rows.forEach((p, i) => {
      const x = sx(p.D);
      const y = sy(Number(p.delta_chi2));
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
    if (s.markers) {
      for (const p of s.rows) {
        ctx.beginPath();
        ctx.arc(sx(p.D), sy(Number(p.delta_chi2)), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }
  if (opts.legend || opts.referenceLegend) {
    const legendPos = opts.legendPosition || "topRight";
    const legendItemCount = (opts.legend ? series.length : 0) +
                            (opts.referenceLegend ? refs.filter(ref => ref && ref.D).length : 0);
    let y = legendPos === "bottomRight" ? h - Math.max(14, legendItemCount * 17) : 28;
    ctx.font = "12px sans-serif";
    if (opts.legend) {
      for (const s of series) {
        ctx.fillStyle = s.color;
        ctx.fillRect(w - 156, y - 9, 10, 10);
        ctx.fillStyle = "#18222d";
        ctx.fillText(s.label, w - 140, y);
        y += 17;
      }
    }
    if (opts.referenceLegend) {
      for (const ref of refs) {
        if (!ref || !ref.D) continue;
        ctx.strokeStyle = ref.color;
        ctx.beginPath();
        ctx.moveTo(w - 156, y - 5);
        ctx.lineTo(w - 146, y - 5);
        ctx.stroke();
        ctx.fillStyle = "#18222d";
        ctx.fillText(ref.label, w - 140, y);
        y += 17;
      }
    }
  }
}
function referenceLines(opts={}) {
  const lines = [];
  if (opts.global && data.bestD && data.bestD.all) {
    lines.push({D: data.bestD.all, color: "#2c8a68", label: "all-tiles best D", annotate: opts.annotate, y: 31});
  }
  if (opts.interior && data.bestD && data.bestD.interior) {
    lines.push({D: data.bestD.interior, color: "#6f63bd", label: "interior best D", annotate: opts.annotate, y: 47});
  }
  if (opts.periphery && data.bestD && data.bestD.periphery) {
    lines.push({D: data.bestD.periphery, color: "#c27a22", label: "periphery best D", annotate: opts.annotate, y: 63});
  }
  return lines;
}
function renderGlobal() {
  drawProfiles(byId("globalCanvas"), [{
    label: "all tiles constant D",
    rows: data.globalProfile,
    color: "#2868b7",
    width: 2,
    markers: true
  }], {
    legend: true,
    legendPosition: "bottomRight",
    yCap: 12,
    referenceLines: referenceLines({
      global: true, interior: true, periphery: true,
      annotate: true
    })
  });
}
function renderAll() {
  const filter = byId("allFilter").value || "all";
  const shown = tileIds.filter(tid => {
    if (filter === "all") return true;
    return data.tileLocation[tid] === filter;
  });
  const series = shown.map((tid, i) => ({
    label: `tile ${tid}`,
    rows: data.tileProfiles[tid] || [],
    color: allColors[i % allColors.length],
    alpha: 0.42,
    width: 1
  }));
  drawProfiles(byId("allCanvas"), series, {
    yCap: 12,
    referenceLines: referenceLines({
      global: true, interior: true, periphery: true,
      annotate: true
    })
  });
}
function renderSelected() {
  const selected = [];
  for (let i = 0; i < 4; i++) {
    const tid = byId(`sel${i}`).value;
    if (!tid) continue;
    selected.push({
      label: `tile ${tid}`,
      rows: data.tileProfiles[tid] || [],
      color: colors[i],
      width: 2,
      markers: true
    });
  }
  drawProfiles(byId("selectedCanvas"), selected, {legend: true, yCap: 12});
  renderTable(selected.map(s => s.label.replace("tile ", "")));
}
function renderTable(selectedIds) {
  const rows = data.summary
    .filter(r => !selectedIds.length || selectedIds.includes(String(r.tile_id)))
    .sort((a,b) => Number(a.tile_id) - Number(b.tile_id));
  const cols = [
    ["tile_id", "tile"],
    ["location", "location"],
    ["D_hat", "D_hat"],
    ["chi2_red", "chi2 red"],
    ["width_decades_dchi1", "width dchi1"],
    ["max_delta_chi2", "max dchi2"],
    ["n_dc", "n dc"],
    ["n_h1", "n h1"],
    ["n_h2", "n h2"]
  ];
  const cell = (r, key) => {
    const value = r[key];
    if (key === "tile_id" || key === "location") return value ?? "";
    return num(value);
  };
  const head = `<thead><tr>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>`;
  const body = rows.map(r => `<tr>${cols.map(c => `<td>${cell(r, c[0])}</td>`).join("")}</tr>`).join("");
  byId("summaryTable").innerHTML = head + `<tbody>${body}</tbody>`;
}
function render() {
  renderGlobal();
  renderAll();
  renderSelected();
}
function setup() {
  const objective = data.defaults.objective === "ordinary_least_squares" ? "ordinary LS" : "weighted LS";
  byId("meta").textContent = `objective=${objective}; variable=${data.defaults.observation_variable}; source=${data.defaults.observation_source}; all best D=${num(data.bestD.all)}; interior best D=${num(data.bestD.interior)}; periphery best D=${num(data.bestD.periphery)}; tiles=${tileIds.length}`;
  setupSelectors();
  window.addEventListener("resize", render);
  render();
}
setup();
</script>
</body>
</html>
"""


def _write_dashboard(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(
        _clean_json_value(payload), allow_nan=False, separators=(",", ":"))
    path.write_text(
        HTML_TEMPLATE.replace("__DATA__", html.escape(data_json, quote=False)),
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run synthetic/GNN-prior tile distensibility inference dashboard.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--graph", default=(str(DEFAULT_SYNTHETIC_GRAPH)
                    if DEFAULT_SYNTHETIC_GRAPH.exists() else None),
                    help="Observation graph. Defaults to synthetic_mosaic_graph.gpickle.")
    ap.add_argument("--tiles", nargs="*", type=int, default=None)
    ap.add_argument("--all-tiles", action="store_true",
                    help="Run all measured tiles. This is also the default "
                         "when --tiles is omitted.")
    ap.add_argument("--out-dir", default=None,
                    help=f"Default: {DEFAULT_OUT_DIR}")
    ap.add_argument("--observation-source", default=DEFAULT_OBSERVATION_SOURCE,
                    help="Label written into output CSV/HTML metadata.")
    ap.add_argument("--observation-variable", choices=("velocity", "flow"),
                    default="velocity",
                    help="Profile against velocity by default. Flow is kept "
                         "for backward-compatible comparisons.")
    ap.add_argument("--tile-harmonics", nargs="+", type=int,
                    default=list(DEFAULT_TILE_PROFILE_HARMONICS),
                    choices=[1, 2])
    ap.add_argument("--D-min", type=float, default=1e-6)
    ap.add_argument("--D-max", type=float, default=1e-1)
    ap.add_argument("--D-count", type=int, default=41)
    ap.add_argument("--a-dc", type=float, default=0.061,
                    help="DC additive noise floor in nL/s.")
    ap.add_argument("--a-h1", type=float, default=0.012,
                    help="H1 additive noise floor in nL/s.")
    ap.add_argument("--a-h2", type=float, default=0.030,
                    help="H2 additive noise floor in nL/s.")
    ap.add_argument("--b-dc", type=float, default=0.29)
    ap.add_argument("--b-h1", type=float, default=0.0)
    ap.add_argument("--b-h2", type=float, default=0.0)
    ap.add_argument("--weighted-least-squares", action="store_true",
                    help="Use sigma-weighted least squares and write outputs "
                         "with _weighted in their filenames. This is also "
                         "the default objective when neither LS flag is set.")
    ap.add_argument("--ordinary-least-squares", action="store_true",
                    help="Use one constant sigma equal to the average "
                         "weighted-noise sigma. This gives ordinary LS "
                         "relative weighting without SI-unit chi2 collapse.")
    ap.add_argument("--pressure-prior-dir",
                    default=(str(DEFAULT_GNN_PRESSURE_PRIOR_DIR)
                             if DEFAULT_GNN_PRESSURE_PRIOR_DIR.exists()
                             else None),
                    help="GNN edge-pressure run directory, or sweep root, with "
                         "node_pressures_physics_gnn.csv.")
    ap.add_argument("--pressure-prior-mode",
                    choices=("off", "absolute", "scaled"), default="scaled",
                    help="Regularize fitted tile boundary pressures toward "
                         "the GNN pressure field exactly or up to scale.")
    ap.add_argument("--lambda-pressure-prior", type=float, default=1.0)
    ap.add_argument("--pressure-prior-sigma-pa", type=float, default=0.0,
                    help="Pressure prior sigma in Pa. If <=0, use the tile "
                         "boundary prior RMS with a 1 Pa floor.")
    args = ap.parse_args()
    if args.weighted_least_squares and args.ordinary_least_squares:
        raise SystemExit("Choose only one of --weighted-least-squares or "
                         "--ordinary-least-squares.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _install_numpy_pickle_compat()
    graph, graph_path = load_graph_from_args(args)
    pressure_prior, pressure_prior_dir = _load_pressure_prior(
        None if args.pressure_prior_mode == "off" else args.pressure_prior_dir)
    tiles = choose_tiles(graph, args.tiles, all_tiles=(args.all_tiles
                         or not args.tiles))
    D_grid = np.logspace(np.log10(args.D_min), np.log10(args.D_max),
                         int(args.D_count))

    print(f"Graph: {graph_path}")
    print(f"Tiles: {tiles}")
    print(f"Output: {out_dir}")
    print("Inference config:")
    print(f"  observation_source={args.observation_source}")
    print(f"  observation_variable={args.observation_variable}")
    print(f"  tile_harmonics={tuple(int(h) for h in args.tile_harmonics)}")
    print(f"  D_grid=[{args.D_min:.3e}, {args.D_max:.3e}], "
          f"count={args.D_count}")
    objective_label = _objective_name(args).replace("_", " ")
    print(f"Tile profile objective: {objective_label}")
    if args.pressure_prior_mode == "off":
        print("Pressure prior: off")
    else:
        print(f"Pressure prior: mode={args.pressure_prior_mode}, "
              f"lambda={args.lambda_pressure_prior:g}, "
              f"source={pressure_prior_dir}")

    t0 = time.time()
    tile_profiles: Dict[int, List[dict]] = {}
    summary_rows: List[dict] = []
    profile_rows: List[dict] = []
    for i, tile_id in enumerate(tiles, start=1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(tiles) - i + 1) if i > 1 else 0
        eta_txt = f", eta={eta / 60.0:.1f} min" if i > 1 else ""
        print(f"[{i}/{len(tiles)}] tile {tile_id}{eta_txt}", flush=True)
        try:
            profile, metrics = _profile_tile_from_observations(
                graph, int(tile_id), D_grid, args, pressure_prior)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary_rows.append({
                "tile_id": int(tile_id), "error": f"{type(e).__name__}: {e}"})
            continue
        tile_profiles[int(tile_id)] = profile
        metrics["location"] = _tile_location(int(tile_id))
        summary_rows.append(metrics)
        chi_min = min(p["chi2"] for p in profile)
        for p in profile:
            profile_rows.append({
                "tile_id": int(tile_id),
                "observation_source": args.observation_source,
                "observation_variable": args.observation_variable,
                "objective": metrics["objective"],
                "weight_mode": metrics["weight_mode"],
                "pressure_prior_mode": metrics["pressure_prior_mode"],
                "lambda_pressure_prior": metrics["lambda_pressure_prior"],
                "pressure_prior_available": metrics["pressure_prior_available"],
                "D": float(p["D"]),
                "chi2": float(p["chi2"]),
                "data_chi2": float(p.get("data_chi2", p["chi2"])),
                "pressure_prior_chi2": float(
                    p.get("pressure_prior_chi2", 0.0)),
                "delta_chi2": float(p["chi2"] - chi_min),
            })
        print(f"  D_hat={metrics['D_hat']:.3e}  "
              f"chi2_red={metrics['chi2_red']:.3g}  "
              f"width1={metrics['width_decades_dchi1']:.3g} decades")

    global_rows = _summed_profile(tile_profiles)
    interior_profiles = {
        tid: prof for tid, prof in tile_profiles.items()
        if _tile_location(tid) == "interior"
    }
    periphery_profiles = {
        tid: prof for tid, prof in tile_profiles.items()
        if _tile_location(tid) == "periphery"
    }
    interior_rows = _summed_profile(interior_profiles)
    periphery_rows = _summed_profile(periphery_profiles)
    best_D = {
        "all": _best_D(global_rows),
        "interior": _best_D(interior_rows),
        "periphery": _best_D(periphery_rows),
    }
    for rows, group in ((global_rows, "all"), (interior_rows, "interior"),
                        (periphery_rows, "periphery")):
        for row in rows:
            row["observation_source"] = args.observation_source
            row["observation_variable"] = args.observation_variable
            row["objective"] = _objective_name(args)
            row["weight_mode"] = ("sigma" if _is_weighted_objective(args)
                                  else "constant_average_sigma")
            row["pressure_prior_mode"] = args.pressure_prior_mode
            row["lambda_pressure_prior"] = float(args.lambda_pressure_prior)
            row["tile_group"] = group
    csv_paths = [
        out_dir / _out_name("gnn_synthetic_global_profile_constant_D.csv", args),
        out_dir / _out_name("gnn_synthetic_interior_profile_constant_D.csv", args),
        out_dir / _out_name("gnn_synthetic_periphery_profile_constant_D.csv", args),
        out_dir / _out_name("gnn_synthetic_tile_profiles.csv", args),
        out_dir / _out_name("gnn_synthetic_tile_profile_summary.csv", args),
    ]
    for path, rows in zip(csv_paths, [
            global_rows, interior_rows, periphery_rows, profile_rows,
            summary_rows]):
        _write_csv(path, rows)
        print(f"Wrote {path}")

    tile_payload = {}
    for tile_id, profile in tile_profiles.items():
        chi_min = min(p["chi2"] for p in profile)
        tile_payload[str(tile_id)] = [
            {"D": float(p["D"]),
             "chi2": float(p["chi2"]),
             "delta_chi2": float(p["chi2"] - chi_min)}
            for p in profile
        ]
    payload = {
        "defaults": {
            "observation_source": args.observation_source,
            "observation_variable": args.observation_variable,
            "graph": str(graph_path),
            "tile_harmonics": [int(h) for h in args.tile_harmonics],
            "D_min": float(args.D_min),
            "D_max": float(args.D_max),
            "D_count": int(args.D_count),
            "objective": _objective_name(args),
            "weighted_least_squares": bool(_is_weighted_objective(args)),
            "weighted_output_names": bool(args.weighted_least_squares),
            "ordinary_least_squares": bool(args.ordinary_least_squares),
            "pressure_prior_mode": args.pressure_prior_mode,
            "lambda_pressure_prior": float(args.lambda_pressure_prior),
            "pressure_prior_sigma_pa": float(args.pressure_prior_sigma_pa),
            "pressure_prior_dir": pressure_prior_dir,
        },
        "globalProfile": global_rows,
        "groupProfiles": {
            "interior": interior_rows,
            "periphery": periphery_rows,
        },
        "bestD": best_D,
        "tileProfiles": tile_payload,
        "tileLocation": {str(t): _tile_location(t) for t in tile_profiles},
        "summary": summary_rows,
    }
    html_path = out_dir / _out_name("infer_default_mosaic_tile_profiles.html",
                                    args)
    _write_dashboard(html_path, payload)
    print(f"Wrote {html_path}")
    print(f"Done in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
