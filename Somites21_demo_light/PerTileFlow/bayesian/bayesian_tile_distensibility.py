"""Deterministic Bayesian tile-wise distensibility inference.

This script estimates tile-wise areal distensibility D0 from measured
harmonic edge-flow phasors without MCMC.  At each D0 grid point it builds a
linear harmonic transfer operator from boundary pressure phasors to edge-flow
phasors.  The high-dimensional boundary pressures are analytically
marginalized under a zero-mean Gaussian prior; the low-dimensional nuisance
parameters (boundary pressure scale, additive noise, proportional noise) are
integrated on log grids with log-sum-exp.

Model for harmonic n, in real-stacked form:

    y_n = T_n(D0) b_n + eps_n
    b_n | tau_b ~ N(0, (tau_b^2 / 2) I)
    eps_n ~ N(0, Sigma_n(a, s^2))

After analytic marginalization of b_n:

    y_n | D0, tau_b, a, s^2 ~ N(0, Sigma_n + T_n P_b T_n.T)

The posterior over log(D0) is then obtained by grid integration over
eta_b=log(tau_b), alpha=log(a), and rho=log(s^2).  H1 is used by default;
H2 can be enabled with ``--harmonics 1 2``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pertile_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for smoke envs.
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = DATASET_ROOT / "emb1" / "config.json"

N_L_PER_M3 = 1.0e12
PX_SIZE_M_DEFAULT = 1.7e-6
MU_DEFAULT = 3.5e-3
F0_HZ_DEFAULT = 1.0
EPS = 1e-300
_WORKER_GRAPH = None
_WORKER_ARGS = None


@dataclass
class TileObservations:
    """Observed complex phasors for one tile and its measured edge set."""

    tile_id: int
    edges: list[tuple[object, object]]
    subgraph: nx.Graph
    harmonics: tuple[int, ...]
    q_obs: dict[int, np.ndarray]
    valid: dict[int, np.ndarray]
    f0_hz: dict[int, float]
    unit_rows: list[dict] = field(default_factory=list)


@dataclass
class JitterStats:
    """Cholesky stabilization diagnostics."""

    attempts: int = 0
    jitter_uses: int = 0
    failures: int = 0
    max_jitter: float = 0.0

    def as_dict(self) -> dict:
        return {
            "cholesky_attempts": int(self.attempts),
            "cholesky_jitter_uses": int(self.jitter_uses),
            "cholesky_failures": int(self.failures),
            "cholesky_max_jitter": float(self.max_jitter),
        }


def _safe_float(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written against either numpy.core or numpy._core."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_graph(args) -> tuple[nx.Graph, Path]:
    """Load the NetworkX graph from --graph or from a PerTileFlow config."""

    graph_path = Path(args.graph).expanduser() if args.graph else None
    if graph_path is None:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path
        with open(cfg_path) as f:
            cfg = json.load(f)
        raw = cfg.get("mosaic_graph")
        if not raw:
            raise SystemExit("Config has no 'mosaic_graph'; pass --graph.")
        graph_path = Path(raw)
        if not graph_path.is_absolute():
            graph_path = cfg_path.parent / graph_path
    graph_path = graph_path.resolve()
    with open(graph_path, "rb") as f:
        graph = _NumpyCompatUnpickler(f).load()
    return graph, graph_path


def collect_tile_ids(graph: nx.Graph) -> list[int]:
    """Return sorted tile IDs present in edge ``measurements_piv`` records."""

    seen: set[int] = set()
    for _, _, ed in graph.edges(data=True):
        for meas in ed.get("measurements_piv") or []:
            try:
                seen.add(int(meas.get("tile_id")))
            except (TypeError, ValueError):
                continue
    return sorted(seen)


def _measurement_for_tile(ed: dict, tile_id: int) -> dict | None:
    best = None
    best_snr = -float("inf")
    for meas in ed.get("measurements_piv") or []:
        try:
            if int(meas.get("tile_id")) != int(tile_id):
                continue
        except (TypeError, ValueError):
            continue
        snr = _safe_float(meas.get("snr_harm_fit_db"),
                          _safe_float(meas.get("Q_H1_snr_db"), 0.0))
        if best is None or snr > best_snr:
            best = meas
            best_snr = snr
    return best


def _edge_sign(ed: dict, u, v, orient: bool = True) -> float:
    if not orient:
        return 1.0
    flow_from = ed.get("flow_from")
    flow_to = ed.get("flow_to")
    if flow_from is None or flow_to is None:
        return 1.0
    return 1.0 if (flow_from == u and flow_to == v) else -1.0


def _phasor_from_measurement(meas: dict, harmonic: int) -> complex | None:
    amp = meas.get(f"Q_H{int(harmonic)}_amp")
    phi = meas.get(f"Q_H{int(harmonic)}_phi")
    amp = _safe_float(amp)
    phi = _safe_float(phi)
    if not (math.isfinite(amp) and math.isfinite(phi)):
        return None
    return complex(float(amp) * np.exp(1j * float(phi)) / N_L_PER_M3)


def _representative_f0(graph: nx.Graph, edges: Sequence[tuple],
                       tile_id: int, default_f0_hz: float) -> float:
    vals = []
    for u, v in edges:
        ed = graph.edges[u, v]
        meas = _measurement_for_tile(ed, tile_id)
        if meas is not None:
            vals.append(_safe_float(meas.get("f0_hz")))
        vals.append(_safe_float(ed.get("f0_hz")))
    vals = [v for v in vals if math.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else float(default_f0_hz)


def build_tile_observations(graph: nx.Graph, tile_id: int,
                            harmonics: Sequence[int], args) -> TileObservations:
    """Collect oriented per-tile harmonic edge-flow phasors.

    Each edge is oriented according to the NetworkX edge tuple used in the
    subgraph and transfer matrix.  If stored ``flow_from``/``flow_to`` indicate
    the measured flow direction is opposite this orientation, the observed
    phasor is multiplied by -1.
    """

    edges: list[tuple[object, object]] = []
    q_obs = {int(h): [] for h in harmonics}
    valid = {int(h): [] for h in harmonics}
    unit_rows: list[dict] = []

    for u, v, ed in graph.edges(data=True):
        meas = _measurement_for_tile(ed, int(tile_id))
        if meas is None:
            continue
        sign = _edge_sign(ed, u, v, orient=not args.no_orient_observations)
        any_valid = False
        per_h: dict[int, complex] = {}
        for h in harmonics:
            ph = _phasor_from_measurement(meas, int(h))
            if ph is None:
                per_h[int(h)] = 0.0 + 0.0j
            else:
                per_h[int(h)] = complex(sign * ph)
                any_valid = True
        if not any_valid:
            continue
        edges.append((u, v))
        for h in harmonics:
            q_obs[int(h)].append(per_h[int(h)])
            valid[int(h)].append(abs(per_h[int(h)]) > 0)

        radius_m, length_m, source = get_edge_geometry(ed, args)
        unit_rows.append({
            "tile_id": int(tile_id),
            "edge_u": str(u),
            "edge_v": str(v),
            "radius_m": radius_m,
            "length_m": length_m,
            "geometry_source": source,
            "flow_sign": sign,
        })

    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    sub = graph.subgraph(nodes).copy()
    # Keep only measured tile edges in the transfer target while retaining
    # attached subgraph anatomy that appears among the measured nodes.
    harmonics = tuple(int(h) for h in harmonics)
    q_arr = {h: np.asarray(q_obs[h], dtype=complex) for h in harmonics}
    valid_arr = {h: np.asarray(valid[h], dtype=bool) for h in harmonics}
    f0 = _representative_f0(graph, edges, int(tile_id), args.default_f0_hz)
    return TileObservations(
        tile_id=int(tile_id),
        edges=edges,
        subgraph=sub,
        harmonics=harmonics,
        q_obs=q_arr,
        valid=valid_arr,
        f0_hz={h: float(f0) for h in harmonics},
        unit_rows=unit_rows,
    )


def choose_tile_boundary_nodes(graph: nx.Graph, obs: TileObservations,
                               args) -> tuple[list[object], list[object]]:
    """Choose cut nodes plus explicit graph boundary nodes as tile boundaries."""

    tile_nodes = set(obs.subgraph.nodes())
    measured_edges = set(tuple(e) for e in obs.edges)
    measured_edges |= {(v, u) for u, v in obs.edges}
    boundary = set()
    for node in tile_nodes:
        if graph.nodes[node].get("boundary_type") is not None:
            boundary.add(node)
            continue
        if args.boundary_mode in ("metadata",):
            continue
        for nbr in graph.neighbors(node):
            if nbr not in tile_nodes:
                boundary.add(node)
                break
    if args.boundary_mode == "all-edge-endpoints":
        for u, v in obs.edges:
            boundary.add(u)
            boundary.add(v)
    if args.min_boundary_nodes and len(boundary) < int(args.min_boundary_nodes):
        # Fallback: use high-degree measured endpoints until the linear system
        # has a meaningful boundary pressure space.
        endpoints = sorted(
            tile_nodes,
            key=lambda n: graph.degree[n],
            reverse=True,
        )
        boundary.update(endpoints[:int(args.min_boundary_nodes)])

    boundary_nodes = sorted(boundary, key=lambda n: str(n))
    interior_nodes = sorted(tile_nodes - set(boundary_nodes), key=lambda n: str(n))
    return boundary_nodes, interior_nodes


def get_edge_geometry(ed: dict, args) -> tuple[float, float, str]:
    """Return edge radius and length in SI units."""

    source = []
    r_raw = ed.get(args.radius_field)
    if r_raw is None and args.radius_field != "radius_adapted_m":
        r_raw = ed.get("radius_adapted_m")
    if r_raw is None:
        r_raw = ed.get("radius_px_true")
        source.append("radius_px_true")
    if r_raw is None:
        r_raw = ed.get("radius")
        source.append("radius")
    else:
        source.append(args.radius_field)
    radius = _safe_float(r_raw)

    l_raw = ed.get(args.length_field)
    if l_raw is None and args.length_field != "length_true":
        l_raw = ed.get("length_true")
    if l_raw is None:
        l_raw = ed.get("length")
        source.append("length")
    else:
        source.append(args.length_field)
    length = _safe_float(l_raw)

    if not (math.isfinite(radius) and radius > 0):
        radius = float("nan")
    elif radius > args.meter_threshold:
        radius *= float(args.radius_scale)

    if not (math.isfinite(length) and length > 0):
        length = float("nan")
    elif length > args.meter_threshold:
        length *= float(args.length_scale)

    return float(radius), float(length), "+".join(source)


def edge_admittances(radius_m: float, length_m: float, D0: float,
                     harmonic: int, f0_hz: float, mu: float) -> tuple[complex, complex, float, float]:
    """Compute compliant tube self/trans admittances for one harmonic.

    The small-xi expansion avoids cancellation around the rigid-tube limit.
    """

    omega_n = 2.0 * math.pi * float(f0_hz) * int(harmonic)
    G = math.pi * radius_m ** 4 / (8.0 * float(mu) * max(length_m, EPS))
    C = 8.0 * float(mu) * float(D0) * omega_n * length_m ** 2 / max(radius_m ** 2, EPS)
    xi = np.sqrt(1j * C)
    if abs(xi) < 1e-6:
        y_self = G * (1.0 + 1j * C / 3.0)
        y_trans = G * (1.0 - 1j * C / 6.0)
    else:
        y_self = G * xi / np.tanh(xi)
        y_trans = G * xi / np.sinh(xi)
    return complex(y_self), complex(y_trans), float(G), float(C)


def assemble_admittance_matrix(graph: nx.Graph, edges: Sequence[tuple],
                               nodes: Sequence[object], D0: float,
                               harmonic: int, f0_hz: float, args) -> tuple[np.ndarray, dict, list[dict]]:
    """Assemble the complex nodal admittance matrix by edge stamping."""

    idx = {node: i for i, node in enumerate(nodes)}
    Y = np.zeros((len(nodes), len(nodes)), dtype=complex)
    edge_params: dict[tuple[object, object], tuple[complex, complex]] = {}
    rows = []
    for u, v in edges:
        if u not in idx or v not in idx:
            continue
        r_m, l_m, source = get_edge_geometry(graph.edges[u, v], args)
        if not (math.isfinite(r_m) and math.isfinite(l_m) and r_m > 0 and l_m > 0):
            continue
        y_self, y_trans, conductance, C = edge_admittances(
            r_m, l_m, float(D0), int(harmonic), float(f0_hz), float(args.mu))
        i = idx[u]
        j = idx[v]
        Y[i, i] += y_self
        Y[j, j] += y_self
        Y[i, j] -= y_trans
        Y[j, i] -= y_trans
        edge_params[(u, v)] = (y_self, y_trans)
        edge_params[(v, u)] = (y_self, y_trans)
        rows.append({
            "edge_u": str(u),
            "edge_v": str(v),
            "harmonic": int(harmonic),
            "radius_m": r_m,
            "length_m": l_m,
            "conductance_m3_per_Pa_s": conductance,
            "C_dimensionless": C,
            "geometry_source": source,
        })
    return Y, edge_params, rows


def solve_transfer_matrix(graph: nx.Graph, obs: TileObservations,
                          boundary_nodes: Sequence[object],
                          interior_nodes: Sequence[object], D0: float,
                          harmonic: int, args) -> tuple[np.ndarray, list[dict]]:
    """Map complex boundary pressure phasors to measured edge-flow phasors."""

    nodes = list(boundary_nodes) + list(interior_nodes)
    n_b = len(boundary_nodes)
    n_i = len(interior_nodes)
    if n_b == 0:
        raise ValueError("no boundary nodes")

    Y, edge_params, geom_rows = assemble_admittance_matrix(
        graph, list(obs.subgraph.edges()), nodes, float(D0), int(harmonic),
        obs.f0_hz[int(harmonic)], args)

    Y_II = Y[n_b:, n_b:]
    Y_IB = Y[n_b:, :n_b]
    P_int_basis = np.zeros((n_i, n_b), dtype=complex)
    if n_i:
        rhs = -Y_IB
        try:
            P_int_basis = np.linalg.solve(Y_II, rhs)
        except np.linalg.LinAlgError:
            P_int_basis, *_ = np.linalg.lstsq(Y_II, rhs, rcond=1e-12)

    P_basis = np.vstack([np.eye(n_b, dtype=complex), P_int_basis])
    node_index = {node: i for i, node in enumerate(nodes)}
    T = np.zeros((len(obs.edges), n_b), dtype=complex)
    for row_i, (u, v) in enumerate(obs.edges):
        params = edge_params.get((u, v))
        if params is None or u not in node_index or v not in node_index:
            continue
        y_self, y_trans = params
        pu = P_basis[node_index[u], :]
        pv = P_basis[node_index[v], :]
        T[row_i, :] = y_self * pu - y_trans * pv
    return T, geom_rows


def complex_to_real_operator(T_complex: np.ndarray,
                             q_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert complex y=Tz to real-stacked y=A b."""

    y = np.concatenate([q_obs.real, q_obs.imag]).astype(float)
    A = np.block([
        [T_complex.real, -T_complex.imag],
        [T_complex.imag, T_complex.real],
    ]).astype(float)
    return y, A


def build_noise_covariance(q_obs: np.ndarray, a: float,
                           s2: float) -> np.ndarray:
    """Diagonal real-stacked covariance for circular complex residual noise."""

    sig2 = float(a) ** 2 + float(s2) * np.abs(q_obs) ** 2
    sig2 = np.maximum(sig2, EPS)
    return np.concatenate([0.5 * sig2, 0.5 * sig2]).astype(float)


def _cholesky_logpdf_zero_mean(y: np.ndarray, C: np.ndarray,
                               jitter_stats: JitterStats,
                               jitter_initial: float = 1e-14) -> float:
    eye = np.eye(C.shape[0])
    scale = float(np.nanmedian(np.diag(C))) if C.size else 1.0
    scale = max(scale, 1e-300)
    jitter = 0.0
    for attempt in range(10):
        jitter_stats.attempts += 1
        try:
            cf = cho_factor(C + jitter * eye, lower=True, check_finite=False)
            alpha = cho_solve(cf, y, check_finite=False)
            logdet = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
            return float(-0.5 * (logdet + y @ alpha + y.size * math.log(2.0 * math.pi)))
        except Exception:
            jitter = max(jitter_initial * scale, 10.0 * jitter if jitter else jitter_initial * scale)
            jitter_stats.jitter_uses += 1
            jitter_stats.max_jitter = max(jitter_stats.max_jitter, float(jitter))
    jitter_stats.failures += 1
    return -float("inf")


def log_marginal_likelihood(y: np.ndarray, T: np.ndarray, q_obs: np.ndarray,
                            tau_b: float, a: float, s2: float,
                            jitter_stats: JitterStats,
                            method: str = "woodbury") -> float:
    """Evaluate log p(y | D0, tau_b, a, s^2) after marginalizing b.

    Boundary pressure b is analytically marginalized here.  The parameters
    tau_b, a, and s^2 are *not* optimized; callers integrate them on grids.
    """

    diag = build_noise_covariance(q_obs, float(a), float(s2))
    p_scale = 0.5 * float(tau_b) ** 2
    if method == "woodbury":
        # C = S + beta T T^T with diagonal S and beta=tau_b^2/2.
        # Matrix determinant lemma:
        #   log|C| = log|S| + log|beta I| + log|beta^-1 I + T^T S^-1 T|
        # Woodbury:
        #   y^T C^-1 y = y^T S^-1 y - v^T K^-1 v,
        # where v=T^T S^-1 y and K=beta^-1 I + T^T S^-1 T.
        inv_diag = 1.0 / np.maximum(diag, EPS)
        beta = max(float(p_scale), EPS)
        K = (T.T * inv_diag[None, :]) @ T
        K += (1.0 / beta) * np.eye(T.shape[1])
        K = 0.5 * (K + K.T)
        v = T.T @ (inv_diag * y)
        jitter = 0.0
        eye = np.eye(K.shape[0])
        scale = float(np.nanmedian(np.diag(K))) if K.size else 1.0
        scale = max(scale, 1e-300)
        for _ in range(10):
            jitter_stats.attempts += 1
            try:
                cf = cho_factor(K + jitter * eye, lower=True,
                                check_finite=False)
                kinv_v = cho_solve(cf, v, check_finite=False)
                logdet_K = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
                logdet = (
                    float(np.sum(np.log(np.maximum(diag, EPS))))
                    + T.shape[1] * math.log(beta)
                    + logdet_K
                )
                quad = float(np.sum(y * y * inv_diag) - v @ kinv_v)
                quad = max(quad, 0.0)
                return float(-0.5 * (
                    logdet + quad + y.size * math.log(2.0 * math.pi)))
            except Exception:
                jitter = max(1e-14 * scale,
                             10.0 * jitter if jitter else 1e-14 * scale)
                jitter_stats.jitter_uses += 1
                jitter_stats.max_jitter = max(jitter_stats.max_jitter,
                                              float(jitter))
        jitter_stats.failures += 1
        return -float("inf")

    C = np.diag(diag) + p_scale * (T @ T.T)
    C = 0.5 * (C + C.T)
    return _cholesky_logpdf_zero_mean(y, C, jitter_stats)


def _normal_logpdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return -0.5 * ((x - mean) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def integrate_nuisance_for_harmonic(
        y: np.ndarray, T: np.ndarray, q_obs: np.ndarray,
        tau_grid: np.ndarray, a_grid: np.ndarray, s2_grid: np.ndarray,
        tau0: float, omega_b: float, jitter_stats: JitterStats,
        likelihood_method: str = "woodbury") -> dict:
    """Grid-integrate eta_b, alpha, and rho for one harmonic at one D0."""

    eta_grid = np.log(np.asarray(tau_grid, dtype=float))
    alpha_grid = np.log(np.asarray(a_grid, dtype=float))
    rho_grid = np.log(np.asarray(s2_grid, dtype=float))
    logp_eta = _normal_logpdf(eta_grid, math.log(float(tau0)), float(omega_b))
    # Uniform in alpha/rho over the chosen integration range.
    logp_alpha = np.full_like(alpha_grid, -math.log(max(alpha_grid[-1] - alpha_grid[0], EPS)))
    logp_rho = np.full_like(rho_grid, -math.log(max(rho_grid[-1] - rho_grid[0], EPS)))
    logw_eta = np.full_like(
        eta_grid,
        math.log(max(abs(float(eta_grid[1] - eta_grid[0])), EPS))
        if len(eta_grid) > 1 else 0.0,
    )
    logw_alpha = np.full_like(
        alpha_grid,
        math.log(max(abs(float(alpha_grid[1] - alpha_grid[0])), EPS))
        if len(alpha_grid) > 1 else 0.0,
    )
    logw_rho = np.full_like(
        rho_grid,
        math.log(max(abs(float(rho_grid[1] - rho_grid[0])), EPS))
        if len(rho_grid) > 1 else 0.0,
    )

    log_terms = np.empty((len(tau_grid), len(a_grid), len(s2_grid)), dtype=float)
    for i, tau in enumerate(tau_grid):
        for j, a in enumerate(a_grid):
            for k, s2 in enumerate(s2_grid):
                ll = log_marginal_likelihood(
                    y, T, q_obs, tau, a, s2, jitter_stats,
                    method=likelihood_method)
                log_terms[i, j, k] = (
                    ll + logp_eta[i] + logp_alpha[j] + logp_rho[k]
                    + logw_eta[i] + logw_alpha[j] + logw_rho[k]
                )

    return {
        "log_marginal": float(logsumexp(log_terms)),
        "log_terms": log_terms,
    }


def normalize_posterior(logD_grid: np.ndarray,
                        logpost: np.ndarray) -> np.ndarray:
    """Normalize posterior mass on a uniform log-D grid."""

    finite = np.isfinite(logpost)
    probs = np.zeros_like(logpost, dtype=float)
    if not finite.any():
        return probs
    dx = float(np.median(np.diff(logD_grid))) if len(logD_grid) > 1 else 1.0
    z = logsumexp(logpost[finite] + math.log(abs(dx)))
    probs[finite] = np.exp(logpost[finite] + math.log(abs(dx)) - z)
    total = float(np.sum(probs))
    return probs / total if total > 0 else probs


def _weighted_quantile_sorted(x: np.ndarray, p: np.ndarray,
                              q: float) -> float:
    if not x.size or float(np.sum(p)) <= 0:
        return float("nan")
    cdf = np.cumsum(p / float(np.sum(p)))
    return float(np.interp(float(q), cdf, x))


def summarize_posterior(tile_id: int, D_grid: np.ndarray,
                        logD_grid: np.ndarray, logpost: np.ndarray,
                        prob: np.ndarray, n_edges: int, n_boundary: int,
                        n_interior: int, harmonics: Sequence[int],
                        args) -> dict:
    """Summarize the normalized tile posterior over D0."""

    prior_sd = float(args.logD_prior_tau)
    finite = np.isfinite(logpost)
    if not finite.any() or float(np.sum(prob)) <= 0:
        raise ValueError("no finite posterior")
    mode_idx = int(np.nanargmax(logpost))
    D_mode = float(D_grid[mode_idx])
    D_mean = float(np.sum(D_grid * prob))
    log_median = _weighted_quantile_sorted(logD_grid, prob, 0.5)
    log_low = _weighted_quantile_sorted(logD_grid, prob, 0.025)
    log_high = _weighted_quantile_sorted(logD_grid, prob, 0.975)
    D_median = float(math.exp(log_median))
    D_low = float(math.exp(log_low))
    D_high = float(math.exp(log_high))
    W95_decades = float(math.log10(D_high / D_low)) if D_low > 0 else float("inf")
    prior_width_decades = float(2.0 * 1.96 * prior_sd / math.log(10.0))
    lower_mass = float(prob[0])
    upper_mass = float(prob[-1])
    width_ratio = float(prior_width_decades / W95_decades) if W95_decades > 0 else float("inf")
    identified = (
        mode_idx > 0
        and mode_idx < len(D_grid) - 1
        and W95_decades < prior_width_decades
        and lower_mass < float(args.boundary_mass_threshold)
        and upper_mass < float(args.boundary_mass_threshold)
        and n_edges >= int(args.min_edges)
        and n_boundary >= int(args.min_boundary_nodes)
    )
    return {
        "tile_id": int(tile_id),
        "n_observed_edges": int(n_edges),
        "n_boundary_nodes": int(n_boundary),
        "n_interior_nodes": int(n_interior),
        "harmonics": "+".join(f"H{int(h)}" for h in harmonics),
        "D_mode": D_mode,
        "D_mean": D_mean,
        "D_median": D_median,
        "D_ci95_low": D_low,
        "D_ci95_high": D_high,
        "W95_decades": W95_decades,
        "posterior_mass_at_lower_boundary": lower_mass,
        "posterior_mass_at_upper_boundary": upper_mass,
        "prior_width_decades": prior_width_decades,
        "prior_to_posterior_width_ratio": width_ratio,
        "identified": bool(identified),
    }


def conditional_boundary_posterior(y: np.ndarray, T: np.ndarray,
                                   q_obs: np.ndarray, tau_b: float,
                                   a: float, s2: float) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean/covariance of boundary pressure at fixed nuisances."""

    diag = build_noise_covariance(q_obs, float(a), float(s2))
    inv_diag = 1.0 / np.maximum(diag, EPS)
    p_inv = 2.0 / max(float(tau_b) ** 2, EPS)
    Lambda = p_inv * np.eye(T.shape[1]) + (T.T * inv_diag[None, :]) @ T
    rhs = T.T @ (inv_diag * y)
    cf = cho_factor(0.5 * (Lambda + Lambda.T), lower=True, check_finite=False)
    b_hat = cho_solve(cf, rhs, check_finite=False)
    cov = cho_solve(cf, np.eye(T.shape[1]), check_finite=False)
    return b_hat, cov


def posterior_predictive_same_tile(
        tile_id: int, harmonic: int, edges: Sequence[tuple],
        y: np.ndarray, T: np.ndarray, q_obs: np.ndarray, tau_b: float,
        a: float, s2: float) -> list[dict]:
    """Same-tile reconstruction using conditional posterior mean b_hat."""

    b_hat, _ = conditional_boundary_posterior(y, T, q_obs, tau_b, a, s2)
    y_pred = T @ b_hat
    n = len(q_obs)
    q_pred = y_pred[:n] + 1j * y_pred[n:]
    sig = np.sqrt(np.maximum(float(a) ** 2 + float(s2) * np.abs(q_obs) ** 2, EPS))
    rows = []
    for i, (edge, obs_q, pred_q, sigma) in enumerate(zip(edges, q_obs, q_pred, sig)):
        resid = obs_q - pred_q
        rows.append({
            "tile_id": int(tile_id),
            "harmonic": int(harmonic),
            "edge_index": int(i),
            "edge_u": str(edge[0]),
            "edge_v": str(edge[1]),
            "q_obs_amp_m3_s": float(abs(obs_q)),
            "q_pred_amp_m3_s": float(abs(pred_q)),
            "q_obs_phase_rad": float(math.atan2(obs_q.imag, obs_q.real)) if abs(obs_q) else float("nan"),
            "q_pred_phase_rad": float(math.atan2(pred_q.imag, pred_q.real)) if abs(pred_q) else float("nan"),
            "resid_real_m3_s": float(resid.real),
            "resid_imag_m3_s": float(resid.imag),
            "resid_abs_m3_s": float(abs(resid)),
            "standardized_resid_abs": float(abs(resid) / max(float(sigma), EPS)),
        })
    return rows


def _logD_prior(logD_grid: np.ndarray, args) -> np.ndarray:
    return _normal_logpdf(
        logD_grid,
        math.log(float(args.logD_prior_median)),
        float(args.logD_prior_tau),
    )


def _make_log_grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(float(lo)), math.log(float(hi)), int(n)))


def _D_grid(args) -> tuple[np.ndarray, np.ndarray]:
    if args.D_count:
        D = np.logspace(math.log10(args.D_min), math.log10(args.D_max),
                        int(args.D_count))
    else:
        decades = math.log10(args.D_max) - math.log10(args.D_min)
        n = int(round(decades / float(args.D_decade_step))) + 1
        D = np.logspace(math.log10(args.D_min), math.log10(args.D_max), n)
    return D, np.log(D)


def _posterior_curve_rows(tile_id: int, D_grid: np.ndarray,
                          logpost: np.ndarray, prob: np.ndarray,
                          loglike: np.ndarray, logprior: np.ndarray) -> list[dict]:
    return [
        {
            "tile_id": int(tile_id),
            "D0": float(D),
            "log_likelihood_marginal": float(ll),
            "log_prior": float(lp),
            "log_posterior": float(post),
            "posterior_prob": float(p),
        }
        for D, ll, lp, post, p in zip(D_grid, loglike, logprior, logpost, prob)
    ]


def _summarize_nuisance_from_logcube(
        tile_id: int, harmonic: int, D_grid: np.ndarray,
        tau_grid: np.ndarray, a_grid: np.ndarray, s2_grid: np.ndarray,
        log_cube: np.ndarray) -> list[dict]:
    """Summarize nuisance posterior marginalized over D and other nuisances."""

    names = [
        ("tau_b", tau_grid, (0, 2, 3)),
        ("a", a_grid, (0, 1, 3)),
        ("s", np.sqrt(s2_grid), (0, 1, 2)),
        ("s2", s2_grid, (0, 1, 2)),
    ]
    rows = []
    z = logsumexp(log_cube)
    if not math.isfinite(float(z)):
        return rows
    for name, grid, axes in names:
        lp = logsumexp(log_cube, axis=axes) - z
        prob = np.exp(lp)
        prob = prob / max(float(np.sum(prob)), EPS)
        mode = float(grid[int(np.argmax(prob))])
        med = _weighted_quantile_sorted(np.asarray(grid, dtype=float), prob, 0.5)
        low = _weighted_quantile_sorted(np.asarray(grid, dtype=float), prob, 0.025)
        high = _weighted_quantile_sorted(np.asarray(grid, dtype=float), prob, 0.975)
        rows.append({
            "tile_id": int(tile_id),
            "harmonic": int(harmonic),
            "parameter": name,
            "mode": mode,
            "median": med,
            "ci95_low": low,
            "ci95_high": high,
        })
    return rows


def _plot_tile(out_dir: Path, tile_id: int, D_grid: np.ndarray,
               logD_grid: np.ndarray, logpost: np.ndarray, prob: np.ndarray,
               loglike: np.ndarray, args) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    prior = np.exp(_logD_prior(logD_grid, args))
    prior = prior / max(np.trapz(prior, logD_grid), EPS)
    post_density = prob / max(float(np.median(np.diff(logD_grid))), EPS)
    ax.plot(D_grid, post_density, label="posterior", color="#2868b7")
    ax.plot(D_grid, prior, label="prior", color="#657487", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("D0 (1/Pa)")
    ax.set_ylabel("density over log(D0)")
    ax.set_title(f"Tile {tile_id} distensibility posterior")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"tile_{int(tile_id):03d}_posterior.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    finite = np.isfinite(loglike)
    delta = np.full_like(loglike, np.nan, dtype=float)
    if finite.any():
        delta[finite] = loglike[finite] - np.nanmax(loglike[finite])
    ax.plot(D_grid, delta, color="#c27a22")
    ax.set_xscale("log")
    ax.set_xlabel("D0 (1/Pa)")
    ax.set_ylabel("log marginal likelihood - max")
    ax.set_title(f"Tile {tile_id} marginal likelihood")
    fig.tight_layout()
    fig.savefig(out_dir / f"tile_{int(tile_id):03d}_log_marginal_likelihood.png",
                dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _best_nuisance_from_cube(D_grid: np.ndarray, tau_grid: np.ndarray,
                             a_grid: np.ndarray, s2_grid: np.ndarray,
                             log_cube: np.ndarray) -> tuple[float, float, float, float]:
    idx = np.unravel_index(int(np.nanargmax(log_cube)), log_cube.shape)
    return (
        float(D_grid[idx[0]]),
        float(tau_grid[idx[1]]),
        float(a_grid[idx[2]]),
        float(s2_grid[idx[3]]),
    )


def run_tile(graph: nx.Graph, tile_id: int, args) -> dict:
    """Run deterministic Bayesian inference for one tile."""

    harmonics = tuple(int(h) for h in args.harmonics)
    D_grid, logD_grid = _D_grid(args)
    obs = build_tile_observations(graph, int(tile_id), harmonics, args)
    boundary_nodes, interior_nodes = choose_tile_boundary_nodes(graph, obs, args)

    valid_edges_any = np.zeros(len(obs.edges), dtype=bool)
    for h in harmonics:
        valid_edges_any |= obs.valid[h]
    n_valid_edges = int(valid_edges_any.sum())
    if n_valid_edges < int(args.min_edges):
        raise ValueError(f"too few valid observed edges ({n_valid_edges})")
    if len(boundary_nodes) < int(args.min_boundary_nodes):
        raise ValueError(f"too few boundary nodes ({len(boundary_nodes)})")

    loglike_total = np.zeros(len(D_grid), dtype=float)
    log_cubes: dict[int, np.ndarray] = {}
    nuisance_rows = []
    geom_rows = []
    jitter = JitterStats()
    predictive_rows = []

    for h in harmonics:
        valid = np.asarray(obs.valid[h], dtype=bool)
        if not valid.any():
            loglike_total += -float("inf")
            continue
        q_h = np.asarray(obs.q_obs[h], dtype=complex)[valid]
        amp_med = float(np.median(np.abs(q_h[np.abs(q_h) > 0]))) if np.any(np.abs(q_h) > 0) else 1e-18
        a_min = float(args.a_min_factor) * amp_med if args.a_min is None else float(args.a_min) / N_L_PER_M3
        a_max = float(args.a_max_factor) * amp_med if args.a_max is None else float(args.a_max) / N_L_PER_M3
        a_min = max(a_min, 1e-30)
        a_max = max(a_max, a_min * 1.001)
        a_grid = _make_log_grid(a_min, a_max, int(args.a_count))
        tau0 = float(args.tau0_h2_pa if int(h) == 2 else args.tau0_h1_pa)
        tau_grid = _make_log_grid(args.tau_min_pa, args.tau_max_pa,
                                  int(args.tau_count))
        s2_grid = _make_log_grid(args.s2_min, args.s2_max,
                                 int(args.s2_count))
        log_cube = np.empty((len(D_grid), len(tau_grid), len(a_grid),
                             len(s2_grid)), dtype=float)

        for i, D in enumerate(tqdm(D_grid, desc=f"tile {tile_id} H{h} D",
                                  leave=False, disable=args.no_progress)):
            T_complex_all, gr = solve_transfer_matrix(
                graph, obs, boundary_nodes, interior_nodes, float(D), int(h),
                args)
            if i == 0:
                geom_rows.extend(gr)
            T_complex = T_complex_all[valid, :]
            y, T_real = complex_to_real_operator(T_complex, q_h)
            integ = integrate_nuisance_for_harmonic(
                y, T_real, q_h, tau_grid, a_grid, s2_grid,
                tau0=tau0, omega_b=float(args.omega_b),
                jitter_stats=jitter,
                likelihood_method=str(args.likelihood_method))
            loglike_total[i] += float(integ["log_marginal"])
            log_cube[i, :, :, :] = integ["log_terms"]

        log_cubes[int(h)] = log_cube
        nuisance_rows.extend(_summarize_nuisance_from_logcube(
            int(tile_id), int(h), D_grid, tau_grid, a_grid, s2_grid, log_cube))

        if args.posterior_predictive:
            D_hat, tau_hat, a_hat, s2_hat = _best_nuisance_from_cube(
                D_grid, tau_grid, a_grid, s2_grid, log_cube)
            T_complex_all, _ = solve_transfer_matrix(
                graph, obs, boundary_nodes, interior_nodes, D_hat, int(h),
                args)
            T_complex = T_complex_all[valid, :]
            y, T_real = complex_to_real_operator(T_complex, q_h)
            pred_edges = [e for e, ok in zip(obs.edges, valid) if bool(ok)]
            predictive_rows.extend(posterior_predictive_same_tile(
                int(tile_id), int(h), pred_edges, y, T_real, q_h, tau_hat,
                a_hat, s2_hat))

    logprior = _logD_prior(logD_grid, args)
    logpost = loglike_total + logprior
    prob = normalize_posterior(logD_grid, logpost)
    summary = summarize_posterior(
        int(tile_id), D_grid, logD_grid, logpost, prob, n_valid_edges,
        len(boundary_nodes), len(interior_nodes), harmonics, args)
    summary.update(jitter.as_dict())

    curve_rows = _posterior_curve_rows(
        int(tile_id), D_grid, logpost, prob, loglike_total, logprior)
    return {
        "tile_id": int(tile_id),
        "curve_rows": curve_rows,
        "summary": summary,
        "nuisance_rows": nuisance_rows,
        "predictive_rows": predictive_rows,
        "unit_rows": obs.unit_rows,
        "geometry_rows": geom_rows,
        "D_grid": D_grid,
        "logD_grid": logD_grid,
        "logpost": logpost,
        "prob": prob,
        "loglike": loglike_total,
    }


def _selected_tiles(graph: nx.Graph, args) -> list[int]:
    if args.tile_id:
        return [int(t) for t in args.tile_id]
    tiles = collect_tile_ids(graph)
    if args.max_tiles:
        tiles = tiles[:int(args.max_tiles)]
    return tiles


def _init_worker(graph: nx.Graph, args) -> None:
    global _WORKER_GRAPH, _WORKER_ARGS
    _WORKER_GRAPH = graph
    _WORKER_ARGS = args


def _run_tile_worker(tile_id: int) -> dict:
    if _WORKER_GRAPH is None or _WORKER_ARGS is None:
        raise RuntimeError("worker was not initialized")
    return run_tile(_WORKER_GRAPH, int(tile_id), _WORKER_ARGS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deterministic Bayesian tile-wise distensibility inference "
                    "with analytic boundary-pressure marginalization.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--graph", default=None)
    ap.add_argument("--out-dir", default="outputs/bayesian_tile_distensibility")
    ap.add_argument("--tile-id", nargs="*", type=int, default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--harmonics", nargs="+", type=int, choices=[1, 2],
                    default=[1])
    ap.add_argument("--D-min", type=float, default=1e-5)
    ap.add_argument("--D-max", type=float, default=1e-1)
    ap.add_argument("--D-count", type=int, default=None)
    ap.add_argument("--D-decade-step", type=float, default=0.1)
    ap.add_argument("--tau-min-pa", type=float, default=0.5)
    ap.add_argument("--tau-max-pa", type=float, default=100.0)
    ap.add_argument("--tau-count", type=int, default=31)
    ap.add_argument("--a-count", type=int, default=21)
    ap.add_argument("--s2-count", type=int, default=21)
    ap.add_argument("--s2-min", type=float, default=1e-4)
    ap.add_argument("--s2-max", type=float, default=10.0 ** 0.4)
    ap.add_argument("--a-min", type=float, default=None,
                    help="Absolute additive noise lower bound in nL/s. "
                         "Default is a-min-factor * median(|q_obs|).")
    ap.add_argument("--a-max", type=float, default=None,
                    help="Absolute additive noise upper bound in nL/s. "
                         "Default is a-max-factor * median(|q_obs|).")
    ap.add_argument("--a-min-factor", type=float, default=1e-3)
    ap.add_argument("--a-max-factor", type=float, default=1.0)
    ap.add_argument("--tau0-h1-pa", type=float, default=7.0)
    ap.add_argument("--tau0-h2-pa", type=float, default=3.0)
    ap.add_argument("--omega-b", type=float, default=math.log(3.0))
    ap.add_argument("--logD-prior-median", type=float, default=1.5e-3)
    ap.add_argument("--logD-prior-tau", type=float, default=math.log(10.0))
    ap.add_argument("--mu", type=float, default=MU_DEFAULT)
    ap.add_argument("--default-f0-hz", type=float, default=F0_HZ_DEFAULT)
    ap.add_argument("--radius-field", default="radius_adapted_m")
    ap.add_argument("--length-field", default="length_true")
    ap.add_argument("--radius-scale", type=float, default=PX_SIZE_M_DEFAULT)
    ap.add_argument("--length-scale", type=float, default=PX_SIZE_M_DEFAULT)
    ap.add_argument("--meter-threshold", type=float, default=1e-3,
                    help="Geometry values above this are treated as pixels "
                         "and multiplied by the relevant scale.")
    ap.add_argument("--boundary-mode",
                    choices=["cut-nodes", "metadata", "all-edge-endpoints"],
                    default="cut-nodes")
    ap.add_argument("--min-boundary-nodes", type=int, default=2)
    ap.add_argument("--min-edges", type=int, default=3)
    ap.add_argument("--boundary-mass-threshold", type=float, default=0.05)
    ap.add_argument("--likelihood-method", choices=["woodbury", "dense"],
                    default="woodbury",
                    help="woodbury is algebraically equivalent to the dense "
                         "covariance likelihood but much faster when the "
                         "number of boundary pressure variables is smaller "
                         "than the number of observed real/imag flows.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of tiles to process concurrently. On Apple "
                         "Silicon, 4-6 is a reasonable starting point.")
    ap.add_argument("--mp-start-method", choices=["fork", "spawn", "forkserver"],
                    default="fork",
                    help="Multiprocessing start method. fork is fastest on "
                         "macOS for this graph-heavy workload; use spawn if "
                         "fork causes issues in your Python environment.")
    ap.add_argument("--no-orient-observations", action="store_true")
    ap.add_argument("--posterior-predictive", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph(args)
    tiles = _selected_tiles(graph, args)
    started = time.time()
    all_curves: list[dict] = []
    summaries: list[dict] = []
    nuisance: list[dict] = []
    predictive: list[dict] = []
    unit_rows: list[dict] = []
    geometry_rows: list[dict] = []
    skipped: list[dict] = []

    def consume_result(result: dict) -> None:
        all_curves.extend(result["curve_rows"])
        summaries.append(result["summary"])
        nuisance.extend(result["nuisance_rows"])
        predictive.extend(result["predictive_rows"])
        unit_rows.extend(result["unit_rows"])
        geometry_rows.extend(result["geometry_rows"])
        if not args.no_plots:
            _plot_tile(out_dir, int(result["tile_id"]), result["D_grid"],
                       result["logD_grid"], result["logpost"],
                       result["prob"], result["loglike"], args)

    if int(args.jobs) <= 1:
        iterator = tqdm(tiles, desc="tiles", disable=args.no_progress)
        for tile_id in iterator:
            try:
                consume_result(run_tile(graph, int(tile_id), args))
            except Exception as exc:
                skipped.append({"tile_id": int(tile_id), "reason": str(exc)})
    else:
        ctx = mp.get_context(str(args.mp_start_method))
        with ProcessPoolExecutor(
                max_workers=int(args.jobs),
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(graph, args)) as pool:
            futures = {pool.submit(_run_tile_worker, int(t)): int(t)
                       for t in tiles}
            iterator = tqdm(as_completed(futures), total=len(futures),
                            desc="tiles", disable=args.no_progress)
            for fut in iterator:
                tile_id = futures[fut]
                try:
                    consume_result(fut.result())
                except Exception as exc:
                    skipped.append({"tile_id": int(tile_id),
                                    "reason": str(exc)})

    _write_csv(out_dir / "tile_D_posterior_curves.csv", all_curves)
    _write_csv(out_dir / "tile_D_posterior_summary.csv", summaries)
    _write_csv(out_dir / "nuisance_posterior_summary.csv", nuisance)
    _write_csv(out_dir / "unit_checks.csv", unit_rows)
    _write_csv(out_dir / "edge_geometry_admittance_checks.csv", geometry_rows)
    _write_csv(out_dir / "skipped_tiles.csv", skipped)
    if predictive:
        _write_csv(out_dir / "posterior_predictive_same_tile.csv", predictive)

    manifest = {
        "script": Path(__file__).name,
        "graph": str(graph_path),
        "out_dir": str(out_dir),
        "tiles_requested": [int(t) for t in tiles],
        "n_tiles_succeeded": len(summaries),
        "n_tiles_skipped": len(skipped),
        "harmonics": [int(h) for h in args.harmonics],
        "elapsed_s": time.time() - started,
        "args": vars(args),
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Also write pandas-friendly copies with stable column order if available.
    if summaries:
        pd.DataFrame(summaries).to_csv(out_dir / "tile_D_posterior_summary_pandas.csv",
                                       index=False)
    print(f"Wrote Bayesian tile distensibility outputs to {out_dir}")
    print(f"Succeeded: {len(summaries)}; skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
