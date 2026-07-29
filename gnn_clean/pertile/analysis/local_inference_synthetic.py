"""Synthetic-data test harness for local pressure/D inference.

Generates ground-truth (D, P_node, Q_edge) by running a global forward
solve on the whole mosaic, then runs ``infer_local`` on a chosen tile
with measurement noise and reports recovery metrics.

Typical use::

    truth = compute_global_truth(G, D_true=1e-3)
    res   = run_synthetic_test(G, tile_id=26, global_truth=truth,
                               D_init=5e-4, noise_rel=0.05, seed=0)
    plot_recovery_summary(res)

Sweep across tiles / noise / seeds::

    df = sweep(G, tiles=[14, 26, 31], noise_levels=[0.02, 0.05, 0.1],
               n_seeds=4, global_truth=truth, D_init=5e-4)

The global forward solve dominates runtime (~1–3 s per harmonic on the
4388-node mosaic).  Compute it once via ``compute_global_truth`` and
pass it to many ``run_synthetic_test`` calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Iterable

import numpy as np
import networkx as nx

from . import local_pressure_inference as lpi


# ──────────────────────────────────────────────────────────────────
# Ground-truth generation: global forward solve
# ──────────────────────────────────────────────────────────────────


@dataclass
class GlobalTruth:
    """Result of a whole-mosaic forward solve.

    Pressures are dicts node→value; flows are dicts (u,v)→value in SI
    units (Pa for pressures, m³/s for flows).
    """
    D_true: float
    mu: float
    f0_hz: float
    px_size_m: float
    harmonics: tuple
    node_P_DC: dict
    node_P_H1: dict
    Q_DC_truth: dict
    Q_H1_truth: dict
    sources: list
    sinks: list


def compute_global_truth(
    graph: nx.Graph,
    D_true: float,
    *,
    mu: float = 2.5e-3,
    f0_hz: float = 2.5,
    px_size_m: float = 1.7e-6,
    harmonics: Sequence[int] = (1,),
    P_DC_source: float = 50.0,
    P_DC_sink: float = -50.0,
    P_H1_source_amp: float = 50.0,
    P_H1_sink_amp: float = 30.0,
    verbose: bool = False,
) -> GlobalTruth:
    """Run the pi-model admittance solve on the entire mosaic.

    Boundary conditions: nodes with ``boundary_type == 'source'`` get
    pressure ``+P_DC_source`` (DC) and an H1 amplitude with a phase that
    rotates around the source set.  Sinks get the negated counterpart.
    The exact BCs don't matter much — they just need to drive nontrivial
    pressure gradients so we have something to recover.
    """
    sources = [n for n, d in graph.nodes(data=True)
               if d.get('boundary_type') == 'source']
    sinks   = [n for n, d in graph.nodes(data=True)
               if d.get('boundary_type') == 'sink']
    g_boundary = sorted(set(sources) | set(sinks))
    g_interior = sorted(set(graph.nodes()) - set(g_boundary))
    if not g_boundary:
        raise ValueError(
            "Graph has no boundary nodes (boundary_type ∈ {source, sink}). "
            "Set boundary_type on at least one node before computing truth.")

    P_DC = {}
    P_H1 = {}
    for i, n in enumerate(sources):
        ang = 2 * np.pi * i / max(len(sources), 1)
        P_DC[n] = P_DC_source + 5.0 * np.cos(ang)
        P_H1[n] = P_H1_source_amp * np.exp(1j * ang)
    for i, n in enumerate(sinks):
        ang = 2 * np.pi * i / max(len(sinks), 1)
        P_DC[n] = P_DC_sink - 5.0 * np.cos(ang)
        P_H1[n] = P_H1_sink_amp * np.exp(1j * (np.pi + ang))

    all_edges = list(graph.edges())
    if verbose:
        print(f"[compute_global_truth] D={D_true:.2e}  "
              f"n_int={len(g_interior)}  n_edges={len(all_edges)}")

    ab = lpi._build_admittance_system(
        graph, all_edges, g_boundary, g_interior,
        D_true, mu, f0_hz, tuple(harmonics), px_size_m)

    P_DC_arr = np.array([P_DC[n] for n in g_boundary], complex)
    L_dc, B_dc, edge_G_dc, _ = ab[0]
    P_int_dc = np.linalg.solve(L_dc, -B_dc @ P_DC_arr).real

    node_P_DC = {n: float(P_DC[n]) for n in g_boundary}
    for i, n in enumerate(g_interior):
        node_P_DC[n] = float(P_int_dc[i])

    node_P_H1 = {n: complex(P_H1[n]) for n in g_boundary}
    for h in harmonics:
        L_h, B_h, _, _ = ab[h]
        P_arr = np.array([P_H1[n] for n in g_boundary], complex)
        P_int_h = np.linalg.solve(L_h, -B_h @ P_arr)
        if h == 1:
            for i, n in enumerate(g_interior):
                node_P_H1[n] = complex(P_int_h[i])

    Q_DC_truth = {}
    Q_H1_truth = {}
    for u, v in all_edges:
        G_e = edge_G_dc.get((u, v), 0.0)
        if G_e <= 0:
            continue
        Q_DC_truth[(u, v)] = G_e * (node_P_DC.get(u, 0.0)
                                     - node_P_DC.get(v, 0.0))
        Q_H1_truth[(u, v)] = G_e * (node_P_H1.get(u, 0+0j)
                                     - node_P_H1.get(v, 0+0j))

    return GlobalTruth(
        D_true=D_true, mu=mu, f0_hz=f0_hz, px_size_m=px_size_m,
        harmonics=tuple(harmonics),
        node_P_DC=node_P_DC, node_P_H1=node_P_H1,
        Q_DC_truth=Q_DC_truth, Q_H1_truth=Q_H1_truth,
        sources=sources, sinks=sinks,
    )


# ──────────────────────────────────────────────────────────────────
# Tile carve-out + truth scaling
# ──────────────────────────────────────────────────────────────────


def _carve_tile(graph, tile_id, inset_frac=0.05):
    """Rectangle-carve the per-tile subgraph into interior/boundary/dropped.

    Tile region is defined by the nodes that touch a tile-``tile_id`` PIV
    edge.  Once that node set is fixed, ``edges_in_tile`` includes ALL
    anatomical edges between those nodes — even those whose PIV record is
    for a different tile or whose PIV failed.  In real-data inference
    you'd be limited to measured edges, but for the synthetic test we
    inject our own measurements, so dropping anatomically-present edges
    just gratuitously fragments the network.

    Returns dict with keys: edges_in_tile, edges_used, nodes_in_tile,
    interior_set, boundary_set, dropped_set, node_xy, bbox.
    """
    # 1. Define the tile region by nodes touching a tile-N PIV edge.
    nodes_in_tile = sorted({n
        for u, v, d in graph.edges(data=True)
        if any(m.get('tile_id') == tile_id
               for m in d.get('measurements_piv', []) or [])
        for n in (u, v)})
    if not nodes_in_tile:
        raise ValueError(f"No edges with tile_id={tile_id} in graph.")

    # 2. Take ALL anatomical edges between those nodes (no PIV filter).
    nodes_set = set(nodes_in_tile)
    edges_in_tile = [(u, v) for u, v in graph.edges()
                     if u in nodes_set and v in nodes_set]
    node_xy = {n: (float(graph.nodes[n].get('x', 0)),
                    float(graph.nodes[n].get('y', 0)))
               for n in nodes_in_tile}

    xs = np.array([node_xy[n][0] for n in nodes_in_tile])
    ys = np.array([node_xy[n][1] for n in nodes_in_tile])
    x_lo = xs.min() + inset_frac * (xs.max() - xs.min())
    x_hi = xs.max() - inset_frac * (xs.max() - xs.min())
    y_lo = ys.min() + inset_frac * (ys.max() - ys.min())
    y_hi = ys.max() - inset_frac * (ys.max() - ys.min())

    inside = {n: (x_lo <= node_xy[n][0] <= x_hi
                   and y_lo <= node_xy[n][1] <= y_hi)
              for n in nodes_in_tile}
    adj = {n: set() for n in nodes_in_tile}
    for u, v in edges_in_tile:
        adj[u].add(v); adj[v].add(u)

    interior_set = {n for n in nodes_in_tile if inside[n]}
    boundary_set = {n for n in nodes_in_tile
                    if not inside[n]
                    and any(inside[m] for m in adj[n])}
    dropped_set = {n for n in nodes_in_tile
                   if not inside[n] and n not in boundary_set}
    edges_used = [(u, v) for u, v in edges_in_tile
                  if u not in dropped_set and v not in dropped_set]

    return {
        'edges_in_tile': edges_in_tile,
        'edges_used': edges_used,
        'nodes_in_tile': nodes_in_tile,
        'interior_set': interior_set,
        'boundary_set': boundary_set,
        'dropped_set': dropped_set,
        'node_xy': node_xy,
        'bbox': (x_lo, x_hi, y_lo, y_hi),
    }


def _real_q_median_si(graph, edges_used, tile_id):
    """Median |Q_DC| from real PIV records for the tile, in SI (m³/s)."""
    from .inference import _meas_phasors_for_edge
    vals = []
    for u, v in edges_used:
        d = graph.edges[u, v]
        m_ref = next((m for m in d.get('measurements_piv', []) or []
                      if m.get('tile_id') == tile_id), None)
        if m_ref is None:
            continue
        try:
            Q_dc, _, _ = _meas_phasors_for_edge((u, v), m_ref, harmonics=(1,))
        except Exception:
            continue
        if np.isfinite(Q_dc):
            vals.append(abs(Q_dc))
    if not vals:
        return None
    return 1e-12 * float(np.median(vals))   # nL/s → m³/s


# ──────────────────────────────────────────────────────────────────
# Result dataclass + main entry point
# ──────────────────────────────────────────────────────────────────


@dataclass
class SyntheticTestResult:
    # Inputs
    tile_id: int
    D_true: float
    D_init: float
    noise_rel: float
    noise_floor_si: float
    seed: int
    scale_factor: float

    # Geometry + truth
    edges_used: list
    boundary_nodes: list
    interior_nodes: list
    node_xy: dict
    bbox: tuple
    P_DC_truth: dict
    P_H1_truth: dict
    Q_DC_clean: np.ndarray         # SI (m³/s)
    Q_H1_clean: np.ndarray         # SI (m³/s)
    Q_DC_noisy: np.ndarray         # SI
    Q_H1_noisy: np.ndarray         # SI

    # Inference output (Q's aligned to edges_used order)
    D_hat: float
    sigma_D: float
    chi2_red: float
    iterations: int
    converged: bool
    cond_DC: float
    cond_H1: float
    pin_node: int
    P_DC_pred: dict                # gauge-pinned → relative to pin's truth
    P_H1_pred: dict
    sigma_P_DC: dict
    sigma_P_H1: dict
    Q_DC_pred: np.ndarray          # SI, aligned to edges_used
    Q_H1_pred: np.ndarray          # SI, aligned to edges_used
    raw_result: object             # the underlying LocalInferenceResult

    # Aggregate metrics
    metrics: dict = field(default_factory=dict)


def _align_q_pred(result, edges_used):
    """Map result.Q_pred (in result.interior_edges order) to edges_used order
    with sign correction for any (v,u) ↔ (u,v) reversal."""
    edge_idx = {(u, v): i for i, (u, v) in enumerate(edges_used)}
    edge_idx.update({(v, u): i for i, (u, v) in enumerate(edges_used)})

    n = len(edges_used)
    Q_DC = np.full(n, np.nan, dtype=float)
    Q_H1 = np.full(n, np.nan, dtype=complex)
    res_qdc = np.asarray(result.Q_pred_DC, dtype=float)
    res_qh1 = np.asarray(result.Q_pred_H1, dtype=complex)
    for j, (u, v) in enumerate(result.interior_edges):
        if (u, v) not in edge_idx:
            continue
        i = edge_idx[(u, v)]
        sgn = +1 if (u, v) == edges_used[i] else -1
        Q_DC[i] = sgn * res_qdc[j]
        Q_H1[i] = sgn * res_qh1[j]
    return Q_DC, Q_H1


def _compute_metrics(res: SyntheticTestResult) -> dict:
    """Compute aggregate recovery metrics + waveform-level diagnostics."""
    m = {}
    Qc, Qp, Qm = res.Q_DC_clean, res.Q_DC_pred, res.Q_DC_noisy
    Hc, Hp, Hm = res.Q_H1_clean, res.Q_H1_pred, res.Q_H1_noisy

    # D recovery
    m['D_true'] = res.D_true
    m['D_hat'] = res.D_hat
    m['sigma_D'] = res.sigma_D
    m['D_rel_err'] = abs(res.D_hat - res.D_true) / max(res.D_true, 1e-30)
    m['D_z'] = (res.D_hat - res.D_true) / max(res.sigma_D, 1e-30)
    m['chi2_red'] = res.chi2_red
    m['iterations'] = res.iterations
    m['converged'] = res.converged
    m['cond_DC'] = res.cond_DC
    m['cond_H1'] = res.cond_H1

    # Pressure errors (gauge-corrected against pin)
    pin = res.pin_node
    P0 = res.P_DC_truth.get(pin, 0.0)
    H0 = res.P_H1_truth.get(pin, 0+0j)
    p_dc_err, p_h1_err = [], []
    for n in res.boundary_nodes:
        if n in res.P_DC_truth and n in res.P_DC_pred:
            p_dc_err.append(res.P_DC_pred[n] - (res.P_DC_truth[n] - P0))
        if n in res.P_H1_truth and n in res.P_H1_pred:
            p_h1_err.append(abs(res.P_H1_pred[n] - (res.P_H1_truth[n] - H0)))
    m['P_DC_max_err'] = float(np.max(np.abs(p_dc_err))) if p_dc_err else np.nan
    m['P_DC_median_err'] = (float(np.median(np.abs(p_dc_err)))
                             if p_dc_err else np.nan)
    m['P_H1_max_err'] = float(np.max(p_h1_err)) if p_h1_err else np.nan

    # Q errors on significant edges (truth amplitude > 5 × noise floor)
    floor = res.noise_floor_si
    sig_dc = (np.abs(Qc) > 5 * floor) & np.isfinite(Qp)
    if sig_dc.any():
        frac = (Qp[sig_dc] - Qc[sig_dc]) / np.abs(Qc[sig_dc])
        m['Q_DC_n_sig'] = int(sig_dc.sum())
        m['Q_DC_frac_err_median'] = float(np.median(np.abs(frac)))
        m['Q_DC_frac_err_p95'] = float(np.percentile(np.abs(frac), 95))
    sig_h1 = (np.abs(Hc) > 0.05 * np.abs(Hc).max()) & np.isfinite(Hp)
    if sig_h1.any():
        frac_h = np.abs(Hp[sig_h1] - Hc[sig_h1]) / np.abs(Hc[sig_h1])
        m['Q_H1_n_sig'] = int(sig_h1.sum())
        m['Q_H1_frac_err_median'] = float(np.median(frac_h))
        m['Q_H1_frac_err_p95'] = float(np.percentile(frac_h, 95))

    # Waveform metrics over the H1 cycle
    N_T = 256
    omega = 2 * np.pi * 1.0   # canonical period; ratio is what matters
    t = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
    n = len(res.edges_used)
    corr = np.full(n, np.nan)
    nrmse = np.full(n, np.nan)
    pp = np.full(n, np.nan)
    PI_tr = np.full(n, np.nan)
    PI_pr = np.full(n, np.nan)
    dphi = np.full(n, np.nan)
    finite = np.isfinite(Qp) & np.isfinite(Hp)
    for i in range(n):
        if not finite[i]:
            continue
        q_t = Qc[i] + 2 * np.real(Hc[i] * np.exp(1j * t))
        q_p = Qp[i] + 2 * np.real(Hp[i] * np.exp(1j * t))
        ptp = float(q_t.max() - q_t.min())
        pp[i] = ptp
        if ptp > 0:
            nrmse[i] = float(np.sqrt(np.mean((q_p - q_t) ** 2)) / ptp)
            if np.std(q_t) > 0 and np.std(q_p) > 0:
                corr[i] = float(np.corrcoef(q_t, q_p)[0, 1])
        if abs(Qc[i]) > 1e-30:
            PI_tr[i] = 2 * abs(Hc[i]) / abs(Qc[i])
        if abs(Qp[i]) > 1e-30:
            PI_pr[i] = 2 * abs(Hp[i]) / abs(Qp[i])
        if abs(Hc[i]) > 0 and abs(Hp[i]) > 0:
            dphi[i] = float(np.angle(np.exp(
                1j * (np.angle(Hp[i]) - np.angle(Hc[i])))))

    sig = np.isfinite(corr) & (pp > 5 * floor)
    if sig.any():
        m['waveform_n_sig'] = int(sig.sum())
        m['waveform_corr_median'] = float(np.nanmedian(corr[sig]))
        m['waveform_corr_p5'] = float(np.nanpercentile(corr[sig], 5))
        m['waveform_nrmse_median'] = float(np.nanmedian(nrmse[sig]))
        m['waveform_nrmse_p95'] = float(np.nanpercentile(nrmse[sig], 95))

    sig_PI = sig & np.isfinite(PI_tr) & np.isfinite(PI_pr) & (PI_tr < 50)
    if sig_PI.any():
        m['PI_log10_err_median'] = float(np.nanmedian(
            np.abs(np.log10(PI_pr[sig_PI] / PI_tr[sig_PI]))))
    sig_ph = sig & np.isfinite(dphi) & (
        np.abs(Hc) > 0.05 * np.nanmax(np.abs(Hc)))
    if sig_ph.any():
        m['phase_residual_median_deg'] = float(np.degrees(
            np.nanmedian(dphi[sig_ph])))
        m['phase_residual_std_deg'] = float(np.degrees(
            np.nanstd(dphi[sig_ph])))

    # Stash arrays so plotting helpers can reuse without recomputing
    m['_arrays'] = {
        'corr': corr, 'nrmse': nrmse, 'pp': pp,
        'PI_tr': PI_tr, 'PI_pr': PI_pr, 'dphi': dphi,
        'sig_waveform': sig,
    }
    return m


def run_synthetic_test(
    graph: nx.Graph,
    tile_id: int,
    *,
    global_truth: Optional[GlobalTruth] = None,
    D_true: Optional[float] = None,
    D_init: float = 5e-4,
    noise_rel: float = 0.05,
    noise_floor_frac: float = 0.05,
    seed: int = 0,
    inset_frac: float = 0.05,
    harmonics: Sequence[int] = (1,),
    mu: float = 2.5e-3,
    f0_hz: float = 2.5,
    px_size_m: float = 1.7e-6,
    max_iter: int = 200,
    tol_rel: float = 1e-6,
    lambda_reg: float = 0.0,
    eps_D: float = 0.10,
    pin_node: Optional[int] = None,
    scale_to_real_q: bool = True,
    synth_tile_id: int = 999,
    verbose: bool = False,
) -> SyntheticTestResult:
    """Run one synthetic recovery test on ``tile_id``.

    Either ``global_truth`` or ``D_true`` must be supplied.  Reusing a
    pre-computed ``GlobalTruth`` avoids the ~1–3 s global solve when
    sweeping noise/seed/tile.
    """
    if global_truth is None:
        if D_true is None:
            raise ValueError("Pass either global_truth or D_true.")
        global_truth = compute_global_truth(
            graph, D_true, mu=mu, f0_hz=f0_hz, px_size_m=px_size_m,
            harmonics=harmonics, verbose=verbose)
    D_true_eff = global_truth.D_true

    rng = np.random.default_rng(seed)
    carve = _carve_tile(graph, tile_id, inset_frac=inset_frac)
    edges_used = carve['edges_used']
    boundary_nodes = sorted(carve['boundary_set'])
    interior_nodes = sorted(carve['interior_set'])
    node_xy = carve['node_xy']

    # Truth values restricted to this tile's boundary + edges
    P_DC_true = {n: global_truth.node_P_DC[n] for n in boundary_nodes}
    P_H1_true = {n: global_truth.node_P_H1[n] for n in boundary_nodes}

    Q_DC_clean = np.zeros(len(edges_used))
    Q_H1_clean = np.zeros(len(edges_used), dtype=complex)
    for i, (u, v) in enumerate(edges_used):
        Q_DC_clean[i] = global_truth.Q_DC_truth.get((u, v), 0.0)
        Q_H1_clean[i] = global_truth.Q_H1_truth.get((u, v), 0+0j)

    # Optionally scale to match real PIV magnitudes for this tile
    scale = 1.0
    if scale_to_real_q:
        sim_med = (float(np.median(np.abs(Q_DC_clean[Q_DC_clean != 0])))
                   if np.any(Q_DC_clean != 0) else 1e-30)
        real_med = _real_q_median_si(graph, edges_used, tile_id)
        if real_med is not None:
            scale = real_med / max(sim_med, 1e-30)
    Q_DC_clean *= scale
    Q_H1_clean *= scale
    P_DC_true = {n: v * scale for n, v in P_DC_true.items()}
    P_H1_true = {n: v * scale for n, v in P_H1_true.items()}

    # Noise + synthetic PIV records
    floor = noise_floor_frac * float(np.median(np.abs(Q_DC_clean)))
    Q_DC_noisy = Q_DC_clean + (
        noise_rel * np.abs(Q_DC_clean) + floor
    ) * rng.standard_normal(len(Q_DC_clean))
    Q_H1_noisy = Q_H1_clean + (
        noise_rel * np.abs(Q_H1_clean) + floor
    ) * (rng.standard_normal(len(Q_H1_clean))
         + 1j * rng.standard_normal(len(Q_H1_clean))) / np.sqrt(2)

    # Build the inference subgraph with synthetic measurements
    G = nx.Graph()
    for n in carve['interior_set'] | carve['boundary_set']:
        nd = graph.nodes[n]
        G.add_node(n, x=float(nd.get('x', 0)), y=float(nd.get('y', 0)),
                   boundary_type=('source' if n in carve['boundary_set']
                                  else None))
    for u, v in edges_used:
        G.add_edge(u, v, **dict(graph.edges[u, v]))
    for i, (u, v) in enumerate(edges_used):
        G.edges[u, v]['flow_from'] = u
        G.edges[u, v]['flow_to'] = v
        G.edges[u, v]['measurements_piv'] = [{
            'tile_id': synth_tile_id,
            'mean_Q': float(Q_DC_noisy[i]),
            'amp_Q': float(abs(Q_H1_noisy[i])),
            'phase': float(np.degrees(np.angle(Q_H1_noisy[i]))),
            'f0_hz': f0_hz,
            'flow_from': u, 'flow_to': v,
            'harmonics': [{
                'k': 1,
                'A': float(Q_H1_noisy[i].real),
                'B': float(-Q_H1_noisy[i].imag),
                'amp': float(abs(Q_H1_noisy[i])),
                'phi': float(np.angle(Q_H1_noisy[i])),
            }],
        }]

    spec = lpi.LocalInferenceSpec(
        D_init=D_init, eps_D=eps_D, lambda_reg=lambda_reg,
        max_iter=max_iter, tol_rel=tol_rel,
        harmonics=tuple(harmonics), use_dc=True,
        mu=mu, f0_hz=f0_hz, pin_node=pin_node, px_size_m=px_size_m,
        verbose=verbose,
    )
    inf = lpi.infer_local(G, synth_tile_id, spec)

    Q_DC_pred, Q_H1_pred = _align_q_pred(inf, edges_used)

    res = SyntheticTestResult(
        tile_id=tile_id,
        D_true=D_true_eff, D_init=D_init,
        noise_rel=noise_rel, noise_floor_si=floor,
        seed=seed, scale_factor=scale,
        edges_used=edges_used,
        boundary_nodes=boundary_nodes,
        interior_nodes=interior_nodes,
        node_xy=node_xy, bbox=carve['bbox'],
        P_DC_truth=P_DC_true, P_H1_truth=P_H1_true,
        Q_DC_clean=Q_DC_clean, Q_H1_clean=Q_H1_clean,
        Q_DC_noisy=Q_DC_noisy, Q_H1_noisy=Q_H1_noisy,
        D_hat=inf.D_hat, sigma_D=inf.sigma_D,
        chi2_red=inf.chi2_red,
        iterations=inf.iterations, converged=inf.converged,
        cond_DC=inf.cond_DC, cond_H1=inf.cond_H1,
        pin_node=inf.pin_node,
        P_DC_pred=dict(inf.P_DC), P_H1_pred=dict(inf.P_H1),
        sigma_P_DC=dict(inf.sigma_P_DC), sigma_P_H1=dict(inf.sigma_P_H1),
        Q_DC_pred=Q_DC_pred, Q_H1_pred=Q_H1_pred,
        raw_result=inf,
    )
    res.metrics = _compute_metrics(res)
    return res


# ──────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────


def plot_recovery_summary(res: SyntheticTestResult, fig=None):
    """Two-panel summary: Q_DC scatter (truth vs pred + noisy) and
    H1 amplitude scatter (truth vs pred + noisy).  Compact, suitable
    for a sweep dashboard."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                                 constrained_layout=True)
    else:
        axes = fig.axes

    Qc, Qp, Qm = res.Q_DC_clean, res.Q_DC_pred, res.Q_DC_noisy
    Hc, Hp, Hm = res.Q_H1_clean, res.Q_H1_pred, res.Q_H1_noisy

    ax = axes[0]
    ax.scatter(Qc * 1e12, Qp * 1e12, s=20, color='#5A4FCF',
               edgecolor='black', lw=0.3, alpha=0.85, label='predicted')
    ax.scatter(Qc * 1e12, Qm * 1e12, s=14, color='#FF7F0E',
               edgecolor='none', alpha=0.4, label='noisy meas')
    finite = np.isfinite(Qp)
    if finite.any():
        ymax = max(np.abs(Qc).max(), np.abs(Qp[finite]).max(),
                   np.abs(Qm).max()) * 1.05 * 1e12
        ax.plot([-ymax, ymax], [-ymax, ymax], 'k--', lw=0.7, alpha=0.5)
        ax.set_xlim(-ymax, ymax); ax.set_ylim(-ymax, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('Q_DC truth [nL/s]'); ax.set_ylabel('Q_DC pred / meas [nL/s]')
    ax.set_title('DC flux recovery')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(np.abs(Hc) * 1e12, np.abs(Hp) * 1e12, s=20,
               color='#5A4FCF', edgecolor='black', lw=0.3, alpha=0.85,
               label='predicted')
    ax.scatter(np.abs(Hc) * 1e12, np.abs(Hm) * 1e12, s=14,
               color='#FF7F0E', edgecolor='none', alpha=0.4,
               label='noisy meas')
    finite = np.isfinite(Hp)
    if finite.any():
        ymax = max(np.abs(Hc).max(), np.abs(Hp[finite]).max(),
                   np.abs(Hm).max()) * 1.05 * 1e12
        ax.plot([0, ymax], [0, ymax], 'k--', lw=0.7, alpha=0.5)
        ax.set_xlim(0, ymax); ax.set_ylim(0, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('|Q_H1| truth [nL/s]'); ax.set_ylabel('|Q_H1| pred / meas [nL/s]')
    ax.set_title('H1 amplitude recovery')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f'tile {res.tile_id}  D_true={res.D_true:.2e}  '
        f'D̂={res.D_hat:.2e} ({100*res.metrics["D_rel_err"]:.1f}% err)  '
        f'noise_rel={res.noise_rel:.2f}  χ²/dof={res.chi2_red:.3f}',
        fontsize=11, fontweight='bold')
    return fig


def plot_waveform_metrics(res: SyntheticTestResult, fig=None):
    """4-panel: waveform corr histogram, NRMSE histogram, PI scatter,
    H1 phase residual histogram."""
    import matplotlib.pyplot as plt
    arr = res.metrics.get('_arrays', {})
    sig = arr.get('sig_waveform')
    if sig is None or not sig.any():
        return None

    if fig is None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2),
                                 constrained_layout=True)
    else:
        axes = fig.axes

    corr = arr['corr']; nrmse = arr['nrmse']
    PI_tr = arr['PI_tr']; PI_pr = arr['PI_pr']; dphi = arr['dphi']

    ax = axes[0]
    ax.hist(corr[sig], bins=np.linspace(-1, 1, 41), color='#5A4FCF',
            edgecolor='black', lw=0.4)
    med = np.nanmedian(corr[sig])
    ax.axvline(med, color='red', lw=1.5, label=f'median = {med:.3f}')
    ax.set_xlabel('Pearson corr(q_truth, q_pred)')
    ax.set_ylabel('# edges')
    ax.set_title('Waveform similarity'); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(np.clip(nrmse[sig], 0, 1.5), bins=np.linspace(0, 1.5, 41),
            color='#1F9E45', edgecolor='black', lw=0.4)
    med = np.nanmedian(nrmse[sig])
    ax.axvline(med, color='red', lw=1.5, label=f'median = {med:.2%}')
    ax.set_xlabel('NRMSE = RMSE / peak-to-peak')
    ax.set_ylabel('# edges')
    ax.set_title('Waveform shape error'); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    sig_PI = sig & np.isfinite(PI_tr) & np.isfinite(PI_pr) & (PI_tr < 50)
    if sig_PI.any():
        lo = max(min(PI_tr[sig_PI].min(), PI_pr[sig_PI].min()), 1e-3)
        hi = max(PI_tr[sig_PI].max(), PI_pr[sig_PI].max()) * 1.1
        ax.scatter(PI_tr[sig_PI], PI_pr[sig_PI], s=20, color='#FF7F0E',
                   edgecolor='black', lw=0.3, alpha=0.85)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
    ax.set_xlabel('PI truth = 2|Q_H1|/|Q_DC|')
    ax.set_ylabel('PI predicted')
    ax.set_title('Pulsatility recovery'); ax.grid(alpha=0.3, which='both')

    ax = axes[3]
    sig_ph = sig & np.isfinite(dphi)
    if sig_ph.any():
        ax.hist(np.degrees(dphi[sig_ph]),
                bins=np.linspace(-180, 180, 37),
                color='#C04040', edgecolor='black', lw=0.4)
        med = np.degrees(np.nanmedian(dphi[sig_ph]))
        ax.axvline(med, color='blue', lw=1.5, label=f'median = {med:+.1f}°')
        ax.legend(fontsize=9)
    ax.set_xlabel('arg(Q_H1_pred) − arg(Q_H1_truth) [°]')
    ax.set_ylabel('# edges')
    ax.set_xlim(-185, 185)
    ax.set_title('H1 phase residual'); ax.grid(alpha=0.3)

    fig.suptitle(
        f'Waveform metrics  —  tile {res.tile_id}  '
        f'noise_rel={res.noise_rel:.2f}  seed={res.seed}',
        fontsize=12, fontweight='bold')
    return fig


def plot_waveforms(res: SyntheticTestResult, n_top: int = 6, fig=None):
    """Top-n edges by |Q_DC| — overlay truth, predicted, noisy waveforms."""
    import matplotlib.pyplot as plt
    Qc = res.Q_DC_clean
    top_idx = np.argsort(np.abs(Qc))[::-1][:n_top]
    f0 = 1.0   # we plot in units of period; absolute scale unimportant
    t = np.linspace(0, 1.0, 200)
    omega = 2 * np.pi * f0
    cols = min(3, n_top)
    rows = int(np.ceil(n_top / cols))
    if fig is None:
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows),
                                 constrained_layout=True)
    else:
        axes = np.array(fig.axes).reshape(rows, cols)
    axes = np.atleast_2d(axes).flatten()
    Qp = res.Q_DC_pred; Qm = res.Q_DC_noisy
    Hc = res.Q_H1_clean; Hp = res.Q_H1_pred; Hm = res.Q_H1_noisy
    for ax, ei in zip(axes, top_idx):
        u, v = res.edges_used[ei]
        if not (np.isfinite(Qp[ei]) and np.isfinite(Hp[ei])):
            ax.text(0.5, 0.5, 'no prediction', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            continue
        q_t = Qc[ei] + 2 * np.real(Hc[ei] * np.exp(1j * omega * t))
        q_p = Qp[ei] + 2 * np.real(Hp[ei] * np.exp(1j * omega * t))
        q_m = Qm[ei] + 2 * np.real(Hm[ei] * np.exp(1j * omega * t))
        ax.plot(t, q_t * 1e12, color='black', lw=1.8, label='truth', zorder=3)
        ax.plot(t, q_p * 1e12, color='#5A4FCF', lw=1.4, ls='--',
                label='inferred', zorder=2)
        ax.plot(t, q_m * 1e12, color='#FF7F0E', lw=0.8, alpha=0.5,
                label='noisy', zorder=1)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax.set_xlabel('t / T')
        ax.set_ylabel('Q [nL/s]')
        ax.set_title(f'({u},{v})  |Q̄|={abs(Qc[ei])*1e12:.3f} nL/s',
                     fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f'Top-{n_top} edge waveforms — tile {res.tile_id}',
                 fontweight='bold', fontsize=12)
    return fig


# ──────────────────────────────────────────────────────────────────
# Sweep helper
# ──────────────────────────────────────────────────────────────────


def sweep(
    graph: nx.Graph,
    *,
    tiles: Iterable[int],
    noise_levels: Iterable[float],
    n_seeds: int,
    global_truth: Optional[GlobalTruth] = None,
    D_true: Optional[float] = None,
    seed_offset: int = 0,
    keep_results: bool = False,
    verbose_progress: bool = True,
    **fixed_kwargs,
):
    """Sweep over (tile × noise × seed) and return (DataFrame, [results]).

    The DataFrame holds one row per run with all aggregate metrics.  If
    ``keep_results=True`` the second return value is the list of
    ``SyntheticTestResult`` objects (memory: ~MB per result), else None.

    Anything in ``fixed_kwargs`` is forwarded to ``run_synthetic_test``.
    """
    import pandas as pd

    if global_truth is None:
        if D_true is None:
            raise ValueError("Pass either global_truth or D_true.")
        global_truth = compute_global_truth(
            graph, D_true,
            mu=fixed_kwargs.get('mu', 2.5e-3),
            f0_hz=fixed_kwargs.get('f0_hz', 2.5),
            px_size_m=fixed_kwargs.get('px_size_m', 1.7e-6),
            harmonics=fixed_kwargs.get('harmonics', (1,)),
            verbose=verbose_progress)

    rows = []
    results = [] if keep_results else None
    tiles = list(tiles); noise_levels = list(noise_levels)
    total = len(tiles) * len(noise_levels) * n_seeds
    done = 0
    for tile_id in tiles:
        for nr in noise_levels:
            for s in range(n_seeds):
                done += 1
                if verbose_progress:
                    print(f"  [{done}/{total}] tile={tile_id} "
                          f"noise={nr:.3f} seed={s}", flush=True)
                try:
                    res = run_synthetic_test(
                        graph, tile_id, global_truth=global_truth,
                        noise_rel=nr, seed=s + seed_offset,
                        verbose=False, **fixed_kwargs)
                except Exception as e:
                    rows.append({
                        'tile_id': tile_id, 'noise_rel': nr,
                        'seed': s + seed_offset,
                        'error': str(e)[:200],
                    })
                    continue
                row = {'tile_id': tile_id, 'noise_rel': nr,
                        'seed': s + seed_offset,
                        'n_boundary': len(res.boundary_nodes),
                        'n_interior': len(res.interior_nodes),
                        'n_edges': len(res.edges_used)}
                row.update({k: v for k, v in res.metrics.items()
                             if not k.startswith('_')})
                rows.append(row)
                if keep_results:
                    results.append(res)

    df = pd.DataFrame(rows)
    return df, results


def plot_sweep(df, x: str = 'noise_rel', y: str = 'D_rel_err',
                hue: str = 'tile_id', logy: bool = True, ax=None):
    """Quick scatter / line plot of one sweep metric vs another.

    Default plots D recovery error vs noise level, one line per tile.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if hue is None or hue not in df.columns:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(hue))
    for key, gdf in groups:
        agg = gdf.groupby(x)[y].agg(['median', 'min', 'max']).reset_index()
        label = f'{hue}={key}' if key is not None else None
        ax.plot(agg[x], agg['median'], 'o-', label=label, lw=1.5)
        ax.fill_between(agg[x], agg['min'], agg['max'], alpha=0.15)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(x); ax.set_ylabel(y)
    if hue:
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax
