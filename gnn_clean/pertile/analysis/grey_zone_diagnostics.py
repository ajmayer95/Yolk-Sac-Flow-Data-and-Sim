"""Diagnostic helpers for the grey-zone Kirchhoff-fractions optimizer.

Designed to be called from a Jupyter notebook so we can inspect the
sign-convention agreement between basis predictions and tile
measurements without going through the viewer's button cycle.

Three top-level entry points:

- `setup_greyzone_inputs(G, target_tile_id, n_harmonics)` — mirrors the
  viewer's `_run_greyzone_optimizer` setup and returns everything
  `optimize_greyzone_kirchhoff_fractions` needs to run.

- `sign_agreement_diagnostic(G, target_tile_id, ...)` — for each red
  node and each external edge, compares the sign of the unit-injection
  basis (B_r) against the sign of the target-tile measurement.  Reports
  per-red and aggregate agreement rates.

- `run_estimator_at_tile(G, target_tile_id, estimator, ...)` — wrapper
  that runs the regression at a given tile and returns the result dict
  (so the notebook can compare ζ across tiles in a single cell).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx


# -----------------------------------------------------------------------
# Setup: re-derive everything optimize_greyzone_kirchhoff_fractions needs
# -----------------------------------------------------------------------


def setup_greyzone_inputs(G: nx.Graph,
                            target_tile_id: Optional[int] = None,
                            n_harmonics: int = 3,
                            ) -> dict:
    """Mirror the viewer's grey-zone setup.  Reads exclusion state from
    `G.graph` (persisted by the viewer's exclusion editor).  Returns a
    dict with all kwargs needed for
    `optimize_greyzone_kirchhoff_fractions`."""
    from .transmission_line import (
        _extract_boundary_harmonics, _classify_boundary_nodes)
    from .config import FRAME_DT_S
    from .harmonic import fit_harmonics as _fh

    excluded = set(G.graph.get('exclusion_nodes', []))
    boundary = set(G.graph.get('exclusion_boundary', []))
    blue_sources = set(G.graph.get('exclusion_blue_sources', []))
    blue_sinks = set(G.graph.get('exclusion_blue_sinks', []))
    if not excluded or not boundary:
        raise ValueError(
            "Graph has no exclusion zone metadata. Run viewer's "
            "exclusion editor + classify-boundary first, save graph.")
    if not blue_sources and not blue_sinks:
        raise ValueError(
            "Blue boundary nodes not classified into sources/sinks.")

    red_nodes = sorted(blue_sources)
    blue_sink_nodes = sorted(blue_sinks)

    # Original arterial / venous boundary nodes (outside grey zone)
    all_bn = [n for n, d in G.nodes(data=True)
              if d.get('boundary_type') is not None]
    src_all, snk_all = _classify_boundary_nodes(G, all_bn)
    arterial_nodes = [n for n in src_all if n not in excluded]
    venous_nodes = [n for n in snk_all if n not in excluded]
    if not venous_nodes:
        raise ValueError("No venous boundary outside exclusion zone")
    v_ref = venous_nodes[-1]
    venous_no_ref = [v for v in venous_nodes if v != v_ref]

    # Frequencies
    tile_f0s = G.graph.get('tile_f0s', {}) or {}
    if target_tile_id is not None and target_tile_id in tile_f0s:
        f0 = float(tile_f0s[target_tile_id])
    elif tile_f0s:
        f0 = float(np.median(list(tile_f0s.values())))
    else:
        f0 = 2.5
    ref_vid = G.graph.get('reference_vid', 14)
    f0_bc_extract = float(tile_f0s.get(int(ref_vid), f0))

    # BC harmonics from the reference tile
    bc_all = _extract_boundary_harmonics(
        G, all_bn, f0_bc_extract, n_harmonics, FRAME_DT_S)
    # Sign-align: arterial DC > 0, venous DC < 0
    for bn in all_bn:
        if bn in snk_all and bc_all[bn][0].real > 0:
            bc_all[bn] = -bc_all[bn]
        elif bn in src_all and bc_all[bn][0].real < 0:
            bc_all[bn] = -bc_all[bn]

    # Constant-CO frequency correction
    if abs(f0_bc_extract - f0) > 1e-6 and f0_bc_extract > 0 and f0 > 0:
        ratio = f0_bc_extract / f0
        if abs(ratio - 1.0) > 1e-4:
            for bn in list(bc_all.keys()):
                c = bc_all[bn].copy()
                c[1:] *= ratio
                bc_all[bn] = c

    bc_arterial = {n: bc_all[n] for n in arterial_nodes if n in bc_all}
    bc_venous = {n: bc_all[n] for n in venous_no_ref if n in bc_all}

    # Blue-sink BCs (incoming flow)
    bc_blue_sinks = {}
    for bn in blue_sink_nodes:
        bc_h = np.zeros(n_harmonics + 1, dtype=complex)
        for nb in G.neighbors(bn):
            if nb in excluded:
                continue
            d = G.edges[bn, nb]
            piv_list = d.get('measurements_piv', [])
            if not piv_list:
                continue
            best = max(piv_list,
                       key=lambda m: m.get('snr_pulse', -np.inf))
            Qt = best.get('Q_t')
            f0_m = best.get('f0_hz', f0)
            if Qt is None or len(Qt) <= 20:
                continue
            Qt_arr = np.asarray(Qt, dtype=float)
            if np.nanmean(Qt_arr) < 0:
                Qt_arr = -Qt_arr
            try:
                hr = _fh(Qt_arr, f0_m, FRAME_DT_S, K=n_harmonics,
                         loss='huber', include_dc=True)
            except Exception:
                continue
            h = np.zeros(n_harmonics + 1, dtype=complex)
            h[0] = hr.get('a0', np.nanmean(Qt_arr))
            for hh in hr.get('harmonics', []):
                k = hh['k']
                if k <= n_harmonics:
                    h[k] = hh['A'] - 1j * hh['B']
            if f0_m > 0 and f0 > 0 and abs(f0_m - f0) > 1e-6:
                h[1:] *= (f0_m / f0)
            bc_h += h
        bc_blue_sinks[bn] = bc_h

    # Build blue → red map by BFS through the grey zone
    blue_red_map = {}
    excluded_set = set(excluded)
    red_set = set(red_nodes)
    for bn in blue_sink_nodes:
        reachable = []
        visited = set()
        queue = [nb for nb in G.neighbors(bn) if nb in excluded_set]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            for nb2 in G.neighbors(n):
                if nb2 in red_set and nb2 not in reachable:
                    reachable.append(nb2)
                elif nb2 in excluded_set and nb2 not in visited:
                    queue.append(nb2)
        if reachable:
            blue_red_map[bn] = sorted(reachable)
    # Assign unconnected reds to nearest blue
    assigned = set().union(*blue_red_map.values()) if blue_red_map else set()
    for rn in red_nodes:
        if rn in assigned:
            continue
        rx, ry = G.nodes[rn].get('x', 0), G.nodes[rn].get('y', 0)
        best_bn = min(
            blue_red_map.keys(),
            key=lambda bn: np.hypot(G.nodes[bn].get('x', 0) - rx,
                                     G.nodes[bn].get('y', 0) - ry))
        blue_red_map[best_bn].append(rn)

    return {
        'G': G,
        'excluded': excluded,
        'blue_red_map': blue_red_map,
        'bc_art': bc_arterial,
        'bc_ven': bc_venous,
        'bc_blue_sinks': bc_blue_sinks,
        'arterial_nodes': arterial_nodes,
        'venous_nodes': venous_no_ref,
        'red_nodes': red_nodes,
        'blue_sink_nodes': blue_sink_nodes,
        'v_ref': v_ref,
        'f0_hz': f0,
        'f0_bc_extract': f0_bc_extract,
        'n_harmonics': n_harmonics,
    }


# -----------------------------------------------------------------------
# Sign-agreement diagnostic
# -----------------------------------------------------------------------


def sign_agreement_diagnostic(G: nx.Graph,
                                target_tile_id: int,
                                D: float = 3.16e-3,
                                mu: float = 3.5e-3,
                                n_harmonics: int = 3,
                                verbose: bool = True,
                                ) -> dict:
    """For each red node, inject a +1 unit at red r alone (with all
    others zeroed and venous Dirichlet-pinned), forward-solve at DC,
    and compare the sign of the predicted edge flow against the sign
    of the target-tile measurement on each external edge.

    Returns a dict with per-edge sign comparison + aggregate stats.
    Used to detect cross-tile sign-convention drift.
    """
    from .transmission_line import (
        _assemble_laplacian, _vessel_admittance, _get_edge_geometry,
        RHO_BLOOD, _classify_boundary_nodes)
    from scipy.sparse.linalg import spsolve

    setup = setup_greyzone_inputs(G, target_tile_id, n_harmonics)
    excluded = setup['excluded']
    blue_red_map = setup['blue_red_map']
    red_nodes = setup['red_nodes']
    bc_ven = setup['bc_ven']
    blue_sinks = setup['blue_sink_nodes']
    v_ref = setup['v_ref']

    # G_mod = full G − grey zone − blue sinks (matches the optimizer)
    G_mod = G.copy()
    G_mod.remove_nodes_from(set(excluded) | set(blue_sinks))
    node_to_idx = {n: i for i, n in enumerate(G_mod.nodes())}
    base_edges = [(u, v) for u, v in G_mod.edges()
                  if _get_edge_geometry(G_mod, u, v)[0] is not None]

    # Collect target-tile measurements per edge.
    # PIV stores Q_t signed in the `flow_from → flow_to` direction.  If
    # those don't match the graph's `(u, v)` storage order, the
    # `mean(Q_t)` value points the opposite direction from what the
    # basis is computing.  Track sign-flip rate so we can see if THIS
    # is the bug.
    meas = {}
    flow_orient_log = {'aligned': 0, 'reversed': 0, 'unknown': 0}
    for u, v, d in G_mod.edges(data=True):
        for m in d.get('measurements_piv', []):
            if m.get('tile_id') != target_tile_id:
                continue
            Qt = m.get('Q_t')
            if Qt is None or len(Qt) < 20:
                continue
            Qt_arr = np.asarray(Qt, dtype=float)
            Qt_fin = Qt_arr[np.isfinite(Qt_arr)]
            if Qt_fin.size < 20:
                continue
            ff = m.get('flow_from')
            ft = m.get('flow_to')
            if ff is not None and ft is not None:
                if ff == u and ft == v:
                    sign = 1.0
                    flow_orient_log['aligned'] += 1
                elif ff == v and ft == u:
                    sign = -1.0    # PIV flow direction is v→u
                    flow_orient_log['reversed'] += 1
                else:
                    # flow_from/to don't match either endpoint —
                    # measurement orientation is uncertain
                    sign = 1.0
                    flow_orient_log['unknown'] += 1
            else:
                sign = 1.0
                flow_orient_log['unknown'] += 1
            meas[(u, v)] = sign * float(np.nanmean(Qt_fin))
            break
    if verbose:
        total_orient = sum(flow_orient_log.values())
        if total_orient > 0:
            print(f"  Edge orientation vs PIV flow_from/to: "
                  f"aligned={flow_orient_log['aligned']} "
                  f"({100*flow_orient_log['aligned']/total_orient:.1f}%), "
                  f"reversed={flow_orient_log['reversed']} "
                  f"({100*flow_orient_log['reversed']/total_orient:.1f}%), "
                  f"unknown={flow_orient_log['unknown']}")

    # Per-red unit-injection DC solve.
    # Mirror the optimizer's `_solve_with_red_bcs`: pin venous to P=0 and
    # add Tikhonov λ·I on non-pinned rows so the otherwise-singular
    # resistive Laplacian stays well-posed.  Without this, spsolve
    # reports MatrixRankWarning + zgstrf info 4283 and returns garbage.
    from scipy.sparse import eye as _sp_eye
    omega = 0.0
    L = _assemble_laplacian(G_mod, omega, base_edges, node_to_idx,
                             mu, RHO_BLOOD, D)
    L = L.tolil()
    N_mod = len(node_to_idx)
    pin_idxs = set()
    for vn in bc_ven.keys():
        if vn in node_to_idx:
            i = node_to_idx[vn]
            L[i, :] = 0
            L[i, i] = 1.0
            pin_idxs.add(i)
    # Tikhonov on non-pinned rows, scaled to median interior diagonal
    # (so lam·P stays small relative to physical Y·P at every red).
    interior_diag = np.abs(np.asarray(L.diagonal()))
    interior_finite = interior_diag[np.isfinite(interior_diag)
                                     & (interior_diag > 0)]
    y_scale = (float(np.median(interior_finite))
               if interior_finite.size else 1.0)
    lam = 1e-6 * y_scale
    L_csr = L.tocsr()
    if lam > 0 and len(pin_idxs) < N_mod:
        diag_mask = np.ones(N_mod)
        for i in pin_idxs:
            diag_mask[i] = 0.0
        L_csr = L_csr + _sp_eye(N_mod, format='csr').multiply(
            diag_mask * lam)

    per_red_basis = {}
    failed = []
    for r in red_nodes:
        if r not in node_to_idx:
            continue
        Q_rhs = np.zeros(N_mod, dtype=complex)
        Q_rhs[node_to_idx[r]] = 1.0 + 0j
        try:
            P = spsolve(L_csr, Q_rhs)
        except Exception:
            failed.append(r)
            continue
        if not np.all(np.isfinite(P)):
            failed.append(r)
            continue
        # Edge flow: Y_self·P_u - Y_trans·P_v at DC ⇒ G·(P_u − P_v)
        edge_flow = {}
        for (u, v) in base_edges:
            R_m, L_m = _get_edge_geometry(G_mod, u, v)
            if R_m is None:
                continue
            Yd, Yo = _vessel_admittance(R_m, L_m, omega, mu, RHO_BLOOD, D)
            edge_flow[(u, v)] = float(np.real(
                Yd * P[node_to_idx[u]] + Yo * P[node_to_idx[v]]))
        per_red_basis[r] = edge_flow
    if failed and verbose:
        print(f"  ⚠️  Per-red basis solve failed for "
              f"{len(failed)} reds: {failed[:6]}")

    # Sign-agreement aggregation
    overall_agree = 0
    overall_count = 0
    per_red_stats = {}
    edge_records = []
    for r, flows in per_red_basis.items():
        agree = disagree = 0
        for (u, v), q_basis in flows.items():
            if abs(q_basis) < 1e-12:
                continue
            edge_key = (u, v) if (u, v) in meas else (
                (v, u) if (v, u) in meas else None)
            if edge_key is None:
                continue
            sign_factor = 1.0 if edge_key == (u, v) else -1.0
            q_meas = sign_factor * meas[edge_key]
            if abs(q_meas) < 1e-12:
                continue
            if np.sign(q_basis) == np.sign(q_meas):
                agree += 1
            else:
                disagree += 1
            edge_records.append({
                'red': r,
                'edge': (u, v),
                'q_basis': q_basis,
                'q_meas': q_meas,
                'sign_agree': np.sign(q_basis) == np.sign(q_meas),
            })
        total = agree + disagree
        per_red_stats[r] = {
            'agree': agree,
            'disagree': disagree,
            'total': total,
            'agree_frac': agree / max(total, 1),
        }
        overall_agree += agree
        overall_count += total

    overall_frac = overall_agree / max(overall_count, 1)

    if verbose:
        print(f"Sign agreement at tile {target_tile_id} "
              f"(D={D:.2e}, μ={mu*1e3:.2f} mPa·s):")
        print(f"  Overall: {overall_agree}/{overall_count} "
              f"({100*overall_frac:.1f}%) edges agree on DC sign")
        print(f"  Per-red breakdown (sorted by disagree count):")
        rows = sorted(per_red_stats.items(),
                       key=lambda kv: -kv[1]['disagree'])
        for r, s in rows[:15]:
            print(f"    red {r:>6}: agree={s['agree']:>3}, "
                  f"disagree={s['disagree']:>3}, "
                  f"frac={100*s['agree_frac']:.0f}%")

    return {
        'overall_agree': overall_agree,
        'overall_count': overall_count,
        'overall_frac': overall_frac,
        'per_red': per_red_stats,
        'edge_records': edge_records,
        'meas': meas,
        'per_red_basis': per_red_basis,
    }


# -----------------------------------------------------------------------
# Estimator-at-tile wrapper
# -----------------------------------------------------------------------


def run_estimator_at_tile(G: nx.Graph,
                            target_tile_id: int,
                            estimator: str = 'zeta_phase_only',
                            D: float = 3.16e-3,
                            mu: float = 3.5e-3,
                            n_harmonics: int = 3,
                            n_harmonics_loss: int = 2,
                            fit_tile_tau: bool = True,
                            zeta_prior_strength: float = 0.01,
                            verbose: bool = False,
                            ) -> dict:
    """One-shot wrapper that runs `optimize_greyzone_kirchhoff_fractions`
    at a given tile with the new estimator/τ machinery.  Useful for
    side-by-side per-tile comparisons in the notebook."""
    from .transmission_line import optimize_greyzone_kirchhoff_fractions
    setup = setup_greyzone_inputs(G, target_tile_id, n_harmonics)
    return optimize_greyzone_kirchhoff_fractions(
        G,
        excluded_nodes=setup['excluded'],
        blue_red_map=setup['blue_red_map'],
        art_nodes=list(setup['bc_art'].keys()),
        ven_nodes=list(setup['bc_ven'].keys()),
        Q_art=setup['bc_art'],
        Q_ven=setup['bc_ven'],
        n_harmonics=n_harmonics,
        f0_hz=setup['f0_hz'],
        mu=mu,
        D_init=D,
        tile_id=target_tile_id,
        per_sheet_scale=False,
        fit_alpha=True,
        n_harmonics_loss=n_harmonics_loss,
        per_red_complex=(estimator != 'alpha'),
        estimator=estimator,
        zeta_prior_strength=zeta_prior_strength,
        fit_tile_tau=fit_tile_tau,
        verbose=verbose,
    )
