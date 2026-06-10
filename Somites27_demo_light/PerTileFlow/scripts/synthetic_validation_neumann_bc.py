"""Synthetic validation sweep with EMPIRICAL boundary flows (Neumann BC).

Replaces the random-Gaussian-P_b setup of the production scan, and the
prescribed-P-amp Dirichlet setup of `synthetic_validation_global_forward.py`,
with the methodology Pilar/Alex actually want:

  1. Read the 4 labeled boundary vessels' real-PIV Q (DC + H1 phasor)
     from the graph.
  2. Enforce mass conservation (per channel) by rescaling sources and
     sinks toward the mean side-sum.
  3. Scale up by 4× to reproduce flow magnitudes.
  4. For each D_true, run a lumped-π forward solve on the network *minus*
     the 4 boundary vessels, with the 4 vessel flows imposed as Neumann
     boundary conditions (current injection at each vessel's interior
     endpoint).
  5. Extract truth Q on every tile edge from that solve.
  6. For each (tile, σ_Q, rep): perturb truth Q by Gaussian noise σ_Q,
     run inversion via `_synthetic_refit`, compare D̂ to D_true.

Output: renders/paper_figures/synthetic_validation_neumann_bc.csv
"""
from __future__ import annotations
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
from pertile.analysis import local_pressure_inference as lpi
from pertile.analysis.local_pressure_inference import (
    _edge_geometry, extract_tile_subgraph_spatial,
    _build_admittance_system, _compute_transfer_matrices,
)

GRAPH_PATH = ("~/Library/CloudStorage/"
              "<your-drive>/My Drive/"
              "Somites27/Mosaic/Graphs/mosaic_graph_canonical.gpickle")

OUT_DIR = PROJECT_ROOT / "renders" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "synthetic_validation_neumann_bc.csv"

TILES = [22, 26, 38]
D_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SIGMA_Q_NLS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05,
                0.08, 0.12, 0.2, 0.3, 0.5]
N_REPS = 8
HARMONICS = (1,)
MU = 3.5e-3
F0_HZ = 2.773
PX_SIZE_M = 1.7e-6
nL_per_m3 = 1.0e12

# Per-channel scale-up of empirical boundary Q values.  Sweep / pilot
# results from earlier showed real bulk plexus flow magnitudes are
# under-represented in the PIV measurement at the labeled boundary
# vessels by a factor of ~4; scaling up brings synth |Q|_med on tiles
# 22, 38 closer to what real-PIV records on those tiles.
BOUNDARY_Q_SCALE = 4.0


def read_boundary_vessel_flows(graph):
    """Return list of (interior_node, sign, Q_DC_si, Q_H1_si) per boundary
    vessel.  `sign` = +1 if vessel injects into network (source-side
    interior endpoint), -1 if it drains (sink-side interior endpoint).
    Q in SI m³/s."""
    out = []
    for n, d in graph.nodes(data=True):
        bt = d.get("boundary_type")
        if bt not in ("source", "sink"):
            continue
        nbrs = list(graph.neighbors(n))
        if len(nbrs) != 1:
            print(f"  WARNING: boundary node {n} has degree "
                  f"{len(nbrs)}, expected 1; skipping")
            continue
        nb = nbrs[0]
        ed = graph.edges[n, nb]
        mean_Q = (ed.get("Q_DC") or ed.get("mean_Q_piv")
                  or ed.get("mean_Q") or ed.get("mean_Q_nL_s"))
        amp_Q = ed.get("Q_H1_amp")
        phase = ed.get("Q_H1_phi")
        if amp_Q is None or phase is None:
            amp_Q = ed.get("amp_Q_h1_piv")
            phase = ed.get("phase_h1_piv")
        if amp_Q is None or phase is None:
            amp_Q = ed.get("amp_Q")
            phase = ed.get("phase")
        if mean_Q is None or amp_Q is None or phase is None:
            print(f"  WARNING: edge {n}↔{nb} missing Q metadata; skipping")
            continue
        # PIV Q stored in nL/s in flow_from→flow_to convention.
        # We need Q at the *interior endpoint* in the convention
        # "+ means injecting into the network".
        ff = ed.get("flow_from"); ft = ed.get("flow_to")
        # If boundary node is flow_from: Q is from boundary→interior =
        # injection at interior.  Sign = +mean_Q (positive flow flowing
        # into the network).
        # If boundary node is flow_to: Q is from interior→boundary =
        # withdrawal from interior.  Sign-flip needed: Q in interior
        # endpoint's "+ inject" convention = -mean_Q.
        if ff == n:
            sign_inject = +1.0   # boundary→interior, positive Q means inflow
        elif ft == n:
            sign_inject = -1.0   # interior→boundary, positive Q means outflow
        else:
            print(f"  WARNING: edge {n}↔{nb} flow_from/flow_to don't include boundary node")
            sign_inject = +1.0 if bt == "source" else -1.0

        Q_DC_nls = float(mean_Q) * sign_inject
        Q_H1_nls = float(amp_Q) * np.exp(1j * float(phase)) * sign_inject
        Q_DC_si = Q_DC_nls / nL_per_m3
        Q_H1_si = Q_H1_nls / nL_per_m3
        out.append((nb, bt, n, Q_DC_si, Q_H1_si))
        print(f"  vessel  {n}({bt})↔{nb}:  "
              f"Q_DC = {Q_DC_nls:+.3f} nL/s,  "
              f"|Q_H1| = {abs(Q_H1_nls):.3f} nL/s,  "
              f"φ_H1 = {np.angle(Q_H1_nls):+.3f} rad")
    return out


def enforce_mass_balance_and_scale(boundary_flows, scale=BOUNDARY_Q_SCALE):
    """Rescale each side to (Σ_in + Σ_out)/2 for both DC and H1,
    then multiply everything by `scale`.  Returns updated list with
    same shape.  Sign convention: +Q at interior endpoint means
    injection."""
    Q_DC = np.array([f[3] for f in boundary_flows], complex)
    Q_H1 = np.array([f[4] for f in boundary_flows], complex)
    sum_DC = Q_DC.sum().real
    sum_H1 = Q_H1.sum()
    print(f"\n  Pre-balance:  Σ Q_DC = {sum_DC*nL_per_m3:+.3f} nL/s  "
          f"|Σ Q_H1| = {abs(sum_H1)*nL_per_m3:.3f} nL/s")

    # DC: enforce strict conservation (additive correction).  Necessary
    # because the resistive solve (no shunt at ω=0) requires Σ i = 0.
    # H1: leave as-is.  Physically the cardiac pulse drives all 4
    # boundary vessels into phase-aligned pulsatile inflow during
    # systole; the network's compliance (AC shunts in the lumped pi
    # model) absorbs this naturally.  Forcing Σ Q_H1 = 0 kills the
    # physiological volumetric-breathing mode.
    n_v = len(Q_DC)
    Q_DC_new = Q_DC - sum_DC / n_v
    Q_H1_new = Q_H1.copy()

    # Verify DC conservation (H1 should still be ≠ 0)
    assert abs(Q_DC_new.sum().real) < 1e-12, "DC mass-balance failed"
    h1_imbal = Q_H1_new.sum()
    print(f"  H1 imbalance left in place: |Σ Q_H1| = "
          f"{abs(h1_imbal)*nL_per_m3:.3f} nL/s  "
          f"(absorbed by AC shunts)")

    # Apply scale-up
    Q_DC_new = Q_DC_new * scale
    Q_H1_new = Q_H1_new * scale

    print(f"  Post-balance + {scale}× scale:")
    out = []
    for (nb, bt, bn, _q_dc, _q_h1), q_dc_new, q_h1_new in zip(
            boundary_flows, Q_DC_new, Q_H1_new):
        out.append((nb, bt, bn, complex(q_dc_new), complex(q_h1_new)))
        print(f"    {bn}({bt})→{nb}: "
              f"Q_DC = {q_dc_new.real*nL_per_m3:+.3f} nL/s,  "
              f"|Q_H1| = {abs(q_h1_new)*nL_per_m3:.3f} nL/s,  "
              f"φ_H1 = {np.angle(q_h1_new):+.3f} rad")
    return out


def neumann_forward_solve(graph, boundary_flows, D, mu, f0_hz,
                            harmonics, px_size_m, pin_node=None,
                            h2_currents=None):
    """Solve the full nodal admittance system with Neumann (current
    injection) boundary conditions at the 4 boundary vessels' interior
    endpoints.  Boundary vessels themselves are excluded from the
    network (their Q is the BC).

    Optional:
      h2_currents: dict {interior_endpoint_node: Q_H2_SI (complex)}
        if provided AND 2 is in `harmonics`, injects these currents at
        the H2 harmonic (ω = 2 · 2π · f₀).  Used by the meeting
        notebook to test H2 sensitivity on synthetic data.

    Returns:
      node_P_DC: dict[node → P (Pa, real)]
      node_P_H[n]: dict[node → P phasor (complex)] for each n in harmonics
      Q_DC_truth[(u, v)]: edge Q (SI m³/s, real) for every non-boundary edge
      Q_H_truth[n][(u, v)]: edge Q phasor for each n in harmonics
    """
    # Identify boundary edges to exclude
    boundary_nodes = {bn for _, _, bn, _, _ in boundary_flows}
    interior_endpoints = {nb: (i, sign_q_dc, sign_q_h1)
                            for i, (nb, _, _, sign_q_dc, sign_q_h1)
                            in enumerate(boundary_flows)}
    exclude_edges = set()
    for nb, _, bn, _, _ in boundary_flows:
        exclude_edges.add(tuple(sorted([bn, nb])))

    # Build node index over all nodes EXCEPT boundary nodes
    all_nodes = [n for n in graph.nodes()
                  if n not in boundary_nodes]
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)
    edges_used = [(u, v) for u, v in graph.edges()
                   if tuple(sorted([u, v])) not in exclude_edges
                   and u in node_idx and v in node_idx]

    # Per-edge geometry → (G_e, C_e_per_unit_D)
    edge_GC = {}
    for u, v in edges_used:
        R_m, L_m = _edge_geometry(graph.edges[u, v], px_size_m)
        if R_m <= 0 or L_m <= 0 or not np.isfinite(R_m) or not np.isfinite(L_m):
            continue
        G_e = float(np.pi * R_m ** 4 / (8.0 * mu * L_m))
        # Areal-distensibility convention (matches production, post-2026-05-18):
        # c = πR²D, so total edge compliance C = c·L = πR²·D·L.
        C_e = float(np.pi * R_m ** 2 * D * L_m)
        edge_GC[(u, v)] = (G_e, C_e)

    # Pin node for gauge — default: an arbitrary high-degree interior node
    if pin_node is None:
        deg = dict(graph.degree())
        # Pick highest-degree interior node
        pin_node = max(node_idx.keys(),
                        key=lambda n: deg.get(n, 0))
    pin_idx = node_idx[pin_node]

    # Solve per harmonic
    node_P_per_h = {}
    Q_per_h = {}
    for n_harm in [0] + list(harmonics):
        omega = 2.0 * np.pi * float(n_harm) * f0_hz
        Y = np.zeros((N, N), dtype=complex)
        for (u, v), (G_e, C_e) in edge_GC.items():
            i, j = node_idx[u], node_idx[v]
            Y[i, j] -= G_e
            Y[j, i] -= G_e
            shunt = 0.5j * omega * C_e
            Y[i, i] += G_e + shunt
            Y[j, j] += G_e + shunt
        # Current injection vector
        i_inj = np.zeros(N, dtype=complex)
        for nb, sign_data in interior_endpoints.items():
            _, q_dc, q_h1 = sign_data
            j = node_idx[nb]
            if n_harm == 0:
                i_inj[j] += q_dc
            elif n_harm == 1:
                i_inj[j] += q_h1
            elif n_harm == 2 and h2_currents is not None:
                i_inj[j] += h2_currents.get(nb, 0.0 + 0.0j)
            # other harmonics: treated as 0 (no BC data)
        # Gauge pin only at DC (Laplacian has all-ones nullspace at ω=0).
        # At AC, shunt admittances make Y full-rank — no pin needed.
        if n_harm == 0:
            Y_modified = Y.copy()
            Y_modified[pin_idx, :] = 0
            Y_modified[pin_idx, pin_idx] = 1.0
            i_modified = i_inj.copy()
            i_modified[pin_idx] = 0.0
            P_vec = np.linalg.solve(Y_modified, i_modified)
        else:
            P_vec = np.linalg.solve(Y, i_inj)
        node_P_per_h[n_harm] = {n: complex(P_vec[idx])
                                  for n, idx in node_idx.items()}
        # Edge Q = G_e (P_u - P_v) for the series component
        Q_dict = {}
        for (u, v), (G_e, C_e) in edge_GC.items():
            i, j = node_idx[u], node_idx[v]
            Q_dict[(u, v)] = G_e * (P_vec[i] - P_vec[j])
        Q_per_h[n_harm] = Q_dict

    # Restructure for output
    node_P_DC = {n: float(node_P_per_h[0][n].real) for n in all_nodes}
    node_P_H = {h: {n: complex(node_P_per_h[h][n])
                       for n in all_nodes}
                 for h in harmonics}
    Q_DC_truth = {e: float(q.real) for e, q in Q_per_h[0].items()}
    Q_H_truth = {h: {e: complex(q) for e, q in Q_per_h[h].items()}
                  for h in harmonics}
    return node_P_DC, node_P_H, Q_DC_truth, Q_H_truth


def joint_lm(sub, edges_used, boundary_nodes, interior_nodes,
              Q_DC, Q_H, sigma_dc_e, sigma_h_e, *, ac_harmonics=HARMONICS,
              D_init=1.3e-3, eps_D=1e-3, D_step_cap=0.5,
              mu=MU, f0_hz=F0_HZ, px_size_m=PX_SIZE_M,
              max_iter=80, lm_mu0=1e-3, lm_factor=3.0,
              pin_dc=True, pin_idx=0,
              tol_rel=1e-8, verbose=False):
    """Joint LM over (D, P_DC, {P_Hn}) with per-measurement σ weights.
    DC pinned (drop pin_idx col); AC fully unpinned — fixes the
    production scan's AC-pin-gauge bug that collapses D̂ to floor on
    realistic boundary-pressure patterns."""
    n_bnd = len(boundary_nodes)
    n_p_dc = n_bnd - 1 if pin_dc else n_bnd
    keep_dc = (np.array([i for i in range(n_bnd) if i != pin_idx])
               if pin_dc else np.arange(n_bnd))
    n_p_ac = n_bnd
    n_ac = len(ac_harmonics)
    n_params = 1 + n_p_dc + 2 * n_p_ac * n_ac

    def ac_off(i_h):
        base = 1 + n_p_dc + 2 * n_p_ac * i_h
        return base, base + n_p_ac

    valid_dc = np.isfinite(Q_DC) & (np.abs(Q_DC) > 0)
    valid_h = {h: (np.isfinite(Q_H[h].real)
                    & np.isfinite(Q_H[h].imag)
                    & (np.abs(Q_H[h]) > 0)) for h in ac_harmonics}
    n_rdc = int(valid_dc.sum())
    n_rh = {h: int(valid_h[h].sum()) for h in ac_harmonics}
    w_dc = 1.0 / np.where(sigma_dc_e[valid_dc] > 0,
                            sigma_dc_e[valid_dc], 1.0)
    w_h_v = {h: 1.0 / np.where(sigma_h_e[h][valid_h[h]] > 0,
                                  sigma_h_e[h][valid_h[h]], 1.0)
              for h in ac_harmonics}
    w = np.concatenate([w_dc,
        *[np.tile(w_h_v[h], 2) for h in ac_harmonics]])
    n_total_rows = int(w.size)

    def build_T(D):
        ab = _build_admittance_system(sub, edges_used, boundary_nodes,
                                       interior_nodes, float(D), mu,
                                       f0_hz, ac_harmonics, px_size_m)
        return _compute_transfer_matrices(ab, edges_used, boundary_nodes,
                                            interior_nodes, verbose=False)

    def unpack(th):
        D_val = max(float(th[0]), 1e-12)
        P_DC_full = np.zeros(n_bnd, dtype=complex)
        P_DC_full[keep_dc] = th[1:1 + n_p_dc]
        P_H_full = {}
        for i_h, h in enumerate(ac_harmonics):
            o_re, o_im = ac_off(i_h)
            P_H_full[h] = (th[o_re:o_re + n_p_ac]
                            + 1j * th[o_im:o_im + n_p_ac]).astype(complex)
        return D_val, P_DC_full, P_H_full

    def residual(th):
        D_val, P_DC_full, P_H_full = unpack(th)
        T = build_T(D_val)
        r_parts = [(Q_DC - (T[0] @ P_DC_full).real)[valid_dc]]
        for h in ac_harmonics:
            r_h = (Q_H[h] - T[h] @ P_H_full[h])[valid_h[h]]
            r_parts.append(r_h.real); r_parts.append(r_h.imag)
        return np.concatenate(r_parts), T

    theta = np.zeros(n_params, dtype=float)
    theta[0] = float(D_init)
    T_init = build_T(D_init)
    if valid_dc.any():
        A = T_init[0][:, keep_dc][valid_dc].real * w_dc[:, None]
        b = Q_DC[valid_dc] * w_dc
        P_dc, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
        theta[1:1 + n_p_dc] = P_dc
    for i_h, h in enumerate(ac_harmonics):
        if valid_h[h].any():
            A = T_init[h][valid_h[h]] * w_h_v[h][:, None]
            b = Q_H[h][valid_h[h]] * w_h_v[h]
            P_h, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
            o_re, o_im = ac_off(i_h)
            theta[o_re:o_re + n_p_ac] = P_h.real
            theta[o_im:o_im + n_p_ac] = P_h.imag

    r_curr, T_curr = residual(theta)
    chi2_prev = float(np.sum((w * r_curr) ** 2))
    mu_lm = float(lm_mu0)
    history = [{'iter': 0, 'D': theta[0], 'chi2': chi2_prev,
                'mu': mu_lm, 'accept': True}]
    consec_rejects = 0
    for it in range(1, max_iter + 1):
        D_val, P_DC_full, P_H_full = unpack(theta)
        T = T_curr
        D_plus = D_val * (1.0 + eps_D)
        T_plus = build_T(D_plus)
        dT = {h: (T_plus[h] - T[h]) / (D_plus - D_val) for h in T}
        J = np.zeros((n_total_rows, n_params), dtype=float)
        row = 0
        if n_rdc > 0:
            jD = -(dT[0] @ P_DC_full).real
            J[row:row + n_rdc, 0] = jD[valid_dc] * w[row:row + n_rdc]
            for k_local, k in enumerate(keep_dc):
                J[row:row + n_rdc, 1 + k_local] = (
                    -T[0][valid_dc, k].real * w[row:row + n_rdc])
            row += n_rdc
        for i_h, h in enumerate(ac_harmonics):
            nrh = n_rh[h]
            if nrh == 0:
                continue
            T_h = T[h]; dT_h = dT[h]; P_h_full = P_H_full[h]
            jD_h = -(dT_h @ P_h_full)
            J[row:row + nrh, 0] = jD_h[valid_h[h]].real * w[row:row + nrh]
            J[row + nrh:row + 2 * nrh, 0] = (
                jD_h[valid_h[h]].imag * w[row + nrh:row + 2 * nrh])
            o_re, o_im = ac_off(i_h)
            for k in range(n_p_ac):
                t_re = T_h[valid_h[h], k].real
                t_im = T_h[valid_h[h], k].imag
                J[row:row + nrh, o_re + k] = -t_re * w[row:row + nrh]
                J[row:row + nrh, o_im + k] = +t_im * w[row:row + nrh]
                J[row + nrh:row + 2 * nrh, o_re + k] = (
                    -t_im * w[row + nrh:row + 2 * nrh])
                J[row + nrh:row + 2 * nrh, o_im + k] = (
                    -t_re * w[row + nrh:row + 2 * nrh])
            row += 2 * nrh
        r_w = w * r_curr
        H = J.T @ J
        g = -(J.T @ r_w)
        diag_H = np.diag(H).copy()
        diag_H = np.where(diag_H > 0, diag_H, 1.0)
        try:
            delta = np.linalg.solve(H + mu_lm * np.diag(diag_H), g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H + mu_lm * np.diag(diag_H), g,
                                     rcond=1e-10)[0]
        theta_trial = theta + delta
        D_cap = D_step_cap * abs(theta[0])
        if abs(theta_trial[0] - theta[0]) > D_cap:
            theta_trial[0] = theta[0] + np.sign(theta_trial[0] - theta[0]) * D_cap
        theta_trial[0] = float(np.clip(theta_trial[0], 1e-12, 1.0))
        r_trial, T_trial = residual(theta_trial)
        chi2_trial = float(np.sum((w * r_trial) ** 2))
        if chi2_trial < chi2_prev:
            theta = theta_trial; r_curr = r_trial; T_curr = T_trial
            chi2_prev = chi2_trial
            mu_lm /= lm_factor
            accept = True; consec_rejects = 0
        else:
            mu_lm *= lm_factor
            accept = False; consec_rejects += 1
        rel_dD = abs(delta[0]) / max(abs(theta[0]), 1e-30)
        chi2_hist = [h['chi2'] for h in history if h.get('accept', False)]
        rel_dchi2 = (abs(chi2_hist[-1] - chi2_prev)
                       / max(abs(chi2_hist[-1]), 1e-30)) if len(chi2_hist) >= 2 else 1.0
        history.append({'iter': it, 'D': theta[0], 'chi2': chi2_prev,
                        'mu': mu_lm, 'accept': accept,
                        'rel_dD': rel_dD, 'rel_dchi2': rel_dchi2})
        if it >= 3 and rel_dD < tol_rel and rel_dchi2 < tol_rel and accept:
            break
        if consec_rejects >= 5:
            break
    H_inv = np.linalg.pinv(H, rcond=1e-12)
    sigma_D = float(np.sqrt(max(H_inv[0, 0], 0.0)))
    D_hat = max(float(theta[0]), 1e-12)
    _, P_DC_full, P_H_full = unpack(theta)
    return dict(D_hat=D_hat, sigma_D=sigma_D, chi2=chi2_prev,
                 iters=len(history)-1, converged=accept,
                 P_DC=P_DC_full, P_H=P_H_full,
                 history=history)


def synth_refit_on_tile(graph, tile_id, Q_DC_truth, Q_H1_truth,
                          sigma_Q_nL, n_reps, rng, sign_clip_dc=True):
    """Run the (fixed) joint_lm inversion on one tile with truth Q
    from the global Neumann forward solve, perturbed by σ_Q noise.
    Uses joint_lm (AC-unpinned + D-step-cap + warm-start P) — does
    NOT use production _synthetic_refit which has the AC-pin-gauge
    bug that collapses D̂ to floor on realistic P_b patterns."""
    spec = lpi.LocalInferenceSpec(
        D_init=1.3e-3, eps_D=1e-3, lambda_reg=0.0, P_scale_Pa=None,
        harmonics=HARMONICS, use_dc=True, use_joint_lm=True,
        include_unmeasured_anatomy=True, mu=MU, px_size_m=PX_SIZE_M,
        f0_hz=F0_HZ, verbose=False, save_to_graph=False,
        prior_mode="magnitude", max_iter=200, tol_rel=1e-7,
        n_outer_iter=1, carve_drop_dangling_boundaries=True)
    edges_in, _, boundary_nodes, interior_nodes = \
        extract_tile_subgraph_spatial(
            graph, int(tile_id),
            inset_frac=float(spec.carve_inset_frac),
            restrict_to_tile_piv_nodes=bool(spec.carve_restrict_to_tile_piv),
            drop_dangling_boundaries=bool(spec.carve_drop_dangling_boundaries))
    interior_set = set(interior_nodes)
    g_attach = {n: 0.0 for n in boundary_nodes}
    for u, v in edges_in:
        Rm, Lm = _edge_geometry(graph.edges[u, v], spec.px_size_m)
        if Rm <= 0 or Lm <= 0:
            continue
        Ge = float(np.pi * Rm ** 4 / (8.0 * spec.mu * Lm))
        if u in g_attach and v in interior_set:
            g_attach[u] += Ge
        if v in g_attach and u in interior_set:
            g_attach[v] += Ge
    pin_node = max(g_attach, key=g_attach.get)
    pin_idx = boundary_nodes.index(pin_node)

    n_edges = len(edges_in)
    Q_DC_clean = np.zeros(n_edges, dtype=float)
    Q_H1_clean = np.zeros(n_edges, dtype=complex)
    for i, (u, v) in enumerate(edges_in):
        # Look up Q in both orderings
        if (u, v) in Q_DC_truth:
            Q_DC_clean[i] = Q_DC_truth[(u, v)]
            Q_H1_clean[i] = Q_H1_truth[(u, v)]
        elif (v, u) in Q_DC_truth:
            # Sign flips when edge ordering reversed (Q(u→v) = -Q(v→u))
            Q_DC_clean[i] = -Q_DC_truth[(v, u)]
            Q_H1_clean[i] = -Q_H1_truth[(v, u)]

    # Build a tile subgraph (joint_lm expects sub as nx.Graph) — only
    # done once per call since it's the same across reps.
    sub = nx.Graph()
    for n in set(interior_nodes) | set(boundary_nodes):
        nd = graph.nodes[n]
        sub.add_node(n, x=float(nd.get("x", 0)), y=float(nd.get("y", 0)),
                      boundary_type=("source" if n in set(boundary_nodes)
                                      else None))
    for u, v in edges_in:
        sub.add_edge(u, v, **dict(graph.edges[u, v]))
    sigma_Q_si = float(sigma_Q_nL) / nL_per_m3
    # Per-row σ vectors (joint_lm expects per-edge σ arrays)
    sigma_dc_e = np.full(n_edges, sigma_Q_si)
    sigma_h1_e = np.full(n_edges, sigma_Q_si)
    rep_records = []
    for _rep in range(n_reps):
        Q_DC_syn = Q_DC_clean + rng.normal(0.0, sigma_Q_si, size=n_edges)
        if sign_clip_dc:
            eps_si = 1e-15
            pos = Q_DC_clean > 0; neg = Q_DC_clean < 0
            Q_DC_syn[pos] = np.maximum(Q_DC_syn[pos], eps_si)
            Q_DC_syn[neg] = np.minimum(Q_DC_syn[neg], -eps_si)
        Q_H1_syn = Q_H1_clean + (
            rng.normal(0.0, sigma_Q_si / np.sqrt(2.0), n_edges)
            + 1j * rng.normal(0.0, sigma_Q_si / np.sqrt(2.0), n_edges))
        result = joint_lm(
            sub, edges_in, boundary_nodes, interior_nodes,
            Q_DC_syn, {1: Q_H1_syn},
            sigma_dc_e, {1: sigma_h1_e},
            ac_harmonics=HARMONICS, pin_dc=True, pin_idx=pin_idx,
            D_init=1.3e-3, verbose=False)
        rep_records.append({
            "D_hat": float(result["D_hat"]) if np.isfinite(result["D_hat"]) else float("nan"),
            "sigma_D": float(result["sigma_D"]),
            "iters": int(result["iters"]),
            "converged": bool(result["converged"]),
        })
    q_med_dc = float(np.median(np.abs(Q_DC_clean[np.abs(Q_DC_clean) > 0]))) \
        * nL_per_m3 if np.any(Q_DC_clean != 0) else float("nan")
    return rep_records, q_med_dc


def main():
    print("Loading graph ...", flush=True)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    print(f"  {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges", flush=True)

    print("\nReading boundary vessel flows ...")
    raw = read_boundary_vessel_flows(graph)
    boundary_flows = enforce_mass_balance_and_scale(raw, scale=BOUNDARY_Q_SCALE)

    print(f"\nGlobal Neumann forward solves at "
          f"D ∈ {D_GRID} ...", flush=True)
    truths = {}
    for D in D_GRID:
        t0 = time.time()
        _, _, Q_DC_tr, Q_H_tr = neumann_forward_solve(
            graph, boundary_flows, D, MU, F0_HZ, HARMONICS, PX_SIZE_M)
        truths[D] = (Q_DC_tr, Q_H_tr[1])
        # Probe synth |Q|_med on tile 22
        from pertile.analysis.local_pressure_inference import \
            extract_tile_subgraph_spatial as _ets
        ei22, _, _, _ = _ets(graph, 22, inset_frac=0.05,
                                 restrict_to_tile_piv_nodes=False,
                                 drop_dangling_boundaries=True)
        q22 = np.array([abs(Q_DC_tr.get((u, v), 0.0)) for u, v in ei22])
        med22 = float(np.median(q22[q22 > 0])) * nL_per_m3
        print(f"  D={D:.1e} done ({time.time()-t0:.1f}s)   "
              f"synth tile-22 |Q_DC|_med = {med22:.4g} nL/s")

    print(f"\nSweep ...", flush=True)
    rows = []
    t_start = time.time()
    for tile in TILES:
        for sigma_Q in SIGMA_Q_NLS:
            print(f"  tile {tile}, σ_Q = {sigma_Q:.4g} nL/s", flush=True)
            for D in D_GRID:
                Q_DC_tr, Q_H1_tr = truths[D]
                rng = np.random.default_rng(
                    int(tile * 1000 + sigma_Q * 1e5 + D * 1e8))
                recs, _q_med = synth_refit_on_tile(
                    graph, tile, Q_DC_tr, Q_H1_tr, sigma_Q, N_REPS, rng)
                for rep, r in enumerate(recs):
                    rows.append({
                        "tile_id": tile, "D_true": float(D),
                        "sigma_Q_nL": float(sigma_Q),
                        "rep": rep,
                        "D_hat": r["D_hat"],
                        "sigma_D": r["sigma_D"],
                        "rel_err": ((r["D_hat"] - D) / D
                                     if np.isfinite(r["D_hat"]) else float("nan")),
                        "iters": r["iters"],
                        "converged": r["converged"],
                    })
            with open(OUT_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows,  "
          f"{(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
