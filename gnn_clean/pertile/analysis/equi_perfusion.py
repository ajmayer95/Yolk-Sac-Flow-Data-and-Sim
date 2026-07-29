"""Equi-perfusion network adaptation (Gounaris/Katifori 2024).

Implements the cost
    H = Σ (ΔJ_ij − ⟨M⟩)² + α · Q²/(2k) + (ω/2) · k^γ
with edge absorption per the Krogh-cylinder formula
    ϕ_ij = 1 / (Q⁺_ij / (π R_ij ξ l_ij) + 1)
    ΔJ_ij = ϕ_ij · ρ_{src(ij)} · Q⁺_ij

JAX is used for autodiff, so the gradient `dH/dR` flows through both the
pressure solve and the concentration solve without finite differences.
For our typical sizes (N ≤ ~5000 nodes) the dense `jnp.linalg.solve` is
fine.  Switch to a sparse solver if you scale up much further.

ξ is the key dial.  Spec recipe:
    Low ξ:   degenerate single path (RMS-shear / Murray-tree regime)
    Mid ξ:   space-filling loopy hierarchy (target regime)
    High ξ:  source-side absorbs everything, downstream prunes

Sweeping ξ across ~4 decades and watching M/J_0 and σ/⟨M⟩ identifies the
intermediate window.

The implementation is intentionally readable rather than maximally fast.
JIT-compilation is left to the caller (the forward pass is jit-able once
debugging is done; see `forward_pass`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ─────────────────────────────────────────────────────────────────────
#  Params
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EquiPerfusionParams:
    """All knobs for the Gounaris/Katifori equi-perfusion adaptation.

    Units: SI throughout (m for lengths/radii, Pa·s for μ, m³/s for Q,
    m/s for ξ).  R_min/R_max also in m.
    """
    # Cost weights
    alpha: float = 1e-2          # dissipation weight
    omega: float = 1e-3          # material-cost weight
    gamma: float = 0.5           # material-cost exponent (0.5 = vol-like)

    # Krogh absorption rate (m/s).  The key dial.
    xi: float = 3e-6

    # Inlet nutrient concentration (units of cost·s/m³, arbitrary norm).
    rho_inlet: float = 1.0

    # Fluid
    mu: float = 3.5e-3           # Pa·s

    # Bounds on R (m); R_max keeps the slender-vessel approximation.
    R_min: float = 1e-6          # 1 µm
    R_max: float = 200e-6        # 200 µm

    # Total volumetric flow through the network in nL/s.  Distributed
    # equally across sources/sinks.  Default 1 nL/s matches yolk-sac data
    # and the rest of the codebase (PIV records, mean_Q, etc. all in nL/s).
    # Converted to SI (m³/s) internally for the Hagen-Poiseuille solve.
    Q_total_nL_per_s: float = 1.0

    # Optimisation
    nu_step: float = 1e-4        # gradient step
    n_iterations: int = 500
    tol_rel: float = 1e-4        # max|ΔR|/R convergence tolerance
    record_every: int = 25

    # Regularisation in case the concentration solve is singular at low ξ
    jitter_abs: float = 1e-12

    # Variance-of-absorption weight.  1.0 = full Gounaris/Katifori
    # equi-perfusion cost.  0.0 = pure dissipation + material cost
    # (Bohn-Magnasco / Hu-Cai); with steady BCs and γ<1 this gives
    # tree-like topologies instead of loopy hierarchy.
    variance_weight: float = 1.0


@dataclass
class EquiPerfusionResult:
    """Output of `run_equi_perfusion`."""
    final_radii_m: Dict[Tuple[int, int], float]
    history_H: np.ndarray
    history_sigma_over_M: np.ndarray
    history_M_over_J0: np.ndarray
    history_Lambda: np.ndarray
    iteration_indices: np.ndarray
    converged: bool
    n_iterations_run: int
    edge_list: List[Tuple[int, int]]
    params: EquiPerfusionParams


# ─────────────────────────────────────────────────────────────────────
#  Forward pass — pure-JAX
# ─────────────────────────────────────────────────────────────────────

def _forward_jax(R, lengths, edges_u, edges_v, n_nodes, Q_B, rho_B,
                  source_mask, pin_node, alpha, omega, gamma, xi,
                  jitter_abs, variance_weight=1.0):
    """Single forward pass in DIMENSIONLESS units.

    All inputs are dimensionless: `R` and `lengths` are O(1), `Q_B` sums
    to ±1 across sources/sinks, `xi` is the dimensionless absorption
    parameter `ξ' = ξ · π R̄ L̄ / Q̄` (so the crossover regime is ξ' ~ 1).

    `edges_u`, `edges_v` are int arrays of edge endpoints (canonical
    orientation).  `source_mask` flags source nodes.

    Returns (H, diag_dict).
    """
    # 1. Conductances k = R⁴ / L (dimensionless).
    k = R ** 4 / lengths

    # 2. Pressure Laplacian (N × N, dense).
    L = jnp.zeros((n_nodes, n_nodes))
    # Add -k on off-diagonals and +k on diagonals.
    L = L.at[edges_u, edges_v].add(-k)
    L = L.at[edges_v, edges_u].add(-k)
    L = L.at[edges_u, edges_u].add(k)
    L = L.at[edges_v, edges_v].add(k)

    # Pin one node: replace its row/col with identity and 0 the RHS entry.
    L = L.at[pin_node, :].set(0.0)
    L = L.at[:, pin_node].set(0.0)
    L = L.at[pin_node, pin_node].set(1.0)
    Q_B_pinned = Q_B.at[pin_node].set(0.0)

    # 3. Solve L p = Q_B
    p_vec = jnp.linalg.solve(L, Q_B_pinned)

    # 4. Signed edge flow Q_ij = k_ij (p_u - p_v) using canonical (u,v).
    #    `Q_mag` is the absolute magnitude (flow regardless of direction).
    Q = k * (p_vec[edges_u] - p_vec[edges_v])
    Q_mag = jnp.abs(Q)

    # 5. Krogh absorption capacity per edge (dimensionless).
    #    capacity' = R' L' × ξ'   (ξ' already absorbs π and the R̄ L̄ scale)
    capacity = R * lengths * xi
    phi = 1.0 / (Q_mag / jnp.maximum(capacity, 1e-30) + 1.0)

    # 6. Concentration solve.  We build L_abs from the directed flow.
    #
    # For an edge with Q_ij > 0 (flow u→v):  v receives nutrient from u
    #   (1 − ϕ_ij) · Q_ij · ρ_u.  Otherwise (Q_ij < 0, flow v→u):
    #   u receives (1 − ϕ_ji) · |Q| · ρ_v.
    #
    # The KCL for concentration at each node j:
    #     (Σ outgoing |Q|) ρ_j = Σ incoming (1 − ϕ) |Q| ρ_src(e)   (+ J_B at sources)
    #
    # We assemble L_abs ρ = -J_B with:
    #   diagonal  L_abs[j,j] = -(Σ outgoing |Q| from j)
    #   off-diag  L_abs[j,src] += (1 − ϕ_e) · |Q_e|   when flow src→j
    abs_Q   = Q_mag
    # Source-of-each-edge per current flow direction:
    src_of_edge = jnp.where(Q >= 0, edges_u, edges_v)
    dst_of_edge = jnp.where(Q >= 0, edges_v, edges_u)
    one_minus_phi_Q = (1.0 - phi) * abs_Q

    # Build L_abs in the convention `L_abs · ρ = -J_B`.
    #
    # At each node i the steady-state concentration balance is
    #     ρ_i · (Σ_edges-out |Q| + max(-Q_B_i, 0))
    #       = Σ_edges-in (1 − ϕ) |Q| ρ_src + max(Q_B_i, 0) · ρ_B_i
    #
    # which corresponds to
    #     diag    L_abs[i,i] = −(outflow_edges_i + sink-BC-outflow_i)
    #     off-d   L_abs[i, src(e)] = +(1 − ϕ_e) |Q_e|
    #     RHS     -J_B with J_B_i = max(Q_B_i, 0) · ρ_B_i (sources only)
    #
    # Adding the sink-BC outflow `max(-Q_B, 0)` to the diagonal is the
    # critical fix vs the bare spec; without it sink rows are singular
    # (no edge outflow → 0·ρ_sink = Σ(1-ϕ)|Q|ρ_src, contradiction).
    L_abs = jnp.zeros((n_nodes, n_nodes))
    L_abs = L_abs.at[src_of_edge, src_of_edge].add(-abs_Q)         # edge outflow
    Q_B_sink_outflow = jnp.maximum(-Q_B, 0.0)
    diag_sink_contrib = jnp.diag(Q_B_sink_outflow)
    L_abs = L_abs - diag_sink_contrib                              # sink BC outflow
    L_abs = L_abs.at[dst_of_edge, src_of_edge].add(one_minus_phi_Q)

    Q_B_source_inflow = jnp.maximum(Q_B, 0.0)
    rhs_abs = -(Q_B_source_inflow * rho_B)

    # Regularise to avoid singular solves at extreme xi.
    eye = jnp.eye(n_nodes)
    L_abs_reg = L_abs + jitter_abs * eye
    rho_node = jnp.linalg.solve(L_abs_reg, rhs_abs)

    # 7. Edge absorption ΔJ_ij = ϕ_ij · ρ_src · |Q_ij|.  Uses magnitude
    #    (not the canonical-orientation signed Q) so absorption is direction-
    #    invariant.
    rho_src = rho_node[src_of_edge]
    Delta_J = phi * rho_src * Q_mag

    # 8. Cost.
    M_mean = jnp.mean(Delta_J)
    var_term = jnp.sum((Delta_J - M_mean) ** 2)
    diss_term = jnp.sum(Q ** 2 / (2.0 * jnp.maximum(k, 1e-30)))
    mat_term = jnp.sum(k ** gamma)
    H = variance_weight * var_term + alpha * diss_term + 0.5 * omega * mat_term

    return H, {
        'M_mean': M_mean,
        'sigma_M': jnp.std(Delta_J),
        'Delta_J': Delta_J,
        'Q': Q,
        'rho_node': rho_node,
        'k': k,
    }


# ─────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────

def _prepare_graph(G: nx.Graph, params: EquiPerfusionParams):
    """Pull edge list + canonical (u, v) orientation + lengths in meters.

    Reads radius from edge attr 'radius' (px), length from 'length' (px),
    boundary nodes from `boundary_type` ∈ {'source', 'sink'}.  Px size
    comes from `pertile.analysis.config.PX_SIZE_UM`.

    Returns a dict with the JAX-ready inputs + accessory info.
    """
    from .config import PX_SIZE_UM
    px_to_m = PX_SIZE_UM * 1e-6

    # Canonical node indexing
    nodes = sorted(G.nodes())
    n_nodes = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edges = list(G.edges())
    n_edges = len(edges)
    edges_u = np.array([node_to_idx[u] for u, v in edges], dtype=np.int32)
    edges_v = np.array([node_to_idx[v] for u, v in edges], dtype=np.int32)

    radii_px = np.array([float(G.edges[u, v].get('radius', 1.0))
                          for u, v in edges])
    lengths_px = np.array([float(G.edges[u, v].get('length', 1.0))
                            for u, v in edges])
    R_m = np.clip(radii_px * px_to_m, params.R_min, params.R_max)
    L_m = lengths_px * px_to_m
    L_m = np.maximum(L_m, 1e-9)

    # Boundary conditions
    sources, sinks = [], []
    for n in nodes:
        bt = G.nodes[n].get('boundary_type')
        if bt == 'source':
            sources.append(node_to_idx[n])
        elif bt == 'sink':
            sinks.append(node_to_idx[n])

    if not sources or not sinks:
        raise ValueError(
            f"Graph must have ≥1 source and ≥1 sink (boundary_type tags). "
            f"Found {len(sources)} sources, {len(sinks)} sinks.")

    # ──────────────────────────────────────────────────────────────
    # Non-dimensionalisation.  The cost function has three terms whose
    # magnitudes depend wildly on the choice of (m, m³/s, Pa·s).  We
    # rescale internally so Q, R, L, k are all O(1) and the spec's
    # default α=1e-2, ω=1e-3 give comparable-magnitude terms.
    #
    # Reference scales:
    #   R̄  = mean(R_init)         (length)
    #   L̄  = mean(L)              (length)
    #   Q̄  = Q_total_si           (volume flow)
    #   μ̄  = mu                   (viscosity)
    #   k̄  = π R̄⁴ / (8 μ̄ L̄)      (conductance)
    #
    # Dimensionless ξ' = ξ × π R̄ L̄ / Q̄.  This makes Q̄/(π R̄ ξ L̄) = 1/ξ',
    # so the crossover Q/(πRξl) ~ 1 corresponds to ξ' ~ 1.
    # ──────────────────────────────────────────────────────────────
    NL_TO_M3S = 1e-12
    Q_total_si = float(params.Q_total_nL_per_s) * NL_TO_M3S
    R_bar = float(np.mean(R_m))
    L_bar = float(np.mean(L_m))
    capacity_bar = np.pi * R_bar * L_bar         # m² (a length scale)
    xi_dimless = float(params.xi) * capacity_bar / max(Q_total_si, 1e-30)

    # Dimensionless quantities for the inner solver.
    R_dim = R_m / R_bar
    L_dim = L_m / L_bar
    Q_B_dim = np.zeros(n_nodes)
    Q_src_each = 1.0 / len(sources)
    Q_snk_each = -1.0 / len(sinks)
    for s in sources:
        Q_B_dim[s] = Q_src_each
    for s in sinks:
        Q_B_dim[s] = Q_snk_each

    # Hagen-Poiseuille conductance in dimensionless units becomes
    #   k' = R'⁴ / L'
    # (a pure geometry term).  μ vanishes from the dimensionless solve
    # because both the pressure and Q rescale to absorb it.
    # See `_forward_jax` for the explicit usage.

    rho_B = np.full(n_nodes, params.rho_inlet)
    source_mask = np.zeros(n_nodes, dtype=bool)
    for s in sources:
        source_mask[s] = True

    # Pin sink-0 (arbitrary choice for the Laplacian).
    pin_node = sinks[0]

    return {
        'edges': edges,
        'edges_u': edges_u,
        'edges_v': edges_v,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        # SI quantities (for unscaling at the end)
        'R_m_init': R_m,
        'lengths_m': L_m,
        'R_bar': R_bar,
        'L_bar': L_bar,
        'Q_total_si': Q_total_si,
        # Dimensionless quantities for the JAX forward pass
        'R_dim': R_dim,
        'L_dim': L_dim,
        'Q_B_dim': Q_B_dim,
        'rho_B': rho_B,
        'source_mask': source_mask,
        'pin_node': pin_node,
        'xi_dimless': xi_dimless,
        'sources': sources,
        'sinks': sinks,
        'node_to_idx': node_to_idx,
    }


def run_equi_perfusion(
    G: nx.Graph,
    params: Optional[EquiPerfusionParams] = None,
    *,
    verbose: bool = True,
) -> EquiPerfusionResult:
    """Run the Gounaris/Katifori equi-perfusion adaptation in JAX.

    Reads R_init (m) + length (m) from the graph; returns final R per edge
    plus optimization history (H, σ/⟨M⟩, M/J_0, Λ).
    """
    if not HAS_JAX:
        raise ImportError("jax not installed; required for equi-perfusion adaptation.")
    if params is None:
        params = EquiPerfusionParams()

    state = _prepare_graph(G, params)
    edges = state['edges']
    edges_u = jnp.asarray(state['edges_u'])
    edges_v = jnp.asarray(state['edges_v'])
    n_nodes = state['n_nodes']
    n_edges = state['n_edges']
    # Use DIMENSIONLESS inputs to the forward pass — see _prepare_graph
    # for the rescaling.  Final radii get multiplied by R_bar to recover SI.
    L_dim = jnp.asarray(state['L_dim'])
    Q_B_dim = jnp.asarray(state['Q_B_dim'])
    rho_B = jnp.asarray(state['rho_B'])
    source_mask = jnp.asarray(state['source_mask'])
    pin_node = int(state['pin_node'])
    xi_dimless = float(state['xi_dimless'])
    R_bar = float(state['R_bar'])
    R_min_dim = float(params.R_min) / R_bar
    R_max_dim = float(params.R_max) / R_bar

    # Forward closure that takes only R (so jax.grad(...) returns dH/dR).
    def H_only(R_):
        H, _ = _forward_jax(
            R_, L_dim, edges_u, edges_v, n_nodes,
            Q_B_dim, rho_B, source_mask, pin_node,
            params.alpha, params.omega, params.gamma, xi_dimless,
            params.jitter_abs)
        return H

    grad_H = jax.grad(H_only)
    forward_full = lambda R_: _forward_jax(
        R_, L_dim, edges_u, edges_v, n_nodes,
        Q_B_dim, rho_B, source_mask, pin_node,
        params.alpha, params.omega, params.gamma, xi_dimless,
        params.jitter_abs, params.variance_weight)

    R = jnp.asarray(state['R_dim'])

    history_H, history_sigma, history_M, history_Lambda, iter_idx = [], [], [], [], []
    converged = False
    n_run = 0
    for it in range(params.n_iterations):
        H_val, diag = forward_full(R)
        g = grad_H(R)

        # Diagnostics
        Delta_J = diag['Delta_J']
        M_mean = float(diag['M_mean'])
        sigma = float(diag['sigma_M'])
        J0 = float(jnp.sum(jnp.where(source_mask,
                                       Q_B_dim * rho_B, 0.0)))     # ∑ Q_B ρ_B at sources
        M_total = float(jnp.sum(Delta_J))
        # Λ = cycles among active edges (dimensionless R > R_min_dim * 1.05).
        active = np.array(R > R_min_dim * 1.05)
        Lambda = int(active.sum() - n_nodes + 1)

        if it % params.record_every == 0 or it == params.n_iterations - 1:
            history_H.append(float(H_val))
            history_sigma.append(sigma / max(abs(M_mean), 1e-30))
            history_M.append(M_total / max(J0, 1e-30))
            history_Lambda.append(Lambda)
            iter_idx.append(it)
            if verbose:
                print(f"  iter {it:4d}:  H={float(H_val):.4e}  "
                      f"σ/⟨M⟩={sigma/max(abs(M_mean),1e-30):.3f}  "
                      f"M/J0={M_total/max(J0,1e-30):.3f}  "
                      f"Λ={Lambda}  |grad|={float(jnp.linalg.norm(g)):.2e}")

        # Gradient step with dimensionless clipping.
        R_new = jnp.clip(R - params.nu_step * g, R_min_dim, R_max_dim)
        rel = float(jnp.max(jnp.abs(R_new - R) / jnp.maximum(R, 1e-30)))
        R = R_new
        n_run = it + 1
        if rel < params.tol_rel and it > 5:
            converged = True
            break

    # Output — unscale dimensionless R → SI by multiplying by R_bar.
    final_radii_m = {e: float(R[i]) * R_bar for i, e in enumerate(edges)}
    return EquiPerfusionResult(
        final_radii_m=final_radii_m,
        history_H=np.asarray(history_H),
        history_sigma_over_M=np.asarray(history_sigma),
        history_M_over_J0=np.asarray(history_M),
        history_Lambda=np.asarray(history_Lambda),
        iteration_indices=np.asarray(iter_idx),
        converged=converged,
        n_iterations_run=n_run,
        edge_list=edges,
        params=params,
    )
