"""
Vascular network adaptation algorithms.

Implements four adaptation modes:

1. **sigma_Q**: Chatterjee & Katifori with dissipation signal.
   dR/dt = a · S^γ / R^{3(1-γ)} − b · R, S = dissipation.

2. **Q2**: Chatterjee & Katifori with ⟨Q²⟩ signal.
   Same ODE, S = ⟨Q²⟩.

3. **perfusion**: Gradient descent on combined cost
   H = Σ D_e + w_p Σ (ΔJ_e − ⟨M⟩)² + w_m Σ k_e^γ
   with nutrient absorption through the vessel wall.

4. **mixed**: sigma_Q Chatterjee rule + nutrient perfusion gradient.

All modes use the frequency-domain transmission line solver for pulsatile
flow, and an iterative concentration solver for nutrient advection.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .transmission_line import (
    TransmissionLineResult,
    solve_transmission_line,
    _get_edge_geometry,
    MU_DEFAULT,
    RHO_BLOOD,
)


# =====================================================================
# Parameters
# =====================================================================

@dataclass
class AdaptationParams:
    """All parameters for vascular adaptation.

    Modes:
    - ``'sigma_Q'``: Chatterjee with dissipation signal (default)
    - ``'Q2'``:      Chatterjee with ⟨Q²⟩ signal
    - ``'perfusion'``: gradient descent on H (dissipation + perfusion + material)
    - ``'mixed'``:   sigma_Q Chatterjee + perfusion gradient
    - ``'sigma_Q_local'``: sigma_Q + local wall-material diffusion on the
      line graph.  Edges can only grow as fast as their neighbours supply
      material; prevents pure A→V trunk collapse.
    """

    # --- Mode selection ---
    mode: str = 'sigma_Q'
    # 'sigma_Q':       dR/dt = a·S^γ/R^{3(1-γ)} − b·R, S=dissipation
    # 'Q2':            same ODE, S=⟨Q²⟩
    # 'perfusion':     gradient descent on H
    # 'mixed':         sigma_Q rule + perfusion gradient
    # 'sigma_Q_local': sigma_Q + ν·Δ(V) wall-material diffusion (V = R²L)

    # --- Murray exponent ---
    gamma: float = 2.0 / 3.0   # 2/3 = blood volume, 4/5 = surface area

    # --- Growth / decay ---
    a: float = 1.0              # growth rate prefactor
    b: float = 1.0              # decay rate

    # --- Cost weights (perfusion / mixed) ---
    w_perfusion: float = 1.0    # weight on nutrient perfusion variance
    w_material: float = 1e-3    # weight on metabolic (material) cost

    # --- Nutrient absorption ---
    xi: float = 0.0             # absorption rate per unit wall area (m/s)
    rho_inlet: float = 1.0     # inlet nutrient concentration (dimensionless)

    # --- Gradient descent (perfusion mode) ---
    nu_step: float = 0.01      # gradient descent step size
    dR_frac: float = 0.01      # fractional perturbation for finite-difference gradient

    # --- Local material exchange (sigma_Q_local mode) ---
    # Material V_e = R_e²·L_e diffuses along the line graph (edges that
    # share a node).  Higher `nu_diffusion` → more spatial smoothing,
    # weaker hierarchy.  Set 0 to disable; the mode then reduces to plain
    # sigma_Q.
    nu_diffusion: float = 0.5
    # When True, after each step rescale all V so Σ V is preserved at
    # its initial value — strict material conservation.  Recommended for
    # sigma_Q_local.
    conserve_volume: bool = True

    # --- Transmission line params ---
    D: float = 1e-6            # distensibility (1/Pa)
    n_harmonics: int = 3
    f0_hz: Optional[float] = None
    mu: float = MU_DEFAULT
    rho_fluid: float = RHO_BLOOD

    # --- Numerics ---
    dt: float = 0.01
    n_iterations: int = 500
    convergence_tol: float = 1e-6
    record_every: int = 10
    shunt_threshold_um: float = 1.0    # hard floor (µm)


# =====================================================================
# Adaptation result
# =====================================================================

@dataclass
class AdaptationResult:
    """Results from radius adaptation simulation."""
    initial_result: TransmissionLineResult
    final_result: TransmissionLineResult
    final_radii_m: Dict[Tuple[int, int], float]
    # Trajectory (recorded every record_every iterations)
    radii_history: np.ndarray          # (n_records, n_edges)
    dissipation_history: np.ndarray    # (n_records,)
    max_dRdt_history: np.ndarray       # (n_records,)
    iteration_indices: np.ndarray      # which iterations were recorded
    # Parameters
    mode: str
    gamma: float
    a: float
    b: float
    dt: float
    n_iterations: int
    shunt_threshold_m: float
    shunted_edges: List[Tuple[int, int]]
    edge_list: List[Tuple[int, int]]
    # Optional nutrient fields
    absorption_history: Optional[np.ndarray] = None    # (n_records,) total absorption
    entropy_history: Optional[np.ndarray] = None       # (n_records,) flow entropy
    final_concentrations: Optional[Dict[int, float]] = None
    final_absorption: Optional[Dict[Tuple[int, int], float]] = None
    params: Optional[AdaptationParams] = None


# =====================================================================
# Nutrient concentration solver
# =====================================================================

def solve_nutrients(
    G: nx.Graph,
    tl_result: TransmissionLineResult,
    edge_list: List[Tuple[int, int]],
    radii_m: Dict[Tuple[int, int], float],
    xi: float = 1e-6,
    rho_inlet: float = 1.0,
    boundary_nodes: Optional[List[int]] = None,
    max_iter: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Solve steady-state nutrient concentration by downstream propagation.

    Nutrients enter at source (arterial) boundary nodes with concentration
    ρ_inlet. As flow passes through vessel (i→j), absorption through the
    wall removes a fraction:

        ΔJ_ij = ρ_i · Q_ij / (Q_ij / (π R_ij ξ L_ij) + 1)

    where ξ is the absorption rate per unit wall area (m/s).

    At each node j, the concentration is the flow-weighted average of
    all incoming nutrient currents.

    Uses DC flows only (nutrients advected by mean flow).

    Parameters
    ----------
    G : nx.Graph
    tl_result : TransmissionLineResult
        Must have mean_Q (nL/s) for flow directions.
    edge_list : list of (u, v)
    radii_m : dict
        Current radii in meters.
    xi : float
        Absorption rate per unit wall area (m/s). 0 = no absorption.
    rho_inlet : float
        Nutrient concentration at source nodes.
    boundary_nodes : list, optional
        Boundary (source) nodes. If None, uses tl_result.boundary_nodes.
    max_iter, tol : int, float
        Iteration limit and convergence tolerance.

    Returns
    -------
    concentrations : dict {node: float}
        Nutrient concentration at each node.
    absorption : dict {(u,v): float}
        Absorption current ΔJ in each edge.
    """
    if xi <= 0:
        # No absorption — all concentrations are rho_inlet
        nodes = set()
        for u, v in edge_list:
            nodes.add(u)
            nodes.add(v)
        conc = {n: rho_inlet for n in nodes}
        absorp = {e: 0.0 for e in edge_list}
        return conc, absorp

    if boundary_nodes is None:
        boundary_nodes = tl_result.boundary_nodes

    # Identify source nodes (net outflow = arteries)
    source_nodes = set()
    for bn in boundary_nodes:
        # Check if this node is a net source (outflow > inflow)
        net_out = 0.0
        for u, v in G.edges(bn):
            q = tl_result.mean_Q.get((u, v), tl_result.mean_Q.get((v, u), 0))
            if u == bn:
                net_out += q
            else:
                net_out -= q
        if net_out > 0:
            source_nodes.add(bn)

    if not source_nodes:
        # Fallback: use all boundary nodes as sources
        source_nodes = set(boundary_nodes)

    # Collect all nodes
    all_nodes = set()
    for u, v in edge_list:
        all_nodes.add(u)
        all_nodes.add(v)

    # Build directed flow info: for each edge, determine upstream → downstream
    # Q_ij > 0 means flow from i to j (as stored in mean_Q)
    flow_dir = {}  # edge -> (upstream, downstream, |Q| in m³/s)
    for e in edge_list:
        u, v = e
        q_nls = tl_result.mean_Q.get(e, 0)
        if q_nls is None:
            q_nls = 0
        q_m3s = abs(q_nls) * 1e-12
        if q_nls >= 0:
            flow_dir[e] = (u, v, q_m3s)
        else:
            flow_dir[e] = (v, u, q_m3s)

    # Get lengths
    lengths_m = {}
    for e in edge_list:
        _, L_m = _get_edge_geometry(G, e[0], e[1])
        lengths_m[e] = L_m if L_m is not None else 1e-6

    # Iterative concentration solver
    conc = {n: rho_inlet for n in all_nodes}

    for iteration in range(max_iter):
        conc_new = {}
        max_change = 0.0

        for node in all_nodes:
            if node in source_nodes:
                conc_new[node] = rho_inlet
                continue

            # Sum incoming nutrient currents and flows
            total_J_in = 0.0
            total_Q_in = 0.0

            for e in edge_list:
                upstream, downstream, Q_abs = flow_dir[e]
                if downstream != node:
                    continue
                if Q_abs < 1e-30:
                    continue

                R_m = radii_m.get(e, 1e-6)
                L_m = lengths_m[e]
                rho_up = conc.get(upstream, rho_inlet)

                # Absorption in this vessel
                # ΔJ = ρ_up · Q / (Q/(π·R·ξ·L) + 1)
                wall_capacity = np.pi * R_m * xi * L_m  # m³/s equivalent
                if wall_capacity > 1e-30:
                    denom = Q_abs / wall_capacity + 1.0
                    delta_J = rho_up * Q_abs / denom
                else:
                    delta_J = 0.0

                J_out = rho_up * Q_abs - delta_J
                total_J_in += J_out
                total_Q_in += Q_abs

            if total_Q_in > 1e-30:
                conc_new[node] = total_J_in / total_Q_in
            else:
                conc_new[node] = 0.0

            max_change = max(max_change, abs(conc_new[node] - conc.get(node, 0)))

        conc = conc_new

        if max_change < tol:
            if verbose:
                print(f"  Nutrients converged in {iteration+1} iterations "
                      f"(max_change={max_change:.2e})")
            break

    # Compute per-edge absorption
    absorption = {}
    for e in edge_list:
        upstream, downstream, Q_abs = flow_dir[e]
        R_m = radii_m.get(e, 1e-6)
        L_m = lengths_m[e]
        rho_up = conc.get(upstream, rho_inlet)

        wall_capacity = np.pi * R_m * xi * L_m
        if wall_capacity > 1e-30 and Q_abs > 1e-30:
            denom = Q_abs / wall_capacity + 1.0
            absorption[e] = rho_up * Q_abs / denom
        else:
            absorption[e] = 0.0

    return conc, absorption


# =====================================================================
# Diagnostics
# =====================================================================

def _flow_entropy(mean_Q: Dict[Tuple[int, int], float]) -> float:
    """Shannon entropy of |Q| distribution (higher = more uniform flow)."""
    flows = np.array([abs(q) for q in mean_Q.values() if np.isfinite(q)])
    if len(flows) == 0 or flows.sum() < 1e-30:
        return 0.0
    p = flows / flows.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


# =====================================================================
# Main adaptation loop
# =====================================================================

def run_adaptation(
    G: nx.Graph,
    params: Optional[AdaptationParams] = None,
    # Legacy keyword arguments for backward compat
    mode: Optional[str] = None,
    gamma: float = 2.0 / 3.0,
    a: float = 1.0,
    b: float = 1.0,
    dt: float = 0.01,
    n_iterations: int = 500,
    D: float = 1e-6,
    n_harmonics: int = 3,
    f0_hz: Optional[float] = None,
    mu: float = MU_DEFAULT,
    rho: float = RHO_BLOOD,
    boundary_nodes: Optional[List[int]] = None,
    bc_harmonics_override: Optional[Dict[int, np.ndarray]] = None,
    shunt_threshold_um: float = 1.0,
    convergence_tol: float = 1e-6,
    record_every: int = 10,
    verbose: bool = True,
) -> AdaptationResult:
    """Run radius adaptation.

    If *params* is provided, it takes precedence over keyword arguments.
    Otherwise, an AdaptationParams is constructed from the keyword arguments
    for backward compatibility with the Chatterjee-only interface.

    Parameters
    ----------
    G : nx.Graph
    params : AdaptationParams, optional
    mode : str, optional
        Legacy: 'Q2', 'sigmaQ', 'gounaris', 'combined',
        or new: 'sigma_Q', 'perfusion', 'mixed'.
    boundary_nodes : list, optional
        If None, auto-detected from graph.
    verbose : bool
    """
    # --- Build params from legacy kwargs if needed ---
    if params is None:
        # Map legacy names
        _mode = mode or 'sigma_Q'
        if _mode == 'sigmaQ':
            _mode = 'sigma_Q'
        elif _mode == 'gounaris':
            _mode = 'perfusion'
        elif _mode == 'combined':
            _mode = 'mixed'
        p = AdaptationParams(
            mode=_mode,
            gamma=gamma, a=a, b=b, dt=dt,
            n_iterations=n_iterations,
            D=D, n_harmonics=n_harmonics, f0_hz=f0_hz,
            mu=mu, rho_fluid=rho,
            shunt_threshold_um=shunt_threshold_um,
            convergence_tol=convergence_tol,
            record_every=record_every,
        )
    else:
        p = params

    shunt_threshold_m = p.shunt_threshold_um * 1e-6

    # --- Initialize radii from graph ---
    edge_list = []
    radii_m = {}
    lengths_m = {}
    for u, v in G.edges():
        R_m, L_m = _get_edge_geometry(G, u, v)
        if R_m is not None:
            edge_list.append((u, v))
            radii_m[(u, v)] = R_m
            lengths_m[(u, v)] = L_m

    n_edges = len(edge_list)
    if verbose:
        print(f"Adaptation: {n_edges} edges, mode={p.mode}, γ={p.gamma:.3f}, "
              f"a={p.a}, b={p.b}, dt={p.dt}")
        if p.mode in ('perfusion', 'mixed') or p.xi > 0:
            print(f"  xi={p.xi:.2e} m/s, "
                  f"w_p={p.w_perfusion}, w_m={p.w_material}")

    # --- Normalization ---
    R0 = np.mean(list(radii_m.values()))

    # Initial solve for normalization constants
    result_init = solve_transmission_line(
        G, D=p.D, n_harmonics=p.n_harmonics, f0_hz=p.f0_hz,
        mu=p.mu, rho=p.rho_fluid,
        boundary_nodes=boundary_nodes, radii_m=radii_m,
        bc_harmonics_override=bc_harmonics_override, verbose=False)

    q_dcs = [abs(result_init.mean_Q.get(e, 0)) for e in edge_list]
    Q0 = max(np.mean(q_dcs), 1e-12)

    diss_vals = [result_init.dissipation.get(e, 0) for e in edge_list]
    diss0 = max(np.mean(diss_vals), 1e-30)

    # Nutrient normalization
    abs0 = 1.0  # will be set after first nutrient solve
    if p.xi > 0:
        _, abs_init = solve_nutrients(
            G, result_init, edge_list, radii_m,
            xi=p.xi, rho_inlet=p.rho_inlet,
            boundary_nodes=boundary_nodes, verbose=verbose)
        abs0 = max(np.mean(list(abs_init.values())), 1e-30)

    if verbose:
        print(f"  R0={R0*1e6:.2f} µm, Q0={Q0:.3f} nL/s, diss0={diss0:.3e} W")
        if p.xi > 0:
            print(f"  abs0={abs0:.3e} (mean absorption per edge)")

    # Normalized radii
    r_norm = np.array([radii_m[e] / R0 for e in edge_list])
    # Cap to keep the trunk from running away on dissipation-greedy rules.
    # 5× the largest initial radius is generous but finite.
    r_max_norm = 5.0 * float(r_norm.max())
    # Normalised edge lengths (used by the sigma_Q_local material-diffusion
    # step; defined here so it's available even when other modes don't need it).
    L_bar = max(float(np.mean(list(lengths_m.values()))), 1e-30)
    L_norm = np.array([lengths_m[e] / L_bar for e in edge_list])
    V_initial_total = float((r_norm ** 2 * L_norm).sum())

    # --- Line-graph adjacency (sigma_Q_local) ---
    # Edges sharing a node are "neighbours" in the line graph.  Build a
    # sparse incidence so the per-iteration Laplacian step is one
    # matrix-vector product.  Cheap to build once.
    from scipy import sparse as _sparse
    node_to_edges = {}
    for i, (u, v) in enumerate(edge_list):
        node_to_edges.setdefault(u, []).append(i)
        node_to_edges.setdefault(v, []).append(i)
    _rows, _cols = [], []
    for incident in node_to_edges.values():
        if len(incident) < 2:
            continue
        for a in incident:
            for b in incident:
                if a != b:
                    _rows.append(a)
                    _cols.append(b)
    if _rows:
        neigh_sum_indices_data = _sparse.csr_matrix(
            (np.ones(len(_rows), dtype=float), (_rows, _cols)),
            shape=(n_edges, n_edges))
        neigh_deg = np.asarray(neigh_sum_indices_data.sum(axis=1)).ravel()
    else:
        neigh_sum_indices_data = _sparse.csr_matrix((n_edges, n_edges))
        neigh_deg = np.zeros(n_edges)

    # --- Trajectory storage ---
    radii_records = []
    diss_records = []
    dRdt_records = []
    iter_records = []
    absorption_records = []
    entropy_records = []
    r_min_norm = shunt_threshold_m / R0

    if verbose:
        print(f"  R_min = {shunt_threshold_m*1e6:.1f} µm (r̃_min = {r_min_norm:.4f})")

    last_concentrations = None
    last_absorption = None

    for iteration in range(p.n_iterations):
        # Build radii_m dict from normalized array
        for i, e in enumerate(edge_list):
            radii_m[e] = r_norm[i] * R0

        # --- Step 1: Solve flow ---
        try:
            result = solve_transmission_line(
                G, D=p.D, n_harmonics=p.n_harmonics, f0_hz=p.f0_hz,
                mu=p.mu, rho=p.rho_fluid,
                boundary_nodes=boundary_nodes, radii_m=radii_m,
                bc_harmonics_override=bc_harmonics_override, verbose=False)
        except Exception as ex:
            if verbose:
                print(f"  iter {iteration}: solve failed ({ex}), stopping")
            break

        # --- Step 2: Solve nutrients (if needed) ---
        if p.xi > 0 and p.mode in ('perfusion', 'mixed'):
            last_concentrations, last_absorption = solve_nutrients(
                G, result, edge_list, radii_m,
                xi=p.xi, rho_inlet=p.rho_inlet,
                boundary_nodes=boundary_nodes)

        # --- Step 3: Compute dR/dt ---
        dRdt = np.zeros(n_edges)
        total_diss = 0.0
        total_abs = 0.0

        if p.mode in ('sigma_Q', 'Q2', 'sigma_Q_local'):
            # Chatterjee & Katifori rule (signal = dissipation, or Q² for 'Q2')
            for i, e in enumerate(edge_list):
                Q_harm = result.edge_flows.get(e)
                if Q_harm is None:
                    continue

                if p.mode == 'Q2':
                    q_dc = Q_harm[0].real
                    Q2_avg = q_dc**2 + 0.5 * sum(
                        abs(Q_harm[k])**2 for k in range(1, p.n_harmonics + 1))
                    signal = Q2_avg / Q0**2
                else:  # sigma_Q or sigma_Q_local
                    diss_e = result.dissipation.get(e, 0.0)
                    total_diss += diss_e
                    signal = diss_e / diss0

                if signal < 1e-30:
                    dRdt[i] = -p.b * r_norm[i]
                else:
                    growth = p.a * signal**p.gamma / max(
                        r_norm[i]**(3*(1-p.gamma)), 1e-30)
                    decay = p.b * r_norm[i]
                    dRdt[i] = growth - decay

            if p.mode == 'Q2':
                total_diss = sum(result.dissipation.get(e, 0) for e in edge_list)

        elif p.mode == 'perfusion':
            # Gradient descent on H = Σ D_e + w_p·Σ(ΔJ_e - ⟨M⟩)² + w_m·Σ k_e^γ
            absorptions = last_absorption or {}
            abs_vals = np.array([absorptions.get(e, 0) for e in edge_list])
            mean_abs = np.mean(abs_vals) if len(abs_vals) > 0 else 0.0
            total_abs = np.sum(abs_vals)

            for i, e in enumerate(edge_list):
                R_m = r_norm[i] * R0
                L_m = lengths_m.get(e, 1e-6)
                diss_e = result.dissipation.get(e, 0.0)
                total_diss += diss_e

                # Conductance k = πR⁴/(8μL)
                k_e = np.pi * R_m**4 / (8 * p.mu * L_m) if L_m > 0 else 1e-30

                dR = max(p.dR_frac * R_m, 1e-9)

                # Dissipation gradient: ∂D_e/∂R ≈ -4·D_e/R (from R⁴ scaling)
                dH_dR = -4.0 * diss_e / max(R_m, 1e-12)

                # Material cost gradient: ∂(k^γ)/∂R = γ·k^(γ-1)·∂k/∂R
                # ∂k/∂R = 4πR³/(8μL)
                if p.w_material > 0:
                    dk_dR = 4 * np.pi * R_m**3 / (8 * p.mu * L_m) if L_m > 0 else 0
                    dH_dR += p.w_material * p.gamma * max(k_e, 1e-30)**(p.gamma - 1) * dk_dR

                # Perfusion gradient (numerical finite difference)
                if p.w_perfusion > 0 and p.xi > 0:
                    abs_e = absorptions.get(e, 0)
                    perf_cost = (abs_e - mean_abs)**2

                    radii_pert = dict(radii_m)
                    radii_pert[e] = R_m + dR
                    _, abs_pert = solve_nutrients(
                        G, result, edge_list, radii_pert,
                        xi=p.xi, rho_inlet=p.rho_inlet,
                        boundary_nodes=boundary_nodes)
                    abs_e_pert = abs_pert.get(e, 0)
                    perf_cost_pert = (abs_e_pert - mean_abs)**2

                    dperf_dR = (perf_cost_pert - perf_cost) / dR
                    dH_dR += p.w_perfusion * dperf_dR

                # δR = -ν · ∂H/∂R (gradient descent)
                dRdt[i] = -p.nu_step * dH_dR / R0

        elif p.mode == 'mixed':
            # sigma_Q Chatterjee rule + perfusion gradient
            absorptions = last_absorption or {}
            abs_vals = np.array([absorptions.get(e, 0) for e in edge_list])
            mean_abs = np.mean(abs_vals) if len(abs_vals) > 0 else 0.0
            total_abs = np.sum(abs_vals)

            for i, e in enumerate(edge_list):
                Q_harm = result.edge_flows.get(e)
                if Q_harm is None:
                    continue

                # Chatterjee dissipation signal
                diss_e = result.dissipation.get(e, 0.0)
                total_diss += diss_e
                signal = diss_e / diss0

                if signal < 1e-30:
                    dRdt[i] = -p.b * r_norm[i]
                else:
                    growth = p.a * signal**p.gamma / max(
                        r_norm[i]**(3*(1-p.gamma)), 1e-30)
                    decay = p.b * r_norm[i]
                    dRdt[i] = growth - decay

                # Add perfusion gradient
                if p.w_perfusion > 0 and p.xi > 0:
                    R_m = r_norm[i] * R0
                    dR = max(p.dR_frac * R_m, 1e-9)
                    abs_e = absorptions.get(e, 0)
                    perf_cost = (abs_e - mean_abs)**2

                    radii_pert = dict(radii_m)
                    radii_pert[e] = R_m + dR
                    _, abs_pert = solve_nutrients(
                        G, result, edge_list, radii_pert,
                        xi=p.xi, rho_inlet=p.rho_inlet,
                        boundary_nodes=boundary_nodes)
                    abs_e_pert = abs_pert.get(e, 0)
                    perf_cost_pert = (abs_e_pert - mean_abs)**2

                    dperf_dR = (perf_cost_pert - perf_cost) / dR
                    dRdt[i] += -p.w_perfusion * dperf_dR / R0

        else:
            raise ValueError(f"Unknown mode: {p.mode}")

        # --- Step 4: Forward Euler update with clamping and hard floor/ceiling ---
        for i in range(n_edges):
            step = p.dt * dRdt[i]
            max_step = 0.2 * r_norm[i]
            step = np.clip(step, -max_step, max_step)
            r_norm[i] = min(max(r_norm[i] + step, r_min_norm), r_max_norm)

        # --- Step 4b (sigma_Q_local only): wall-material diffusion + cons ---
        # V_e = r_norm² · L_e represents normalised wall material on edge
        # e.  Diffuse on the line graph: dV_e/dt = ν · Σ_{f~e} (V_f − V_e).
        # This bounds how fast a high-signal edge can grow by what its
        # neighbours can supply.  Optionally rescale to preserve Σ V.
        if p.mode == 'sigma_Q_local' and p.nu_diffusion > 0:
            V = r_norm ** 2 * L_norm
            # Edge-Laplacian: Δ_e = Σ_{f~e} V_f − deg_e · V_e
            laplacian = neigh_sum_indices_data @ V - neigh_deg * V
            V_new = V + p.dt * p.nu_diffusion * laplacian
            # Floor V at r_min_norm² · L_norm to prevent negatives.
            V_floor = (r_min_norm ** 2) * L_norm
            V_new = np.maximum(V_new, V_floor)
            if p.conserve_volume:
                V_total_post = float(V_new.sum())
                if V_total_post > 0:
                    V_new *= (V_initial_total / V_total_post)
            r_norm = np.sqrt(V_new / np.maximum(L_norm, 1e-30))
            r_norm = np.clip(r_norm, r_min_norm, r_max_norm)

        # Count edges at the floor
        at_floor = r_norm <= r_min_norm * 1.01
        active_mask = ~at_floor
        n_active = int(np.sum(active_mask))

        # Check for NaN
        if np.any(~np.isfinite(dRdt)):
            if verbose:
                n_nan = int(np.sum(~np.isfinite(dRdt)))
                print(f"  iter {iteration}: {n_nan} NaN in dR/dt — stopping")
            dRdt = np.nan_to_num(dRdt, nan=0.0)
            break

        max_dRdt = np.max(np.abs(dRdt[active_mask])) if n_active > 0 else 0.0

        # Record trajectory
        if iteration % p.record_every == 0 or iteration == p.n_iterations - 1:
            radii_records.append(r_norm.copy() * R0)
            diss_records.append(total_diss)
            dRdt_records.append(max_dRdt)
            iter_records.append(iteration)
            absorption_records.append(total_abs)
            entropy_records.append(_flow_entropy(result.mean_Q))

        if verbose and iteration % max(p.n_iterations // 20, 1) == 0:
            r_um = r_norm[r_norm > 0] * R0 * 1e6
            extra = ""
            if total_abs > 0:
                extra = f", abs={total_abs:.3e}"
            print(f"  iter {iteration:4d}: max|dR/dt|={max_dRdt:.2e}, "
                  f"diss={total_diss:.3e}, active={n_active}/{n_edges}, "
                  f"R=[{r_um.min():.1f}, {np.median(r_um):.1f}, "
                  f"{r_um.max():.1f}] µm{extra}")

        # Convergence
        if n_active > 0 and max_dRdt < p.convergence_tol:
            if verbose:
                print(f"  Converged at iteration {iteration} "
                      f"(max|dR/dt|={max_dRdt:.2e})")
            break

        if n_active < 0.01 * n_edges:
            if verbose:
                print(f"  iter {iteration}: only {n_active}/{n_edges} active — "
                      f"stopping (network collapsed)")
            break

    # --- Final solve ---
    for i, e in enumerate(edge_list):
        radii_m[e] = r_norm[i] * R0

    final_result = solve_transmission_line(
        G, D=p.D, n_harmonics=p.n_harmonics, f0_hz=p.f0_hz,
        mu=p.mu, rho=p.rho_fluid,
        boundary_nodes=boundary_nodes, radii_m=radii_m,
        bc_harmonics_override=bc_harmonics_override, verbose=verbose)

    # Final nutrient solve
    final_conc = None
    final_abs = None
    if p.xi > 0:
        final_conc, final_abs = solve_nutrients(
            G, final_result, edge_list, radii_m,
            xi=p.xi, rho_inlet=p.rho_inlet,
            boundary_nodes=boundary_nodes, verbose=verbose)

    mode_str = p.mode

    return AdaptationResult(
        initial_result=result_init,
        final_result=final_result,
        final_radii_m=dict(radii_m),
        radii_history=np.array(radii_records),
        dissipation_history=np.array(diss_records),
        max_dRdt_history=np.array(dRdt_records),
        iteration_indices=np.array(iter_records),
        mode=mode_str,
        gamma=p.gamma,
        a=p.a,
        b=p.b,
        dt=p.dt,
        n_iterations=len(iter_records),
        shunt_threshold_m=shunt_threshold_m,
        shunted_edges=[e for i, e in enumerate(edge_list)
                       if r_norm[i] <= r_min_norm * 1.01],
        edge_list=edge_list,
        absorption_history=np.array(absorption_records) if absorption_records else None,
        entropy_history=np.array(entropy_records) if entropy_records else None,
        final_concentrations=final_conc,
        final_absorption=final_abs,
        params=p,
    )


# =====================================================================
# Plotting
# =====================================================================

def plot_adaptation_result(
    G: nx.Graph,
    adapt: AdaptationResult,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 12),
):
    """Plot adaptation convergence, dissipation, radius trajectories, and network map."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from .config import PX_SIZE_UM

    iters = adapt.iteration_indices

    # Decide layout: 2x2 or 2x3 if we have absorption data
    has_nutrients = (adapt.absorption_history is not None
                     and len(adapt.absorption_history) > 0
                     and np.any(adapt.absorption_history > 0))
    ncols = 3 if has_nutrients else 2
    fig, axes = plt.subplots(2, ncols, figsize=(figsize[0] * ncols / 2, figsize[1]))

    # --- Panel 1: Convergence ---
    ax = axes[0, 0]
    ax.semilogy(iters, adapt.max_dRdt_history, 'k-', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('max |dR/dt|')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Total dissipation ---
    ax = axes[0, 1]
    ax.plot(iters, adapt.dissipation_history, 'r-', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total dissipation (W)')
    ax.set_title('Network dissipation')
    ax.grid(True, alpha=0.3)

    # --- Panel 3 (optional): Absorption + entropy ---
    if has_nutrients:
        ax = axes[0, 2]
        ax.plot(iters, adapt.absorption_history, 'g-', linewidth=1, label='Total absorption')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total absorption')
        ax.set_title('Nutrient absorption')
        ax.grid(True, alpha=0.3)
        if adapt.entropy_history is not None:
            ax2 = ax.twinx()
            ax2.plot(iters, adapt.entropy_history, 'b--', linewidth=0.8, label='Flow entropy')
            ax2.set_ylabel('Flow entropy', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

    # --- Panel 4: Radius trajectories ---
    ax = axes[1, 0]
    n_records, n_edges = adapt.radii_history.shape
    R_um = adapt.radii_history * 1e6
    R_final = R_um[-1, :]
    R_max = np.percentile(R_final[R_final > 0], 95) if np.any(R_final > 0) else 1.0
    for j in range(n_edges):
        if R_final[j] <= 0:
            ax.plot(iters, R_um[:, j], 'r--', linewidth=0.3, alpha=0.3)
        else:
            c = plt.cm.viridis(R_final[j] / R_max)
            ax.plot(iters, R_um[:, j], color=c, linewidth=0.3, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Radius (µm)')
    ax.set_title(f'Radius trajectories ({n_edges} edges, '
                 f'{len(adapt.shunted_edges)} at floor)')
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Network colored by R_final/R_initial ---
    ax = axes[1, 1]
    ratio = np.zeros(n_edges)
    for j in range(n_edges):
        r0 = adapt.radii_history[0, j]
        rf = adapt.radii_history[-1, j]
        if r0 > 0:
            ratio[j] = rf / r0

    segments = []
    seg_ratios = []
    seg_widths = []
    for j, (u, v) in enumerate(adapt.edge_list):
        if ratio[j] <= 0:
            continue
        data = G.edges[u, v]
        path = data.get('path')
        if path is not None and len(path) > 1:
            pts = np.array(path)
            seg = pts[:, ::-1] if pts.shape[1] == 2 else pts
        else:
            pos_u = G.nodes[u].get('pos', G.nodes[u].get('o'))
            pos_v = G.nodes[v].get('pos', G.nodes[v].get('o'))
            if pos_u is None or pos_v is None:
                continue
            seg = np.array([[pos_u[1], pos_u[0]], [pos_v[1], pos_v[0]]])

        points = seg.reshape(-1, 2)
        if len(points) < 2:
            continue
        segs = np.concatenate([points[:-1, np.newaxis, :],
                               points[1:, np.newaxis, :]], axis=1)
        segments.extend(segs)
        seg_ratios.extend([ratio[j]] * len(segs))
        seg_widths.extend([R_final[j]] * len(segs))

    if segments:
        seg_ratios = np.array(seg_ratios)
        seg_widths = np.array(seg_widths)

        w_min, w_max = 0.3, 6.0
        r_max_um = seg_widths.max() if seg_widths.max() > 0 else 1.0
        lw = w_min + (w_max - w_min) * seg_widths / r_max_um

        vmin_r = 0.0
        vmax_r = min(np.percentile(seg_ratios, 95), 3.0)
        if vmax_r <= 1.0:
            norm = Normalize(vmin=vmin_r, vmax=max(vmax_r, 0.01))
        else:
            norm = TwoSlopeNorm(vmin=vmin_r, vcenter=1.0, vmax=vmax_r)
        lc = LineCollection(segments, cmap='RdBu_r', norm=norm, linewidths=lw)
        lc.set_array(seg_ratios)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, shrink=0.8, label='R_final / R_initial')

        all_pts = np.concatenate([s.reshape(-1, 2) for s in segments])
        margin = 50
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].max() + margin, all_pts[:, 1].min() - margin)

    for bn in adapt.final_result.boundary_nodes:
        pos = G.nodes[bn].get('pos', G.nodes[bn].get('o'))
        if pos is not None:
            pos = np.array(pos)
            ax.plot(pos[1], pos[0], 'r*', ms=12, zorder=5)

    ax.set_aspect('equal')
    ax.set_title(f'Radius change (mode={adapt.mode}, γ={adapt.gamma:.2f})')

    # --- Panel 6 (optional): Absorption map ---
    if has_nutrients and adapt.final_absorption:
        ax = axes[1, 2]
        seg_abs = []
        seg_list = []
        for j, (u, v) in enumerate(adapt.edge_list):
            abs_val = adapt.final_absorption.get((u, v), 0)
            data = G.edges[u, v]
            path = data.get('path')
            if path is not None and len(path) > 1:
                pts = np.array(path)
                seg = pts[:, ::-1] if pts.shape[1] == 2 else pts
            else:
                pos_u = G.nodes[u].get('pos', G.nodes[u].get('o'))
                pos_v = G.nodes[v].get('pos', G.nodes[v].get('o'))
                if pos_u is None or pos_v is None:
                    continue
                seg = np.array([[pos_u[1], pos_u[0]], [pos_v[1], pos_v[0]]])
            points = seg.reshape(-1, 2)
            if len(points) < 2:
                continue
            segs = np.concatenate([points[:-1, np.newaxis, :],
                                   points[1:, np.newaxis, :]], axis=1)
            seg_list.extend(segs)
            seg_abs.extend([abs_val] * len(segs))

        if seg_list:
            seg_abs = np.array(seg_abs)
            vmax_a = np.percentile(seg_abs[seg_abs > 0], 95) if np.any(seg_abs > 0) else 1
            lc2 = LineCollection(seg_list, cmap='YlOrRd',
                                 norm=Normalize(vmin=0, vmax=max(vmax_a, 1e-30)),
                                 linewidths=1.5)
            lc2.set_array(seg_abs)
            ax.add_collection(lc2)
            plt.colorbar(lc2, ax=ax, shrink=0.8, label='Absorption ΔJ')

            all_pts = np.concatenate([s.reshape(-1, 2) for s in seg_list])
            margin = 50
            ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
            ax.set_ylim(all_pts[:, 1].max() + margin, all_pts[:, 1].min() - margin)

        ax.set_aspect('equal')
        ax.set_title('Final nutrient absorption')

    fig.suptitle(
        f'Adaptation: {adapt.mode}, γ={adapt.gamma:.2f}, '
        f'a={adapt.a}, b={adapt.b}, dt={adapt.dt}, '
        f'{adapt.n_iterations} iters, {len(adapt.shunted_edges)} at floor',
        fontsize=11, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig
