"""Coupled nonlinear-heart + linear-plexus time-domain simulator.

Wraps any networkx plexus graph G (with ``boundary_type ∈ {'source','sink'}``
on its boundary nodes and ``radius`` / ``length`` edge attrs) with a ghost
heart subsystem:

    venous sink(s) → SV → A ── AV valve ── V ── outflow valve ── DA → arterial source(s)
                                                                          │
                                                                          └──► plexus ─── venous sink

Ghost elements (not in G):
  - A, V: two chamber nodes with time-varying (double-Hill) elastance.
    Pressure is algebraic: p = E(t)·(V − V0).
  - AV valve: state-based (Mynard & Smolich 2012).  A state variable
    ζ ∈ [0,1] tracks valve openness; flow Q = G_max·ζ·Δp.  Opens under
    positive forward Δp at rate K_vo, closes under negative Δp at K_vc.
    ζ evolves on a cardiac-cycle-ish time scale, smoothing the diode
    discontinuity cleanly (no stiffness penalty).
  - outflow valve: same state-based model at each V→source conduit,
    one ζ_out per source node.
  - DA conduit: linear resistor in series with the outflow valve
    (effective G combined with valve G_max).
  - SV return: linear resistor per venous sink (no valve — embryonic
    venous system is valveless; Re ≪ 1 anyway).

Plexus compliance C_wall at each node is *kept* as a dynamical element
because the distributed wall compliance is what low-pass-filters
pulsatility across the network — that physics is central to the paper.

State vector (dim = 2 + 1 + N_sources + N_plexus):
    [V_A, V_V,
     ζ_AV,
     ζ_out_1, ζ_out_2, ..., ζ_out_Ns,
     p_plex_0, p_plex_1, ...,  p_plex_{Np-1}]

In ``single_chamber=True`` mode, ζ_AV is clamped to 1 and E_A(t) ≡ E_V(t),
collapsing the two chambers into one effective compartment without
changing the state-vector layout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix, eye as _sp_eye
from scipy.sparse.linalg import spsolve


# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------


@dataclass
class HeartParams:
    """Heart + conduit parameters (dimensionless or loosely physiological).

    Two elastance families supported:
      * half-sin² (legacy, ``use_double_hill=False``)
      * double-Hill (Mynard & Smolich 2015 Eq. T9–T11, default)

    Valve model:
      * state-based (Mynard et al. 2012): ζ ∈ [0,1] with opening /
        closing rates K_vo, K_vc.
    """

    # ---- Cycle -------------------------------------------------------
    T: float = 1.0
    d: float = 0.3                         # legacy sin² duty fraction
    phi: float = 0.15                      # AV delay (fraction of T)

    # ---- Elastance (shared min/max) ----------------------------------
    E_A_min: float = 0.1
    E_A_max: float = 1.0
    E_V_min: float = 0.05
    E_V_max: float = 5.0

    # ---- Double-Hill shape parameters (per chamber) ------------------
    # Defaults from Mynard & Smolich 2015 Table 3, scaled to T.  Users
    # may override for embryonic-specific fits.
    use_double_hill: bool = True
    # Ventricle: tau_1 ≈ 0.27·T, m_1 ≈ 1.3 (contraction); tau_2 ≈ 0.45·T,
    # m_2 ≈ 27 (steep relaxation).
    tau_1_V_frac: float = 0.27
    m_1_V: float = 1.32
    tau_2_V_frac: float = 0.45
    m_2_V: float = 27.4
    # Atrium: shorter, earlier contraction.
    tau_1_A_frac: float = 0.11
    m_1_A: float = 1.32
    tau_2_A_frac: float = 0.18
    m_2_A: float = 13.1

    # Onset times (absolute, in T units). Atrium fires first; ventricle
    # at t = phi·T after atrial onset.
    t_onset_A: float = 0.0
    t_onset_V: float = 0.15

    # ---- Unstressed chamber volumes --------------------------------
    V_A0: float = 0.0
    V_V0: float = 0.0

    # ---- Conductances ----------------------------------------------
    G_max_AV: float = 10.0                 # fully-open AV valve conductance
    G_max_out: float = 10.0                # fully-open outflow conductance
    G_DA: float = 5.0                      # DA series conduit (per source)
    G_SV: float = 10.0                     # SV linear return (per sink)

    # ---- Valve state dynamics (Mynard et al. 2012) -----------------
    # Opening / closing rate constants [1/(pressure · time)].
    # Larger K_vo, K_vc → snappier valves.  K_vc usually larger than K_vo
    # (closure faster than opening) — defaults ~Mynard 2012 for adult.
    K_vo_AV: float = 30.0
    K_vc_AV: float = 30.0
    K_vo_out: float = 30.0
    K_vc_out: float = 30.0

    # ---- Plexus compliance -----------------------------------------
    C_wall: float = 0.01

    # ---- Initial chamber state -------------------------------------
    V_A_init: float = 1.0
    V_V_init: float = 1.0

    # ---- Topology mode --------------------------------------------
    # single_chamber: collapse atrium + ventricle into one compartment
    #   (ζ_AV held at 1, E_A ≡ E_V). Useful as a sanity-check to see
    #   whether two chambers are needed.
    single_chamber: bool = False

    # ---- Solver knobs ---------------------------------------------
    n_cycles: int = 15
    rtol: float = 1e-4
    atol: float = 1e-7
    max_step: float = 0.02


# -----------------------------------------------------------------------
# Elastance functions
# -----------------------------------------------------------------------


def _double_hill_unnormalized(tau_rel, tau_1, m_1, tau_2, m_2):
    """f(τ) = g1/(1+g1) · 1/(1+g2) where g_i = (τ/τ_i)^m_i.

    τ_rel: time since onset, already wrapped to [0, T).
    Returns scalar or ndarray matching input shape.
    """
    tr = np.clip(tau_rel, 1e-12, None)
    g1 = (tr / tau_1) ** m_1
    g2 = (tr / tau_2) ** m_2
    return (g1 / (1.0 + g1)) * (1.0 / (1.0 + g2))


def _compute_double_hill_k(tau_1, m_1, tau_2, m_2, T):
    """Precompute normalization k so the peak of the shape function
    equals 1 over one cycle. Scan a fine grid."""
    tau_grid = np.linspace(1e-6, T, 2000)
    f = _double_hill_unnormalized(tau_grid, tau_1, m_1, tau_2, m_2)
    fmax = float(np.max(f))
    return 1.0 / fmax if fmax > 0 else 1.0


def _double_hill_elastance(t, t_onset, T, E_min, E_max,
                            tau_1, m_1, tau_2, m_2, k_norm):
    """Mynard–Smolich double-Hill: E(t) = E_min + (E_max−E_min)·k·f(τ_rel)
    where τ_rel = (t − t_onset) mod T."""
    tau_rel = (t - t_onset) % T
    f = _double_hill_unnormalized(tau_rel, tau_1, m_1, tau_2, m_2)
    return E_min + (E_max - E_min) * k_norm * f


def _sin2_elastance(t, t0, E_min, E_max, d, T):
    """Legacy half-sin² (used when use_double_hill=False)."""
    tau = (t - t0) % T
    active = tau < d * T
    phase = np.pi * tau / (d * T)
    return np.where(active,
                    E_min + (E_max - E_min) * np.sin(phase) ** 2,
                    E_min)


# -----------------------------------------------------------------------
# Plexus conductance helper
# -----------------------------------------------------------------------

MU_DEFAULT = 3.5e-3  # Pa·s


def _edge_conductance(G: nx.Graph, u, v, mu: float,
                      radii_override: Optional[Dict] = None) -> float:
    """Poiseuille G = πR⁴/(8μL)."""
    d = G.edges[u, v]
    if radii_override is not None:
        R = radii_override.get((u, v), radii_override.get((v, u)))
        if R is None:
            R = d.get('radius')
    else:
        R = d.get('radius')
    L = d.get('length', d.get('length_true'))
    if R is None or L is None or not np.isfinite(R) or not np.isfinite(L):
        return 0.0
    if R <= 0 or L <= 0:
        return 0.0
    return float(np.pi * R ** 4 / (8.0 * mu * L))


# -----------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------


@dataclass
class CoupledResult:
    t: np.ndarray
    V_A: np.ndarray
    V_V: np.ndarray
    p_A: np.ndarray
    p_V: np.ndarray
    zeta_AV: np.ndarray
    zeta_out: Dict[int, np.ndarray]       # source node -> ζ trace
    node_p: Dict[int, np.ndarray]
    edge_Q: Dict[Tuple[int, int], np.ndarray]
    valve_Q: Dict[str, np.ndarray]
    params: HeartParams
    source_nodes: List[int]
    sink_nodes: List[int]
    conservation: Dict[str, float] = field(default_factory=dict)

    def last_cycle_slice(self) -> slice:
        t_end = self.t[-1]
        i0 = int(np.searchsorted(self.t, t_end - self.params.T))
        return slice(i0, None)


# -----------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------


class HeartCoupledSim:
    """Heart + plexus coupled ODE integrator with Mynard-style elastance
    and valve-state dynamics."""

    def __init__(self, G: nx.Graph, params: HeartParams = None,
                 mu: float = MU_DEFAULT,
                 radii_override: Optional[Dict] = None):
        self.G = G
        self.params = params or HeartParams()
        self.mu = mu
        self.radii_override = radii_override

        # -------- Boundary classification --------
        self.source_nodes = sorted(n for n, d in G.nodes(data=True)
                                    if d.get('boundary_type') == 'source')
        self.sink_nodes = sorted(n for n, d in G.nodes(data=True)
                                  if d.get('boundary_type') == 'sink')
        if not self.source_nodes:
            raise ValueError("No source boundary nodes in G.")
        if not self.sink_nodes:
            raise ValueError("No sink boundary nodes in G.")
        self.N_src = len(self.source_nodes)

        # -------- Plexus indexing --------
        self.plexus_nodes = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.plexus_nodes)}
        self.N_plexus = len(self.plexus_nodes)

        # -------- Edge conductances --------
        self.edge_list = list(G.edges())
        self.edge_G = np.array([_edge_conductance(G, u, v, self.mu,
                                                    self.radii_override)
                                for u, v in self.edge_list], dtype=float)
        self._iu = np.array([self.node_to_idx[u]
                             for u, _ in self.edge_list], dtype=np.int64)
        self._iv = np.array([self.node_to_idx[v]
                             for _, v in self.edge_list], dtype=np.int64)

        # -------- Sparse Laplacian --------
        rows = np.concatenate([self._iu, self._iv, self._iu, self._iv])
        cols = np.concatenate([self._iv, self._iu, self._iu, self._iv])
        data = np.concatenate([-self.edge_G, -self.edge_G,
                                self.edge_G, self.edge_G])
        self._L = csr_matrix((data, (rows, cols)),
                              shape=(self.N_plexus, self.N_plexus))

        self._src_idx = np.array([self.node_to_idx[n]
                                   for n in self.source_nodes],
                                  dtype=np.int64)
        self._snk_idx = np.array([self.node_to_idx[n]
                                   for n in self.sink_nodes],
                                  dtype=np.int64)

        # -------- State-vector layout --------
        # [V_A, V_V, ζ_AV, ζ_out_1, ..., ζ_out_Ns, p_plex_0, ...]
        self.IDX_VA = 0
        self.IDX_VV = 1
        self.IDX_ZAV = 2
        self.IDX_ZOUT = 3                    # ζ_out_1 starts here
        self.IDX_P = 3 + self.N_src          # plexus pressures start here
        self.N_states = self.IDX_P + self.N_plexus

        # -------- Double-Hill normalization --------
        p = self.params
        T = p.T
        self._tau_1_A = p.tau_1_A_frac * T
        self._tau_2_A = p.tau_2_A_frac * T
        self._tau_1_V = p.tau_1_V_frac * T
        self._tau_2_V = p.tau_2_V_frac * T
        if p.use_double_hill:
            self._k_A = _compute_double_hill_k(
                self._tau_1_A, p.m_1_A, self._tau_2_A, p.m_2_A, T)
            self._k_V = _compute_double_hill_k(
                self._tau_1_V, p.m_1_V, self._tau_2_V, p.m_2_V, T)
        else:
            self._k_A = self._k_V = 1.0

        # Profiling
        self._rhs_calls = 0
        self._progress_every = None

    # ------------------------------------------------------------------

    def _E_A(self, t):
        p = self.params
        if p.single_chamber:
            return self._E_V(t)
        if p.use_double_hill:
            return _double_hill_elastance(
                t, p.t_onset_A, p.T, p.E_A_min, p.E_A_max,
                self._tau_1_A, p.m_1_A, self._tau_2_A, p.m_2_A, self._k_A)
        return _sin2_elastance(t, p.t_onset_A,
                                p.E_A_min, p.E_A_max, p.d, p.T)

    def _E_V(self, t):
        p = self.params
        if p.use_double_hill:
            return _double_hill_elastance(
                t, p.t_onset_V, p.T, p.E_V_min, p.E_V_max,
                self._tau_1_V, p.m_1_V, self._tau_2_V, p.m_2_V, self._k_V)
        return _sin2_elastance(t, p.t_onset_V,
                                p.E_V_min, p.E_V_max, p.d, p.T)

    def _chamber_pressures(self, t, y):
        p = self.params
        V_A = y[self.IDX_VA]
        V_V = y[self.IDX_VV]
        pA = float(self._E_A(t)) * (V_A - p.V_A0)
        pV = float(self._E_V(t)) * (V_V - p.V_V0)
        return pA, pV

    def _G_art_series(self):
        """Series combo of fully-open outflow valve + DA conduit."""
        p = self.params
        return 1.0 / (1.0 / p.G_max_out + 1.0 / p.G_DA)

    # ------------------------------------------------------------------

    def rhs(self, t, y):
        p = self.params
        pA, pV = self._chamber_pressures(t, y)
        p_plex = y[self.IDX_P:]

        # Valve states (clipped into [0,1] for safety; ODE keeps them
        # bounded but numerical noise can nudge them slightly outside).
        zAV = np.clip(y[self.IDX_ZAV], 0.0, 1.0)
        zout = np.clip(y[self.IDX_ZOUT:self.IDX_ZOUT + self.N_src],
                       0.0, 1.0)
        if p.single_chamber:
            zAV = 1.0   # atrium = ventricle, valve always open

        dy = np.zeros(self.N_states)
        self._rhs_calls += 1

        # ---- Plexus internal flows (sparse matvec) ----
        dQ_net = -(self._L @ p_plex)

        # ---- Outflow valves: V → (valve) → source ----
        G_art_series = self._G_art_series()
        dp_out = pV - p_plex[self._src_idx]                  # per-source Δp
        Q_V_to_src = G_art_series * zout * dp_out            # signed; ζ∈[0,1]
        np.add.at(dQ_net, self._src_idx, Q_V_to_src)

        # ---- Venous returns (linear, no valve) ----
        G_ven = p.G_SV
        dp_ven = p_plex[self._snk_idx] - pA
        Q_sink_to_A = G_ven * dp_ven
        np.add.at(dQ_net, self._snk_idx, -Q_sink_to_A)

        # ---- Plexus pressure ODE ----
        dy[self.IDX_P:] = dQ_net / p.C_wall

        # ---- AV valve flow ----
        dp_AV = pA - pV
        Q_AV = p.G_max_AV * zAV * dp_AV

        # ---- Chamber volume ODEs ----
        dy[self.IDX_VA] = float(Q_sink_to_A.sum()) - Q_AV
        dy[self.IDX_VV] = Q_AV - float(Q_V_to_src.sum())

        # ---- Valve-state ODEs (Mynard et al. 2012) ----
        # dζ/dt = K_vo·(1−ζ)·Δp  if Δp > 0   (opening)
        #       = K_vc·ζ·Δp       if Δp ≤ 0   (closing)
        if not p.single_chamber:
            dy[self.IDX_ZAV] = (
                p.K_vo_AV * (1.0 - zAV) * dp_AV if dp_AV > 0
                else p.K_vc_AV * zAV * dp_AV
            )
        else:
            dy[self.IDX_ZAV] = 0.0
        # Outflow valves (vectorised over sources)
        dz_out = np.where(
            dp_out > 0,
            p.K_vo_out * (1.0 - zout) * dp_out,
            p.K_vc_out * zout * dp_out,
        )
        dy[self.IDX_ZOUT:self.IDX_ZOUT + self.N_src] = dz_out

        # ---- Progress ticker ----
        if self._progress_every and self._rhs_calls % self._progress_every == 0:
            elapsed = time.perf_counter() - self._t_start
            print(f"    rhs calls: {self._rhs_calls}, "
                  f"t_sim: {t:.3f} / {self._t_end:.3f}, "
                  f"wall: {elapsed:.1f}s")
        return dy

    # ------------------------------------------------------------------

    def _precondition(self, y0):
        """Seed plexus pressures with the DC quasi-static solve at
        heart-mean pressures, and valve states at their linear-regime
        equilibrium. Skips the long capacitive/valve transient."""
        p = self.params

        # Mean chamber pressures (time-average of E·(V−V0); use peak/2).
        E_A_mean = 0.5 * (p.E_A_min + p.E_A_max)
        E_V_mean = 0.5 * (p.E_V_min + p.E_V_max)
        pA_mean = E_A_mean * (p.V_A_init - p.V_A0)
        pV_mean = E_V_mean * (p.V_V_init - p.V_V0)

        # DC solve: treat outflow as always-open (ζ=1) at mean Δp.
        G_art_series = self._G_art_series()
        G_ven = p.G_SV
        L = self._L.copy().tolil()
        b = np.zeros(self.N_plexus)
        for i in self._src_idx:
            L[i, i] += G_art_series
            b[i] += G_art_series * pV_mean
        for i in self._snk_idx:
            L[i, i] += G_ven
            b[i] += G_ven * pA_mean
        diag = np.abs(L.diagonal())
        lam = 1e-8 * float(diag.max() if diag.size else 1.0)
        L = L.tocsr() + _sp_eye(self.N_plexus, format='csr') * lam
        try:
            p_plex_ss = spsolve(L, b)
            y0[self.IDX_P:] = p_plex_ss
        except Exception as _e:
            print(f"    preconditioner failed ({_e}); plexus starts at 0")

        # Quasi-steady valve ICs: set each ζ to 1 if the initial Δp
        # favours opening, 0 otherwise. Avoids the ~1/K_vo transient
        # that 0.5 ICs would spend settling to the physical state.
        p_plex_ss = y0[self.IDX_P:]
        zav_ic = 1.0 if pA_mean > pV_mean else 0.0
        y0[self.IDX_ZAV] = zav_ic
        for k, i in enumerate(self._src_idx):
            dp_init = pV_mean - p_plex_ss[i]
            y0[self.IDX_ZOUT + k] = 1.0 if dp_init > 0 else 0.0
        if p.single_chamber:
            y0[self.IDX_ZAV] = 1.0

    # ------------------------------------------------------------------

    def _build_jac_sparsity(self):
        N = self.N_states
        jac = lil_matrix((N, N), dtype=float)

        # ---- Chamber ODEs ----
        jac[self.IDX_VA, self.IDX_VA] = 1
        jac[self.IDX_VA, self.IDX_VV] = 1
        jac[self.IDX_VA, self.IDX_ZAV] = 1
        jac[self.IDX_VV, self.IDX_VA] = 1
        jac[self.IDX_VV, self.IDX_VV] = 1
        jac[self.IDX_VV, self.IDX_ZAV] = 1
        for k in range(self.N_src):
            jac[self.IDX_VV, self.IDX_ZOUT + k] = 1
            jac[self.IDX_VV, self.IDX_P + self._src_idx[k]] = 1
        for i in self._snk_idx:
            jac[self.IDX_VA, self.IDX_P + i] = 1

        # ---- AV valve ODE ----
        jac[self.IDX_ZAV, self.IDX_ZAV] = 1
        jac[self.IDX_ZAV, self.IDX_VA] = 1
        jac[self.IDX_ZAV, self.IDX_VV] = 1

        # ---- Outflow valve ODEs ----
        for k in range(self.N_src):
            row = self.IDX_ZOUT + k
            jac[row, row] = 1
            jac[row, self.IDX_VV] = 1
            jac[row, self.IDX_P + self._src_idx[k]] = 1

        # ---- Plexus rows ----
        for iu, iv in zip(self._iu, self._iv):
            jac[self.IDX_P + iu, self.IDX_P + iu] = 1
            jac[self.IDX_P + iv, self.IDX_P + iv] = 1
            jac[self.IDX_P + iu, self.IDX_P + iv] = 1
            jac[self.IDX_P + iv, self.IDX_P + iu] = 1
        for k, i in enumerate(self._src_idx):
            jac[self.IDX_P + i, self.IDX_VV] = 1
            jac[self.IDX_P + i, self.IDX_ZOUT + k] = 1
            jac[self.IDX_P + i, self.IDX_P + i] = 1
        for i in self._snk_idx:
            jac[self.IDX_P + i, self.IDX_VA] = 1
            jac[self.IDX_P + i, self.IDX_P + i] = 1
        return jac.tocsr()

    # ------------------------------------------------------------------

    def run(self, n_cycles: Optional[int] = None,
            verbose: bool = True) -> CoupledResult:
        p = self.params
        n_cyc = n_cycles or p.n_cycles

        y0 = np.zeros(self.N_states)
        y0[self.IDX_VA] = p.V_A_init
        y0[self.IDX_VV] = p.V_V_init
        self._precondition(y0)

        t_end = n_cyc * p.T
        n_samples = int(400 * n_cyc) + 1
        t_eval = np.linspace(0.0, t_end, n_samples)

        if verbose:
            print(f"  HeartCoupledSim: {self.N_plexus} plexus nodes, "
                  f"{len(self.edge_list)} edges, "
                  f"{self.N_src} sources, {len(self.sink_nodes)} sinks "
                  f"→ {self.N_states} ODE states")
            mode = ('single-chamber' if p.single_chamber
                    else 'two-chamber')
            elastance = ('double-Hill' if p.use_double_hill
                         else 'half-sin²')
            print(f"  Mode: {mode}, elastance: {elastance}, "
                  f"integrating {n_cyc} cycles (T={p.T}, "
                  f"{t_eval.size} samples)")

        self._rhs_calls = 0
        self._t_start = time.perf_counter()
        self._t_end = t_end
        expected = max(200, int(t_end / max(p.max_step, 1e-6)) * 6)
        self._progress_every = max(expected // 20, 500) if verbose else None

        jac_sparsity = self._build_jac_sparsity()
        sol = solve_ivp(
            lambda t, y: self.rhs(t, y),
            (0.0, t_end), y0,
            method='BDF', t_eval=t_eval,
            rtol=p.rtol, atol=p.atol, max_step=p.max_step,
            jac_sparsity=jac_sparsity,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")

        wall = time.perf_counter() - self._t_start
        if verbose:
            print(f"  Solve: {wall:.2f}s wall, "
                  f"{self._rhs_calls} RHS calls "
                  f"({1e6 * wall / max(self._rhs_calls, 1):.1f} µs/call)")

        # -------- Reconstruct results --------
        V_A = sol.y[self.IDX_VA, :]
        V_V = sol.y[self.IDX_VV, :]
        zeta_AV = sol.y[self.IDX_ZAV, :]
        zeta_out = {sn: sol.y[self.IDX_ZOUT + k, :]
                    for k, sn in enumerate(self.source_nodes)}

        pA_t = np.empty_like(sol.t)
        pV_t = np.empty_like(sol.t)
        for i, ti in enumerate(sol.t):
            pA_t[i], pV_t[i] = self._chamber_pressures(ti, sol.y[:, i])

        node_p = {n: sol.y[self.IDX_P + self.node_to_idx[n], :]
                  for n in self.plexus_nodes}
        edge_Q = {}
        for k, (u, v) in enumerate(self.edge_list):
            edge_Q[(u, v)] = self.edge_G[k] * (node_p[u] - node_p[v])

        G_art_series = self._G_art_series()
        # Valve flows are Q = G_max · ζ · Δp (signed).
        valve_Q = {'AV': p.G_max_AV * zeta_AV * (pA_t - pV_t)}
        for sn in self.source_nodes:
            dp = pV_t - node_p[sn]
            valve_Q[f'V_{sn}'] = G_art_series * zeta_out[sn] * dp
        for kn in self.sink_nodes:
            valve_Q[f'{kn}_A'] = p.G_SV * (node_p[kn] - pA_t)

        # -------- Conservation diagnostic --------
        T_cyc = p.T
        mask = sol.t >= t_end - T_cyc - 1e-12
        t_last = sol.t[mask]
        T_last = t_last[-1] - t_last[0]

        def _mean(x):
            return float(np.trapz(x[mask], t_last) / T_last)

        Q_AV_mean = _mean(valve_Q['AV'])
        Q_out_mean = sum(_mean(valve_Q[f'V_{sn}'])
                          for sn in self.source_nodes)
        Q_ret_mean = sum(_mean(valve_Q[f'{kn}_A'])
                          for kn in self.sink_nodes)
        mean_ref = max(abs(Q_AV_mean), abs(Q_out_mean), abs(Q_ret_mean),
                        1e-30)
        # ---- Valve timing: cycle phase (in [0, 1)) where ζ crosses 0.5 ----
        def _phase_crossings(zeta_trace):
            """Return (open_phase, close_phase) within the last cycle,
            computed as the cycle-relative time where ζ crosses 0.5
            upward (opening) or downward (closing). NaN if no crossing."""
            z = zeta_trace[mask]
            above = z > 0.5
            op = np.nan
            cl = np.nan
            for i in range(1, len(z)):
                if not above[i - 1] and above[i]:
                    op = (t_last[i] - t_last[0]) / T_last
                    break
            for i in range(1, len(z)):
                if above[i - 1] and not above[i]:
                    cl = (t_last[i] - t_last[0]) / T_last
                    break
            return float(op), float(cl)

        av_open, av_close = _phase_crossings(zeta_AV)
        out_phases = {sn: _phase_crossings(zeta_out[sn])
                      for sn in self.source_nodes}

        # ---- Energy balance per cycle ----
        # W_heart   = ∫ p_V · Σ(Q_V→src) dt  (work delivered by ventricle)
        # W_heart_A = ∫ p_A · Σ(Q_v→A) dt    (work by atrium reclaiming vol)
        # W_plex    = ∫ Σ_edges G_e (p_u−p_v)² dt  (viscous dissipation)
        # W_valve   = ∫ (G_max·ζ) · Δp²  for each valve
        t_last_arr = t_last
        # Ventricular output work
        Q_V_total_t = np.zeros_like(t_last_arr)
        for sn in self.source_nodes:
            Q_V_total_t += valve_Q[f'V_{sn}'][mask]
        W_heart_V = float(np.trapz(pV_t[mask] * Q_V_total_t, t_last_arr))
        # Atrial work (recovery)
        Q_A_total_t = np.zeros_like(t_last_arr)
        for kn in self.sink_nodes:
            Q_A_total_t += valve_Q[f'{kn}_A'][mask]
        W_heart_A = float(np.trapz(pA_t[mask] * Q_A_total_t, t_last_arr))
        # Viscous dissipation in plexus = Σ_e G_e (p_u−p_v)² integrated
        W_plex_diss = 0.0
        for k_e, (u, v) in enumerate(self.edge_list):
            dp = (node_p[u] - node_p[v])[mask]
            W_plex_diss += float(
                self.edge_G[k_e] * np.trapz(dp * dp, t_last_arr))
        # Valve dissipation = ∫ Q · Δp (signed; should be ≥ 0 for diodes)
        W_AV_diss = float(np.trapz(valve_Q['AV'][mask] *
                                     (pA_t - pV_t)[mask], t_last_arr))
        W_out_diss = sum(
            float(np.trapz(valve_Q[f'V_{sn}'][mask] *
                           (pV_t - node_p[sn])[mask], t_last_arr))
            for sn in self.source_nodes)
        W_SV_diss = sum(
            float(np.trapz(valve_Q[f'{kn}_A'][mask] *
                           (node_p[kn] - pA_t)[mask], t_last_arr))
            for kn in self.sink_nodes)
        W_in  = W_heart_V + W_heart_A
        W_out = W_plex_diss + W_AV_diss + W_out_diss + W_SV_diss
        W_ref = max(abs(W_in), abs(W_out), 1e-30)
        energy_residual = abs(W_in - W_out) / W_ref

        residuals = {
            'Q_AV':   Q_AV_mean,
            'Q_V→src': Q_out_mean,
            'Q_sink→A': Q_ret_mean,
            'rel_err_AV_vs_out': abs(Q_AV_mean - Q_out_mean) / mean_ref,
            'rel_err_out_vs_ret': abs(Q_out_mean - Q_ret_mean) / mean_ref,
            'V_A_periodicity': abs(V_A[mask][-1] - V_A[mask][0])
                               / max(abs(V_A[mask]).max(), 1e-30),
            'V_V_periodicity': abs(V_V[mask][-1] - V_V[mask][0])
                               / max(abs(V_V[mask]).max(), 1e-30),
            'stroke_V':        float(V_V[mask].max() - V_V[mask].min()),
            # Valve timing (fractions of the cycle, 0..1)
            'phase_AV_open':   av_open,
            'phase_AV_close':  av_close,
            'phase_out_open':  {sn: p[0] for sn, p in out_phases.items()},
            'phase_out_close': {sn: p[1] for sn, p in out_phases.items()},
            # Energy balance
            'W_heart_V':       W_heart_V,
            'W_heart_A':       W_heart_A,
            'W_plex_diss':     W_plex_diss,
            'W_AV_diss':       W_AV_diss,
            'W_out_diss':      W_out_diss,
            'W_SV_diss':       W_SV_diss,
            'energy_in':       W_in,
            'energy_out':      W_out,
            'energy_residual': energy_residual,
        }

        if verbose:
            print(f"  Mean Q:  AV={Q_AV_mean:+.4f}  "
                  f"V→src={Q_out_mean:+.4f}  sink→A={Q_ret_mean:+.4f}")
            print(f"  Conservation residuals: "
                  f"AV–out={residuals['rel_err_AV_vs_out']:.2%}, "
                  f"out–ret={residuals['rel_err_out_vs_ret']:.2%}")
            print(f"  Periodicity: |ΔV_A|={residuals['V_A_periodicity']:.2%}, "
                  f"|ΔV_V|={residuals['V_V_periodicity']:.2%}  "
                  f"stroke_V={residuals['stroke_V']:.3f}")
            # Valve timing (cycle-relative phase, 0..1)
            def _fmt(x):
                return '—' if not np.isfinite(x) else f'{x:.3f}'
            print(f"  Valve timing (fraction of cycle):")
            print(f"    AV:   open@{_fmt(av_open)}  close@{_fmt(av_close)}")
            for sn in self.source_nodes:
                op, cl = out_phases[sn]
                print(f"    out[{sn}]: open@{_fmt(op)}  close@{_fmt(cl)}")
            # Energy balance
            print(f"  Energy balance (per cycle):")
            print(f"    W_in  = {W_in:.3e}  "
                  f"(V={W_heart_V:.3e}, A={W_heart_A:.3e})")
            print(f"    W_out = {W_out:.3e}  "
                  f"(plex={W_plex_diss:.3e}, AV={W_AV_diss:.3e}, "
                  f"out={W_out_diss:.3e}, SV={W_SV_diss:.3e})")
            print(f"    residual = {energy_residual:.2%}")
            if max(residuals['rel_err_AV_vs_out'],
                   residuals['rel_err_out_vs_ret']) > 0.05:
                print(f"  ⚠️  Loop not at steady state — increase n_cycles "
                      f"or check parameter regime.")
            if energy_residual > 0.05:
                print(f"  ⚠️  Energy balance residual > 5% — tighten "
                      f"rtol/atol or increase n_cycles (periodic steady "
                      f"state not fully reached).")

        return CoupledResult(
            t=sol.t, V_A=V_A, V_V=V_V, p_A=pA_t, p_V=pV_t,
            zeta_AV=zeta_AV, zeta_out=zeta_out,
            node_p=node_p, edge_Q=edge_Q, valve_Q=valve_Q,
            params=p,
            source_nodes=list(self.source_nodes),
            sink_nodes=list(self.sink_nodes),
            conservation=residuals,
        )


# -----------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------


def run_coupled_simulation(G: nx.Graph,
                            params: Optional[HeartParams] = None,
                            mu: float = MU_DEFAULT,
                            radii_override: Optional[Dict] = None,
                            verbose: bool = True) -> CoupledResult:
    """Build + run the coupled sim. Returns a CoupledResult."""
    sim = HeartCoupledSim(G, params=params, mu=mu,
                          radii_override=radii_override)
    return sim.run(verbose=verbose)
