"""
Transmission line network solver for pulsatile blood flow.

Implements the Fancher & Katifori (2022) frequency-domain transmission line
equations. Each vessel is a distributed RLC element with:
  - Viscous resistance  r = 8μ/(πR⁴)
  - Inertance           ℓ = ρ/(πR²)
  - Compliance          c = πR²D   (areal distensibility, ΔA/A = D·ΔP)

The dispersion relation is:
  k(ω) = sqrt(-iω c (r + iω ℓ))

For each harmonic n of the cardiac cycle (ω_n = 2π n f0), the network
Laplacian L(ω_n) is assembled and the system solved:
  L(ω_n) P_n = Q_n^(bc)

where Q_n^(bc) are harmonic coefficients of flow at boundary nodes.

DC limit (ω=0) recovers Poiseuille: L(0) = diag(G_ij) Kirchhoff Laplacian.

Parameters
----------
ρ : float
    Blood density ≈ 1060 kg/m³
μ : float
    Blood viscosity = 3.5 mPa·s = 3.5e-3 Pa·s
D : float
    Vessel wall AREAL distensibility (1/Pa), defined by
    ΔA/A = D·ΔP.  Free parameter; empirically ≈ 1.3×10⁻³ 1/Pa for
    HH-stage yolk sac.  NOTE: 2026-05-18 switched from the radius
    convention (ΔR/R = D_R·ΔP, c = 2πR²D_R) to the areal convention
    used in vascular-biomechanics literature.  D_areal = 2 · D_radius
    for the same physics.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


# Physical constants
RHO_BLOOD = 1060.0        # kg/m³
MU_DEFAULT = 3.5e-3       # Pa·s (3.5 mPa·s)


@dataclass
class TransmissionLineResult:
    """Results from transmission line network solve."""
    # Per-edge predicted flow (complex, per harmonic)
    edge_flows: Dict[Tuple[int, int], np.ndarray]  # (u,v) -> [Q_dc, Q_1, Q_2, ...]
    # Per-node predicted pressure (complex, per harmonic)
    node_pressures: Dict[int, np.ndarray]           # node -> [P_dc, P_1, P_2, ...]
    # Derived quantities per edge
    mean_Q: Dict[Tuple[int, int], float]            # nL/s
    amp_Q: Dict[Tuple[int, int], float]             # nL/s (fundamental amplitude)
    PI: Dict[Tuple[int, int], float]                # pulsatility index
    phase: Dict[Tuple[int, int], float]             # phase of fundamental forward-direction Q (rad, sign-canonicalised)
    pressure_phase: Dict[Tuple[int, int], float]    # legacy: midpoint-pressure H1 phase (kept for revertibility)
    WSS: Dict[Tuple[int, int], float]               # time-avg wall shear stress (Pa)
    dissipation: Dict[Tuple[int, int], float]       # r*L*<Q²> (W) viscous dissipation
    pulsatile_cost: Dict[Tuple[int, int], float]    # <Q²>/Q̄² (1 = steady, 2 = PI≈2)
    RPSI: Dict[Tuple[int, int], float]              # max(dQ/dt) / Q̄  (rate of pulsatility)
    eta: Dict[Tuple[int, int], float]               # waveform sharpness η = RPSI/(PI·πf0)
    # Compliance storage per edge
    Q_stored: Dict[Tuple[int, int], np.ndarray]     # complex [0, Q̂_1^stored, Q̂_2^stored, ...]
    storage_fraction: Dict[Tuple[int, int], float]  # |Q̂_1^stored| / |Q̂_1^(μ)|  (fundamental)
    # Parameters used
    f0_hz: float
    n_harmonics: int
    D: float                                         # distensibility (1/Pa)
    mu: float                                        # viscosity (Pa·s)
    # Boundary info
    boundary_nodes: List[int]
    boundary_Q_harmonics: Dict[int, np.ndarray]     # node -> [Q_dc, Q_1, ...]
    # Network info
    n_edges: int
    n_nodes: int


def compute_rpsi_from_harmonics(Q_harmonics, f0_hz, n_eval=1000):
    """Compute RPSI = (Q_sys - Q_dia) / Δt_rise / Q̄ from harmonic coefficients.

    Reconstructs Q(t) over one cycle, finds diastolic min → systolic max,
    computes the average slope of the systolic upstroke normalized by mean Q.

    Parameters
    ----------
    Q_harmonics : array of complex [Q_dc, Q_1, Q_2, ...]
    f0_hz : float, fundamental frequency
    n_eval : int, grid points for reconstruction

    Returns
    -------
    float — RPSI value (1/s units)
    """
    q_dc = Q_harmonics[0].real
    if abs(q_dc) < 1e-15:
        return np.nan
    omega0 = 2.0 * np.pi * f0_hz
    T_cycle = 1.0 / f0_hz
    t_grid = np.linspace(0, T_cycle, n_eval, endpoint=False)

    # Reconstruct Q(t) = Q_dc + Re[ Σ_n Q̂_n exp(i n ω₀ t) ]
    qt = np.full(n_eval, q_dc, dtype=float)
    for n in range(1, len(Q_harmonics)):
        qt += np.real(Q_harmonics[n] * np.exp(1j * n * omega0 * t_grid))

    qt_abs = np.abs(qt)

    # Find diastolic min → systolic max (double to avoid wrap)
    qt_double = np.concatenate([qt_abs, qt_abs])
    i_dia = int(np.argmin(qt_double[:n_eval]))
    search_end = min(i_dia + n_eval, len(qt_double))
    i_sys = i_dia + int(np.argmax(qt_double[i_dia:search_end]))

    q_dia = qt_double[i_dia]
    q_sys = qt_double[i_sys]
    dt_rise = (i_sys - i_dia) * T_cycle / n_eval

    if dt_rise < 1e-10 or q_sys <= q_dia:
        return np.nan

    return float((q_sys - q_dia) / dt_rise / abs(q_dc))


def compute_rpsi_from_qt(Q_t, f0_hz, frame_dt_s, smooth_frames=3):
    """Compute RPSI = max(dQ/dt) / Q̄ from a measured Q(t) time series.

    Cycle-averages Q(t) first, then differentiates the averaged waveform.

    Parameters
    ----------
    Q_t : (T,) array — per-frame flow
    f0_hz : float — cardiac frequency
    frame_dt_s : float — frame interval
    smooth_frames : int — moving average before cycle folding

    Returns
    -------
    float — RPSI value (1/s units)
    """
    qt = np.asarray(Q_t, dtype=float)
    # Sign-flip so mean is positive (preserves waveform shape)
    if np.nanmean(qt) < 0:
        qt = -qt
    finite = np.isfinite(qt)
    if finite.sum() < 20:
        return np.nan

    q_mean = float(np.nanmean(qt))
    if q_mean < 1e-15:
        return np.nan

    # Light smoothing
    if smooth_frames > 1:
        from scipy.ndimage import uniform_filter1d
        qt_s = uniform_filter1d(qt, smooth_frames)
    else:
        qt_s = qt

    # Fold onto single cardiac cycle
    cycle_samples = int(round(1.0 / (f0_hz * frame_dt_s)))
    if cycle_samples < 10:
        return np.nan
    n_cycles = len(qt_s) // cycle_samples
    if n_cycles < 2:
        return np.nan

    # Trim to whole cycles and reshape
    qt_trim = qt_s[:n_cycles * cycle_samples].reshape(n_cycles, cycle_samples)
    qt_avg = np.nanmean(qt_trim, axis=0)  # averaged single cycle

    # Find diastolic min → next systolic max on the averaged cycle
    # Double the cycle to avoid wrap-around issues
    qt_double = np.concatenate([qt_avg, qt_avg])

    i_min = int(np.argmin(qt_double[:cycle_samples]))  # diastolic trough

    # Search for systolic peak AFTER the diastolic trough (within one cycle)
    search_end = min(i_min + cycle_samples, len(qt_double))
    i_max = i_min + int(np.argmax(qt_double[i_min:search_end]))

    q_dia = qt_double[i_min]
    q_sys = qt_double[i_max]
    dt_rise = (i_max - i_min) * frame_dt_s

    if dt_rise < frame_dt_s or q_sys <= q_dia:
        return np.nan

    return float((q_sys - q_dia) / dt_rise / q_mean)


def compute_eta_from_harmonics(Q_harmonics, f0_hz, n_eval=1000):
    r"""Compute waveform sharpness index η from harmonic coefficients.

    η = (Q_sys - Q_dia) / Δt_rise / Q_rms

    Same upstroke slope as RPSI, but normalized by Q_rms = √⟨Q²⟩
    instead of Q̄.  Units: s⁻¹.  Well-defined even when Q̄ → 0.

    Parameters
    ----------
    Q_harmonics : array of complex [Q_dc, Q_1, Q_2, ...]
    f0_hz : float, fundamental frequency
    n_eval : int, grid points for reconstruction

    Returns
    -------
    float — η (s⁻¹)
    """
    omega0 = 2.0 * np.pi * f0_hz
    T_cycle = 1.0 / f0_hz
    t_grid = np.linspace(0, T_cycle, n_eval, endpoint=False)

    # Reconstruct Q(t)
    qt = np.full(n_eval, Q_harmonics[0].real, dtype=float)
    for n in range(1, len(Q_harmonics)):
        qt += np.real(Q_harmonics[n] * np.exp(1j * n * omega0 * t_grid))

    q_rms = float(np.sqrt(np.mean(qt**2)))
    if q_rms < 1e-30:
        return np.nan

    # Find diastolic min → systolic max (same as RPSI)
    qt_double = np.concatenate([qt, qt])
    i_dia = int(np.argmin(qt_double[:n_eval]))
    search_end = min(i_dia + n_eval, len(qt_double))
    i_sys = i_dia + int(np.argmax(qt_double[i_dia:search_end]))

    q_dia = qt_double[i_dia]
    q_sys = qt_double[i_sys]
    dt_rise = (i_sys - i_dia) * T_cycle / n_eval

    if dt_rise < 1e-10 or q_sys <= q_dia:
        return np.nan

    upstroke_slope = (q_sys - q_dia) / dt_rise
    return float(upstroke_slope / q_rms)


def compute_eta_from_qt(Q_t, f0_hz, frame_dt_s, smooth_frames=3):
    r"""Compute waveform sharpness index η from a measured Q(t) time series.

    η = (Q_sys - Q_dia) / Δt_rise / Q_rms

    Same upstroke slope as RPSI, but normalized by Q_rms = √⟨Q²⟩
    instead of Q̄.  Units: s⁻¹.  Well-defined even when Q̄ → 0.

    Parameters
    ----------
    Q_t : (T,) array — per-frame flow
    f0_hz : float — cardiac frequency
    frame_dt_s : float — frame interval
    smooth_frames : int — moving average before cycle folding

    Returns
    -------
    float — η (s⁻¹)
    """
    qt = np.asarray(Q_t, dtype=float)
    if np.nanmean(qt) < 0:
        qt = -qt
    finite = np.isfinite(qt)
    if finite.sum() < 20:
        return np.nan

    # Light smoothing
    if smooth_frames > 1:
        from scipy.ndimage import uniform_filter1d
        qt_s = uniform_filter1d(qt, smooth_frames)
    else:
        qt_s = qt

    # Fold onto single cardiac cycle
    cycle_samples = int(round(1.0 / (f0_hz * frame_dt_s)))
    if cycle_samples < 10:
        return np.nan
    n_cycles = len(qt_s) // cycle_samples
    if n_cycles < 2:
        return np.nan

    qt_trim = qt_s[:n_cycles * cycle_samples].reshape(n_cycles, cycle_samples)
    qt_avg = np.nanmean(qt_trim, axis=0)

    # Smooth the cycle average for stable slope detection
    from scipy.ndimage import uniform_filter1d as uf1d
    qt_smooth = uf1d(qt_avg, max(3, cycle_samples // 30))

    q_rms = float(np.sqrt(np.mean(qt_smooth**2)))
    if q_rms < 1e-30:
        return np.nan

    # Find diastolic min → systolic max (same logic as RPSI)
    qt_double = np.concatenate([qt_smooth, qt_smooth])
    i_dia = int(np.argmin(qt_double[:cycle_samples]))
    search_end = min(i_dia + cycle_samples, len(qt_double))
    i_sys = i_dia + int(np.argmax(qt_double[i_dia:search_end]))

    q_dia = qt_double[i_dia]
    q_sys = qt_double[i_sys]
    dt_rise = (i_sys - i_dia) * frame_dt_s

    if dt_rise < frame_dt_s or q_sys <= q_dia:
        return np.nan

    upstroke_slope = (q_sys - q_dia) / dt_rise
    return float(upstroke_slope / q_rms)


def _get_edge_geometry(G: nx.Graph, u: int, v: int,
                       radii_m: Optional[Dict[Tuple[int, int], float]] = None,
                       ) -> Tuple[float, float]:
    """Get radius (m) and length (m) for an edge.

    If *radii_m* is provided, use the override radius (already in meters)
    instead of reading from the graph.  Length always comes from the graph.
    """
    from .config import PX_SIZE_UM
    px_to_m = PX_SIZE_UM * 1e-6  # µm/px * m/µm

    data = G.edges[u, v]

    # --- Radius ---
    R_m = None
    if radii_m is not None:
        R_m = radii_m.get((u, v))
        if R_m is None:
            R_m = radii_m.get((v, u))
        if R_m is not None and (R_m <= 0 or not np.isfinite(R_m)):
            return None, None

    if R_m is None:
        # Prefer tile-corrected radius if available
        R_px = data.get('radius_px_true')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            R_px = data.get('R_fit_px')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            R_px = data.get('radius_px')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            R_px = data.get('radius')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            return None, None
        R_m = R_px * px_to_m

    # --- Length ---
    # Prefer tile-corrected length if available
    L_px = data.get('length_true')
    if L_px is None or not np.isfinite(L_px) or L_px <= 0:
        L_px = data.get('length')
    if L_px is None or not np.isfinite(L_px) or L_px <= 0:
        L_px = data.get('path_length_px')
    if L_px is None or not np.isfinite(L_px) or L_px <= 0:
        return None, None

    L_m = L_px * px_to_m

    return R_m, L_m


def _per_length_params(R_m: float, mu: float = MU_DEFAULT,
                       rho: float = RHO_BLOOD, D=1e-6):
    """Compute per-unit-length transmission line parameters.

    Parameters
    ----------
    R_m : float
        Vessel radius in meters
    mu : float
        Viscosity in Pa·s
    rho : float
        Density in kg/m³
    D : float or callable
        Distensibility in 1/Pa, or a callable D(R_m) implementing a
        compliance model (e.g. thin-wall: D = 2R / (E*h) → D(R) = 2R/(E*h)).

    Returns
    -------
    r : float
        Resistance per unit length [Pa·s/m⁴]
    ell : float
        Inertance per unit length [kg/m⁵]
    c : float
        Compliance per unit length [m³/(Pa·m)] = [m²/Pa]
    """
    D_eff = D(R_m) if callable(D) else D
    A = np.pi * R_m**2
    r = 8 * mu / (np.pi * R_m**4)      # Poiseuille resistance/length
    ell = rho / A                        # inertance/length
    # Compliance/length = ∂A/∂P.  D is the AREAL distensibility
    # (ΔA/A = D·ΔP), so ∂A/∂P = A·D = πR²·D directly.
    # NOTE: prior convention was radius distensibility (ΔR/R = D·ΔP),
    # which gave c = 2πR²D.  Switched 2026-05-18 to areal convention
    # to match common vascular-biomechanics literature.  D values
    # under the new convention are 2× the old ones for the same
    # physics (D_areal = 2·D_radius).
    c = np.pi * R_m**2 * D_eff           # compliance/length: ∂A/∂P = πR²D [m²/Pa]
    return r, ell, c


def _vessel_admittance(R_m: float, L_m: float, omega: float,
                       mu: float = MU_DEFAULT, rho: float = RHO_BLOOD,
                       D: float = 1e-6) -> complex:
    """Compute the 2-port admittance for a vessel at frequency omega.

    For a transmission line segment of length L with propagation constant
    k(ω) = sqrt((r + iωℓ)(iωc)), the characteristic impedance is
    Z_c = sqrt((r + iω ℓ) / (iω c)), and the 2-port admittance matrix is:

        Y = (1/Z_c) * [[coth(kL), -csch(kL)],
                        [-csch(kL), coth(kL)]]

    For the Kirchhoff Laplacian assembly, the diagonal contribution is
    Y_diag = coth(kL)/Z_c and off-diagonal is Y_off = -csch(kL)/Z_c.

    At DC (omega=0): Y_diag = Y_off = G = πR⁴/(8μL) (Poiseuille conductance).

    Returns
    -------
    Y_diag, Y_off : complex
        Diagonal and off-diagonal admittance entries.
    """
    r, ell, c = _per_length_params(R_m, mu, rho, D)

    if abs(omega) < 1e-12:
        # DC limit: pure Poiseuille conductance
        G = np.pi * R_m**4 / (8 * mu * L_m)
        return G, -G

    # Propagation constant: k = sqrt((r + iωℓ)(iωc))
    z = r + 1j * omega * ell       # series impedance per length
    y = 1j * omega * c             # shunt admittance per length
    gamma_sq = z * y               # k² = z*y
    gamma = np.sqrt(gamma_sq)      # propagation constant k

    # Characteristic impedance: Z_c = sqrt(z/y)
    Z_c = np.sqrt(z / y)

    kL = gamma * L_m

    # Numerical safety for small kL (Taylor expand)
    if abs(kL) < 1e-6:
        # coth(x) ≈ 1/x + x/3, csch(x) ≈ 1/x - x/6
        # Y_diag = (1/Z_c)(1/kL + kL/3), Y_off = -(1/Z_c)(1/kL - kL/6)
        Y_diag = (1.0 / Z_c) * (1.0 / kL + kL / 3.0)
        Y_off = -(1.0 / Z_c) * (1.0 / kL - kL / 6.0)
    elif abs(kL.real) > 500:
        # Heavily damped: coth(kL) → 1, csch(kL) → 0
        # Vessel is so lossy that the far end is effectively decoupled.
        Y_diag = 1.0 / Z_c
        Y_off = 0.0
        return Y_diag, Y_off
    else:
        # Use exp form for numerical stability when Re(kL) is large
        e_pos = np.exp(kL)
        e_neg = np.exp(-kL)
        sinh_kL = (e_pos - e_neg) / 2.0
        cosh_kL = (e_pos + e_neg) / 2.0

        if abs(sinh_kL) < 1e-30:
            # Degenerate case
            G = np.pi * R_m**4 / (8 * mu * L_m)
            return G, -G

        coth_kL = cosh_kL / sinh_kL
        csch_kL = 1.0 / sinh_kL

        Y_diag = coth_kL / Z_c
        Y_off = -csch_kL / Z_c

    return Y_diag, Y_off


def _assemble_laplacian(G: nx.Graph, omega: float,
                        edge_list: List[Tuple[int, int]],
                        node_to_idx: Dict[int, int],
                        mu: float, rho: float, D: float,
                        radii_m: Optional[Dict[Tuple[int, int], float]] = None,
                        ) -> csr_matrix:
    """Assemble the frequency-dependent network admittance matrix (Laplacian).

    L(ω)_ii = Σ_j Y_diag(edge_ij, ω)
    L(ω)_ij = Y_off(edge_ij, ω)
    """
    N = len(node_to_idx)
    L = lil_matrix((N, N), dtype=complex)

    for u, v in edge_list:
        R_m, L_m = _get_edge_geometry(G, u, v, radii_m=radii_m)
        if R_m is None:
            continue

        Y_diag, Y_off = _vessel_admittance(R_m, L_m, omega, mu, rho, D)

        i = node_to_idx[u]
        j = node_to_idx[v]

        L[i, i] += Y_diag
        L[j, j] += Y_diag
        L[i, j] += Y_off
        L[j, i] += Y_off

    return L.tocsr()


def _extract_boundary_harmonics(
    G: nx.Graph,
    boundary_nodes: List[int],
    f0_hz: float,
    n_harmonics: int,
    frame_dt: float,
) -> Dict[int, np.ndarray]:
    """Extract flow harmonic coefficients at boundary nodes from Q_t data.

    For each boundary node, find all edges connected to it that have Q_t.
    Sum the flows (with correct sign convention) to get net Q(t) at the node.
    Then decompose into harmonics via fit_harmonics.

    Returns
    -------
    boundary_Q : dict
        node -> complex array [Q_dc, Q_1, Q_2, ..., Q_n_harmonics]
        Q_dc is real (mean flow in nL/s)
        Q_k is complex (A_k - i*B_k convention)
    """
    from .harmonic import fit_harmonics

    boundary_Q = {}

    # Determine reference tile (tile 14 by default, or from graph metadata)
    ref_tile = G.graph.get('reference_vid', 14)

    for node in boundary_nodes:
        # For boundary edges, prefer the reference tile's PIV measurement
        # to ensure consistent f0 and Q_t across all boundaries.
        # IMPORTANT: do NOT take abs() — preserve the phase information
        # in the waveform. Q_t from PIV is already positive (magnitude).
        Q_raw = None

        for neighbor in G.neighbors(node):
            data = G.edges[node, neighbor]

            # Try reference tile's PIV Q_t first
            piv_list = data.get('measurements_piv', [])
            ref_meas = [m for m in piv_list if m.get('tile_id') == ref_tile]
            if ref_meas:
                Qt_src = ref_meas[0].get('Q_t')
                if Qt_src is not None and len(Qt_src) > 20:
                    Q_raw = np.asarray(Qt_src, dtype=float)
                    continue

            # Fallback: top-level Q_t
            Q_t = data.get('Q_t')
            if Q_t is None or len(Q_t) == 0:
                continue
            Q_t = np.asarray(Q_t, dtype=float)

            if Q_raw is None:
                Q_raw = Q_t.copy()
            else:
                n_min = min(len(Q_raw), len(Q_t))
                Q_raw = Q_raw[:n_min] + Q_t[:n_min]

        if Q_raw is None or len(Q_raw) < 20:
            boundary_Q[node] = np.zeros(n_harmonics + 1, dtype=complex)
            continue

        # Ensure Q is positive magnitude (PIV stores positive, but
        # kymograph may have signed values)
        if np.nanmean(Q_raw) < 0:
            Q_raw = -Q_raw

        # Solver convention: positive Q = flow injected into network.
        # Sources inject (positive), sinks extract (negative).
        bt = G.nodes[node].get('boundary_type', 'source')
        solver_sign = -1.0 if bt == 'sink' else +1.0
        Q_total = solver_sign * Q_raw

        # Direct harmonic fit to raw Q_t (no cycle-averaging — that inflates
        # harmonic amplitudes by phase-locking noisy cycles)
        hr = fit_harmonics(Q_total, frame_dt, f0_hz, K=n_harmonics,
                           loss='huber', include_dc=True)

        # Pack into complex array: [DC, H1, H2, ..., Hn]
        coeffs = np.zeros(n_harmonics + 1, dtype=complex)
        coeffs[0] = float(np.nanmean(Q_total))  # DC from nanmean

        # fit_harmonics convention: signal = A·cos(kωt) + B·sin(kωt)
        # Complex Fourier convention: Re[Q̂·e^(ikωt)] = A·cos(kωt) - B·sin(kωt)
        # So Q̂ = A - iB (negate the imaginary part)
        for h in hr['harmonics']:
            k = h['k']
            if k <= n_harmonics:
                coeffs[k] = complex(h['A'], -h['B'])

        boundary_Q[node] = coeffs

    return boundary_Q


def _classify_boundary_nodes(
    G: nx.Graph,
    boundary_nodes: List[int],
) -> Tuple[List[int], List[int]]:
    """Classify boundary nodes as sources (arterial) or sinks (venous).

    Sources: flow leaves the boundary node into the network (flow_from == node).
    Sinks:   flow enters the boundary node from the network (flow_to == node).

    Hard-coded override for 21 somites (4 boundary nodes):
      Sources: 59735, 59736  (arterial inlets, lower y / upper in image)
      Sinks:   59734, 59737  (venous outlets)

    Returns
    -------
    sources, sinks : list of int
    """
    # Hard-coded for 21 somites
    _KNOWN_SOURCES = {59735, 59736}
    _KNOWN_SINKS = {59734, 59737}

    # First try: use boundary_type node attribute if available
    sources = [n for n in boundary_nodes
               if G.nodes[n].get('boundary_type') == 'source']
    sinks = [n for n in boundary_nodes
             if G.nodes[n].get('boundary_type') == 'sink']
    if sources and sinks:
        return sources, sinks

    # Hard-coded for 21 somites
    bnd_set = set(boundary_nodes)
    if bnd_set == (_KNOWN_SOURCES | _KNOWN_SINKS):
        sources = [n for n in boundary_nodes if n in _KNOWN_SOURCES]
        sinks = [n for n in boundary_nodes if n in _KNOWN_SINKS]
        return sources, sinks

    # Fallback: classify from measured flow direction
    sources = []
    sinks = []
    for node in boundary_nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            sinks.append(node)
            continue
        # Check majority flow direction across all incident edges
        net_outflow = 0.0
        for nb in neighbors:
            data = G.edges[node, nb]
            flow_from = data.get('flow_from')
            mean_Q = data.get('mean_Q', 0)
            if flow_from == node:
                net_outflow += abs(mean_Q)
            else:
                net_outflow -= abs(mean_Q)
        if net_outflow > 0:
            sources.append(node)
        else:
            sinks.append(node)

    return sources, sinks


def solve_transmission_line(
    G: nx.Graph,
    D=1e-6,
    n_harmonics: int = 3,
    f0_hz: Optional[float] = None,
    mu: float = MU_DEFAULT,
    rho: float = RHO_BLOOD,
    boundary_nodes: Optional[List[int]] = None,
    bc_harmonics_override: Optional[Dict[int, np.ndarray]] = None,
    radii_m: Optional[Dict[Tuple[int, int], float]] = None,
    verbose: bool = True,
    E_wall: Optional[float] = None,
    h_wall: float = 1e-6,
    sink_pressure_bc: Optional[float] = None,
    sink_impedance=None,
    merged_boundary: bool = False,
) -> TransmissionLineResult:
    """Solve the transmission line network at DC + n_harmonics.

    Boundary conditions:
      - Q(t) prescribed at ALL boundary nodes (sources and sinks)
      - The outlier boundary node (largest |Q_dc| deviation) is uniformly
        rescaled so that DC flow is conserved: ΣQ_source = ΣQ_sink

    If *sink_pressure_bc* is not None:
      - Sink (venous) boundary nodes get Dirichlet P = sink_pressure_bc (Pa)
        instead of Q BCs.  Only source (arterial) nodes keep Q BCs.
      - DC conservation adjustment is applied only among source nodes.
      - The P-BC nodes serve as the DC gauge (no separate pinning needed).

    If *sink_impedance* is not None:
      - Sink nodes get impedance BCs: P = Z_ven(ω) · Q at each harmonic.
        Implemented by adding ground admittance Y = 1/Z_ven to the
        Laplacian diagonal at sink nodes (Q_rhs = 0).
      - Can be a scalar (frequency-independent R) or callable Z(omega)
        returning complex impedance.  For a Windkessel:
        Z(ω) = R_ven / (1 + iω R_ven C_ven)
      - Z → 0 approaches P = 0.  Z → ∞ approaches free node.
      - Takes precedence over sink_pressure_bc if both are set.

    If *merged_boundary* is True:
      - All boundary nodes (arterial + venous) are merged into a single
        node with shared pressure P_u.  Total arterial Q is prescribed
        at this merged node.  Venous flows become predictions.
      - DC: P_u pinned to 0 (gauge).  Flow distributes among veins by topology.
      - AC: P_u is solved (shared oscillation).  Venous pulsatility predicted.
      - Takes precedence over all other sink BC modes.
      - AC harmonics: compliance makes L(ω) non-singular, so Q prescribed
        at all boundary nodes works directly (no pressure pinning needed)
      - DC: L(0) is singular (Kirchhoff), so pin P=0 at one interior node

    Algorithm:
    1. Identify and classify boundary nodes (source vs sink)
    2. Extract Q(t) harmonics at ALL boundary nodes
    3. Rescale outlier node's harmonics to enforce DC conservation
    4. For each harmonic n:
       a. Assemble L(ω_n)
       b. DC only: pin P=0 at one interior node
       c. Prescribe Q at all boundary nodes in RHS
       d. Solve for all pressures
    5. Reconstruct edge flows from pressure differences

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with measurements (Q_t, geometry, boundary_type)
    D : float
        Distensibility in 1/Pa. This is the free parameter.
    n_harmonics : int
        Number of cardiac harmonics to solve (plus DC)
    f0_hz : float, optional
        Heart rate. If None, read from G.graph['tile_f0s'] (median).
    mu : float
        Blood viscosity in Pa·s
    rho : float
        Blood density in kg/m³
    boundary_nodes : list, optional
        Override boundary node detection
    bc_harmonics_override : dict, optional
        Pre-computed boundary harmonics {node_id: complex_array}.
        If provided, skips automatic extraction from Q_t data.
        Each array is [Q_dc, Q_1, ..., Q_n] with n_harmonics+1 entries.
    E_wall : float, optional
        If given, overrides D with the thin-wall compliance model
        D(R) = 2R / (E_wall * h_wall).  E_wall in Pa (e.g. 20e3 = 20 kPa).
    h_wall : float
        Endothelial wall thickness in meters (default 1 µm).
        Only used when E_wall is provided.
    verbose : bool

    Returns
    -------
    TransmissionLineResult
    """
    from .config import FRAME_DT_S

    # Thin-wall compliance model: D(R) = 2R / (E_wall * h_wall)
    if E_wall is not None:
        D = lambda R, _E=E_wall, _h=h_wall: 2.0 * R / (_E * _h)

    # --- 1. Identify boundary nodes ---
    if boundary_nodes is None:
        boundary_nodes = [n for n, d in G.nodes(data=True)
                          if d.get('boundary_type') is not None]

    if len(boundary_nodes) < 2:
        raise ValueError(f"Need >= 2 boundary nodes, found {len(boundary_nodes)}")

    # --- 2. Classify sources vs sinks ---
    source_nodes, sink_nodes = _classify_boundary_nodes(G, boundary_nodes)

    if not source_nodes:
        raise ValueError("No source (arterial) boundary nodes found")
    if not sink_nodes:
        raise ValueError("No sink (venous) boundary nodes found")

    # --- 3. Determine f0 ---
    if f0_hz is None:
        tile_f0s = G.graph.get('tile_f0s', {})
        if tile_f0s:
            f0_hz = float(np.median(list(tile_f0s.values())))
        else:
            raise ValueError("No f0_hz provided and no tile_f0s in graph")

    if verbose:
        D_disp = f"{D(10e-6):.2e} @10µm" if callable(D) else f"{D:.2e}"
        print(f"Transmission line solve: D={D_disp} 1/Pa, f0={f0_hz:.3f} Hz, "
              f"{n_harmonics} harmonics, µ={mu*1e3:.1f} mPa·s")
        print(f"  Sources: {source_nodes}")
        print(f"  Sinks:   {sink_nodes}")

    # --- 4. Build node/edge lists ---
    boundary_set = set(boundary_nodes)
    boundary_edges = set()
    # Keep track of boundary edges for post-hoc assignment, but do NOT
    # exclude them from the solve.  BCs are applied directly at boundary nodes.
    bc_remap = {}  # identity: boundary node -> itself (no remap)
    for bn in boundary_nodes:
        nbrs = list(G.neighbors(bn))
        if len(nbrs) == 1:
            boundary_edges.add((bn, nbrs[0]))
            boundary_edges.add((nbrs[0], bn))
        elif len(nbrs) > 1:
            best_nbr = max(nbrs,
                           key=lambda n: abs(G.edges[bn, n].get('mean_Q', 0)))
            boundary_edges.add((bn, best_nbr))
            boundary_edges.add((best_nbr, bn))

    all_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    # All edges with valid geometry (boundary edges included)
    edge_list = []
    for u, v in G.edges():
        R_m, L_m = _get_edge_geometry(G, u, v, radii_m=radii_m)
        if R_m is not None:
            edge_list.append((u, v))

    if verbose:
        print(f"  {len(edge_list)}/{G.number_of_edges()} edges in solve "
              f"({len(boundary_nodes)} boundary nodes, "
              f"{len(boundary_edges)//2} boundary edges)")

    # --- 5. Extract flow harmonics at ALL boundary nodes ---
    if bc_harmonics_override is not None:
        # Use user-supplied harmonics, padding/truncating to n_harmonics+1
        bc_harmonics = {}
        for n in boundary_nodes:
            if n in bc_harmonics_override:
                arr = np.asarray(bc_harmonics_override[n], dtype=complex)
                target_len = n_harmonics + 1
                if len(arr) >= target_len:
                    bc_harmonics[n] = arr[:target_len]
                else:
                    padded = np.zeros(target_len, dtype=complex)
                    padded[:len(arr)] = arr
                    bc_harmonics[n] = padded
            else:
                bc_harmonics[n] = np.zeros(n_harmonics + 1, dtype=complex)
        if verbose:
            print("  Using user-supplied boundary harmonics")
    else:
        bc_harmonics = _extract_boundary_harmonics(
            G, boundary_nodes, f0_hz, n_harmonics, FRAME_DT_S)

    if verbose:
        for node, coeffs in bc_harmonics.items():
            role = 'Source' if node in source_nodes else 'Sink'
            print(f"  {role} {node}: Q_dc={coeffs[0].real:.3f} nL/s, "
                  f"|Q_1|={abs(coeffs[1]):.3f} nL/s")

    # --- 6. Determine sink BC mode ---
    # Priority: merged_boundary > sink_impedance > sink_pressure_bc > Q (default)
    use_merged_bc = merged_boundary
    use_impedance_bc = sink_impedance is not None and not use_merged_bc
    use_pressure_bc = (sink_pressure_bc is not None
                       and not use_impedance_bc and not use_merged_bc)
    sink_has_passive_bc = use_impedance_bc or use_pressure_bc or use_merged_bc

    if use_merged_bc:
        q_bc_nodes = list(source_nodes)  # only sources prescribe Q
        p_bc_nodes = []
        z_bc_nodes = []
        if verbose:
            print(f"  Merged boundary BC: all boundary nodes share P_u")
            print(f"    Arterial Q prescribed (total), venous Q predicted")
    elif use_impedance_bc:
        q_bc_nodes = list(source_nodes)
        p_bc_nodes = []
        z_bc_nodes = list(sink_nodes)
        z_ven = sink_impedance
        if verbose:
            z_disp = f"{z_ven:.2e}" if not callable(z_ven) else "Z(ω)"
            print(f"  Sink BC: impedance Z={z_disp} at {z_bc_nodes}")
    elif use_pressure_bc:
        q_bc_nodes = list(source_nodes)
        p_bc_nodes = list(sink_nodes)
        z_bc_nodes = []
        p_bc_value = float(sink_pressure_bc)
        if verbose:
            print(f"  Sink BC: Dirichlet P={p_bc_value:.2f} Pa at {p_bc_nodes}")
    else:
        q_bc_nodes = list(boundary_nodes)
        p_bc_nodes = []
        z_bc_nodes = []

    # --- 6a. DC conservation (among Q-BC nodes only) ---
    # When sinks have passive BCs (P or Z), conservation is automatic.
    # With all-Q BCs, distribute imbalance proportionally.
    Q_dc_total = sum(bc_harmonics[n][0].real for n in q_bc_nodes)

    if abs(Q_dc_total) > 1e-10 and not sink_has_passive_bc:
        abs_dc = {n: abs(bc_harmonics[n][0].real) for n in q_bc_nodes}
        total_abs = sum(abs_dc.values())

        if verbose:
            print(f"  DC imbalance: {Q_dc_total:.3f} nL/s")

        if total_abs > 1e-10:
            for n in q_bc_nodes:
                share = abs_dc[n] / total_abs
                correction = share * Q_dc_total
                Q_dc_old = bc_harmonics[n][0].real
                Q_dc_new = Q_dc_old - correction
                if abs(Q_dc_old) > 1e-10:
                    scale = Q_dc_new / Q_dc_old
                    bc_harmonics[n] = bc_harmonics[n] * scale
                    if verbose:
                        print(f"    Node {n}: {Q_dc_old:.3f} -> {Q_dc_new:.3f} nL/s "
                              f"(x{scale:.3f})")
    elif sink_has_passive_bc and verbose:
        bc_type = ('merged' if use_merged_bc else
                   'impedance' if use_impedance_bc else 'pressure')
        print(f"  Source Q_dc total: {Q_dc_total:.3f} nL/s "
              f"(sink flows determined by {bc_type} solve)")

    # --- 6b. Apply BCs directly at boundary nodes (no remap) ---
    bc_harmonics_remapped = {bn: bc_harmonics[bn].copy() for bn in q_bc_nodes}
    solve_bc_nodes = list(q_bc_nodes)
    # Nodes whose boundary edge flow comes from the solve (not overridden)
    solve_flow_bc_nodes = set(p_bc_nodes) | set(z_bc_nodes)
    if use_merged_bc:
        # In merged mode, ALL boundary node flows come from the solve
        solve_flow_bc_nodes = set(boundary_nodes)

    # --- 7. Solve at each frequency ---
    # Include ALL graph nodes in node_P for pressure storage (even excluded ones)
    node_P = {n: np.zeros(n_harmonics + 1, dtype=complex) for n in G.nodes()}
    bc_set = set(solve_bc_nodes)

    frequencies = [0.0] + [2 * np.pi * k * f0_hz for k in range(1, n_harmonics + 1)]

    # Pick an interior node to pin P=0 at DC (needed because Kirchhoff L is singular)
    pin_node = None
    for n in all_nodes:
        if n not in bc_set:
            pin_node = n
            break

    # Precompute merged-BC penalty admittance
    if use_merged_bc:
        if verbose:
            print(f"  Merged BC: {len(boundary_nodes)} boundary nodes "
                  f"connected by penalty admittance (shared P_u)")
        B_idxs = [node_to_idx[bn] for bn in boundary_nodes]

    for harm_idx, omega in enumerate(frequencies):
        # Assemble full Laplacian on all N nodes
        L_full = _assemble_laplacian(G, omega, edge_list, node_to_idx,
                                     mu, rho, D, radii_m=radii_m)

        if use_merged_bc:
            # --- Merged boundary: penalty-edge approach ---
            # Add large admittance between all pairs of boundary nodes
            # to force them to share a common pressure P_u.
            # Prescribe Q at arterial nodes, Q=0 at venous nodes.
            L_solve = L_full.tolil()

            # Determine penalty magnitude: ~1000× max diagonal entry
            diag_abs = np.abs(L_full.diagonal())
            Y_big = 1e3 * max(diag_abs.max(), 1e-20)

            # Add penalty edges between all boundary pairs
            for i_b in range(len(B_idxs)):
                for j_b in range(i_b + 1, len(B_idxs)):
                    bi, bj = B_idxs[i_b], B_idxs[j_b]
                    L_solve[bi, bj] -= Y_big   # off-diagonal (negative)
                    L_solve[bj, bi] -= Y_big
                    L_solve[bi, bi] += Y_big   # diagonal correction
                    L_solve[bj, bj] += Y_big

            # RHS: arterial Q at source nodes, 0 at sinks and interior
            Q_solve = np.zeros(N, dtype=complex)
            for sn in source_nodes:
                idx = node_to_idx[sn]
                Q_solve[idx] = bc_harmonics[sn][harm_idx] * 1e-12

            # DC: pin P=0 at one boundary node (gauge reference)
            if harm_idx == 0:
                pin_b = B_idxs[0]
                L_solve[pin_b, :] = 0
                L_solve[pin_b, pin_b] = 1.0
                Q_solve[pin_b] = 0.0

            L_solve = L_solve.tocsr()
            try:
                P = spsolve(L_solve, Q_solve)
            except Exception as e:
                if verbose:
                    print(f"  WARNING: Merged solve failed at harmonic "
                          f"{harm_idx}: {e}")
                continue

            if verbose and harm_idx == 0:
                P_abs = np.abs(P)
                P_boundary = [P[bi] for bi in B_idxs]
                P_spread = max(abs(p) for p in P_boundary) - min(abs(p) for p in P_boundary)
                n_nonzero = np.count_nonzero(P_abs > 1e-20)
                print(f"  Merged DC: {n_nonzero}/{N} nodes |P|>0, "
                      f"max|P|={P_abs.max():.4e}, "
                      f"boundary P spread={P_spread:.2e} "
                      f"(Y_big={Y_big:.2e})")
            elif verbose and harm_idx == 1:
                P_u_mean = np.mean([P[bi] for bi in B_idxs])
                print(f"  Merged AC h=1: |P_u|={abs(P_u_mean):.4e} Pa")
        else:
            # --- Standard per-node solve ---
            # Build RHS: Q at Q-BC nodes, 0 elsewhere
            Q_rhs = np.zeros(N, dtype=complex)
            for node in solve_bc_nodes:
                idx = node_to_idx[node]
                Q_rhs[idx] = bc_harmonics_remapped[node][harm_idx]

            Q_rhs_si = Q_rhs * 1e-12
            L_solve = L_full.tolil()
            Q_solve = Q_rhs_si.copy()

            # Impose Dirichlet P at pressure-BC nodes (sinks)
            for pn in p_bc_nodes:
                idx = node_to_idx[pn]
                L_solve[idx, :] = 0
                L_solve[idx, idx] = 1.0
                Q_solve[idx] = p_bc_value if harm_idx == 0 else 0.0

            # Impose impedance BC at sink nodes
            for zn in z_bc_nodes:
                idx = node_to_idx[zn]
                z_val = z_ven(omega) if callable(z_ven) else z_ven
                if abs(z_val) > 1e-30:
                    Y_ground = 1.0 / z_val
                    L_solve[idx, idx] += Y_ground

            # DC: pin P=0 at one interior node
            if harm_idx == 0 and pin_node is not None and not sink_has_passive_bc:
                idx = node_to_idx[pin_node]
                L_solve[idx, :] = 0
                L_solve[idx, idx] = 1.0
                Q_solve[idx] = 0.0

            L_solve = L_solve.tocsr()

            try:
                P = spsolve(L_solve, Q_solve)
            except Exception as e:
                if verbose:
                    print(f"  WARNING: Solve failed at harmonic {harm_idx} "
                          f"(ω={omega:.2f}): {e}")
                continue

        # Store pressures
        for node in all_nodes:
            node_P[node][harm_idx] = P[node_to_idx[node]]

        if verbose and harm_idx == 0 and not use_merged_bc:
            P_abs = np.abs(P)
            n_nonzero = np.count_nonzero(P_abs > 1e-20)
            print(f"  DC solve: {n_nonzero}/{len(P)} nodes with |P|>0, "
                  f"max|P|={P_abs.max():.4e}, "
                  f"Q_rhs nonzero={np.count_nonzero(np.abs(Q_solve) > 1e-20)}, "
                  f"max|Q_rhs|={np.abs(Q_solve).max():.4e}, "
                  f"L nnz={L_solve.nnz}, pin={pin_node}")

    # --- 8. Reconstruct edge flows from pressure differences ---
    edge_flows = {}
    mean_Q_dict = {}
    amp_Q_dict = {}
    PI_dict = {}
    RPSI_dict = {}
    eta_dict = {}
    phase_dict = {}
    pressure_phase_dict = {}
    WSS_dict = {}
    dissipation_dict = {}
    pulsatile_cost_dict = {}
    Q_stored_dict = {}
    storage_frac_dict = {}

    for u, v in edge_list:
        R_m, L_m = _get_edge_geometry(G, u, v, radii_m=radii_m)
        if R_m is None:
            continue

        Q_harmonics = np.zeros(n_harmonics + 1, dtype=complex)

        for harm_idx, omega in enumerate(frequencies):
            Y_diag, Y_off = _vessel_admittance(R_m, L_m, omega, mu, rho, D)

            P_u = node_P[u][harm_idx]
            P_v = node_P[v][harm_idx]

            # Flow from u to v (current out of port u)
            Q_uv = Y_diag * P_u + Y_off * P_v  # m³/s
            Q_harmonics[harm_idx] = Q_uv * 1e12  # -> nL/s

        edge_flows[(u, v)] = Q_harmonics

        # Summary statistics
        q_dc = Q_harmonics[0].real
        q1 = Q_harmonics[1] if n_harmonics >= 1 else 0.0

        mean_Q_dict[(u, v)] = q_dc
        amp_Q_dict[(u, v)] = abs(q1)
        if abs(q_dc) > 1e-12:
            PI_dict[(u, v)] = 2.0 * abs(q1) / abs(q_dc)
        else:
            PI_dict[(u, v)] = np.nan

        # RPSI: max(dQ/dt) / Q̄ — analytical from harmonics
        RPSI_dict[(u, v)] = compute_rpsi_from_harmonics(Q_harmonics, f0_hz)
        # η: waveform sharpness = max|dQ/dt| / (πf0 (Qmax-Qmin))
        eta_dict[(u, v)] = compute_eta_from_harmonics(Q_harmonics, f0_hz)

        # Phase = angle of forward-direction Q_H1 (sign-canonicalised so
        # DC ≥ 0).  Previously this was midpoint pressure phase, which
        # is also orientation-independent but bakes in a ~180° structural
        # split between source-side and sink-side edges; that masked the
        # small physiological A→V phase lag.  Sign-canonicalising Q
        # restores orientation-independence without that split.
        # CHANGED 2026-05-27 — to revert, replace below with:
        #   P1_mid = (node_P[u][1] + node_P[v][1]) / 2.0 if n_harmonics >= 1 else 0.0
        #   phase_dict[(u, v)] = float(np.angle(P1_mid))
        if n_harmonics >= 1:
            Q_canon = Q_harmonics if Q_harmonics[0].real >= 0 else -Q_harmonics
            phase_dict[(u, v)] = float(np.angle(Q_canon[1]))
        else:
            phase_dict[(u, v)] = 0.0
        # Keep the legacy pressure-phase under a separate key so anything
        # downstream that wants it can still find it.
        P1_mid_legacy = ((node_P[u][1] + node_P[v][1]) / 2.0
                         if n_harmonics >= 1 else 0.0)
        pressure_phase_dict[(u, v)] = float(np.angle(P1_mid_legacy))

        # Wall shear stress: TAWSS = 4μ/(πR³) · ⟨|Q(t)|⟩
        # Reconstruct Q(t) over one cycle from harmonics, then mean(|Q|)
        _n_pts = 128
        _t_cyc = np.linspace(0, 2 * np.pi, _n_pts, endpoint=False)
        _Q_t = np.full(_n_pts, Q_harmonics[0].real)  # DC
        for _nh in range(1, n_harmonics + 1):
            _Q_t += (Q_harmonics[_nh].real * np.cos(_nh * _t_cyc)
                     - Q_harmonics[_nh].imag * np.sin(_nh * _t_cyc))
        _Q_mean_abs_m3s = float(np.mean(np.abs(_Q_t))) * 1e-12
        WSS_dict[(u, v)] = 4.0 * mu * _Q_mean_abs_m3s / (np.pi * R_m**3)

        # Viscous dissipation: Φ = r·L·⟨Q²⟩  (Watts)
        Q_dc_sq = (q_dc * 1e-12)**2
        Q_ac_sq = 0.5 * sum(abs(Q_harmonics[n] * 1e-12)**2
                            for n in range(1, n_harmonics + 1))
        Q2_avg = Q_dc_sq + Q_ac_sq
        r_per_length = 8.0 * mu / (np.pi * R_m**4)
        dissipation_dict[(u, v)] = r_per_length * L_m * Q2_avg

        # Pulsatile cost: ⟨Q²⟩/Q̄²
        pulsatile_cost_dict[(u, v)] = (Q2_avg / Q_dc_sq
                                       if Q_dc_sq > 1e-30 else np.nan)

        # Compliance storage: Q̂_n^stored for each AC harmonic
        # Q̂_stored = (γ/r) · (P_u + P_v)(cosh(γL) - 1) / sinh(γL)
        Q_stored_harmonics = np.zeros(n_harmonics + 1, dtype=complex)
        for harm_idx in range(1, n_harmonics + 1):
            omega_h = 2.0 * np.pi * harm_idx * f0_hz
            r_pl, ell_pl, c_pl = _per_length_params(R_m, mu, rho, D)
            z_h = r_pl + 1j * omega_h * ell_pl
            y_h = 1j * omega_h * c_pl
            gamma_h = np.sqrt(z_h * y_h)
            kL_h = gamma_h * L_m

            P_u_h = node_P[u][harm_idx]
            P_v_h = node_P[v][harm_idx]

            if abs(kL_h) < 1e-6:
                # Small kL: lumped approx Q_stored ≈ iωcL · (P_u+P_v)/2
                Q_stored_harmonics[harm_idx] = (
                    1j * omega_h * c_pl * L_m * (P_u_h + P_v_h) / 2.0
                ) * 1e12  # m³/s → nL/s
            elif abs(kL_h.real) > 500:
                # Heavily damped: negligible storage
                Q_stored_harmonics[harm_idx] = 0.0
            else:
                e_pos = np.exp(kL_h)
                e_neg = np.exp(-kL_h)
                sinh_kL = (e_pos - e_neg) / 2.0
                cosh_kL = (e_pos + e_neg) / 2.0
                if abs(sinh_kL) > 1e-30:
                    Q_stored_harmonics[harm_idx] = (
                        (gamma_h / r_pl)
                        * (P_u_h + P_v_h) * (cosh_kL - 1.0) / sinh_kL
                    ) * 1e12  # → nL/s

        Q_stored_dict[(u, v)] = Q_stored_harmonics
        # Storage fraction: |Q̂_1^stored| / |Q̂_1^(μ)| at fundamental
        q1_in = abs(Q_harmonics[1]) if n_harmonics >= 1 else 0.0
        q1_stored = abs(Q_stored_harmonics[1]) if n_harmonics >= 1 else 0.0
        storage_frac_dict[(u, v)] = (q1_stored / q1_in
                                     if q1_in > 1e-30 else np.nan)

    # Assign boundary edges:
    #  - Q-BC nodes (sources): override from BC harmonics (as before)
    #  - P-BC nodes (sinks):   Q already computed from pressure solve, no override
    n_be_assigned = 0
    for bn in boundary_nodes:
        if bn in solve_flow_bc_nodes:
            # P-BC node: boundary edge flow comes from the pressure solve.
            # It's already in edge_flows from the reconstruction loop above.
            # Just ensure derived metrics are computed.
            nbrs = list(G.neighbors(bn))
            interior = nbrs[0] if len(nbrs) == 1 else max(
                nbrs, key=lambda n: abs(G.edges[bn, n].get('mean_Q', 0)))
            if G.has_edge(bn, interior):
                be = (bn, interior)
            elif G.has_edge(interior, bn):
                be = (interior, bn)
            else:
                continue
            if be in edge_flows:
                Q_harm_be = edge_flows[be]
                q_dc = Q_harm_be[0].real
                q1 = Q_harm_be[1] if n_harmonics >= 1 else 0.0
                mean_Q_dict[be] = q_dc
                amp_Q_dict[be] = abs(q1)
                PI_dict[be] = 2.0 * abs(q1) / abs(q_dc) if abs(q_dc) > 1e-12 else np.nan
                RPSI_dict[be] = compute_rpsi_from_harmonics(Q_harm_be, f0_hz)
                eta_dict[be] = compute_eta_from_harmonics(Q_harm_be, f0_hz)
                bc_type = 'P-BC' if bn in set(p_bc_nodes) else 'Z-BC'
                if verbose:
                    print(f"  {bc_type} edge {be}: Q_dc={q_dc:.3f} (from solve), "
                          f"|Q_1|={abs(q1):.3f}, PI={PI_dict[be]:.3f}")
            n_be_assigned += 1
            continue

        # Q-BC node (source): override from BC harmonics
        nbrs = list(G.neighbors(bn))
        interior = nbrs[0] if len(nbrs) == 1 else max(
            nbrs, key=lambda n: abs(G.edges[bn, n].get('mean_Q', 0)))
        if interior is None:
            continue
        if G.has_edge(bn, interior):
            be = (bn, interior)
        elif G.has_edge(interior, bn):
            be = (interior, bn)
        else:
            continue
        Q_bc = bc_harmonics[bn]
        edge_flows[be] = Q_bc
        q_dc = Q_bc[0].real
        q1 = Q_bc[1] if n_harmonics >= 1 else 0.0
        mean_Q_dict[be] = q_dc
        amp_Q_dict[be] = abs(q1)
        PI_dict[be] = 2.0 * abs(q1) / abs(q_dc) if abs(q_dc) > 1e-12 else np.nan
        RPSI_dict[be] = compute_rpsi_from_harmonics(Q_bc, f0_hz)
        eta_dict[be] = compute_eta_from_harmonics(Q_bc, f0_hz)
        n_be_assigned += 1
        if verbose:
            print(f"  Q-BC edge {be}: Q_dc={q_dc:.3f}, |Q_1|={abs(q1):.3f}, "
                  f"PI={PI_dict[be]:.3f}")
        # Phase = forward-direction Q_H1 (sign-canonicalised so DC ≥ 0).
        # CHANGED 2026-05-27 (see note at the interior-edge phase block).
        if n_harmonics >= 1 and abs(q1) > 0:
            Q_canon_be = Q_bc if Q_bc[0].real >= 0 else -Q_bc
            phase_dict[be] = float(np.angle(Q_canon_be[1]))
        else:
            phase_dict[be] = np.nan
        # Legacy pressure-phase kept alongside.
        interior_legacy = bc_remap.get(bn, bn)
        if interior_legacy in node_P and n_harmonics >= 1:
            pressure_phase_dict[be] = float(np.angle(node_P[interior_legacy][1]))
        else:
            pressure_phase_dict[be] = np.nan
        # WSS and dissipation for boundary edge
        R_m, L_m = _get_edge_geometry(G, be[0], be[1], radii_m=radii_m)
        if R_m is not None:
            _Q_t_be = np.full(_n_pts, Q_bc[0].real)
            for _nh in range(1, n_harmonics + 1):
                _Q_t_be += (Q_bc[_nh].real * np.cos(_nh * _t_cyc)
                            - Q_bc[_nh].imag * np.sin(_nh * _t_cyc))
            _Q_mean_abs_be = float(np.mean(np.abs(_Q_t_be))) * 1e-12
            WSS_dict[be] = 4.0 * mu * _Q_mean_abs_be / (np.pi * R_m ** 3)
            Q_dc_sq = (q_dc * 1e-12) ** 2
            Q_ac_sq = 0.5 * sum(abs(Q_bc[n] * 1e-12) ** 2
                                for n in range(1, n_harmonics + 1))
            r_pl = 8.0 * mu / (np.pi * R_m ** 4)
            dissipation_dict[be] = r_pl * L_m * (Q_dc_sq + Q_ac_sq)
            pulsatile_cost_dict[be] = ((Q_dc_sq + Q_ac_sq) / Q_dc_sq
                                       if Q_dc_sq > 1e-30 else np.nan)

    if verbose:
        print(f"  Assigned {n_be_assigned} boundary edges from BCs")
        n_finite = sum(1 for v in PI_dict.values() if np.isfinite(v))
        print(f"  Solved: {len(edge_flows)} edges, {n_finite} with finite PI")
        pis = [v for v in PI_dict.values() if np.isfinite(v)]
        if pis:
            print(f"  PI: median={np.median(pis):.3f}, "
                  f"mean={np.mean(pis):.3f}, range=[{min(pis):.3f}, {max(pis):.3f}]")
        qs = [abs(v) for v in mean_Q_dict.values() if abs(v) > 1e-12]
        if qs:
            print(f"  |mean_Q|: median={np.median(qs):.3f} nL/s, "
                  f"range=[{min(qs):.3f}, {max(qs):.3f}]")

        # Flow conservation check at boundary nodes
        for node in boundary_nodes:
            total_q = 0.0
            for nb in G.neighbors(node):
                if (node, nb) in mean_Q_dict:
                    total_q += mean_Q_dict[(node, nb)]
                elif (nb, node) in mean_Q_dict:
                    total_q -= mean_Q_dict[(nb, node)]
            role = 'Source' if node in source_nodes else 'Sink'
            print(f"  {role} {node}: net Q = {total_q:.3f} nL/s")

    if verbose and WSS_dict:
        wss_vals = [v for v in WSS_dict.values() if np.isfinite(v)]
        if wss_vals:
            print(f"  WSS: median={np.median(wss_vals):.3f} Pa, "
                  f"range=[{min(wss_vals):.3f}, {max(wss_vals):.3f}]")
        diss_vals = list(dissipation_dict.values())
        total_diss = sum(diss_vals)
        print(f"  Total dissipation: {total_diss:.3e} W")
        pc_vals = [v for v in pulsatile_cost_dict.values() if np.isfinite(v)]
        if pc_vals:
            print(f"  Pulsatile cost ⟨Q²⟩/Q̄²: median={np.median(pc_vals):.2f}, "
                  f"range=[{min(pc_vals):.2f}, {max(pc_vals):.2f}]")

        # Compliance storage diagnostics
        sf_vals = [v for v in storage_frac_dict.values() if np.isfinite(v)]
        if sf_vals:
            print(f"  Storage fraction |Q̂₁ˢᵗᵒʳᵉᵈ|/|Q̂₁ⁱⁿ|: "
                  f"median={np.median(sf_vals):.4f}, "
                  f"range=[{min(sf_vals):.4f}, {max(sf_vals):.4f}]")
            # Global conservation check: ΣQ̂_n^stored = ΣQ̂_n^ext
            for nh in range(1, n_harmonics + 1):
                total_stored = sum(
                    Q_stored_dict[k][nh] for k in Q_stored_dict) * 1e-12
                total_ext = sum(
                    bc_harmonics[bn][nh] for bn in q_bc_nodes) * 1e-12
                err = abs(total_stored - total_ext)
                rel = err / max(abs(total_ext), 1e-30)
                print(f"  Conservation h={nh}: "
                      f"|ΣQ_stored|={abs(total_stored)*1e12:.4f}, "
                      f"|ΣQ_ext|={abs(total_ext)*1e12:.4f}, "
                      f"rel_err={rel:.2e}")

    # Store representative scalar D for display (evaluate at R=10 µm if callable)
    D_scalar = D(10e-6) if callable(D) else D

    return TransmissionLineResult(
        edge_flows=edge_flows,
        node_pressures=node_P,
        mean_Q=mean_Q_dict,
        amp_Q=amp_Q_dict,
        PI=PI_dict,
        RPSI=RPSI_dict,
        eta=eta_dict,
        phase=phase_dict,
        pressure_phase=pressure_phase_dict,
        WSS=WSS_dict,
        dissipation=dissipation_dict,
        pulsatile_cost=pulsatile_cost_dict,
        Q_stored=Q_stored_dict,
        storage_fraction=storage_frac_dict,
        f0_hz=f0_hz,
        n_harmonics=n_harmonics,
        D=D_scalar,
        mu=mu,
        boundary_nodes=boundary_nodes,
        boundary_Q_harmonics=bc_harmonics,
        n_edges=len(edge_flows),
        n_nodes=N,
    )


def optimize_transmission_line(
    G: nx.Graph,
    n_harmonics: int = 3,
    f0_hz: Optional[float] = None,
    D_init: float = 1e-3,
    mu_init: float = 3.5e-3,
    bc_harmonics_ref: Optional[Dict[int, np.ndarray]] = None,
    radii_m: Optional[Dict[Tuple[int, int], float]] = None,
    D_bounds: Tuple[float, float] = (1e-5, 1e-2),
    mu_bounds: Tuple[float, float] = (1e-3, 5e-3),
    s_bounds: Tuple[float, float] = (0.3, 3.0),
    max_iter: int = 200,
    verbose: bool = True,
    **solver_kwargs,
) -> dict:
    """Optimize D and boundary scales using harmonic amplitude ratios.

    For each candidate D, solves the network once per boundary node
    (Green's function approach).  Predicted edge flows are superpositions:
        Q̂_n,e = Σ_b  s_b · Q̂_n,e^(b)(D)
    The loss compares harmonic amplitude ratios ρ₂/₁ and ρ₃/₁ between
    predicted and measured, weighted by pulsatile power.

    Parameters
    ----------
    G : nx.Graph with measurements_piv on edges and boundary nodes
    n_harmonics : number of AC harmonics (default 3)
    f0_hz : cardiac frequency (Hz)
    D_init : initial distensibility guess
    mu_init : viscosity (fixed, not optimized — well constrained by biology)
    bc_harmonics_ref : reference boundary harmonics
    radii_m : optional radius overrides
    D_bounds : distensibility bounds
    mu_bounds : viscosity bounds
    s_bounds : boundary scale factor bounds
    max_iter : max optimizer iterations
    solver_kwargs : passed to solve_transmission_line

    Returns
    -------
    dict with 'D', 'mu', 'scales', 'loss', 'result', 'history', 'residuals'
    """
    from scipy.optimize import minimize
    from .config import FRAME_DT_S

    # --- Identify boundary nodes ---
    boundary_nodes = [n for n, d in G.nodes(data=True)
                      if d.get('boundary_type') is not None]
    if len(boundary_nodes) < 2:
        raise ValueError(f"Need >= 2 boundary nodes, found {len(boundary_nodes)}")

    source_nodes, sink_nodes = _classify_boundary_nodes(G, boundary_nodes)

    # --- Extract reference BCs (with correct signs) ---
    if bc_harmonics_ref is None:
        bc_harmonics_ref = _extract_boundary_harmonics(
            G, boundary_nodes, f0_hz, n_harmonics, FRAME_DT_S)
    for bn in boundary_nodes:
        if bn in sink_nodes and bc_harmonics_ref[bn][0].real > 0:
            bc_harmonics_ref[bn] = -bc_harmonics_ref[bn]
        elif bn in source_nodes and bc_harmonics_ref[bn][0].real < 0:
            bc_harmonics_ref[bn] = -bc_harmonics_ref[bn]

    # --- Extract measured harmonics from PIV data ---
    meas_edges = {}
    for u, v, d in G.edges(data=True):
        piv_list = d.get('measurements_piv', [])
        if not piv_list:
            continue
        best = max(piv_list, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        f0_m = best.get('f0_hz', f0_hz)
        if Qt is None or len(Qt) < 20 or f0_m is None:
            continue

        from .harmonic import fit_harmonics
        Qt_arr = np.asarray(Qt, dtype=float)
        if np.nanmean(Qt_arr) < 0:
            Qt_arr = -Qt_arr

        hr = fit_harmonics(Qt_arr, f0_m, FRAME_DT_S,
                           K=n_harmonics, loss='huber', include_dc=True)
        q_dc = hr.get('a0', float(np.nanmean(Qt_arr)))
        if not np.isfinite(q_dc) or abs(q_dc) < 0.1:
            continue

        harmonics_complex = np.zeros(n_harmonics + 1, dtype=complex)
        harmonics_complex[0] = q_dc
        for h in hr.get('harmonics', []):
            k = h['k']
            if k <= n_harmonics:
                harmonics_complex[k] = h['A'] - 1j * h['B']

        # Compute measured amplitude ratios ρ₂/₁, ρ₃/₁
        amp1 = abs(harmonics_complex[1]) if n_harmonics >= 1 else 0
        if amp1 < 1e-12:
            continue
        ratios_meas = []
        for n in range(2, n_harmonics + 1):
            ratios_meas.append(abs(harmonics_complex[n]) / amp1)

        meas_edges[(u, v)] = {
            'q_dc': q_dc,
            'harmonics': harmonics_complex,
            'amp1': amp1,
            'ratios': ratios_meas,  # [ρ₂/₁, ρ₃/₁, ...]
            'f0_meas': f0_m,
        }

    if not meas_edges:
        raise ValueError("No edges with measurable pulsatility found")

    # Weights: w_e ∝ (|Q̄|² + |Q̂₁|²) × freq_proximity
    # Frequency proximity: Gaussian centered on sim f0, σ = 5% of f0
    # Vessels at the same frequency as the sim get weight 1.0;
    # vessels with very different f0 are down-weighted.
    f0_sigma = 0.05 * f0_hz  # 5% bandwidth
    weights = {}
    for key, m in meas_edges.items():
        w_power = abs(m['q_dc'])**2 + m['amp1']**2
        df = m['f0_meas'] - f0_hz
        w_freq = np.exp(-0.5 * (df / f0_sigma)**2) if f0_sigma > 0 else 1.0
        weights[key] = w_power * w_freq
    w_total = sum(weights.values())
    if w_total > 0:
        weights = {k: v / w_total for k, v in weights.items()}

    N_meas = len(meas_edges)
    N_boundary = len(boundary_nodes)
    bn_list = sorted(boundary_nodes)
    # Fix one boundary scale to 1.0 (only relative scales matter)
    fixed_bn = bn_list[0]
    free_bn = bn_list[1:]
    N_free_s = len(free_bn)

    edge_keys = list(meas_edges.keys())

    if verbose:
        _eq = '=' * 60
        print(f"\n{_eq}")

        print(f"TRANSMISSION LINE OPTIMIZATION (Green's function)")
        print(f"{_eq}")
        print(f"  {N_meas} edges with |Q̄|≥0.1 and |Q̂₁|>0")
        print(f"  {N_boundary} boundary nodes ({len(source_nodes)} sources, "
              f"{len(sink_nodes)} sinks)")
        print(f"  Parameters: D, µ, {N_free_s} relative scales = "
              f"{2 + N_free_s} total")
        print(f"  Fixed scale: node {fixed_bn} = 1.0")
        print(f"  D_init={D_init:.2e}, µ_init={mu_init*1e3:.1f} mPa·s")
        for bn in bn_list:
            role = 'Source' if bn in source_nodes else 'Sink'
            print(f"  {role} {bn}: Q_dc_ref={bc_harmonics_ref[bn][0].real:.3f} nL/s")

    # --- Parameter packing ---
    # theta = [log10(D), log10(mu), s_1, s_2, ..., s_{N-1}]
    # (s_0 = fixed_bn is always 1.0)

    def pack(D, mu, scales):
        return np.concatenate([
            [np.log10(D), np.log10(mu)],
            [scales.get(bn, 1.0) for bn in free_bn],
        ])

    def unpack(theta):
        D = 10**theta[0]
        mu = 10**theta[1]
        scales = {fixed_bn: 1.0}
        for i, bn in enumerate(free_bn):
            scales[bn] = theta[2 + i]
        return D, mu, scales

    theta0 = pack(D_init, mu_init, {bn: 1.0 for bn in bn_list})
    bounds = (
        [(np.log10(D_bounds[0]), np.log10(D_bounds[1])),
         (np.log10(mu_bounds[0]), np.log10(mu_bounds[1]))]
        + [(s_bounds[0], s_bounds[1])] * N_free_s
    )

    # --- Green's function cache ---
    # For each D, solve once per boundary node with unit excitation
    # at that node only.  Cache keyed by (D, mu) rounded.
    _green_cache = {}

    def _compute_greens(D, mu):
        """Solve once per boundary node → per-edge harmonic responses."""
        cache_key = (round(np.log10(D), 6), round(np.log10(mu), 6))
        if cache_key in _green_cache:
            return _green_cache[cache_key]

        # For each boundary node, solve with only that node's BC active
        greens = {}  # bn -> {edge_key: complex array [Q_dc, Q_1, ...]}
        for bn in bn_list:
            bc_single = {b: np.zeros(n_harmonics + 1, dtype=complex)
                         for b in bn_list}
            bc_single[bn] = bc_harmonics_ref[bn].copy()

            try:
                result = solve_transmission_line(
                    G, D=D, n_harmonics=n_harmonics, f0_hz=f0_hz,
                    mu=mu, bc_harmonics_override=bc_single,
                    radii_m=radii_m, verbose=False,
                    **solver_kwargs,
                )
            except Exception:
                greens[bn] = {}
                continue

            bn_flows = {}
            for ek in edge_keys:
                eu, ev = ek
                Q = result.edge_flows.get((eu, ev),
                        result.edge_flows.get((ev, eu)))
                if Q is not None:
                    bn_flows[ek] = Q.copy()
            greens[bn] = bn_flows

        _green_cache[cache_key] = greens
        return greens

    # --- Loss function ---
    history = []
    eval_count = [0]

    def loss_fn(theta):
        D, mu, scales = unpack(theta)
        greens = _compute_greens(D, mu)

        L = 0.0
        n_used = 0

        for ek, m in meas_edges.items():
            w = weights.get(ek, 0.0)
            ratios_meas = m['ratios']

            # Superpose: Q̂_n,e = Σ_b s_b · Q̂_n,e^(b)
            Q_super = np.zeros(n_harmonics + 1, dtype=complex)
            for bn in bn_list:
                Q_bn = greens.get(bn, {}).get(ek)
                if Q_bn is not None:
                    Q_super += scales[bn] * Q_bn

            # Predicted amplitude ratios
            amp1_pred = abs(Q_super[1]) if n_harmonics >= 1 else 0
            if amp1_pred < 1e-15:
                continue

            n_used += 1
            for i, n in enumerate(range(2, n_harmonics + 1)):
                rho_pred = abs(Q_super[n]) / amp1_pred
                rho_meas = ratios_meas[i] if i < len(ratios_meas) else 0
                L += w * (rho_pred - rho_meas)**2

        eval_count[0] += 1
        if verbose and eval_count[0] % 5 == 0:
            s_str = ', '.join(f'{scales[bn]:.2f}' for bn in bn_list)
            print(f"    eval {eval_count[0]}: loss={L:.6f} "
                  f"(n={n_used}/{N_meas}) "
                  f"D={D:.2e}, µ={mu*1e3:.2f}mPa·s, s=[{s_str}]")
        history.append((eval_count[0], L))
        return L

    # --- Optimize ---
    if verbose:
        print(f"  Starting L-BFGS-B optimization (max_iter={max_iter})...")
        loss0 = loss_fn(theta0)
        print(f"  Initial loss: {loss0:.6f}")

    opt = minimize(loss_fn, theta0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})

    D_opt, mu_opt, scales_opt = unpack(opt.x)

    if verbose:
        print(f"\nOptimization {'converged' if opt.success else 'stopped'}: "
              f"{opt.message}")
        print(f"  Final loss: {opt.fun:.6f} ({eval_count[0]} evaluations)")
        print(f"  D = {D_opt:.4e} 1/Pa  (init {D_init:.2e})")
        print(f"  µ = {mu_opt*1e3:.3f} mPa·s  (init {mu_init*1e3:.1f})")
        for bn in bn_list:
            role = 'Source' if bn in source_nodes else 'Sink'
            q_ref = bc_harmonics_ref[bn][0].real
            print(f"  {role} {bn}: s={scales_opt[bn]:.3f}  "
                  f"(Q_dc: {q_ref:.3f} → {q_ref*scales_opt[bn]:.3f} nL/s)")

    # --- Final solve at optimum with scaled BCs ---
    bc_opt = {bn: bc_harmonics_ref[bn] * scales_opt[bn] for bn in bn_list}
    result_opt = solve_transmission_line(
        G, D=D_opt, n_harmonics=n_harmonics, f0_hz=f0_hz,
        mu=mu_opt, bc_harmonics_override=bc_opt,
        radii_m=radii_m, verbose=verbose,
        **solver_kwargs,
    )

    # --- Per-edge residuals at optimum ---
    residuals = {}
    for ek, m in meas_edges.items():
        eu, ev = ek
        key = (eu, ev)
        key_r = (ev, eu)
        Q_pred = result_opt.edge_flows.get(key, result_opt.edge_flows.get(key_r))
        if Q_pred is None:
            continue
        amp1_pred = abs(Q_pred[1])
        if amp1_pred < 1e-15:
            continue
        r_e = 0.0
        for i, n in enumerate(range(2, n_harmonics + 1)):
            rho_pred = abs(Q_pred[n]) / amp1_pred
            rho_meas = m['ratios'][i] if i < len(m['ratios']) else 0
            r_e += (rho_pred - rho_meas)**2
        residuals[ek] = float(r_e)

    if verbose:
        r_vals = [v for v in residuals.values() if np.isfinite(v)]
        if r_vals:
            print(f"  Per-edge residuals: median={np.median(r_vals):.4f}, "
                  f"mean={np.mean(r_vals):.4f}, max={max(r_vals):.4f}")
        else:
            print(f"  WARNING: No finite per-edge residuals computed")
        print(f"{_eq}")

    return {
        'D': D_opt,
        'mu': mu_opt,
        'scales': scales_opt,
        'loss': float(opt.fun),
        'result': result_opt,
        'history': history,
        'residuals': residuals,
        'bc_harmonics': bc_opt,
        'optimizer_result': opt,
    }


def optimize_greyzone_refine(
    G: nx.Graph,
    excluded_nodes: set,
    vessel_groups: Dict[Tuple[int, int], int],
    art_nodes: List[int],
    ven_nodes: List[int],
    v_ref: int,
    Q_art: Dict[int, np.ndarray],
    Q_ven: Dict[int, np.ndarray],
    radii_m_base: Dict[Tuple[int, int], float],
    n_harmonics: int = 3,
    f0_hz: float = 2.5,
    mu: float = MU_DEFAULT,
    D_init: float = 1e-3,
    tile_id: Optional[int] = None,
    q_min: float = 0.1,
    lambda_prior: float = 1.0,
    lambda_murray: float = 0.01,
    sigma_R_frac: float = 0.3,
    scale_bounds: Tuple[float, float] = (0.5, 2.0),
    max_iter: int = 200,
    max_hops_from_boundary: Optional[int] = None,
    global_scale_first: bool = True,
    verbose: bool = True,
) -> dict:
    """Grey-zone optimizer: preserve topology, optimize per-vessel radius scales.

    Keeps every vessel in the grey zone in the network. For each grey-zone
    vessel (chain of edges between junctions), a single scale factor s_v is
    optimized such that R_edge = s_v * R_meas. All edges in the same vessel
    share the same scale — dimensionality = number of vessels, not edges.

    Loss:
        L = L_shape(external waveforms)                              # data
          + λ_prior · Σ_vessels ((s_v - 1) / σ_R_frac)²               # stay near measured
          + λ_murray · Σ_grey_junctions (ΣR³_children - R³_parent)² / R⁶  # Murray
    Shape loss uses the L2 complex ratio (Q̂_n / Q̂_1) over edges outside the
    grey zone, weighted by pulsatile power and graph proximity to the zone.

    Parameters
    ----------
    vessel_groups : dict {(u,v) (sorted): vessel_id}
    radii_m_base : baseline measured radii per edge (meters)
    tile_id : if given, restrict the loss to edges with a PIV measurement
              from this tile. The reference f0 is that tile's f0.
    sigma_R_frac : fractional std of the measured-radius prior (0.3 = 30%)
    scale_bounds : per-vessel scale bounds (below 1 = narrower than meas)

    Returns
    -------
    dict with 'D', 'scales' (per-vessel), 'result', 'loss', 'residuals',
    'radii_m_opt' (optimized per-edge radii).
    """
    from .config import FRAME_DT_S
    from scipy.optimize import minimize as sp_min

    omega0 = 2.0 * np.pi * f0_hz

    # Identify grey-zone edges and their vessel_ids
    grey_edges = []
    grey_vessels = {}  # vessel_id -> list of edge keys (sorted tuples)
    for u, v in G.edges():
        if u in excluded_nodes or v in excluded_nodes:
            ek = tuple(sorted([u, v]))
            vid_ves = vessel_groups.get(ek, -1)
            grey_edges.append(ek)
            grey_vessels.setdefault(vid_ves, []).append(ek)

    all_vessel_ids = sorted(grey_vessels.keys())

    # Restrict optimization to vessels within max_hops of grey-external boundary.
    # Boundary nodes: excluded nodes that have at least one external neighbor.
    if max_hops_from_boundary is not None and max_hops_from_boundary >= 0:
        boundary_in_grey = {n for n in excluded_nodes
                            if any(nb not in excluded_nodes
                                   for nb in G.neighbors(n))}
        from collections import deque
        node_hops_grey = {n: 0 for n in boundary_in_grey}
        queue = deque([(n, 0) for n in boundary_in_grey])
        while queue:
            n, h = queue.popleft()
            if h >= max_hops_from_boundary:
                continue
            for nb in G.neighbors(n):
                if nb not in excluded_nodes:
                    continue
                if nb not in node_hops_grey or node_hops_grey[nb] > h + 1:
                    node_hops_grey[nb] = h + 1
                    queue.append((nb, h + 1))
        # A vessel is "near" if ANY of its edges touches a near-boundary node
        near_vessel_ids = set()
        for vid_v, eks in grey_vessels.items():
            for u, v in eks:
                if u in node_hops_grey or v in node_hops_grey:
                    near_vessel_ids.add(vid_v)
                    break
        vessel_ids_ordered = sorted(near_vessel_ids)
    else:
        vessel_ids_ordered = all_vessel_ids

    N_vessels = len(vessel_ids_ordered)
    N_all_vessels = len(all_vessel_ids)
    vessel_ids_frozen = sorted(set(all_vessel_ids) - set(vessel_ids_ordered))

    if verbose:
        _eq = '=' * 60
        print(f"\n{_eq}")
        print("GREY-ZONE REFINE OPTIMIZER (per-vessel scale)")
        print(f"{_eq}")
        print(f"  Grey edges: {len(grey_edges)}, grey vessels: {N_all_vessels}")
        if max_hops_from_boundary is not None:
            print(f"  Optimizing {N_vessels} vessels (within {max_hops_from_boundary} "
                  f"hops of boundary); {len(vessel_ids_frozen)} interior vessels "
                  f"frozen at s=1")
        else:
            print(f"  Optimizing all {N_vessels} vessels")
        print(f"  Scale bounds: {scale_bounds}, prior σ_R={sigma_R_frac*100:.0f}%")
        print(f"  Art: {art_nodes}, Ven: {ven_nodes}, gauge: {v_ref}")
        if tile_id is not None:
            print(f"  Tile-based loss: only edges with tile {tile_id} PIV")

    # Extract measured harmonics — optionally tile-filtered
    meas_edges = {}
    for u, v, d in G.edges(data=True):
        if u in excluded_nodes and v in excluded_nodes:
            continue  # don't use grey-interior measurements to fit its own scales
        piv_list = d.get('measurements_piv', [])
        if not piv_list:
            continue
        if tile_id is not None:
            tile_piv = [m for m in piv_list if m.get('tile_id') == tile_id]
            if not tile_piv:
                continue
            best = tile_piv[0]
        else:
            best = max(piv_list, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        f0_m = best.get('f0_hz', f0_hz)
        if Qt is None or len(Qt) < 20:
            continue
        from .harmonic import fit_harmonics
        Qt_arr = np.asarray(Qt, dtype=float)
        if np.nanmean(Qt_arr) < 0:
            Qt_arr = -Qt_arr
        hr = fit_harmonics(Qt_arr, f0_m, FRAME_DT_S,
                           K=n_harmonics, loss='huber', include_dc=True)
        q_dc = hr.get('a0', float(np.nanmean(Qt_arr)))
        if not np.isfinite(q_dc) or abs(q_dc) < q_min:
            continue
        harmonics = np.zeros(n_harmonics + 1, dtype=complex)
        harmonics[0] = q_dc
        for h in hr.get('harmonics', []):
            kk = h['k']
            if kk <= n_harmonics:
                harmonics[kk] = h['A'] - 1j * h['B']
        meas_edges[(u, v)] = {'harmonics': harmonics, 'f0_meas': f0_m}

    N_meas = len(meas_edges)
    if N_meas == 0:
        raise ValueError("No measured edges available for loss evaluation"
                         + (f" (tile {tile_id})" if tile_id else ""))
    edge_keys = list(meas_edges.keys())

    # Weights: pulsatile power × freq proximity × BFS proximity to grey zone
    from collections import deque
    node_hops = {}
    bfs_queue = deque()
    for ex in excluded_nodes:
        node_hops[ex] = 0
        bfs_queue.append((ex, 0))
    max_hops = 10
    while bfs_queue:
        n, h = bfs_queue.popleft()
        if h >= max_hops:
            continue
        for nb in G.neighbors(n):
            if nb not in node_hops or node_hops[nb] > h + 1:
                node_hops[nb] = h + 1
                bfs_queue.append((nb, h + 1))

    f0_sigma = 0.05 * f0_hz
    weights = np.zeros(N_meas)
    for ei, ek in enumerate(edge_keys):
        m = meas_edges[ek]
        h = m['harmonics']
        w_p = abs(h[0])**2 + abs(h[1])**2
        df = m['f0_meas'] - f0_hz
        w_f = np.exp(-0.5 * (df / f0_sigma)**2) if f0_sigma > 0 else 1.0
        eu, ev = ek
        h_u = node_hops.get(eu, max_hops + 1)
        h_v = node_hops.get(ev, max_hops + 1)
        min_h = min(h_u, h_v)
        w_prox = np.exp(-0.5 * (min_h / 3.0)**2)
        weights[ei] = w_p * w_f * w_prox
    w_sum = weights.sum()
    if w_sum > 0:
        weights /= w_sum

    # Murray structure: for each junction inside grey zone, find
    # incident vessel radii (use parent = largest).
    grey_junctions = [n for n in excluded_nodes if G.degree(n) >= 3]
    junction_neighbors = {}
    for jn in grey_junctions:
        incident = []
        for nb in G.neighbors(jn):
            ek = tuple(sorted([jn, nb]))
            vid_v = vessel_groups.get(ek, -1)
            R_m = radii_m_base.get((jn, nb), radii_m_base.get((nb, jn)))
            if vid_v >= 0 and R_m is not None and R_m > 0:
                incident.append((vid_v, R_m))
        if len(incident) >= 2:
            junction_neighbors[jn] = incident

    def eval_once(scales_vec, D):
        """Build radii, solve, compute shape loss + priors."""
        # Build radii_m from scales
        radii_m_new = dict(radii_m_base)
        for vi, vid_v in enumerate(vessel_ids_ordered):
            s = scales_vec[vi]
            for ek in grey_vessels[vid_v]:
                u, v = ek
                R_base = radii_m_base.get((u, v), radii_m_base.get((v, u)))
                if R_base is not None:
                    radii_m_new[(u, v)] = s * R_base

        bc_override = dict(Q_art)
        bc_override.update(Q_ven)
        try:
            result = solve_transmission_line(
                G, D=D, n_harmonics=n_harmonics, f0_hz=f0_hz, mu=mu,
                bc_harmonics_override=bc_override,
                radii_m=radii_m_new, verbose=False)
        except Exception:
            return 1e10, None, radii_m_new

        # Unnormalized harmonic loss — sums |Q̂_n (pred) − Q̂_n (meas)|²
        # over DC + all AC harmonics. Keeps magnitude information so that
        # per-vessel scales can be identified by absolute flow differences.
        # Sign-align pred to meas DC to avoid global sign ambiguity.
        L_shape = 0.0
        for ei, ek in enumerate(edge_keys):
            Q_pred = result.edge_flows.get(ek,
                        result.edge_flows.get((ek[1], ek[0])))
            if Q_pred is None:
                continue
            Qm = meas_edges[ek]['harmonics']
            # Sign flip so pred DC matches meas DC sign
            sign = 1.0
            if (Qm[0].real != 0 and Q_pred[0].real != 0 and
                    np.sign(Qm[0].real) != np.sign(Q_pred[0].real)):
                sign = -1.0
            for n in range(0, n_harmonics + 1):
                L_shape += weights[ei] * abs(sign * Q_pred[n] - Qm[n])**2

        # Prior: scales near 1 (measured radius)
        L_prior = float(np.sum(((scales_vec - 1.0) / sigma_R_frac)**2))

        # Murray penalty at grey-zone junctions:
        # Parent = largest vessel (largest R_base); children = others
        L_murray = 0.0
        for jn, incident in junction_neighbors.items():
            # Compute current radii after scaling
            r_cubes = []
            for vid_v, R_base in incident:
                try:
                    i_vid = vessel_ids_ordered.index(vid_v)
                    s = scales_vec[i_vid]
                except ValueError:
                    s = 1.0
                r_cubes.append((s * R_base) ** 3)
            if len(r_cubes) >= 2:
                r_cubes_sorted = sorted(r_cubes, reverse=True)
                r_par3 = r_cubes_sorted[0]
                r_chi3 = sum(r_cubes_sorted[1:])
                if r_par3 > 1e-30:
                    L_murray += ((r_chi3 - r_par3) / r_par3)**2

        L_total = L_shape + lambda_prior * L_prior + lambda_murray * L_murray
        return L_total, result, radii_m_new

    # Optimize scales at fixed D_init
    eval_count = [0]
    history = []

    def _loss(scales_vec):
        L, _, _ = eval_once(scales_vec, D_init)
        eval_count[0] += 1
        if verbose and eval_count[0] % 5 == 0:
            print(f"    eval {eval_count[0]}: L={L:.6f}  "
                  f"s∈[{scales_vec.min():.2f}, {scales_vec.max():.2f}]")
        history.append((eval_count[0], L))
        return L

    s0 = np.ones(N_vessels)
    bounds = [scale_bounds] * N_vessels
    if verbose:
        L0 = _loss(s0)
        print(f"  Initial loss (all s=1): {L0:.6f}  ({N_vessels}-dim problem)")

    # Stage 1: scalar global scale warm-start (1-D, fast)
    # Fit a single scale factor for all near-boundary vessels together,
    # then use that as the starting point for the per-vessel pass.
    if global_scale_first and N_vessels > 1:
        if verbose:
            print(f"\n  STAGE 1: global scale warm-start (1-D, ~15 evals)")

        def _loss_global(s_scalar_vec):
            s_vec = np.full(N_vessels, s_scalar_vec[0])
            L, _, _ = eval_once(s_vec, D_init)
            eval_count[0] += 1
            if verbose:
                print(f"    global eval {eval_count[0]}: "
                      f"s={s_scalar_vec[0]:.3f}, L={L:.6f}")
            history.append((eval_count[0], L))
            return L

        opt_g = sp_min(_loss_global, np.array([1.0]),
                       method='L-BFGS-B',
                       bounds=[scale_bounds],
                       options={'maxiter': 20, 'eps': 0.02, 'ftol': 1e-5})
        s_global = float(opt_g.x[0])
        s0 = np.full(N_vessels, s_global)
        if verbose:
            print(f"  STAGE 1 done: global s* = {s_global:.3f}, L = {opt_g.fun:.6f}")

    # --- Stage 2: analytic-adjoint gradient ---
    # ∂L/∂s_v = explicit + implicit:
    #   explicit (through Y_e for measured edges on vessel v):
    #       Σ_{e∈v} 2·w_e·Re[conj(r_e) · (∂Y_diag,e/∂s_v · P_u + ∂Y_off,e/∂s_v · P_v)]
    #   implicit (through P via Y(s)):
    #       -λ^T · (dY/ds_v) · P    where  Y λ = ∂L/∂P, Y^T = Y (complex-symmetric)
    # Per harmonic: 1 forward + 1 adjoint solve (reuses LU factorization).
    # Total per gradient eval: 2 solves × (n_harmonics+1) instead of N_vessels solves.
    from scipy.sparse.linalg import splu as _splu
    import warnings as _warn

    # Build mod_node_to_idx (includes all graph nodes — same as solve_transmission_line)
    all_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    N_nodes = len(all_nodes)
    gauge_idx = node_to_idx[v_ref] if v_ref in node_to_idx else 0
    frequencies = [0.0] + [omega0 * k for k in range(1, n_harmonics + 1)]

    # Edges on each optimized vessel (for gradient accumulation)
    vessel_edge_indices = {}
    for vi, vid_v in enumerate(vessel_ids_ordered):
        vessel_edge_indices[vi] = list(grey_vessels[vid_v])

    # Map edge_keys (measured) to (ui, vi) for quick access
    meas_edge_uv = []
    for ek in edge_keys:
        eu, ev = ek
        if eu in node_to_idx and ev in node_to_idx:
            meas_edge_uv.append((ek, node_to_idx[eu], node_to_idx[ev]))

    def loss_and_grad(scales_vec):
        """Return (L, grad) using analytic adjoint. One forward + one adjoint per harmonic."""
        # Build per-edge radii from scales
        radii_m_new = dict(radii_m_base)
        for vi, vid_v in enumerate(vessel_ids_ordered):
            s = scales_vec[vi]
            for ek in grey_vessels[vid_v]:
                u, v = ek
                R_base = radii_m_base.get((u, v), radii_m_base.get((v, u)))
                if R_base is not None:
                    radii_m_new[(u, v)] = s * R_base

        # Accumulate L and grad over harmonics
        L_shape = 0.0
        grad_shape = np.zeros(N_vessels)

        for harm_idx, omega in enumerate(frequencies):
            # Build Y
            Y_full = _assemble_laplacian(
                G, omega,
                [(u, v) for u, v in G.edges()
                 if _get_edge_geometry(G, u, v, radii_m=radii_m_new)[0] is not None],
                node_to_idx, mu, RHO_BLOOD, D_init,
                radii_m=radii_m_new)
            Y_lil = Y_full.tolil()

            # Build RHS (injected Q at art/ven nodes in nL/s → m³/s)
            Q_rhs = np.zeros(N_nodes, dtype=complex)
            for n_node in list(Q_art.keys()) + list(Q_ven.keys()):
                if n_node in node_to_idx:
                    bc = Q_art.get(n_node, Q_ven.get(n_node,
                                    np.zeros(n_harmonics + 1)))
                    Q_rhs[node_to_idx[n_node]] = bc[harm_idx] * 1e-12

            # Apply gauge (pin P=0 at v_ref for DC)
            if harm_idx == 0:
                Y_lil[gauge_idx, :] = 0
                Y_lil[:, gauge_idx] = 0
                Y_lil[gauge_idx, gauge_idx] = 1.0
                Q_rhs[gauge_idx] = 0.0
            Y_csr = Y_lil.tocsr()

            # Factorize once, reuse for forward + adjoint
            try:
                Y_lu = _splu(Y_csr.tocsc())
            except Exception:
                return 1e10, np.zeros(N_vessels)

            # Forward solve
            with _warn.catch_warnings():
                _warn.simplefilter("ignore")
                P = Y_lu.solve(Q_rhs)
            if not np.all(np.isfinite(P)):
                return 1e10, np.zeros(N_vessels)

            # Compute residuals + ∂L/∂P (from measured edges)
            dL_dP = np.zeros(N_nodes, dtype=complex)
            residuals_arr = np.zeros(len(meas_edge_uv), dtype=complex)
            Y_diag_e_arr = np.zeros(len(meas_edge_uv), dtype=complex)
            Y_off_e_arr = np.zeros(len(meas_edge_uv), dtype=complex)
            signs = np.ones(len(meas_edge_uv))

            for ei, (ek, ui, vi_i) in enumerate(meas_edge_uv):
                Rm, Lm = _get_edge_geometry(G, *ek, radii_m=radii_m_new)
                if Rm is None:
                    continue
                Yd_e, Yo_e = _vessel_admittance(
                    Rm, Lm, omega, mu, RHO_BLOOD, D_init)
                Q_e = (Yd_e * P[ui] + Yo_e * P[vi_i]) * 1e12  # nL/s
                Qm = meas_edges[ek]['harmonics']
                # Global sign alignment from DC
                if harm_idx == 0:
                    if (Qm[0].real != 0 and Q_e.real != 0 and
                            np.sign(Qm[0].real) != np.sign(Q_e.real)):
                        signs[ei] = -1.0
                sign = signs[ei]
                r_e = sign * Q_e - Qm[harm_idx]
                w_e = weights[ei]
                L_shape += w_e * abs(r_e)**2

                # ∂L/∂P contributions (propagating the 1e12 scale back)
                # Q_e = (Yd P_u + Yo P_v) * 1e12
                # dL/dP_u = 2 w_e sign · conj(r_e) · Yd · 1e12
                dL_dQ = 2.0 * w_e * sign * np.conj(r_e) * 1e12
                dL_dP[ui] += dL_dQ * Yd_e
                dL_dP[vi_i] += dL_dQ * Yo_e

                residuals_arr[ei] = r_e
                Y_diag_e_arr[ei] = Yd_e
                Y_off_e_arr[ei] = Yo_e

            # Adjoint solve: Y^T λ = ∂L/∂P.  Y is complex-symmetric → Y^T = Y.
            # But gauge row was zeroed in Y, which is correct for forward; for adjoint
            # with symmetric Y this also works (λ[gauge]=0 forced).
            dL_dP[gauge_idx] = 0.0
            try:
                lam = Y_lu.solve(dL_dP)
            except Exception:
                lam = np.zeros(N_nodes, dtype=complex)
            lam[gauge_idx] = 0.0

            # Gradient contributions per vessel v
            # For each edge on vessel v, compute dY_e/ds_v and accumulate.
            # dR/ds_v = R_base (since R = s · R_base, dR/ds = R_base = R / s)
            for vi, vid_v in enumerate(vessel_ids_ordered):
                s_v = scales_vec[vi]
                if s_v < 1e-12:
                    continue
                for (eu_v, ev_v) in grey_vessels[vid_v]:
                    if eu_v not in node_to_idx or ev_v not in node_to_idx:
                        continue
                    Rm, Lm = _get_edge_geometry(
                        G, eu_v, ev_v, radii_m=radii_m_new)
                    if Rm is None:
                        continue
                    R_base_e = radii_m_base.get(
                        (eu_v, ev_v), radii_m_base.get((ev_v, eu_v)))
                    if R_base_e is None:
                        continue
                    # dY/dR via finite difference (small eps on R)
                    eps_R = Rm * 1e-5
                    Yd_p, Yo_p = _vessel_admittance(
                        Rm + eps_R, Lm, omega, mu, RHO_BLOOD, D_init)
                    Yd_m, Yo_m = _vessel_admittance(
                        Rm - eps_R, Lm, omega, mu, RHO_BLOOD, D_init)
                    dYd_dR = (Yd_p - Yd_m) / (2 * eps_R)
                    dYo_dR = (Yo_p - Yo_m) / (2 * eps_R)
                    # dY/ds = dY/dR * dR/ds = dY/dR * R_base
                    dYd_ds = dYd_dR * R_base_e
                    dYo_ds = dYo_dR * R_base_e

                    ui_e = node_to_idx[eu_v]
                    vi_e = node_to_idx[ev_v]

                    # Implicit term: -Re(λ^T · (dY_e/ds_v) · P)
                    # (dY_e/ds_v) · P has nonzero at rows ui_e, vi_e:
                    #   row ui_e: dYd_ds · P[ui_e] + dYo_ds · P[vi_e]
                    #   row vi_e: dYo_ds · P[ui_e] + dYd_ds · P[vi_e]
                    dYP_u = dYd_ds * P[ui_e] + dYo_ds * P[vi_e]
                    dYP_v = dYo_ds * P[ui_e] + dYd_ds * P[vi_e]
                    implicit = -(lam[ui_e] * dYP_u + lam[vi_e] * dYP_v).real

                    # Explicit term (if this edge is also a measured edge):
                    explicit = 0.0
                    for ei_meas, (ek_m, uim, vim) in enumerate(meas_edge_uv):
                        if (ek_m == (eu_v, ev_v) or ek_m == (ev_v, eu_v)):
                            w_e = weights[ei_meas]
                            sign = signs[ei_meas]
                            r_e = residuals_arr[ei_meas]
                            dQ_ds = (dYd_ds * P[uim] + dYo_ds * P[vim]) * 1e12
                            explicit += (2.0 * w_e * sign *
                                         (np.conj(r_e) * dQ_ds).real)

                    grad_shape[vi] += implicit + explicit

        # Prior + Murray (both analytic — trivial)
        L_prior = float(np.sum(((scales_vec - 1.0) / sigma_R_frac)**2))
        grad_prior = 2.0 * (scales_vec - 1.0) / (sigma_R_frac**2)

        L_murray = 0.0
        grad_murray = np.zeros(N_vessels)
        for jn, incident in junction_neighbors.items():
            r_cubes = []
            incident_info = []  # (i_vid in vessel_ids_ordered or -1, R_base, s, R³)
            for vid_v, R_base in incident:
                try:
                    i_vid = vessel_ids_ordered.index(vid_v)
                    s = scales_vec[i_vid]
                except ValueError:
                    i_vid = -1
                    s = 1.0
                R_cur = s * R_base
                r_cubes.append(R_cur ** 3)
                incident_info.append((i_vid, R_base, s, R_cur**3))
            if len(r_cubes) < 2:
                continue
            # Parent = index of largest R³; children = rest
            i_par = int(np.argmax(r_cubes))
            r_par3 = r_cubes[i_par]
            r_chi3 = sum(r_cubes) - r_par3
            if r_par3 < 1e-30:
                continue
            diff = (r_chi3 - r_par3) / r_par3
            L_murray += diff**2
            # ∂L_murray/∂s_i for each incident edge i
            # f(s) = (Σ_{j≠par} R³_j - R³_par) / R³_par = Σ(R³_j - R³_par)/R³_par... tricky
            # Use numerical simplification: d(diff)/ds_i where s_i scales R_i
            # R³_i = (s_i R_base_i)^3 → d R³_i / ds_i = 3 s_i^2 R_base_i^3
            # d(diff)/ds_i depends on whether i is parent or child
            for idx, (i_vid, R_base, s, R3) in enumerate(incident_info):
                if i_vid < 0:
                    continue
                dR3_ds = 3.0 * (s**2) * (R_base**3)
                if idx == i_par:
                    # R³_par in denominator AND in numerator sum
                    # diff = (r_chi3 - r_par3)/r_par3 = r_chi3/r_par3 - 1
                    # d(diff)/ds_par = -r_chi3 / r_par3^2 · dR3_ds
                    ddiff_ds = -r_chi3 / (r_par3**2) * dR3_ds
                else:
                    ddiff_ds = dR3_ds / r_par3
                grad_murray[i_vid] += 2.0 * diff * ddiff_ds

        L_total = L_shape + lambda_prior * L_prior + lambda_murray * L_murray
        grad = grad_shape + lambda_prior * grad_prior + lambda_murray * grad_murray
        return L_total, grad

    # Use analytic gradient
    eval_count_ad = [0]

    def _loss_and_grad_wrapped(scales_vec):
        L, g = loss_and_grad(scales_vec)
        eval_count_ad[0] += 1
        eval_count[0] += 1
        if verbose:
            print(f"    iter {eval_count_ad[0]}: L={L:.6f}  "
                  f"s∈[{scales_vec.min():.2f}, {scales_vec.max():.2f}]  "
                  f"|grad|={np.linalg.norm(g):.3e}")
        history.append((eval_count[0], L))
        return L, g

    if verbose:
        print(f"\n  STAGE 2: per-vessel refinement ({N_vessels}-D, adjoint gradients)")
        print(f"  ~2 solves per iter × {n_harmonics+1} harmonics = "
              f"{2*(n_harmonics+1)} solves regardless of dim.")

    opt = sp_min(_loss_and_grad_wrapped, s0, method='L-BFGS-B', bounds=bounds,
                 jac=True,
                 options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-7,
                          'disp': False})
    s_opt = opt.x

    # Final solve at optimum
    L_final, result_final, radii_m_opt = eval_once(s_opt, D_init)

    if verbose:
        print(f"  Optimization {'converged' if opt.success else 'stopped'}: "
              f"{opt.message}")
        print(f"  Final loss: {L_final:.6f} ({eval_count[0]} evals)")
        print(f"  Scale range: [{s_opt.min():.3f}, {s_opt.max():.3f}], "
              f"mean={s_opt.mean():.3f}")

    # Per-edge residuals
    residuals = {}
    for ei, ek in enumerate(edge_keys):
        if result_final is None:
            continue
        Q_pred = result_final.edge_flows.get(ek,
                    result_final.edge_flows.get((ek[1], ek[0])))
        if Q_pred is None or len(Q_pred) < 2 or abs(Q_pred[1]) < 1e-15:
            continue
        Qm = meas_edges[ek]['harmonics']
        if abs(Qm[1]) < 1e-15:
            continue
        r_e = 0.0
        for n in range(2, n_harmonics + 1):
            r_e += abs(Q_pred[n] / Q_pred[1] - Qm[n] / Qm[1])**2
        residuals[ek] = float(r_e)

    scales_per_vessel = {vid_v: float(s_opt[vi])
                         for vi, vid_v in enumerate(vessel_ids_ordered)}
    # Frozen interior vessels keep s=1
    for vid_v in vessel_ids_frozen:
        scales_per_vessel[vid_v] = 1.0

    return {
        'D_opt': D_init,
        'scales': scales_per_vessel,
        'scales_vec': s_opt,
        'vessel_ids': vessel_ids_ordered,
        'radii_m_opt': radii_m_opt,
        'result': result_final,
        'loss': float(opt.fun),
        'history': history,
        'residuals': residuals,
        'tile_id': tile_id,
    }


def _build_sheet_admittance(
    coords: np.ndarray,
    kappa_h_over_mu: float,
    a_cutoff: float,
) -> np.ndarray:
    """Build the Darcy-sheet admittance matrix Y_grey (N×N, dense, symmetric).

    Green's function of 2D Laplacian: G_ij = ln|x_i − x_j| / (2π), i ≠ j.
    Diagonal: G_ii = ln(a_cutoff) / (2π).
    Admittance: Y_grey = (κh/µ) · G⁻¹.

    Parameters
    ----------
    coords : (N, 2) positions of grey-zone boundary nodes (px or any unit;
        a_cutoff must be in the same units).
    kappa_h_over_mu : float
        Sheet permeability times thickness divided by viscosity  (has units
        of length²·time/mass in SI if you want physical Q; for the relative
        optimization it's an abstract scalar).
    a_cutoff : float
        Diagonal regularization cutoff. Typical: half the median inter-node
        spacing.
    """
    N = len(coords)
    dx = coords[:, 0:1] - coords[None, :, 0]
    dy = coords[:, 1:2] - coords[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, a_cutoff)
    Gm = np.log(np.maximum(dist, a_cutoff * 1e-3)) / (2.0 * np.pi)
    Ginv = np.linalg.inv(Gm)
    Y = kappa_h_over_mu * Ginv
    # Symmetrize (should already be ≈ symmetric)
    Y = 0.5 * (Y + Y.T)
    return Y


def optimize_greyzone_kirchhoff_fractions(
    G: 'nx.Graph',
    excluded_nodes: set,
    blue_red_map: Dict[int, List[int]],
    art_nodes: List[int],
    ven_nodes: List[int],
    Q_art: Dict[int, np.ndarray],
    Q_ven: Dict[int, np.ndarray],
    n_harmonics: int = 3,
    f0_hz: float = 2.5,
    mu: float = MU_DEFAULT,
    D_init: float = 1e-3,
    radii_m: Optional[Dict[Tuple[int, int], float]] = None,
    tile_id: Optional[int] = None,
    per_sheet_scale: bool = True,
    fit_alpha: bool = True,
    n_harmonics_loss: int = 0,
    fit_distensibility: bool = False,
    D_scan_grid: Optional[np.ndarray] = None,
    per_red_complex: bool = False,
    zeta_prior_strength: float = 0.0,
    zeta_sheet_phase_strength: float = 0.0,
    fit_tile_tau: bool = False,
    n_tau_iterations: int = 3,
    estimator: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Fractional-distribution Kirchhoff: fix red-share ratios from KCL, scale
    a single Q_in per sheet to best match tile measurements.

    Workflow:
      1. Per sheet: compute share_i = |Q_kcl_red_i| / Σ|Q_kcl_red_i|
         (using KCL magnitudes, single-tile-per-sheet consistency).
      2. Build G_mod = G − excluded_nodes − blue_sinks. Arterial dorsal
         aortae are disconnected from the main plexus and not used here.
      3. Impose source BC at each red: Q_red_i = α_sheet · share_i.
         Venous nodes keep their measured BC.
      4. Forward solve at DC; measure Q on tile-measured edges.
      5. Linear regression: α s.t. Σ_e Σ_k w_{e,k} |Q_m_{e,k} −
         α·b_{e,k} − Q_ven_{e,k}|² is minimized.

    Parameters
    ----------
    n_harmonics_loss : int (default 0)
        Number of AC harmonics to include in the α-fit loss.  0 = DC-only
        (legacy behavior).  ≥1 = include H1..H_{n} complex residuals,
        weighted by the per-edge measured magnitude at each harmonic.
        Closed-form α* still applies (loss is quadratic in α).
    fit_distensibility : bool (default False)
        If True, scan the global plexus distensibility D over `D_scan_grid`,
        re-running the basis solves at each D and picking D* that gives
        the lowest weighted-L2 loss.  Requires `n_harmonics_loss ≥ 1`
        because D is unidentifiable from DC alone.
    D_scan_grid : np.ndarray or None
        D values to scan when `fit_distensibility=True`.  Default is
        `np.logspace(log10(D_init)−1.5, log10(D_init)+1.5, 21)` — a
        ±1.5-decade window around the prior.
    per_red_complex : bool (default False)
        Replace the global-α (single real scale × fixed shares) fit with a
        per-red complex-amplitude regression.  Each red gets its own
        complex coefficient ζ_{b,i} ∈ ℂ that multiplies the (fixed)
        arterial waveform shape uniformly across harmonics — so per-red
        magnitude AND phase are inferred from the data.  Forward solve
        is linear in {ζ}, so closed-form weighted complex LS applies:
            ζ* = (B* W B + λΣ⁻¹)⁻¹ (B* W r + λΣ⁻¹ ζ_prior)
        with prior ζ_prior_r = KCL share (real, zero phase).
        Requires `n_harmonics_loss ≥ 1` (DC alone has no phase
        information).  Disables α-fit and D-scan in this path.
    zeta_prior_strength : float (default 0)
        Tikhonov prior strength λ pulling ζ toward the KCL-magnitude /
        zero-phase prior.  0 = no prior (pure WLS, can be ill-
        conditioned with few constraints).  Larger = stronger pull
        toward KCL shares.
    zeta_sheet_phase_strength : float (default 0)
        Soft intra-sheet similarity prior λ_sheet penalizing deviations
        of each red's ζ_r from the sheet's anchor (largest-share red).
        Quadratic penalty Σ_b Σ_{r∈b, r≠anchor} λ_sheet · ‖ζ_r − ζ_anchor‖²
        added as augmented rows in the normal equations.  In
        `zeta_phase_only` mode this primarily pulls phases together
        (magnitudes are already pinned); in `zeta_per_red` it pulls
        both magnitudes and phases.  Use to suppress lone anti-phase
        reds within a sheet that look like recirculation artifacts.
        0 = off; typical values 0.01–1.0.
    fit_tile_tau : bool (default False)
        Co-fit a per-tile global temporal phase τ along with α/ζ.  Each
        tile's video starts at an unknown moment in the cardiac cycle,
        so its measured Q̂^(k) carries an unknown rotation
        e^{i·k·ω₀·τ} relative to the arterial reference.  Within-tile
        relative phases are physically meaningful, but absolute phases
        are not.  Fitting τ as a nuisance parameter prevents the
        regression from absorbing tile-timing offsets into α/ζ.
        Requires `n_harmonics_loss ≥ 1`; when off (legacy default), the
        loss assumes τ = 0.
    n_tau_iterations : int (default 3)
        Number of α↔τ alternation iterations.  At each step, α (or ζ)
        is closed-form-fit at the current τ, then τ is updated by 1D
        optimization (Brent's method over [0, T_period)) at the new α.
        Converges in 2–3 iterations for typical cases.
    estimator : str | None (default None)
        Mutually-exclusive estimator selector that supersedes the
        legacy boolean flags.  One of:
          'alpha'                — global-α single-real scale, KCL shares fixed
          'zeta_per_red'         — per-red ζ ∈ ℂ (magnitude + phase free)
          'zeta_phase_only'      — per-red ∠ζ free, |ζ| pinned to KCL share
        When None, the estimator is resolved from the legacy flags
        (`per_red_complex`).  Passing both `estimator` and a legacy
        flag that disagrees raises ValueError to surface the conflict.
        The chosen estimator is recorded in `result['estimator']`.
    """
    from .config import FRAME_DT_S
    from .harmonic import fit_harmonics
    from scipy.sparse import eye as _sp_eye
    from scipy.sparse.linalg import spsolve as _spsolve

    # ------------------------------------------------------------------
    # Resolve the estimator selector.  Mutually exclusive at the API
    # level so that the previous "boolean flags silently supersede each
    # other" footgun cannot recur.  Legacy flags map onto the selector
    # when `estimator=None`, but contradicting `estimator` against a
    # legacy flag raises so the user knows what they actually got.
    # ------------------------------------------------------------------
    _VALID_ESTIMATORS = ('alpha', 'zeta_per_red', 'zeta_phase_only')
    if estimator is None:
        estimator = ('zeta_per_red' if per_red_complex else 'alpha')
    else:
        if estimator not in _VALID_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {_VALID_ESTIMATORS}, "
                f"got {estimator!r}")
        if (estimator in ('zeta_per_red', 'zeta_phase_only')
                and not per_red_complex):
            per_red_complex = True
        if estimator == 'alpha' and per_red_complex:
            raise ValueError(
                "estimator='alpha' conflicts with per_red_complex=True. "
                "Pass exactly one — they are mutually exclusive.")
    if verbose:
        print(f"  Estimator: {estimator} "
              f"(n_harmonics_loss={n_harmonics_loss}, "
              f"fit_tile_tau={bool(fit_tile_tau)}, "
              f"fit_distensibility={bool(fit_distensibility)})")

    excluded = set(excluded_nodes)
    all_blue = sorted(blue_red_map.keys())

    # --- 1. Compute per-sheet shares from KCL magnitudes --------------
    # (reuses the absolute-value, red-red-exclude, single-tile semantics
    # the Kirchhoff-mode printout uses)
    shares = {}             # red_node -> share within its sheet
    sheet_tile = {}         # blue -> tile used
    sheet_of_red = {}       # red -> blue
    for blue, reds in blue_red_map.items():
        reds_set = set(reds)
        # Best-coverage tile for this sheet
        from collections import Counter as _C
        tc = _C()
        for rn in reds:
            for nb in G.neighbors(rn):
                if nb in excluded or nb in reds_set:
                    continue
                for m in G.edges[rn, nb].get('measurements_piv', []):
                    Qt = m.get('Q_t')
                    tid = m.get('tile_id')
                    if tid is None or Qt is None or len(Qt) < 20:
                        continue
                    tc[tid] += 1
        stile = tc.most_common(1)[0][0] if tc else None
        sheet_tile[blue] = stile

        # |Q| at each red (same tile)
        q_mag = {}
        for rn in reds:
            total_mag = 0.0
            for nb in G.neighbors(rn):
                if nb in excluded or nb == rn or nb in reds_set:
                    continue
                piv = G.edges[rn, nb].get('measurements_piv', [])
                m_s = next((m for m in piv if m.get('tile_id') == stile),
                           None) if stile else None
                if m_s is None:
                    if piv:
                        m_s = max(piv,
                                  key=lambda m: m.get('snr_pulse', -np.inf))
                    else:
                        continue
                Qt = np.asarray(m_s.get('Q_t', []), dtype=float)
                if Qt.size < 20:
                    continue
                q_dc = float(np.nanmean(Qt))
                if np.isfinite(q_dc):
                    total_mag += abs(q_dc)
            q_mag[rn] = total_mag

        total = sum(q_mag.values())
        if total <= 1e-15:
            if verbose:
                print(f"  Sheet blue={blue}: zero total magnitude; skip")
            continue
        for rn in reds:
            shares[rn] = q_mag[rn] / total
            sheet_of_red[rn] = blue

    if verbose:
        print(f"\n  Fractional KCL: {len(all_blue)} sheets, "
              f"{len(shares)} reds with valid shares")
        for blue, reds in blue_red_map.items():
            ssum = sum(shares.get(r, 0.0) for r in reds)
            print(f"    sheet {blue} (tile t{sheet_tile[blue]}): "
                  f"Σ share = {ssum:.3f}")
            for rn in sorted(reds, key=lambda r: -shares.get(r, 0)):
                print(f"      red {rn:>6}: share = {shares.get(rn, 0.0):.4f}")

    # --- 2. Build G_mod (grey + blue sinks removed) -------------------
    blue_set = set(all_blue)
    G_mod = G.copy()
    G_mod.remove_nodes_from(excluded | blue_set)
    all_mod_nodes = list(G_mod.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_mod_nodes)}
    N_mod = len(all_mod_nodes)
    base_edge_list = [(u, v) for u, v in G_mod.edges()
                      if _get_edge_geometry(G_mod, u, v, radii_m=radii_m)[0] is not None]
    if verbose:
        print(f"  G_mod: {N_mod} nodes, {len(base_edge_list)} edges "
              f"(after removing grey zone + blue sinks)")

    # --- 3. Forward solve per harmonic, with parametric α per sheet ----
    # For each harmonic k (DC + AC), inject at each red:
    #   Q_red_k[rn] = share_rn · α_sheet · w_k
    # where w_k is the waveform shape (w_0 = 1, w_k = Q_art_mean[k]/Q_art_mean[0]
    # for k ≥ 1, complex — carries amplitude + phase from the arteries).
    # DC magnitude (loss-determined via α·share) is preserved, AC shape is
    # inherited from the arterial waveform. Venous nodes stay pinned to
    # P = 0 at every harmonic (gauge).
    def _solve_with_red_bcs(red_Q_map, ven_Q_map, omega_k, D_val=None):
        """Solve L(ω)·P = Q_rhs on G_mod with red sources and venous pins.

        Scale-invariant: Q_rhs units → Y·P units (Y and 1/Y cancel). Tikhonov
        λ scales to the interior (non-pinned) diagonal so λ·P ≪ Q_rhs at
        reds. Pin rows (diag=1 by construction) are NOT regularized.

        `D_val` is the per-edge distensibility used when assembling L; if
        None, falls back to the closure's `D_init`.  Threading D through
        is required for the profile-likelihood D scan.
        """
        D_use = D_init if D_val is None else D_val
        L = _assemble_laplacian(G_mod, omega_k, base_edge_list, node_to_idx,
                                 mu, RHO_BLOOD, D_use, radii_m=radii_m)
        L = L.tolil()
        Q_rhs = np.zeros(N_mod, dtype=complex)
        for rn, q in red_Q_map.items():
            if rn in node_to_idx:
                Q_rhs[node_to_idx[rn]] += q

        interior_diag = np.abs(np.asarray(L.diagonal()))
        interior_finite = interior_diag[np.isfinite(interior_diag)
                                        & (interior_diag > 0)]
        y_scale = float(np.median(interior_finite)) if interior_finite.size \
            else 1.0

        pin_idxs = set()
        for vn in ven_Q_map.keys():
            if vn in node_to_idx:
                idx = node_to_idx[vn]
                L[idx, :] = 0
                L[idx, idx] = 1.0
                Q_rhs[idx] = 0.0
                pin_idxs.add(idx)
        L_csr = L.tocsr()

        lam = 1e-6 * y_scale
        if lam > 0 and len(pin_idxs) < N_mod:
            diag_mask = np.ones(N_mod)
            for i in pin_idxs:
                diag_mask[i] = 0.0
            L_csr = L_csr + _sp_eye(N_mod, format='csr').multiply(
                diag_mask * lam)

        try:
            return _spsolve(L_csr, Q_rhs)
        except Exception:
            return None

    def _edge_Q(P, omega_k, D_val=None):
        """Reconstruct per-edge Q at a given harmonic from pressures.
        `D_val` overrides the closure's D_init for the admittance
        evaluation — required so basis solves at different D in the
        profile-likelihood scan are self-consistent."""
        D_use = D_init if D_val is None else D_val
        out = {}
        for u, v in base_edge_list:
            R_m, L_m = _get_edge_geometry(G_mod, u, v, radii_m=radii_m or {})
            if R_m is None:
                continue
            Yd, Yo = _vessel_admittance(R_m, L_m, omega_k, mu, RHO_BLOOD, D_use)
            iu, iv = node_to_idx[u], node_to_idx[v]
            out[(u, v)] = Yd * P[iu] + Yo * P[iv]
        return out

    # --- Waveform shape from arterial BCs (mean of all DAs) ------------
    # w_0 = 1 (DC anchor), w_k = mean(Q_art[an][k] / Q_art[an][0]) for k≥1.
    # If arterial BCs are missing or have zero DC, w_k = 0 (AC stays flat).
    waveform = np.zeros(n_harmonics + 1, dtype=complex)
    waveform[0] = 1.0
    per_art_ratios = {}
    waveform_source = "Q_art (boundary-extracted, ref_tile)"

    # Per-artery waveform: search every tile's PIV measurement on the
    # DA's incident edges and pick the one with the LARGEST |w_1|/|DC|
    # (most pulsatile). Rationale: the DA is biologically pulsatile; the
    # smallest-AC measurement is the one most corrupted by partial-volume,
    # centerline drift, out-of-focus, etc. Taking the max over tiles gives
    # the best approximation to the true DA waveform shape. Each Q_t is
    # fit at its OWN tile's f0 (no spectral leakage); coefficients are
    # dimensionless ratios, so the sim can apply them at its target f0
    # (same-shape-higher-frequency).
    if n_harmonics >= 1 and art_nodes:
        from .harmonic import fit_harmonics as _fit_h
        # Map tile_id → f0 if present in the graph metadata.
        tile_f0s = G.graph.get('tile_f0s', {}) or {}
        picked_per_art = {}   # an -> (tile_id, f0_used, |w1|, ratios_complex)
        for an in art_nodes:
            if an not in G.nodes:
                continue
            best_w1 = -1.0
            best = None
            for nb in G.neighbors(an):
                d = G.edges[an, nb]
                for m in d.get('measurements_piv', []):
                    t_id = m.get('tile_id')
                    Qt = m.get('Q_t')
                    if t_id is None or Qt is None or len(Qt) < 20:
                        continue
                    Qt_arr = np.asarray(Qt, dtype=float)
                    Qt_fin = Qt_arr[np.isfinite(Qt_arr)]
                    if Qt_fin.size < 20:
                        continue
                    # Use that tile's native f0 if known; else the sim f0
                    f0_native = float(tile_f0s.get(int(t_id),
                                                    tile_f0s.get(t_id, f0_hz)))
                    try:
                        hr = _fit_h(Qt_fin, f0_native, FRAME_DT_S,
                                    K=n_harmonics, include_dc=True,
                                    loss='huber')
                    except Exception:
                        continue
                    coeffs = np.zeros(n_harmonics + 1, dtype=complex)
                    coeffs[0] = float(np.nanmean(Qt_fin))
                    for h in hr.get('harmonics', []):
                        k = int(h.get('k', 0))
                        if 1 <= k <= n_harmonics:
                            coeffs[k] = complex(h.get('A', 0.0),
                                                 -h.get('B', 0.0))
                    dc = coeffs[0]
                    if abs(dc) < 1e-15:
                        continue
                    ratios = coeffs / dc
                    w1 = float(abs(ratios[1]))
                    if w1 > best_w1:
                        best_w1 = w1
                        best = (int(t_id), f0_native, w1, ratios)
            if best is not None:
                picked_per_art[an] = best
                per_art_ratios[an] = best[3]
        if per_art_ratios:
            picks = ", ".join(f"art {an}→t{best[0]} @f0={best[1]:.2f} "
                              f"|w1|={best[2]:.3f}"
                              for an, best in picked_per_art.items())
            waveform_source = f"most-pulsatile DA PIV ({picks})"

    # Fallback: use the supplied Q_art (ref_tile / boundary extraction).
    if not per_art_ratios and Q_art and n_harmonics >= 1:
        for an, coeffs in Q_art.items():
            if coeffs is None or len(coeffs) < n_harmonics + 1:
                continue
            dc = coeffs[0]
            if abs(dc) < 1e-15:
                continue
            per_art_ratios[an] = coeffs[:n_harmonics + 1] / dc

    if per_art_ratios:
        stacked = np.vstack(list(per_art_ratios.values()))
        mean_ratio = np.mean(stacked, axis=0)
        waveform[1:] = mean_ratio[1:]

    if verbose:
        if per_art_ratios:
            print(f"  Waveform source: {waveform_source}")
            print(f"  Per-artery waveform ratios (Q_art[k] / Q_art[0]):")
            for an, r in per_art_ratios.items():
                parts = ", ".join(f"k={k}: |{abs(v):.3f}|∠{np.angle(v, deg=True):+.0f}°"
                                  for k, v in enumerate(r) if k >= 1)
                print(f"    artery {an}: {parts}")
        print(f"  AC waveform shape (per-harmonic |w_k| / ∠w_k):")
        for k in range(n_harmonics + 1):
            print(f"    k={k}: |w|={abs(waveform[k]):.3f}  "
                  f"∠w={np.angle(waveform[k], deg=True):+.1f}°")
        # Characteristic-attenuation sanity check: warn if D makes AC
        # decay too quickly across typical vessel lengths.
        if n_harmonics >= 1:
            omega1 = 2 * np.pi * f0_hz
            R_typ = 10e-6  # 10 µm typical radius
            r_typ = 8.0 * mu / (np.pi * R_typ ** 4)
            # Areal distensibility convention: c = πR²D (was 2πR²D pre-2026-05-18)
            c_typ = np.pi * R_typ ** 2 * D_init
            k_mag = np.sqrt(omega1 * r_typ * c_typ / 2.0)
            att_per_100um = float(np.exp(-k_mag * 1e-4))
            print(f"  AC attenuation check (D={D_init:.1e}): "
                  f"|w| per 100 µm at f0 ≈ {att_per_100um:.2f}")
            if att_per_100um < 0.5:
                print(f"    ⚠️  AC damped >2× per 100 µm — D may be too "
                      f"high for this network. Typical embryonic "
                      f"D ≈ 1e-8 to 1e-5.")

    # ------------------------------------------------------------------
    # Basis-and-α-fit machinery as a function of distensibility D.
    # Multi-harmonic loss + profile-likelihood D inference share this
    # closed-form evaluator; we call it once when fit_distensibility=False
    # and across a D grid when True.  Per-sheet basis is needed at every
    # harmonic (not just DC) when n_harmonics_loss > 0 so the α fit can
    # use AC residuals.  Additionally we now solve a venous-only basis
    # at each AC harmonic so we can subtract that contribution from the
    # measured Q before regressing against the red-driven basis.
    # ------------------------------------------------------------------
    omegas = [0.0] + [2.0 * np.pi * k * f0_hz for k in range(1, n_harmonics + 1)]

    def _build_basis(D_val):
        """Solve per-sheet (unit-α) basis + venous-only basis at D=D_val.
        Returns (basis_Q, basis_P, ven_Q).
        - basis_Q[blue][k][edge]  : Q at unit α for sheet, harm k
        - basis_P[blue][k][node]  : P at unit α for sheet, harm k
        - ven_Q[k][edge]          : Q from venous-BC-only forcing, harm k
        """
        b_Q, b_P = {}, {}
        for blue, reds in blue_red_map.items():
            per_harm_Q, per_harm_P = {}, {}
            for k, omega_k in enumerate(omegas):
                red_bc = {rn: complex(shares.get(rn, 0.0)) * waveform[k]
                          for rn in reds}
                # Pass Q_ven for DC (gauge pin), empty at AC so the
                # red-only basis doesn't double-count venous flow.
                ven_for_solve = Q_ven if k == 0 else {}
                P = _solve_with_red_bcs(red_bc, ven_for_solve, omega_k,
                                         D_val=D_val)
                if P is None:
                    continue
                per_harm_Q[k] = _edge_Q(P, omega_k, D_val=D_val)
                per_harm_P[k] = P
            b_Q[blue] = per_harm_Q
            b_P[blue] = per_harm_P

        # Venous-only AC basis: red sources zero, venous BCs measured.
        # At DC the venous Dirichlet pin is already inside the red-basis
        # solve (k=0 gauge), so we only need k≥1 here.
        v_Q = {}
        for k, omega_k in enumerate(omegas):
            if k == 0 or not Q_ven:
                continue
            rhs = {}
            for vn in ven_nodes:
                if vn in node_to_idx and vn in Q_ven:
                    coeffs = np.asarray(Q_ven[vn], dtype=complex)
                    if len(coeffs) > k:
                        rhs[vn] = complex(coeffs[k])
            if not rhs:
                continue
            P = _solve_with_red_bcs(rhs, {}, omega_k, D_val=D_val)
            if P is None:
                continue
            v_Q[k] = _edge_Q(P, omega_k, D_val=D_val)
        return b_Q, b_P, v_Q

    basis_Q, basis_P, venous_basis_Q = _build_basis(D_init)
    venous_basis_P = {}    # legacy slot, unused

    # --- 4. Collect tile measurements on external edges ---------------
    # `meas_edges` keeps the legacy DC-magnitude entries (used by the
    # backward-compat DC-only α fit and the residual-field outputs).
    # `meas_edges_harm` carries the full complex harmonic spectrum from
    # fitting Q_t at the tile's f0 — this is what the multi-harmonic
    # α* fit regresses against.
    from .config import FRAME_DT_S as _FDT
    from .harmonic import fit_harmonics as _fit_h_loss
    meas_edges = {}
    meas_edges_harm = {}
    tile_f0s = G.graph.get('tile_f0s', {}) or {}
    # Track how many edges have PIV flow_from/flow_to reversed relative
    # to the graph's (u, v) storage order.  Without canonicalization,
    # the regression sees signed measurements in mixed conventions and
    # converges to a wrong-sign solution (the per-red ζ all-flipped
    # pathology).  See `grey_zone_diagnostics.sign_agreement_diagnostic`.
    n_oriented_aligned = 0
    n_oriented_reversed = 0
    n_oriented_unknown = 0
    for u, v, d in G_mod.edges(data=True):
        piv = d.get('measurements_piv', [])
        if not piv:
            continue
        if tile_id is not None:
            tile_piv = [m for m in piv if m.get('tile_id') == tile_id]
            if not tile_piv:
                continue
            best = tile_piv[0]
        else:
            best = max(piv, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        if Qt is None or len(Qt) < 20:
            continue
        Qt_arr = np.asarray(Qt, dtype=float)
        Qt_fin = Qt_arr[np.isfinite(Qt_arr)]
        if Qt_fin.size < 20:
            continue
        # PIV stores Q_t signed in `flow_from → flow_to`.  Canonicalize
        # to the graph's (u, v) order: flip sign if reversed.
        ff = best.get('flow_from')
        ft = best.get('flow_to')
        if ff is not None and ft is not None:
            if ff == u and ft == v:
                orient_sign = 1.0
                n_oriented_aligned += 1
            elif ff == v and ft == u:
                orient_sign = -1.0
                n_oriented_reversed += 1
            else:
                orient_sign = 1.0
                n_oriented_unknown += 1
        else:
            orient_sign = 1.0
            n_oriented_unknown += 1
        q_dc_signed = orient_sign * float(np.nanmean(Qt_fin))
        q_dc = abs(q_dc_signed)
        if not np.isfinite(q_dc):
            continue
        meas_edges[(u, v)] = q_dc
        # Fit harmonics for AC residuals (only when needed)
        if n_harmonics_loss >= 1:
            f0_native = float(tile_f0s.get(int(best.get('tile_id', tile_id))
                                            if best.get('tile_id') is not None
                                            else (tile_id or -1), f0_hz))
            try:
                hr = _fit_h_loss(Qt_fin, f0_native, _FDT,
                                 K=n_harmonics_loss,
                                 include_dc=True, loss='huber')
                coeffs = np.zeros(n_harmonics_loss + 1, dtype=complex)
                coeffs[0] = q_dc_signed
                # Apply same orient_sign to AC harmonics (Q_t→−Q_t flips
                # every harmonic uniformly, so the orientation flip is a
                # uniform sign factor on coeffs[1:] as well).
                for h in hr.get('harmonics', []):
                    k = int(h.get('k', 0))
                    if 1 <= k <= n_harmonics_loss:
                        coeffs[k] = orient_sign * complex(
                            h.get('A', 0.0), -h.get('B', 0.0))
                meas_edges_harm[(u, v)] = coeffs
            except Exception:
                pass
    if verbose:
        print(f"  Tile measurements: {len(meas_edges)} edges"
              + (f" (tile {tile_id})" if tile_id else " (best SNR)"))
        total_o = (n_oriented_aligned + n_oriented_reversed
                   + n_oriented_unknown)
        if total_o > 0:
            print(f"  Edge orientation vs PIV flow_from/to: "
                  f"aligned={n_oriented_aligned} "
                  f"({100*n_oriented_aligned/total_o:.0f}%), "
                  f"reversed={n_oriented_reversed} "
                  f"({100*n_oriented_reversed/total_o:.0f}%), "
                  f"unknown={n_oriented_unknown}  "
                  f"(measurements canonicalized to (u,v) order)")
        if n_harmonics_loss >= 1:
            print(f"  Per-edge AC harmonics extracted: "
                  f"{len(meas_edges_harm)} edges, "
                  f"K={n_harmonics_loss} harmonics each")

    # --- 5a. Fit α to tile measurements (closed-form weighted L2) ------
    # Two paths:
    #
    # n_harmonics_loss = 0 (legacy):
    #   L(α) = Σ_e |Q_m_e| · (α·|b_e| − |Q_m_e|)²
    #   uses DC magnitudes only.  α* = Σ w·|b_e|·|Q_m_e| / Σ w·|b_e|²
    #
    # n_harmonics_loss ≥ 1 (multi-harmonic):
    #   L(α) = Σ_e Σ_k w_{e,k} |Q_m_{e,k} − α·b_{e,k} − Q_v_{e,k}|²
    #   with w_{e,k} = |Q_m_{e,k}| (per-harmonic magnitude weighting).
    #   Closed-form (loss is real-valued and quadratic in real α):
    #     α* = Σ Σ w · Re((Q_m − Q_v) · conj(b)) / Σ Σ w · |b|²
    #   Subtracting Q_v_{e,k} (the venous-only basis response) makes the
    #   regression isolate the red-driven contribution from the venous-
    #   driven contribution at each harmonic.
    def _alpha_star_at_D(D_val, b_Q_local, v_Q_local, meas_dc, meas_harm,
                          tau_tile=0.0):
        """Closed-form α* and weighted loss given basis at D=D_val.

        `tau_tile` (seconds) rotates each measured complex harmonic by
        e^{-i·k·ω₀·τ} before forming the residual.  This represents the
        unknown global cardiac-phase offset of the tile's video relative
        to the arterial reference; absolute AC phase is tile-specific
        and not physically meaningful, but within-tile relative phases
        are.  τ=0 reproduces the no-correction case.
        """
        # DC unit-α basis (signed real, summed over sheets) — used by both
        # paths because DC always contributes to the loss.
        all_keys = set()
        for blue in b_Q_local:
            all_keys.update(b_Q_local[blue].get(0, {}).keys())
        b_dc_total = {}
        for e in all_keys:
            tot = complex(0.0)
            for blue in b_Q_local:
                v = b_Q_local[blue].get(0, {}).get(e)
                if v is not None:
                    tot += v
            b_dc_total[e] = abs(float(np.real(tot)))

        num = 0.0
        den = 0.0
        used = {}
        # DC term — τ rotation has no effect at k=0 (e^0 = 1)
        for e, q_m in meas_dc.items():
            b_e = b_dc_total.get(e)
            if b_e is None:
                b_e = b_dc_total.get((e[1], e[0]))
            if b_e is None or b_e <= 0:
                continue
            q_m_abs = abs(float(q_m))
            w = q_m_abs
            num += w * b_e * q_m_abs
            den += w * b_e * b_e
            used[e] = (b_e, q_m_abs)

        # AC terms — measurements rotated by e^{-i·k·ω₀·τ}
        omega0 = 2.0 * np.pi * f0_hz
        if n_harmonics_loss >= 1 and meas_harm:
            for e, coeffs in meas_harm.items():
                e_key = e if e in b_dc_total else (
                    (e[1], e[0]) if (e[1], e[0]) in b_dc_total else None)
                sign = 1.0 if e_key == e else -1.0
                if e_key is None:
                    continue
                for k in range(1, n_harmonics_loss + 1):
                    if k > n_harmonics:
                        break
                    b_e_k = complex(0.0)
                    for blue in b_Q_local:
                        v_k = b_Q_local[blue].get(k, {}).get(e_key)
                        if v_k is not None:
                            b_e_k += v_k
                    if abs(b_e_k) < 1e-30:
                        continue
                    q_v_k = v_Q_local.get(k, {}).get(e_key, complex(0.0))
                    rot = np.exp(-1j * k * omega0 * tau_tile)
                    q_m_k = sign * complex(coeffs[k]) * rot
                    R_k = q_m_k - q_v_k
                    w_k = abs(q_m_k)
                    if w_k < 1e-30:
                        continue
                    num += w_k * float(np.real(np.conj(b_e_k) * R_k))
                    den += w_k * float(abs(b_e_k) ** 2)

        if den < 1e-20:
            return None, np.inf, used
        a_star = float(num / den)

        # Total loss at α* (DC + AC, weighted)
        loss = 0.0
        for (b_e, qm) in used.values():
            loss += qm * (a_star * b_e - qm) ** 2
        if n_harmonics_loss >= 1 and meas_harm:
            for e, coeffs in meas_harm.items():
                e_key = e if e in b_dc_total else (
                    (e[1], e[0]) if (e[1], e[0]) in b_dc_total else None)
                sign = 1.0 if e_key == e else -1.0
                if e_key is None:
                    continue
                for k in range(1, n_harmonics_loss + 1):
                    if k > n_harmonics:
                        break
                    b_e_k = complex(0.0)
                    for blue in b_Q_local:
                        v_k = b_Q_local[blue].get(k, {}).get(e_key)
                        if v_k is not None:
                            b_e_k += v_k
                    if abs(b_e_k) < 1e-30:
                        continue
                    q_v_k = v_Q_local.get(k, {}).get(e_key, complex(0.0))
                    rot = np.exp(-1j * k * omega0 * tau_tile)
                    q_m_k = sign * complex(coeffs[k]) * rot
                    w_k = abs(q_m_k)
                    if w_k < 1e-30:
                        continue
                    pred_k = a_star * b_e_k + q_v_k
                    loss += w_k * abs(q_m_k - pred_k) ** 2
        return a_star, float(loss), used

    def _fit_tau_from_rows(M_rows, pred_rows, k_rows, w_rows):
        """Generic 1D τ optimizer for per-(e,k) measurements + predictions.

        Minimizes Σ_i w_i |M_i · e^{-i·k_i·ω₀·τ} − pred_i|².  All inputs
        are flat arrays/lists indexed by row.  Used by both the global-α
        path (predictions = α·b + Q_ven) and the per-red ζ path
        (predictions = Σ_r ζ_r · B_r + Q_ven).  Returns (τ_opt, loss_at_τ).
        """
        from scipy.optimize import minimize_scalar
        if len(M_rows) == 0:
            return 0.0, np.inf
        omega0 = 2.0 * np.pi * f0_hz
        T_period = 2.0 * np.pi / omega0
        # Per-harmonic accumulators for the τ-dependent term:
        #   Loss(τ) = const − 2 Re(Σ_k C_k e^{-ikω₀τ})
        # where C_k = Σ_{i: k_i=k} w_i · M_i · conj(pred_i)
        max_k = max(k_rows) if k_rows else 0
        C = np.zeros(max_k + 1, dtype=complex)
        const_term = 0.0
        for M_i, p_i, k_i, w_i in zip(M_rows, pred_rows, k_rows, w_rows):
            C[k_i] += w_i * M_i * np.conj(p_i)
            const_term += w_i * (abs(M_i) ** 2 + abs(p_i) ** 2)

        def neg_obj(tau):
            return -float(np.real(np.sum([
                C[k] * np.exp(-1j * k * omega0 * tau)
                for k in range(1, max_k + 1)
            ])))

        # Coarse grid + Brent refine to avoid local minima from the
        # multi-harmonic landscape (up to max_k bumps per period).
        n_grid = 60
        taus = np.linspace(0.0, T_period, n_grid, endpoint=False)
        vals = np.array([neg_obj(t) for t in taus])
        i_min = int(np.argmin(vals))
        lo = taus[(i_min - 1) % n_grid]
        hi = taus[(i_min + 1) % n_grid]
        if lo > hi:
            lo -= T_period
        try:
            res = minimize_scalar(neg_obj, bracket=(lo, taus[i_min], hi))
            tau_opt = float(res.x % T_period)
        except Exception:
            tau_opt = float(taus[i_min])
        loss_at_tau = const_term + 2.0 * neg_obj(tau_opt)
        return tau_opt, loss_at_tau

    def _fit_tau_for_alpha(D_val, alpha_val, b_Q_local, v_Q_local, meas_harm):
        """Given α (and basis), find τ that minimizes the multi-harmonic
        AC loss.  Optimization is a 1D Brent search over [0, T_period).
        Returns (τ*, loss*).  At α=0 or no AC harmonics, returns 0."""
        from scipy.optimize import minimize_scalar
        if alpha_val == 0.0 or not meas_harm or n_harmonics_loss < 1:
            return 0.0, np.inf
        # Pre-collect (M^(k)_e, pred^(k)_e, w_{e,k}) per harmonic so the
        # 1D objective is cheap.  pred^(k) = α·b^(k) + Q_ven^(k).
        all_keys = set()
        for blue in b_Q_local:
            all_keys.update(b_Q_local[blue].get(0, {}).keys())
        # Per-harmonic accumulator: C_k = Σ_e w_{e,k} M^(k) conj(pred^(k))
        omega0 = 2.0 * np.pi * f0_hz
        C = np.zeros(n_harmonics_loss + 1, dtype=complex)
        const_term = 0.0   # Σ_e w_{e,k} (|M|² + |pred|²); independent of τ
        for e, coeffs in meas_harm.items():
            e_key = e if e in all_keys else (
                (e[1], e[0]) if (e[1], e[0]) in all_keys else None)
            if e_key is None:
                continue
            sign = 1.0 if e_key == e else -1.0
            for k in range(1, n_harmonics_loss + 1):
                if k > n_harmonics:
                    break
                b_e_k = complex(0.0)
                for blue in b_Q_local:
                    v_k = b_Q_local[blue].get(k, {}).get(e_key)
                    if v_k is not None:
                        b_e_k += v_k
                if abs(b_e_k) < 1e-30:
                    continue
                q_v_k = v_Q_local.get(k, {}).get(e_key, complex(0.0))
                pred_k = alpha_val * b_e_k + q_v_k
                M_k = sign * complex(coeffs[k])
                w_k = abs(M_k)
                if w_k < 1e-30:
                    continue
                C[k] += w_k * M_k * np.conj(pred_k)
                const_term += w_k * (abs(M_k) ** 2 + abs(pred_k) ** 2)

        # Loss(τ) = const_term - 2 Re(Σ_k C_k e^{-ikω₀τ}); minimize.
        T_period = 2.0 * np.pi / omega0
        def neg_obj(tau):
            return -float(np.real(np.sum([
                C[k] * np.exp(-1j * k * omega0 * tau)
                for k in range(1, n_harmonics_loss + 1)
            ])))
        # Brent on [0, T_period) — coarse grid first, then local refine
        # so we don't get trapped at a local minimum (multi-harmonic loss
        # has up to n_harmonics_loss bumps per period).
        n_grid = 60
        taus = np.linspace(0.0, T_period, n_grid, endpoint=False)
        vals = np.array([neg_obj(t) for t in taus])
        i_min = int(np.argmin(vals))
        # Bracket around the discrete minimum
        lo = taus[(i_min - 1) % n_grid]
        hi = taus[(i_min + 1) % n_grid]
        if lo > hi:
            lo -= T_period
        try:
            res = minimize_scalar(neg_obj, bracket=(lo, taus[i_min], hi))
            tau_opt = float(res.x % T_period)
        except Exception:
            tau_opt = float(taus[i_min])
        # Loss at τ*
        loss_at_tau = const_term + 2.0 * neg_obj(tau_opt)
        return tau_opt, loss_at_tau

    # --- Profile-likelihood D scan (when fit_distensibility=True) ------
    # Without AC residuals, D is unidentifiable from the loss (DC is
    # purely resistive), so silently fall back to no-scan if user asked
    # for D-fitting but didn't include AC harmonics.
    D_scan_result = None
    if fit_distensibility and n_harmonics_loss >= 1:
        if D_scan_grid is None:
            log_D = np.log10(D_init) if D_init > 0 else -3.0
            D_scan_grid = np.logspace(log_D - 1.5, log_D + 1.5, 21)
        D_grid = np.asarray(D_scan_grid, dtype=float)
        D_alphas = np.full(D_grid.size, np.nan)
        D_losses = np.full(D_grid.size, np.inf)
        if verbose:
            print(f"  Profile-likelihood D scan: {D_grid.size} values "
                  f"from {D_grid.min():.2e} to {D_grid.max():.2e}, "
                  f"K={n_harmonics_loss} AC harmonics in loss")
        for i, D_try in enumerate(D_grid):
            try:
                bQ_i, _, vQ_i = _build_basis(D_try)
            except Exception as _e:
                if verbose:
                    print(f"    D={D_try:.2e}: basis solve failed ({_e})")
                continue
            a_i, L_i, _ = _alpha_star_at_D(
                D_try, bQ_i, vQ_i, meas_edges, meas_edges_harm)
            if a_i is not None:
                D_alphas[i] = a_i
                D_losses[i] = L_i
        if np.any(np.isfinite(D_losses)):
            i_best = int(np.argmin(D_losses))
            D_best = float(D_grid[i_best])
            if verbose:
                print(f"  D* = {D_best:.3e}  (α*={D_alphas[i_best]:.4f}, "
                      f"loss={D_losses[i_best]:.4g})")
                # Print nearby grid for context
                lo = max(0, i_best - 3)
                hi = min(D_grid.size, i_best + 4)
                for j in range(lo, hi):
                    marker = ' ←' if j == i_best else ''
                    print(f"    D={D_grid[j]:.2e}  α={D_alphas[j]:.4f}  "
                          f"loss={D_losses[j]:.4g}{marker}")
            # Re-build basis at D* and use those for downstream solve.
            basis_Q, basis_P, venous_basis_Q = _build_basis(D_best)
            D_init = D_best
            D_scan_result = {'D_grid': D_grid,
                             'alpha_grid': D_alphas,
                             'loss_grid': D_losses,
                             'D_best': D_best,
                             'i_best': i_best}
        elif verbose:
            print(f"  ⚠️  D scan: no valid (D, α*) — keeping D_init")
    elif fit_distensibility and n_harmonics_loss < 1 and verbose:
        print(f"  ⚠️  fit_distensibility requires n_harmonics_loss ≥ 1 "
              f"(D unidentifiable from DC alone) — keeping D_init")

    # Final α* at the chosen D (D_init either as supplied or as D*).
    # If `fit_tile_tau`, alternate α↔τ for `n_tau_iterations` rounds.
    # Skipped entirely when an estimator other than 'alpha' is active —
    # the per-red ζ paths supersede α and produce their own diagnostics,
    # so running this block would just emit confusing parallel outputs.
    alpha_fit = None
    alpha_sweep = None
    tau_fit = 0.0
    fit_used_edges = {}
    if estimator == 'alpha' and fit_alpha and meas_edges:
        # Iteration 0: α at τ=0
        a_star, _, used = _alpha_star_at_D(
            D_init, basis_Q, venous_basis_Q, meas_edges, meas_edges_harm,
            tau_tile=0.0)
        fit_used_edges = used
        # Optional alternation: refine τ then α
        if (fit_tile_tau and n_harmonics_loss >= 1
                and meas_edges_harm and a_star is not None):
            omega0 = 2.0 * np.pi * f0_hz
            T_period = 2.0 * np.pi / omega0
            for it in range(int(n_tau_iterations)):
                tau_new, _ = _fit_tau_for_alpha(
                    D_init, a_star, basis_Q, venous_basis_Q,
                    meas_edges_harm)
                a_new, _, used = _alpha_star_at_D(
                    D_init, basis_Q, venous_basis_Q, meas_edges,
                    meas_edges_harm, tau_tile=tau_new)
                if a_new is None:
                    break
                d_alpha = abs(a_new - a_star)
                d_tau = abs(((tau_new - tau_fit + T_period / 2)
                              % T_period) - T_period / 2)
                if verbose:
                    print(f"    α↔τ iter {it+1}: α={a_new:.4f}, "
                          f"τ={tau_new*1e3:.2f} ms "
                          f"({np.degrees(omega0*tau_new):+.1f}° at f0), "
                          f"Δα={d_alpha:.4g}, Δτ={d_tau*1e3:.2f} ms")
                a_star, tau_fit = a_new, tau_new
                fit_used_edges = used
                if d_alpha < 1e-6 and d_tau < 1e-5:
                    break
        if a_star is not None:
            alpha_fit = a_star
            n_used = len(used)
            if verbose:
                kk = ('DC + ' + ', '.join(f'H{k}'
                                           for k in range(1, n_harmonics_loss + 1))
                      if n_harmonics_loss >= 1 else 'DC')
                tau_str = ''
                if fit_tile_tau and n_harmonics_loss >= 1:
                    omega0 = 2.0 * np.pi * f0_hz
                    tau_str = (f", τ_tile={tau_fit*1e3:.2f} ms "
                               f"({np.degrees(omega0*tau_fit):+.1f}° at f0)")
                print(f"  α FIT to tile {tile_id} measurements ({kk}, "
                      f"w=|Q_m|{tau_str}): α* = {alpha_fit:.4f} "
                      f"({n_used} edges used)")

            # Loss landscape L(α) — sweep around α* using DC residual
            # (cheaper than recomputing AC contributions; AC terms are
            # also quadratic in α, so the curvature is preserved up to a
            # rescaling of the parabolic width).
            if alpha_fit > 0:
                alpha_vals = np.concatenate([
                    np.linspace(0.0, 0.5 * alpha_fit, 15),
                    np.linspace(0.5 * alpha_fit, 1.5 * alpha_fit, 40),
                    np.linspace(1.5 * alpha_fit, 3.0 * alpha_fit, 20)
                ])
                alpha_vals = np.unique(alpha_vals)
                loss_vals = np.zeros_like(alpha_vals)
                for i, a in enumerate(alpha_vals):
                    L_a = 0.0
                    for (b_e, qm) in fit_used_edges.values():
                        L_a += qm * (a * b_e - qm) ** 2
                    loss_vals[i] = L_a
                alpha_sweep = (alpha_vals, loss_vals)
        else:
            if verbose:
                print(f"  ⚠️  α fit skipped: insufficient tile-measured "
                      f"edges with finite basis response.")

    # ------------------------------------------------------------------
    # 5b. Per-red complex amplitude regression (optional, supersedes α*)
    # ------------------------------------------------------------------
    # Parameterization:
    #   Q_red_r^(k) = ζ_r · w_art^(k)        (ζ_r ∈ ℂ, k-independent)
    # Forward solve linear in {ζ_r} ⇒ stack measurements vs. per-red basis
    # and solve weighted complex normal equations.  Replaces the single
    # global α with K complex parameters (K = total red count).  D fixed
    # at D_init (no scan in this path).
    zeta_per_red = None       # dict {red_id: complex}
    zeta_prior   = None       # dict {red_id: real}  KCL shares
    zeta_diagnostics = None
    per_red_basis_Q = None    # B[r][k][edge] for downstream reconstruction
    zeta_tau_fit = 0.0        # per-tile temporal phase from ζ-mode fit
    if per_red_complex and meas_edges_harm and n_harmonics_loss >= 1:
        if verbose:
            print(f"  Per-red complex amplitude regression: "
                  f"{sum(len(rs) for rs in blue_red_map.values())} reds, "
                  f"K={n_harmonics_loss} AC harmonics, "
                  f"prior λ={zeta_prior_strength}")

        red_list = sorted([r for blue in blue_red_map
                            for r in blue_red_map[blue]])
        K_red = len(red_list)
        red_idx = {r: i for i, r in enumerate(red_list)}

        # ---- Per-red basis: one unit-injection solve per red per harm
        # B_per_red[r][k][edge] = w_art^(k) · response on edge to a unit
        # injection at red r alone (with all other reds zero, venous BCs
        # active at DC, zero at AC for this basis).
        B_per_red = {}
        for r in red_list:
            per_harm = {}
            for k, omega_k in enumerate(omegas):
                red_bc = {r: complex(waveform[k])}
                ven_for_solve = Q_ven if k == 0 else {}
                P = _solve_with_red_bcs(red_bc, ven_for_solve, omega_k,
                                         D_val=D_init)
                if P is None:
                    continue
                per_harm[k] = _edge_Q(P, omega_k, D_val=D_init)
            B_per_red[r] = per_harm
        per_red_basis_Q = B_per_red

        # ---- Stack design matrix B, data r, weights W ------------------
        # One row per (edge, harmonic ≥ 0) pair with finite measurement.
        # Sign-flip orientation for measured Q if edge stored reversed in
        # the basis (rare; matches existing logic).
        rows_B   = []
        rows_M   = []   # raw signed measurement (un-rotated) per row
        rows_Qv  = []   # venous-only contribution per row (subtracted)
        rows_k   = []   # harmonic index per row (0 = DC)
        weights  = []
        # All edges that have AT LEAST one column basis entry — used to
        # detect orientation alignment.
        all_basis_edges = set()
        for r in red_list:
            for k_map in B_per_red[r].values():
                all_basis_edges.update(k_map.keys())

        for e, coeffs in meas_edges_harm.items():
            e_key = e if e in all_basis_edges else (
                (e[1], e[0]) if (e[1], e[0]) in all_basis_edges else None)
            if e_key is None:
                continue
            sign = 1.0 if e_key == e else -1.0

            # DC row: signed real Q_meas (signed because the per-red
            # bases carry sign information).  τ rotation has no effect
            # at k=0 (e^0 = 1).
            q_dc_signed = sign * complex(coeffs[0])
            row = np.zeros(K_red, dtype=complex)
            for r in red_list:
                v = B_per_red[r].get(0, {}).get(e_key)
                if v is not None:
                    row[red_idx[r]] = complex(v)
            if np.abs(row).sum() > 1e-30:
                w_dc = abs(q_dc_signed)
                rows_B.append(row)
                rows_M.append(q_dc_signed)
                rows_Qv.append(complex(0.0))
                rows_k.append(0)
                weights.append(w_dc)

            # AC rows (k = 1..n_harmonics_loss): complex residuals.
            # Track raw M and Q_v separately so τ rotation can be applied
            # to M only (it represents the unknown tile-time offset of
            # the measurement, not the simulation prediction).
            for k in range(1, n_harmonics_loss + 1):
                if k > n_harmonics:
                    break
                q_m_k = sign * complex(coeffs[k])
                q_v_k = venous_basis_Q.get(k, {}).get(e_key, complex(0.0))
                row = np.zeros(K_red, dtype=complex)
                for r in red_list:
                    v = B_per_red[r].get(k, {}).get(e_key)
                    if v is not None:
                        row[red_idx[r]] = complex(v)
                if np.abs(row).sum() < 1e-30:
                    continue
                w_k = abs(q_m_k)
                if w_k < 1e-30:
                    continue
                rows_B.append(row)
                rows_M.append(q_m_k)
                rows_Qv.append(q_v_k)
                rows_k.append(k)
                weights.append(w_k)

        # Per-row residual vector at given τ:
        #   r_i = M_i · e^{-i·k_i·ω₀·τ} − Q_v_i
        # used by both the WLS solve and (for τ refit) the τ-from-rows
        # objective.  k=0 rows leave M unchanged.
        rows_r = [m - qv for m, qv in zip(rows_M, rows_Qv)]

        if not rows_B:
            if verbose:
                print(f"  ⚠️  Per-red ζ regression: no valid "
                      f"(edge, harmonic) constraints — skipping.")
        else:
            B_mat = np.array(rows_B, dtype=complex)
            M_vec = np.array(rows_M, dtype=complex)
            Qv_vec = np.array(rows_Qv, dtype=complex)
            k_vec = np.array(rows_k, dtype=int)
            w_vec = np.array(weights, dtype=float)

            # ---- KCL-share prior: ζ_prior_r = ŝ_r (real, zero phase)
            zeta_prior = {r: float(shares.get(r, 0.0)) for r in red_list}
            zeta_prior_vec = np.array(
                [complex(zeta_prior[r]) for r in red_list], dtype=complex)

            # ---- Identify "dead" columns: reds whose basis response
            # is essentially zero across all measured (e, k).  These
            # are reds with no external visibility (ŝ_kcl ≈ 0), and
            # their columns make B*WB rank-deficient.  Drop them from
            # the active subproblem and fix their ζ at the prior value.
            col_norms = np.sqrt(
                np.sum((np.abs(B_mat) ** 2) * w_vec[:, None], axis=0))
            col_scale = float(col_norms.max()) if col_norms.size else 0.0
            active_mask = col_norms > max(col_scale * 1e-10, 1e-30)
            n_active = int(active_mask.sum())
            n_dead = K_red - n_active
            if verbose and n_dead > 0:
                dead_reds = [red_list[i] for i in range(K_red)
                              if not active_mask[i]]
                print(f"  Per-red ζ: dropping {n_dead} dead column(s) "
                      f"(no basis response): "
                      f"{dead_reds[:6]}{'...' if len(dead_reds) > 6 else ''}")

            B_act = B_mat[:, active_mask]
            zeta_prior_act = zeta_prior_vec[active_mask]
            sqrtW = np.sqrt(w_vec)

            # ---- Pre-build sheet-similarity rows (augmented LS) ------
            # For each sheet b with K_b reds, pick anchor = the red with
            # largest KCL share, then for each non-anchor red r in b add
            # one row penalizing (ζ_r − ζ_anchor): row has +√λ_sheet at
            # col(r) and −√λ_sheet at col(anchor), RHS = 0.
            # Stays in active-column subspace; reds dropped as dead are
            # skipped.  This is a graph-Laplacian-like soft constraint:
            # within a sheet, all ζ values pull toward the anchor.
            sheet_aug_A = None
            sheet_aug_b = None
            if zeta_sheet_phase_strength > 0 and n_active >= 2:
                # Map full-list red index → active-list red index
                active_indices = np.where(active_mask)[0]
                full_to_active = {full: i_active
                                   for i_active, full in enumerate(active_indices)}
                aug_rows = []
                sqrt_lam_s = float(np.sqrt(zeta_sheet_phase_strength))
                for blue, reds_in_sheet in blue_red_map.items():
                    if len(reds_in_sheet) < 2:
                        continue
                    # Anchor: red with largest KCL share in this sheet
                    anchor = max(reds_in_sheet,
                                  key=lambda rr: shares.get(rr, 0.0))
                    a_full = red_idx.get(anchor)
                    if a_full is None or a_full not in full_to_active:
                        continue
                    a_act = full_to_active[a_full]
                    for rr in reds_in_sheet:
                        if rr == anchor:
                            continue
                        r_full = red_idx.get(rr)
                        if r_full is None or r_full not in full_to_active:
                            continue
                        r_act = full_to_active[r_full]
                        row = np.zeros(n_active, dtype=complex)
                        row[r_act] = sqrt_lam_s
                        row[a_act] = -sqrt_lam_s
                        aug_rows.append(row)
                if aug_rows:
                    sheet_aug_A = np.vstack(aug_rows)
                    sheet_aug_b = np.zeros(len(aug_rows), dtype=complex)
                    if verbose:
                        n_sheets_active = sum(
                            1 for blue, rs in blue_red_map.items()
                            if len(rs) >= 2)
                        print(f"  Sheet-similarity prior: λ_sheet="
                              f"{zeta_sheet_phase_strength:.4g}  "
                              f"({len(aug_rows)} pair rows across "
                              f"{n_sheets_active} sheets)")

            def _solve_zeta(tau_val):
                """WLS solve for ζ given current τ rotation of the AC
                measurements.  τ has no effect at k=0.

                When `estimator == 'zeta_phase_only'`, the magnitudes
                |ζ_r| are pinned at the KCL prior ŝ_r and only ∠ζ_r
                (per-red phase) plus τ (already an outer variable) are
                fit.  Closed-form for that variant per red:
                    ∠ζ_r = arg(Σ_i w_i · conj(B_{i,r}) · r̃_i)
                    |ζ_r| = ŝ_r
                where r̃_i = M_i e^{-ikω₀τ} − Q_v_i − Σ_{r'≠r} ζ_{r'}·B_{i,r'}.
                Iterating over reds (or equivalently, solving the
                constrained problem in Cartesian space and projecting)
                converges; for parameter counts this small a single
                pass is fine if we initialize phases from the
                magnitude-free WLS solution.

                Sheet-similarity prior (when zeta_sheet_phase_strength
                > 0) adds rows to the augmented system penalizing each
                non-anchor red's deviation from its sheet's anchor.
                """
                omega0 = 2.0 * np.pi * f0_hz
                rot = np.exp(-1j * k_vec * omega0 * tau_val)
                M_rot = M_vec * rot
                r_vec_local = M_rot - Qv_vec

                # Step 1: unconstrained WLS (gives both magnitudes and
                # phases).  Used directly in 'zeta_per_red'; serves as
                # the phase initialization for 'zeta_phase_only'.
                # Augmented system stacks: (a) data rows, (b) Tikhonov
                # KCL prior rows, (c) sheet-similarity rows.
                A_blocks = [sqrtW[:, None] * B_act]
                b_blocks = [sqrtW * r_vec_local]
                if zeta_prior_strength > 0:
                    A_blocks.append(np.sqrt(zeta_prior_strength)
                                     * np.eye(n_active, dtype=complex))
                    b_blocks.append(np.sqrt(zeta_prior_strength)
                                     * zeta_prior_act)
                if sheet_aug_A is not None:
                    A_blocks.append(sheet_aug_A)
                    b_blocks.append(sheet_aug_b)
                A_aug = np.vstack(A_blocks)
                b_aug = np.concatenate(b_blocks)
                z_act, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
                z_full = zeta_prior_vec.copy()
                z_full[active_mask] = z_act

                # Step 2: phase-only refinement.  Pin |ζ_r| at the KCL
                # share ŝ_r; sweep the active reds, optimizing each
                # phase given the others.  Single Gauss–Seidel pass.
                if estimator == 'zeta_phase_only':
                    # Initial phases from the unconstrained solve.
                    phases = np.angle(z_full)
                    mags = np.abs(zeta_prior_vec).astype(float)
                    z_full = mags * np.exp(1j * phases)
                    # Coordinate descent over phases.
                    for _pass in range(2):     # 2 sweeps for stability
                        for j_idx in np.where(active_mask)[0]:
                            B_col = B_mat[:, j_idx]
                            others = z_full.copy()
                            others[j_idx] = 0.0
                            r_partial = r_vec_local - B_mat @ others
                            num = np.sum(
                                w_vec * np.conj(B_col) * r_partial)
                            phases[j_idx] = float(np.angle(num))
                            z_full[j_idx] = mags[j_idx] * np.exp(
                                1j * phases[j_idx])

                    # Step 3: hard symmetry break.  The (ζ→−ζ, τ→τ+T/2)
                    # transformation is approximately invariant — only
                    # the venous-term contributions at even k break it,
                    # and those are usually small relative to the red-
                    # driven terms.  Magnitude pinning anchors |ζ_r|
                    # but NOT the global rotation, so the optimizer can
                    # land in the −1 basin even though KCL prefers +1.
                    # Always evaluate both basins and pick the one with
                    # lower loss; tie-break to the KCL-aligned sign
                    # (positive projection onto ŝ).  Active-only
                    # projection so dead reds don't contribute zero
                    # noise to the decision.
                    z_flipped = -z_full
                    resid_orig = r_vec_local - B_mat @ z_full
                    resid_flip = r_vec_local - B_mat @ z_flipped
                    wls_orig = float(np.real(np.sum(
                        w_vec * np.abs(resid_orig) ** 2)))
                    wls_flip = float(np.real(np.sum(
                        w_vec * np.abs(resid_flip) ** 2)))
                    proj = float(np.real(np.sum(
                        z_full[active_mask]
                        * np.conj(zeta_prior_vec[active_mask]))))
                    rel_diff = (abs(wls_flip - wls_orig)
                                / max(wls_orig, 1e-30))
                    # Decision logic (verbose-logged below):
                    flipped = False
                    reason = 'kept original (lower loss)'
                    if wls_flip < wls_orig:
                        z_full = z_flipped
                        flipped = True
                        reason = 'flipped (lower loss)'
                    elif rel_diff < 1e-2 and proj < 0.0:
                        z_full = z_flipped
                        flipped = True
                        reason = 'flipped (loss tied, KCL alignment)'
                    elif proj < 0.0:
                        reason = ('kept original (loss prefers −1 basin '
                                   'by >1%, despite KCL-misaligned sign)')
                    if verbose:
                        print(f"    ζ-symmetry-break: "
                              f"wls_orig={wls_orig:.4g}  "
                              f"wls_flip={wls_flip:.4g}  "
                              f"rel_diff={rel_diff:.2%}  "
                              f"proj_on_ŝ={proj:+.4g}  → {reason}")

                # Loss at this τ
                resid = r_vec_local - B_mat @ z_full
                wls = float(np.real(np.sum(w_vec * np.abs(resid) ** 2)))
                return z_full, wls

            # Initial WLS at τ=0
            try:
                zeta_vec, wls_loss = _solve_zeta(zeta_tau_fit)
                zeta_per_red = {r: complex(zeta_vec[red_idx[r]])
                                for r in red_list}
            except np.linalg.LinAlgError as _e:
                if verbose:
                    print(f"  ⚠️  Per-red ζ lstsq failed: {_e}; "
                          f"falling back to KCL prior.")
                zeta_vec = zeta_prior_vec.copy()
                wls_loss = np.inf
                zeta_per_red = {r: complex(zeta_prior_vec[red_idx[r]])
                                for r in red_list}

            # ---- Optional ζ↔τ alternation ---------------------------------
            if (fit_tile_tau and n_harmonics_loss >= 1
                    and len(rows_B) > 0):
                omega0 = 2.0 * np.pi * f0_hz
                T_period = 2.0 * np.pi / omega0
                ac_idx = np.where(k_vec >= 1)[0]
                M_ac = M_vec[ac_idx]
                Qv_ac = Qv_vec[ac_idx]
                B_ac = B_mat[ac_idx, :]
                k_ac = k_vec[ac_idx]
                w_ac = w_vec[ac_idx]
                for it in range(int(n_tau_iterations)):
                    # Predictions per AC row at current ζ
                    pred_ac = B_ac @ zeta_vec + Qv_ac
                    tau_new, _ = _fit_tau_from_rows(
                        M_ac.tolist(), pred_ac.tolist(),
                        k_ac.tolist(), w_ac.tolist())
                    z_new, wls_new = _solve_zeta(tau_new)
                    d_zeta = float(np.max(np.abs(z_new - zeta_vec))) \
                        if zeta_vec.size else np.inf
                    d_tau = abs(((tau_new - zeta_tau_fit + T_period / 2)
                                  % T_period) - T_period / 2)
                    zeta_tau_fit = tau_new
                    zeta_vec = z_new
                    zeta_per_red = {r: complex(zeta_vec[red_idx[r]])
                                    for r in red_list}
                    wls_loss = wls_new
                    if verbose:
                        print(f"    ζ↔τ iter {it+1}: τ_tile="
                              f"{tau_new*1e3:.2f} ms "
                              f"({np.degrees(omega0*tau_new):+.1f}° at f0), "
                              f"loss={wls_new:.4g}, "
                              f"max|Δζ|={d_zeta:.4g}, "
                              f"Δτ={d_tau*1e3:.2f} ms")
                    if d_zeta < 1e-6 and d_tau < 1e-5:
                        break

            # ---- Loss + diagnostics ------------------------------------
            # ---- Per-harmonic residual diagnostic ----------------------
            # Sanity check on the "universal shape" assumption: split
            # the residual norm by harmonic and check whether errors are
            # roughly white across k.  Systematic growth with k flags
            # path dispersion that the k-independent ζ parameterization
            # cannot capture.
            omega0 = 2.0 * np.pi * f0_hz
            rot_final = np.exp(-1j * k_vec * omega0 * zeta_tau_fit)
            M_rot_final = M_vec * rot_final
            r_final = (M_rot_final - Qv_vec) - B_mat @ zeta_vec
            ks_present = sorted(set(k_vec.tolist()))
            per_harm_resid = {}
            for kk in ks_present:
                mask_k = (k_vec == kk)
                if mask_k.sum() == 0:
                    continue
                # Weighted relative residual at this harmonic
                wr = w_vec[mask_k] * np.abs(r_final[mask_k]) ** 2
                wd = w_vec[mask_k] * np.abs(M_rot_final[mask_k]) ** 2
                per_harm_resid[int(kk)] = {
                    'wls_loss_k': float(np.real(np.sum(wr))),
                    'rel_residual_k': float(
                        np.sqrt(np.sum(wr) / max(np.sum(wd), 1e-30))),
                    'n_rows_k': int(mask_k.sum()),
                }

            zeta_diagnostics = {
                'red_list': red_list,
                'zeta_kcl_prior': zeta_prior,
                'zeta_complex': zeta_per_red,
                'wls_loss': wls_loss,
                'n_constraints': len(rows_B),
                'n_active_reds': n_active,
                'n_dead_reds': n_dead,
                'estimator': estimator,
                # Conditioning of the active design matrix (after dead-
                # column removal); reflects identifiability of the
                # parameters that were actually fit.
                'condition_number': float(
                    np.linalg.cond(B_act) if B_act.size else np.inf),
                'per_harmonic_residuals': per_harm_resid,
            }
            if verbose:
                print(f"  Per-harmonic relative residuals "
                      f"(diagnostic for k-independent shape assumption):")
                for kk in sorted(per_harm_resid):
                    d = per_harm_resid[kk]
                    print(f"    k={kk}: rel_resid={d['rel_residual_k']:.3f}  "
                          f"({d['n_rows_k']} rows, "
                          f"loss={d['wls_loss_k']:.4g})")
            if verbose:
                print(f"  ζ* solved: {n_active}/{K_red} active reds, "
                      f"{len(rows_B)} (edge,harm) constraints, "
                      f"cond(B_act)={zeta_diagnostics['condition_number']:.2e}, "
                      f"loss={wls_loss:.4g}")
                print(f"  Per-red ζ vs KCL prior:")
                # Print sorted by |ζ| descending; show magnitude + phase
                # alongside KCL share for direct comparison.
                for r in sorted(red_list, key=lambda r: -abs(zeta_per_red[r])):
                    z = zeta_per_red[r]
                    s_kcl = zeta_prior[r]
                    mag = abs(z)
                    phi = np.degrees(np.angle(z))
                    print(f"    red {r:>6}: |ζ|={mag:.4f}  "
                          f"∠ζ={phi:+6.1f}°   ŝ_kcl={s_kcl:.4f}   "
                          f"|ζ|/ŝ={mag/s_kcl if s_kcl>1e-30 else float('inf'):.2f}")
    elif per_red_complex and verbose:
        print(f"  ⚠️  per_red_complex requires n_harmonics_loss ≥ 1 "
              f"and tile measurements — falling back to global-α path.")

    # --- 5c. Legacy α per sheet = Σ |Q_kcl_red| (total KCL-inferred
    #        outflow for that sheet). Used as fallback if fit_alpha
    #        disabled or no tile measurements. Also kept for diagnostic
    #        comparison vs. the fitted α.
    alpha_per_sheet_direct = {}
    for blue, reds in blue_red_map.items():
        total_mag = 0.0
        reds_set = set(reds)
        stile = sheet_tile.get(blue)
        for rn in reds:
            for nb in G.neighbors(rn):
                if nb in excluded or nb == rn or nb in reds_set:
                    continue
                piv = G.edges[rn, nb].get('measurements_piv', [])
                m_s = next((m for m in piv if m.get('tile_id') == stile),
                           None) if stile else None
                if m_s is None and piv:
                    m_s = max(piv,
                              key=lambda m: m.get('snr_pulse', -np.inf))
                if m_s is None:
                    continue
                Qt = np.asarray(m_s.get('Q_t', []), dtype=float)
                if Qt.size < 20:
                    continue
                q_dc = float(np.nanmean(Qt))
                if np.isfinite(q_dc):
                    total_mag += abs(q_dc)
        alpha_per_sheet_direct[blue] = total_mag

    # Enforce symmetry across grey zones (same α per sheet) unless
    # per_sheet_scale=True explicitly disables this. Use the MAX of the
    # per-sheet totals rather than the mean — so the bigger sheet's
    # measured KCL magnitude is preserved instead of being scaled down
    # by the smaller sheet's contribution.
    if not per_sheet_scale:
        per_sheet_values = list(alpha_per_sheet_direct.values())
        if per_sheet_values:
            alpha_sym = float(max(per_sheet_values))
        else:
            alpha_sym = 0.0
        if verbose:
            print(f"  Per-sheet Σ|Q_kcl| before symmetry:")
            for blue, a in alpha_per_sheet_direct.items():
                print(f"    sheet {blue}: Σ|Q_kcl| = {a:.4f}")
            print(f"  Symmetry: α = max across sheets = {alpha_sym:.4f}")
        alpha_per_sheet_direct = {b: alpha_sym for b in alpha_per_sheet_direct}
    if verbose:
        print(f"  Direct α per sheet (Σ |Q_kcl| → no tile fit):")
        for blue, a in alpha_per_sheet_direct.items():
            print(f"    α[sheet {blue}] = {a:.4f}")

    # Prefer the closed-form fit when available; otherwise fall back to
    # the KCL Σ magnitude. A single global α applies to every sheet
    # (absolute DA scale, distributed to reds by fixed shares).
    if alpha_fit is not None:
        alpha_per_sheet = {b: alpha_fit for b in alpha_per_sheet_direct}
        if verbose:
            print(f"  Using FITTED α = {alpha_fit:.4f} (Σ|Q_kcl| α would "
                  f"have been {max(alpha_per_sheet_direct.values(), default=0):.4f})")
    else:
        alpha_per_sheet = alpha_per_sheet_direct
    alpha_global = float(np.mean(list(alpha_per_sheet.values()))) \
        if alpha_per_sheet else 0.0

    # --- 6. Combined AC solve (superposition of all sources) -----------
    # With α fixed, we can collapse the per-sheet AC bases + separate
    # venous-driven AC basis into ONE solve per harmonic:
    #   Q_rhs[red]   = α · share · w_k      (arterial waveform propagated)
    #   Q_rhs[venous]= Q_ven[k]              (measured venous waveform)
    #   Q_rhs[other] = 0
    # No Dirichlet pin at AC (compliance makes the system non-singular,
    # Tikhonov inside _solve_with_red_bcs handles near-singular edge
    # cases). At DC we keep the per-sheet basis result (venous Dirichlet-
    # pinned) because it's the correct mass-balance gauge.
    combined_Q = {}     # k → dict(edge → complex Q), AC only
    for k, omega_k in enumerate(omegas):
        if k == 0:
            continue    # DC handled by per-sheet basis aggregation
        rhs = {}
        # Red injections (summed over sheets with their α)
        for blue, reds in blue_red_map.items():
            a = alpha_per_sheet.get(blue, 0.0)
            if a == 0.0:
                continue
            for rn in reds:
                if rn in node_to_idx:
                    rhs[rn] = rhs.get(rn, 0.0) \
                        + a * complex(shares.get(rn, 0.0)) * waveform[k]
        # Venous measured Q-BC (signed, already carries sink sign
        # convention from _extract_boundary_harmonics)
        if Q_ven:
            for vn in ven_nodes:
                if vn in node_to_idx and vn in Q_ven:
                    coeffs = np.asarray(Q_ven[vn], dtype=complex)
                    if len(coeffs) > k:
                        rhs[vn] = rhs.get(vn, 0.0) + complex(coeffs[k])
        if not rhs:
            continue
        # Pass empty ven_Q_map → no Dirichlet pin, system closed by
        # compliance + Tikhonov.
        P_comb = _solve_with_red_bcs(rhs, {}, omega_k)
        if P_comb is None:
            continue
        combined_Q[k] = _edge_Q(P_comb, omega_k)

    if verbose and combined_Q:
        print(f"  Combined-source AC solves: {len(combined_Q)} harmonics "
              f"({'reds' if blue_red_map else ''}"
              f"{' + venous' if Q_ven else ''}) — propagates both "
              f"arterial and venous waveforms network-wide.")

    # --- 7. Final combined prediction + residuals --------------------
    all_edges = set()
    for blue_map in basis_Q.values():
        for h_map in blue_map.values():
            all_edges.update(h_map.keys())
    for k_map in combined_Q.values():
        all_edges.update(k_map.keys())
    if per_red_basis_Q is not None:
        for r_map in per_red_basis_Q.values():
            for h_map in r_map.values():
                all_edges.update(h_map.keys())

    edge_flows = {}
    if zeta_per_red is not None and per_red_basis_Q is not None:
        # Per-red complex amplitudes: edge_flow = Σ_r ζ_r · B_r^(k) at every
        # harmonic, plus venous-only contribution at AC.  At DC the venous
        # Dirichlet pin is folded into each per-red basis already (via the
        # k=0 ven_for_solve = Q_ven branch), so no extra venous term is
        # needed at k=0.
        for e in all_edges:
            q_harm = np.zeros(n_harmonics + 1, dtype=complex)
            for k in range(n_harmonics + 1):
                tot = complex(0.0)
                for r, z_r in zeta_per_red.items():
                    v = per_red_basis_Q.get(r, {}).get(k, {}).get(e)
                    if v is not None:
                        tot += z_r * v
                # Venous-only contribution at AC (k≥1)
                if k >= 1:
                    v_ven = venous_basis_Q.get(k, {}).get(e)
                    if v_ven is not None:
                        tot += v_ven
                q_harm[k] = tot
            edge_flows[e] = q_harm
    else:
        for e in all_edges:
            q_harm = np.zeros(n_harmonics + 1, dtype=complex)
            # DC from per-sheet basis (venous-pinned gauge)
            for blue, per_harm in basis_Q.items():
                a = alpha_per_sheet.get(blue, 0.0)
                if a == 0.0:
                    continue
                v = per_harm.get(0, {}).get(e)
                if v is not None:
                    q_harm[0] += a * v
            # AC from combined solve
            for k, h_map in combined_Q.items():
                v = h_map.get(e)
                if v is not None:
                    q_harm[k] = v
            edge_flows[e] = q_harm

    # Override BOUNDARY-EDGE flows with the measured venous BC waveform.
    # Dirichlet pinning at the venous node (P=0) loses the measured AC
    # shape — only the DC magnitude survives through mass balance. For
    # faithful venous waveforms, overwrite each venous boundary edge's
    # complex harmonics with Q_ven directly, matching what the normal
    # `solve_transmission_line` does with `bc_harmonics`. Same treatment
    # for arterial boundary edges when Q_art is supplied.
    def _override_boundary_edges(bc_dict, node_list, label):
        n_over = 0
        for bn in node_list:
            if bn not in G.nodes or bn not in bc_dict:
                continue
            coeffs = np.asarray(bc_dict[bn], dtype=complex)
            if coeffs.size < n_harmonics + 1:
                padded = np.zeros(n_harmonics + 1, dtype=complex)
                padded[:coeffs.size] = coeffs
                coeffs = padded
            for nb in G.neighbors(bn):
                # Only override edges that made it into G_mod (non-grey)
                if (bn, nb) in edge_flows:
                    edge_flows[(bn, nb)] = coeffs[:n_harmonics + 1].copy()
                    n_over += 1
                elif (nb, bn) in edge_flows:
                    # Reverse edge orientation: flow u→v = −(v→u)
                    edge_flows[(nb, bn)] = \
                        -coeffs[:n_harmonics + 1].copy()
                    n_over += 1
        if verbose and n_over:
            print(f"  Overrode {n_over} {label} boundary edges with "
                  f"measured BC harmonics.")

    _override_boundary_edges(Q_ven, ven_nodes, 'venous')
    _override_boundary_edges(Q_art, art_nodes, 'arterial')

    mean_Q = {e: float(np.real(q[0])) for e, q in edge_flows.items()}
    amp_Q = {e: float(abs(q[1])) if n_harmonics >= 1 else 0.0
             for e, q in edge_flows.items()}
    PI = {}
    phase = {}
    for e, q in edge_flows.items():
        dc = q[0].real
        q1 = q[1] if n_harmonics >= 1 else 0.0
        PI[e] = float(2 * abs(q1) / abs(dc)) if abs(dc) > 1e-12 else np.nan
        phase[e] = float(np.angle(q1)) if n_harmonics >= 1 else np.nan

    # RPSI / η from harmonics (real-valued waveform diagnostics)
    RPSI = {}
    for e, q in edge_flows.items():
        try:
            RPSI[e] = float(compute_rpsi_from_harmonics(q, f0_hz))
        except Exception:
            RPSI[e] = np.nan

    # Per-edge WSS and dissipation using tile geometry
    WSS = {}
    dissipation = {}
    pulsatile_cost = {}
    for e, q_h in edge_flows.items():
        u, v = e
        R_m, L_m = _get_edge_geometry(G_mod, u, v, radii_m=radii_m or {})
        if R_m is None:
            WSS[e] = np.nan
            dissipation[e] = np.nan
            pulsatile_cost[e] = np.nan
            continue
        # Reconstruct Q(t) over one cycle in nL/s
        n_pts = 128
        t_cyc = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        Q_t = np.full(n_pts, q_h[0].real)
        for nh in range(1, n_harmonics + 1):
            Q_t += (q_h[nh].real * np.cos(nh * t_cyc)
                    - q_h[nh].imag * np.sin(nh * t_cyc))
        Q_mean_abs_m3s = float(np.mean(np.abs(Q_t))) * 1e-12
        WSS[e] = 4.0 * mu * Q_mean_abs_m3s / (np.pi * R_m**3)
        Q_dc_sq = (q_h[0].real * 1e-12) ** 2
        Q_ac_sq = 0.5 * sum(abs(q_h[n] * 1e-12) ** 2
                            for n in range(1, n_harmonics + 1))
        r_per_length = 8.0 * mu / (np.pi * R_m**4)
        dissipation[e] = r_per_length * L_m * (Q_dc_sq + Q_ac_sq)
        pulsatile_cost[e] = ((Q_dc_sq + Q_ac_sq) / Q_dc_sq
                             if Q_dc_sq > 1e-30 else np.nan)

    residuals = {}
    for e, q_meas in meas_edges.items():
        q_signed = mean_Q.get(e)
        if q_signed is None:
            q_signed = -mean_Q.get((e[1], e[0]), 0.0)
        residuals[e] = abs(abs(q_signed) - q_meas)

    class _SR:
        pass
    result = _SR()
    result.f0_hz = f0_hz
    result.D = D_init
    result.mu = mu
    result.n_harmonics = n_harmonics
    result.mean_Q = mean_Q
    result.amp_Q = amp_Q
    result.PI = PI
    result.RPSI = RPSI
    result.phase = phase
    result.WSS = WSS
    result.dissipation = dissipation
    result.pulsatile_cost = pulsatile_cost
    result.storage_fraction = {k: np.nan for k in mean_Q}

    # Aggregate per-harmonic node pressures across sheets (α-weighted sum,
    # matching the edge_flows aggregation) so Q_out at v-ports can be
    # reconstructed in waveform-click views.
    node_pressures = {n: np.zeros(n_harmonics + 1, dtype=complex)
                      for n in all_mod_nodes}
    for blue, per_harm_P in basis_P.items():
        a = alpha_per_sheet.get(blue, 0.0)
        if a == 0.0:
            continue
        for k, P_vec in per_harm_P.items():
            for n, idx in node_to_idx.items():
                node_pressures[n][k] += a * P_vec[idx]

    # Q_stored per edge per harmonic: Q_u + Q_v (both defined as flow INTO
    # the edge), reconstructed from pressures and vessel admittance.
    Q_stored = {}
    for e in edge_flows:
        u, v = e
        R_m, L_m = _get_edge_geometry(G_mod, u, v, radii_m=radii_m or {})
        if R_m is None:
            Q_stored[e] = np.zeros(n_harmonics + 1, dtype=complex)
            continue
        q_st = np.zeros(n_harmonics + 1, dtype=complex)
        for k, omega_k in enumerate(omegas):
            Yd, Yo = _vessel_admittance(R_m, L_m, omega_k, mu, RHO_BLOOD, D_init)
            P_u = node_pressures.get(u, np.zeros(n_harmonics+1))[k]
            P_v = node_pressures.get(v, np.zeros(n_harmonics+1))[k]
            Q_u = Yd * P_u + Yo * P_v
            Q_v = Yo * P_u + Yd * P_v
            q_st[k] = Q_u + Q_v
        Q_stored[e] = q_st
    result.Q_stored = Q_stored
    result.node_pressures = node_pressures
    result.boundary_nodes = list(ven_nodes) + list(blue_red_map.keys())
    result.edge_flows = edge_flows

    # Weighted tile-fit loss (w=|Q_m|) — the objective that was minimized
    # in closed form. Reported alongside the unweighted L2 for comparison.
    weighted_loss = 0.0
    for e, q_m in meas_edges.items():
        q_signed = mean_Q.get(e)
        if q_signed is None:
            q_signed = -mean_Q.get((e[1], e[0]), 0.0)
        weighted_loss += abs(q_m) * (abs(q_signed) - abs(q_m)) ** 2

    # Per-edge tile residuals — for each measured edge, report absolute
    # and relative error (sim − meas, signed) so the viewer can colour
    # the network and find systematically mis-fit regions.
    tile_residuals_abs = {}        # signed (sim − meas)
    tile_residuals_sq  = {}        # squared (gives the pointwise loss)
    tile_residuals_rel = {}        # (sim − meas) / meas
    for e, q_m in meas_edges.items():
        q_signed = mean_Q.get(e)
        if q_signed is None:
            q_signed = -mean_Q.get((e[1], e[0]), 0.0)
        q_sim_mag = abs(q_signed)
        q_meas_mag = abs(float(q_m))
        err = q_sim_mag - q_meas_mag
        tile_residuals_abs[e] = float(err)
        tile_residuals_sq[e]  = float(err * err)
        tile_residuals_rel[e] = (float(err / q_meas_mag)
                                  if q_meas_mag > 1e-12 else np.nan)

    return {
        'D_opt': D_init,
        's': float(alpha_global) if np.isfinite(alpha_global) else np.nan,
        'alpha_per_sheet': alpha_per_sheet,
        'alpha_global': float(alpha_global) if np.isfinite(alpha_global) else np.nan,
        'alpha_fit': alpha_fit,
        'alpha_kcl': (max(alpha_per_sheet_direct.values())
                      if alpha_per_sheet_direct else np.nan),
        'alpha_sweep': alpha_sweep,        # (α_grid, L(α))
        'fit_used_edges': fit_used_edges,  # {edge → (|b|, |Q_m|)}
        'shares': shares,
        'sheet_tile': sheet_tile,
        'sheet_of_red': sheet_of_red,
        'loss': sum(r ** 2 for r in residuals.values()),
        'loss_weighted': weighted_loss,
        'result': result,
        'residuals': residuals,
        'tile_residuals_abs': tile_residuals_abs,
        'tile_residuals_sq':  tile_residuals_sq,
        'tile_residuals_rel': tile_residuals_rel,
        'meas_edges': dict(meas_edges),
        'meas_edges_harm': dict(meas_edges_harm),  # complex per-harm
        'n_harmonics_loss': n_harmonics_loss,
        'D_scan': D_scan_result,           # None if no scan was run
        # Per-red complex amplitude regression results (None if not run)
        'per_red_complex': bool(zeta_per_red is not None),
        'zeta_per_red': zeta_per_red,            # {red_id: complex}
        'zeta_kcl_prior': zeta_prior,            # {red_id: real ŝ}
        'zeta_diagnostics': zeta_diagnostics,    # cond, loss, n_constraints
        # Per-tile temporal-phase nuisance parameter (seconds).  Tile-
        # specific cardiac-phase offset; non-zero means the tile's video
        # was imaged with this lag relative to the arterial reference.
        'tau_tile_alpha': float(tau_fit),
        'tau_tile_zeta': float(zeta_tau_fit),
        # API-level metadata.  Frame indicates the time-reference frame
        # in which simulated edge flows are reported.  All sim outputs
        # (mean_Q_sim, edge_flows, etc.) are in the *arterial* reference
        # frame regardless of estimator; measured values are in the
        # *tile* reference frame, offset by `tau_tile_*`.  Comparing
        # simulated vs measured at AC harmonics requires applying the
        # rotation `M̃ = M · exp(-i·k·ω₀·τ_tile_*)` first.
        'estimator': estimator,
        'frame': 'arterial',
        # Per-estimator τ to use when comparing measurements to sim:
        'tau_tile_active': float(
            zeta_tau_fit if estimator in ('zeta_per_red', 'zeta_phase_only')
            else tau_fit),
        'tile_id': tile_id,
    }


def optimize_greyzone_kirchhoff(
    G: 'nx.Graph',
    excluded_nodes: set,
    blue_red_map: Dict[int, List[int]],
    art_nodes: List[int],
    ven_nodes: List[int],
    Q_art: Dict[int, np.ndarray],
    Q_ven: Dict[int, np.ndarray],
    n_harmonics: int = 3,
    f0_hz: float = 2.5,
    tile_id: Optional[int] = None,
    per_sheet: bool = True,
    verbose: bool = True,
) -> dict:
    """Pure-Kirchhoff grey-zone conservation.

    For each red boundary node: infer the flow leaving into the grey zone
    by summing signed measured Q on its external (non-grey-zone) edges —
    KCL guarantees that equals the net grey-zone outflow at that node.

    For each sheet (blue sink + its red nodes):
        Σ_reds Q_kirchhoff_red = α_sheet · Q_measured_blue
    Solve for α per sheet (or a single global α).

    No admittance optimization; no forward solve. Linear scalar-per-sheet.
    Returns a sim-result compatible dict whose edge Q's come from the
    inferred red outflows (propagated through the network by simple sign).
    """
    from .config import FRAME_DT_S
    from .harmonic import fit_harmonics

    excluded = set(excluded_nodes)
    all_blue = sorted(blue_red_map.keys())

    def _edge_dc_signed(u, v, d, toward_node):
        """Signed DC Q on edge flowing toward `toward_node`, or nan.

        Uses best-SNR measurement regardless of tile — KCL accounting at
        the grey-zone boundary needs whatever measurement we have per edge.
        If tile_id is set, prefer that tile's data but fall back to best-
        SNR if the selected tile didn't measure this edge.
        """
        piv = d.get('measurements_piv', [])
        if not piv:
            # Fallback to the promoted edge-level fields if present
            q_dc = d.get('mean_Q')
            if q_dc is None or not np.isfinite(q_dc):
                return np.nan
            flow_from = d.get('flow_from', u)
            flow_to = d.get('flow_to', v)
            q_abs = abs(float(q_dc))
            if toward_node == flow_to:
                return +q_abs
            elif toward_node == flow_from:
                return -q_abs
            return np.nan
        best = None
        if tile_id is not None:
            tile_piv = [m for m in piv if m.get('tile_id') == tile_id]
            if tile_piv:
                best = tile_piv[0]
        if best is None:
            best = max(piv, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        if Qt is None or len(Qt) < 20:
            # Fall back to stored mean_Q scalar on the measurement
            q_dc = best.get('mean_Q')
            if q_dc is None or not np.isfinite(q_dc):
                return np.nan
            flow_from = best.get('flow_from', u)
            flow_to = best.get('flow_to', v)
            q_abs = abs(float(q_dc))
            if toward_node == flow_to:
                return +q_abs
            elif toward_node == flow_from:
                return -q_abs
            return np.nan
        Qt_arr = np.asarray(Qt, dtype=float)
        q_dc = float(np.nanmean(Qt_arr))
        if not np.isfinite(q_dc):
            return np.nan
        # Canonical direction: stored mean_Q >= 0, flowing from flow_from → flow_to
        flow_from = best.get('flow_from', u)
        flow_to = best.get('flow_to', v)
        q_abs = abs(q_dc)
        if toward_node == flow_to:
            return +q_abs
        elif toward_node == flow_from:
            return -q_abs
        return np.nan

    def _kirchhoff_at_red(rn):
        """Inferred DC Q leaving the grey zone at red node rn (nL/s).

        KCL at rn:  Σ Q_toward_rn(all edges) = 0
            =>  Q_toward_rn(grey edge) = -Σ Q_toward_rn(external edges)
        Flow LEAVING grey at rn = Q_toward_rn(grey edge) = -Σ externals.

        To keep KCL consistent, picks a single *tile* whose PIV measurements
        cover the most external edges at rn and uses only that tile's data.
        Falls back to best-SNR per edge if no single tile is dominant.
        """
        # Collect external edges (non-excluded neighbors, non-self)
        ext_edges = []
        for nb in G.neighbors(rn):
            if nb in excluded or nb == rn:
                continue
            ext_edges.append(nb)
        if not ext_edges:
            return 0.0, 0, None

        # Tile coverage: which tiles have measurements for how many of these edges?
        tile_coverage = {}  # tile_id -> set of edge-nb indices
        for i, nb in enumerate(ext_edges):
            piv = G.edges[rn, nb].get('measurements_piv', [])
            for m in piv:
                tid = m.get('tile_id')
                if tid is None:
                    continue
                Qt = m.get('Q_t')
                if Qt is None or len(Qt) < 20:
                    continue
                tile_coverage.setdefault(tid, set()).add(i)
        best_tile = None
        if tile_coverage:
            best_tile = max(tile_coverage.keys(),
                            key=lambda t: len(tile_coverage[t]))

        total_toward = 0.0
        n_edges = 0
        for i, nb in enumerate(ext_edges):
            # Try best_tile first, then fall back
            q = np.nan
            if best_tile is not None and i in tile_coverage.get(best_tile, set()):
                piv = G.edges[rn, nb].get('measurements_piv', [])
                m_tile = next((m for m in piv if m.get('tile_id') == best_tile), None)
                if m_tile is not None:
                    Qt = np.asarray(m_tile.get('Q_t', []), dtype=float)
                    if Qt.size >= 20:
                        q_dc = float(np.nanmean(Qt))
                        if np.isfinite(q_dc):
                            flow_from = m_tile.get('flow_from', rn)
                            flow_to = m_tile.get('flow_to', nb)
                            q_abs = abs(q_dc)
                            if rn == flow_to:
                                q = +q_abs
                            elif rn == flow_from:
                                q = -q_abs
            if not np.isfinite(q):
                # Fall back
                q = _edge_dc_signed(rn, nb, G.edges[rn, nb], toward_node=rn)
            if np.isfinite(q):
                total_toward += q
                n_edges += 1
        return -total_toward, n_edges, best_tile

    if verbose:
        print(f"  Kirchhoff mode: {len(all_blue)} sheets, "
              f"{sum(len(r) for r in blue_red_map.values())} red nodes total")

    # Per-sheet conservation
    alphas = {}
    red_Q = {}       # red_node -> inferred Q (nL/s, positive = into grey)
    sheet_info = {}  # blue -> {'Q_red_total', 'Q_blue', 'alpha', 'n_valid_reds'}

    # Total aorta / venous DC for global scale fallback
    total_Q_art_dc = sum(Q_art[b][0].real for b in art_nodes if b in Q_art)
    total_Q_ven_dc = sum(Q_ven[b][0].real for b in ven_nodes if b in Q_ven)

    red_tiles = {}  # red_node -> tile_id used
    for blue, reds in blue_red_map.items():
        # Pick ONE tile for the entire sheet — the one covering the most
        # external edges across ALL reds in this sheet. This ensures the
        # measurements are simultaneously consistent (KCL holds), which
        # doesn't happen when each red picks its own tile.
        from collections import Counter as _Counter2
        sheet_tile_coverage = _Counter2()
        for rn in reds:
            for nb in G.neighbors(rn):
                if nb in excluded or nb == rn:
                    continue
                for m in G.edges[rn, nb].get('measurements_piv', []):
                    tid = m.get('tile_id')
                    Qt = m.get('Q_t')
                    if tid is None or Qt is None or len(Qt) < 20:
                        continue
                    sheet_tile_coverage[tid] += 1
        sheet_tile = None
        if sheet_tile_coverage:
            sheet_tile = sheet_tile_coverage.most_common(1)[0][0]

        def _edge_dc_for_sheet(rn, nb):
            """Signed DC Q toward rn, using ONLY sheet_tile if available."""
            piv = G.edges[rn, nb].get('measurements_piv', [])
            m_s = next((m for m in piv
                        if m.get('tile_id') == sheet_tile), None) if sheet_tile else None
            if m_s is None:
                return np.nan
            Qt = np.asarray(m_s.get('Q_t', []), dtype=float)
            if Qt.size < 20:
                return np.nan
            q_dc = float(np.nanmean(Qt))
            if not np.isfinite(q_dc):
                return np.nan
            flow_from = m_s.get('flow_from', rn)
            flow_to = m_s.get('flow_to', nb)
            q_abs = abs(q_dc)
            if rn == flow_to:
                return +q_abs
            elif rn == flow_from:
                return -q_abs
            return np.nan

        # Red-to-red edges within this sheet are plexus-internal — exclude.
        reds_set = set(reds)

        Q_red_sum = 0.0
        n_valid = 0
        for rn in reds:
            total_mag = 0.0
            n_edges = 0
            for nb in G.neighbors(rn):
                if nb in excluded or nb == rn:
                    continue
                # Skip edges that connect two red nodes in the same sheet
                if nb in reds_set:
                    continue
                q = _edge_dc_for_sheet(rn, nb)
                if not np.isfinite(q):
                    q = _edge_dc_signed(rn, nb, G.edges[rn, nb], toward_node=rn)
                if np.isfinite(q):
                    # Assume flow is always OUT of the sheet at this red →
                    # sum magnitudes of external (non-red-to-red) edges.
                    total_mag += abs(q)
                    n_edges += 1
            q_out = total_mag
            if n_edges > 0:
                red_Q[rn] = q_out
                red_tiles[rn] = sheet_tile
                Q_red_sum += q_out
                n_valid += 1

        # Get blue's "inflow" prescribed Q (measured Q at the aorta)
        # Blue sinks are fed by the arterial supply; for conservation we
        # want the Q that would enter the grey zone at blue.
        Q_blue = 0.0
        for nb in G.neighbors(blue):
            if nb in excluded:
                continue
            d = G.edges[blue, nb]
            q = _edge_dc_signed(blue, nb, d, toward_node=blue)
            if np.isfinite(q):
                Q_blue += q   # positive = into blue = into grey at this node

        if abs(Q_blue) > 1e-12:
            alpha = Q_red_sum / Q_blue
        else:
            alpha = np.nan
        alphas[blue] = float(alpha)
        sheet_info[blue] = {
            'Q_red_total': float(Q_red_sum),
            'Q_blue': float(Q_blue),
            'alpha': float(alpha),
            'n_valid_reds': n_valid,
            'n_reds': len(reds),
        }
        if verbose:
            from collections import Counter as _Counter
            tile_counts = _Counter(red_tiles.get(rn) for rn in reds
                                    if red_tiles.get(rn) is not None)
            tile_summary = ", ".join(f"t{t}:{c}" for t, c in tile_counts.most_common())
            print(f"    Blue {blue}: Q_blue={Q_blue:.3f}  Σ Q_reds="
                  f"{Q_red_sum:.3f}  α={alpha:.3f}  "
                  f"({n_valid}/{len(reds)} reds; tiles used: {tile_summary})")
            # Per-red breakdown — sorted by magnitude so dominant outliers show
            red_q_pairs = [(rn, red_Q[rn]) for rn in reds if rn in red_Q]
            red_q_pairs.sort(key=lambda x: -abs(x[1]))
            for rn, q in red_q_pairs:
                frac = (q / Q_red_sum * 100.0) if abs(Q_red_sum) > 1e-12 else np.nan
                print(f"      red {rn:>6}: Q_out={q:+.4f}  ({frac:+.1f}% of Σ)")

    # Global α
    alpha_global = np.nan
    total_red_all = sum(s['Q_red_total'] for s in sheet_info.values())
    total_blue_all = sum(s['Q_blue'] for s in sheet_info.values())
    if abs(total_blue_all) > 1e-12:
        alpha_global = total_red_all / total_blue_all
    if verbose:
        print(f"  Global: Σ Q_blue={total_blue_all:.3f}  "
              f"Σ Q_red={total_red_all:.3f}  α_global={alpha_global:.3f}")

    # Build sim-result-compatible dict: edge mean_Q comes from measured
    # external edges (as-is). Grey-interior edges remain NaN. Red-sheet
    # internal flow cannot be inferred from KCL alone.
    mean_Q = {}
    amp_Q = {}
    PI = {}
    RPSI = {}
    phase = {}
    WSS = {}
    dissipation = {}
    pulsatile_cost = {}
    storage_fraction = {}

    for u, v, d in G.edges(data=True):
        # Skip grey-interior edges
        if u in excluded and v in excluded:
            continue
        piv = d.get('measurements_piv', [])
        if tile_id is not None:
            piv = [m for m in piv if m.get('tile_id') == tile_id]
        if not piv:
            continue
        best = (piv[0] if tile_id is not None
                else max(piv, key=lambda m: m.get('snr_pulse', -np.inf)))
        Qt = best.get('Q_t')
        if Qt is None or len(Qt) < 20:
            continue
        Qt_arr = np.asarray(Qt, dtype=float)
        if np.nanmean(Qt_arr) < 0:
            Qt_arr = -Qt_arr
        hr = fit_harmonics(Qt_arr, best.get('f0_hz', f0_hz), FRAME_DT_S,
                           K=n_harmonics, loss='huber', include_dc=True)
        q_dc = hr.get('a0', float(np.nanmean(Qt_arr)))
        q1 = 0.0 + 0.0j
        for h in hr.get('harmonics', []):
            if h['k'] == 1:
                q1 = h['A'] - 1j * h['B']
                break
        mean_Q[(u, v)] = float(abs(q_dc))
        amp_Q[(u, v)] = float(abs(q1))
        PI[(u, v)] = (2 * abs(q1) / abs(q_dc)) if abs(q_dc) > 1e-15 else np.nan
        RPSI[(u, v)] = np.nan
        phase[(u, v)] = float(np.angle(q1)) if abs(q1) > 0 else np.nan
        WSS[(u, v)] = np.nan
        dissipation[(u, v)] = np.nan
        pulsatile_cost[(u, v)] = np.nan
        storage_fraction[(u, v)] = np.nan

    class _SimResult:
        pass
    result = _SimResult()
    result.f0_hz = f0_hz
    result.mean_Q = mean_Q
    result.amp_Q = amp_Q
    result.PI = PI
    result.RPSI = RPSI
    result.phase = phase
    result.WSS = WSS
    result.dissipation = dissipation
    result.pulsatile_cost = pulsatile_cost
    result.storage_fraction = storage_fraction
    result.node_pressures = {}
    result.edge_flows = {k: np.array([mean_Q[k]] + [0j] * n_harmonics)
                         for k in mean_Q}

    # Residuals: |1 - α| per sheet tells you how far from conservation
    residuals = {}
    for blue, s in sheet_info.items():
        residuals[blue] = abs(1.0 - s['alpha']) if np.isfinite(s['alpha']) else np.nan

    return {
        'D_opt': np.nan,
        's': float(alpha_global),
        'alpha_per_sheet': alphas,
        'alpha_global': float(alpha_global),
        'sheet_info': sheet_info,
        'red_Q': red_Q,
        'red_tiles': red_tiles,
        'loss': float(sum(abs(1.0 - a)**2 for a in alphas.values()
                          if np.isfinite(a))),
        'result': result,
        'residuals': residuals,
        'tile_id': tile_id,
    }


def optimize_greyzone_sheet(
    G: 'nx.Graph',
    excluded_nodes: set,
    grey_zone_nodes: Optional[List[int]] = None,
    blue_red_map: Optional[Dict[int, List[int]]] = None,
    art_nodes: Optional[List[int]] = None,
    ven_nodes: Optional[List[int]] = None,
    v_ref: Optional[int] = None,
    Q_art: Optional[Dict[int, np.ndarray]] = None,
    Q_ven: Optional[Dict[int, np.ndarray]] = None,
    n_harmonics: int = 3,
    f0_hz: float = 2.5,
    mu: float = MU_DEFAULT,
    D_init: float = 1e-3,
    radii_m: Optional[Dict[Tuple[int, int], float]] = None,
    tile_id: Optional[int] = None,
    q_min: float = 0.1,
    kh_log_range: Tuple[float, float] = (-14.0, -6.0),
    n_kh: int = 20,
    alpha_sweep: Optional[List[float]] = None,
    a_cutoff_frac: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Grey-zone optimizer: 2D Darcy-sheet model (single scalar κh).

    Replaces the excluded region with a 2D homogeneous sheet parameterized
    by a single permeability·thickness value κh. Uses the Green's-function
    admittance matrix constructed from the 2D positions of the grey-zone
    boundary nodes. Same Y_grey applied at all harmonics (rigid sheet).

    Parameters
    ----------
    grey_zone_nodes : list of node IDs at the grey-zone boundary
        Typically: aorta outlets (blue sinks in existing terminology) +
        plexus boundary nodes (red / blue sources). These are the nodes
        where the grey-zone sheet connects to the resolved network.
    kh_log_range : (lo, hi) log10 range for κh sweep
    n_kh : number of κh values in the log-spaced sweep
    alpha_sweep : optional list of α scaling factors for Q_art (2D grid)
    a_cutoff_frac : diagonal regularization a = frac × median spacing
    """
    from scipy.sparse import csr_matrix as _csr, eye as _sp_eye
    from scipy.sparse.linalg import spsolve as _spsolve
    from .config import FRAME_DT_S, PX_SIZE_UM

    omega0 = 2.0 * np.pi * f0_hz

    # ── 1. Build per-sheet boundary sets ────────────────────────────────
    # Each sheet = {one blue sink} ∪ {red nodes reachable through that sheet}
    # If blue_red_map not supplied, fall back to a single sheet using
    # grey_zone_nodes as a flat list.
    from scipy.spatial.distance import pdist
    sheets = []  # list of dicts: {blue, reds, nodes, coords, Ginv, a_cutoff}
    if blue_red_map is not None and len(blue_red_map) > 0:
        for blue, reds in blue_red_map.items():
            nodes_s = sorted(set([blue]) | set(reds))
            sheets.append({'blue': blue, 'reds': sorted(reds),
                           'nodes': nodes_s})
    else:
        if grey_zone_nodes is None or len(grey_zone_nodes) < 2:
            raise ValueError("Supply blue_red_map or grey_zone_nodes")
        sheets.append({'blue': None, 'reds': None,
                       'nodes': sorted(set(grey_zone_nodes))})

    all_gz_nodes = sorted({n for s in sheets for n in s['nodes']})

    # Build per-sheet Green's matrices (cached — don't depend on κh)
    for si, s in enumerate(sheets):
        coords_s = np.array([
            [G.nodes[n].get('x', 0.0), G.nodes[n].get('y', 0.0)]
            for n in s['nodes']
        ], dtype=float)
        if len(coords_s) < 2:
            raise ValueError(f"Sheet {si}: need ≥2 nodes")
        med_s = float(np.median(pdist(coords_s)))
        a_s = max(a_cutoff_frac * med_s, 1.0)
        dx = coords_s[:, 0:1] - coords_s[None, :, 0]
        dy = coords_s[:, 1:2] - coords_s[None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, a_s)
        Gm = np.log(np.maximum(dist, a_s * 1e-3)) / (2.0 * np.pi)
        Ginv_s = np.linalg.inv(Gm)
        s['coords'] = coords_s
        s['Ginv'] = Ginv_s
        s['a_cutoff'] = a_s
        s['med_spacing'] = med_s
        s['cond'] = float(np.linalg.cond(Gm))
        if verbose:
            print(f"  Sheet {si}: {len(s['nodes'])} boundary nodes, "
                  f"med spacing {med_s:.1f} px, a={a_s:.1f} px, "
                  f"cond(G)={s['cond']:.2e}")
    N_sheets = len(sheets)

    # ── 2. Build modified graph (remove grey interior, keep boundary) ──
    G_mod = G.copy()
    grey_interior = set(excluded_nodes) - set(all_gz_nodes)
    G_mod.remove_nodes_from(grey_interior)

    all_mod_nodes = list(G_mod.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_mod_nodes)}
    N_mod = len(all_mod_nodes)
    # Per-sheet indices into modified graph
    for s in sheets:
        s['idx'] = [node_to_idx[n] for n in s['nodes']]

    # Precompute base edge list
    base_edge_list = [(u, v) for u, v in G_mod.edges()
                      if _get_edge_geometry(G_mod, u, v, radii_m=radii_m)[0] is not None]
    if verbose:
        total_gz = sum(len(s['nodes']) for s in sheets)
        print(f"  Base edges: {len(base_edge_list)}, "
              f"{N_sheets} sheet(s), {total_gz} total boundary nodes")

    # ── 3. DC conservation: rescale venous ───────────────────────────────
    Q_art_work = {k: v.copy() for k, v in Q_art.items()}
    Q_ven_work = {k: v.copy() for k, v in Q_ven.items()}
    Q_art_dc = sum(Q_art_work.get(j, np.zeros(1))[0].real for j in art_nodes)
    Q_ven_dc = sum(Q_ven_work.get(j, np.zeros(1))[0].real for j in ven_nodes)
    if abs(Q_ven_dc) > 1e-12 and abs(Q_art_dc) > 1e-12:
        corr = -Q_art_dc / Q_ven_dc
        for j in ven_nodes:
            if j in Q_ven_work:
                Q_ven_work[j] = Q_ven_work[j] * corr
        if verbose and abs(corr - 1.0) > 0.01:
            print(f"  DC conservation: venous rescaled by {corr:.3f}")

    # ── 4. Measured harmonics (tile-filtered) ────────────────────────────
    meas_edges = {}
    for u, v, d in G_mod.edges(data=True):
        piv_list = d.get('measurements_piv', [])
        if not piv_list:
            continue
        if tile_id is not None:
            tile_piv = [m for m in piv_list if m.get('tile_id') == tile_id]
            if not tile_piv:
                continue
            best = tile_piv[0]
        else:
            best = max(piv_list, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        f0_m = best.get('f0_hz', f0_hz)
        if Qt is None or len(Qt) < 20:
            continue
        from .harmonic import fit_harmonics
        Qt_arr = np.asarray(Qt, dtype=float)
        if np.nanmean(Qt_arr) < 0:
            Qt_arr = -Qt_arr
        hr = fit_harmonics(Qt_arr, f0_m, FRAME_DT_S,
                           K=n_harmonics, loss='huber', include_dc=True)
        q_dc = hr.get('a0', float(np.nanmean(Qt_arr)))
        if not np.isfinite(q_dc) or abs(q_dc) < q_min:
            continue
        harmonics = np.zeros(n_harmonics + 1, dtype=complex)
        harmonics[0] = q_dc
        for h in hr.get('harmonics', []):
            kk = h['k']
            if kk <= n_harmonics:
                harmonics[kk] = h['A'] - 1j * h['B']
        meas_edges[(u, v)] = {'harmonics': harmonics, 'f0_meas': f0_m}
    N_meas = len(meas_edges)
    if verbose:
        print(f"  {N_meas} measured edges"
              + (f" (tile {tile_id})" if tile_id else ""))
    if N_meas == 0:
        raise ValueError("No measured edges for sheet loss")

    # ── 5. Helpers: forward solve at given Y_grey ────────────────────────
    frequencies = [0.0] + [omega0 * k for k in range(1, n_harmonics + 1)]
    radii_dict = radii_m if radii_m is not None else {}

    def _solve_with_sheets(Y_grey_list, Q_art_s, Q_ven_s, D, harmonic_idx):
        """Single-harmonic forward solve on modified graph + per-sheet admittances.

        Y_grey_list : list of (N_si × N_si) matrices, one per sheet, same order
                      as `sheets`.
        """
        omega = frequencies[harmonic_idx]
        L_base = _assemble_laplacian(
            G_mod, omega, base_edge_list, node_to_idx,
            mu, RHO_BLOOD, D, radii_m=radii_dict,
        )
        L = L_base.tolil()
        Q_rhs = np.zeros(N_mod, dtype=complex)

        # Inject each sheet's admittance block into the global Laplacian
        for s, Y_grey in zip(sheets, Y_grey_list):
            idx_s = s['idx']
            N_s = len(idx_s)
            for a in range(N_s):
                ia = idx_s[a]
                row_sum = 0.0
                for b in range(N_s):
                    if a == b:
                        continue
                    ib = idx_s[b]
                    L[ia, ib] += -Y_grey[a, b]
                    row_sum += Y_grey[a, b]
                L[ia, ia] += row_sum

        # Arterial Q injection, venous P=0 (sink Dirichlet)
        for j, Qh in Q_art_s.items():
            if j in node_to_idx:
                Q_rhs[node_to_idx[j]] += Qh[harmonic_idx]
        for j in Q_ven_s.keys():
            if j in node_to_idx:
                idx = node_to_idx[j]
                L[idx, :] = 0
                L[idx, idx] = 1.0
                Q_rhs[idx] = 0.0

        # Gauge: pin a non-BC, non-gz interior node if no venous Dirichlet
        if v_ref in node_to_idx and not Q_ven_s:
            idx = node_to_idx[v_ref]
            L[idx, :] = 0
            L[idx, idx] = 1.0
            Q_rhs[idx] = 0.0

        L_csr = L.tocsr()
        # Tikhonov
        diag_abs = np.abs(L_csr.diagonal())
        lam = 1e-10 * max(diag_abs.max(), 1e-30)
        L_csr = L_csr + _sp_eye(L_csr.shape[0], format='csr') * lam
        try:
            P = _spsolve(L_csr, Q_rhs)
        except Exception:
            return None
        return P

    def _edge_Q_from_P(P, harmonic_idx):
        omega = frequencies[harmonic_idx]
        Q_edge = {}
        for u, v in base_edge_list:
            R_m, L_m = _get_edge_geometry(G_mod, u, v, radii_m=radii_dict)
            if R_m is None:
                continue
            Y_diag, Y_off = _vessel_admittance(
                R_m, L_m, omega, mu, RHO_BLOOD, D_init,
            )
            iu, iv = node_to_idx[u], node_to_idx[v]
            # Q entering at u: Y_diag·P_u + Y_off·P_v
            Q_edge[(u, v)] = Y_diag * P[iu] + Y_off * P[iv]
        return Q_edge

    # ── 6. Sweep κh (and optional α) ────────────────────────────────────
    kh_values = np.logspace(kh_log_range[0], kh_log_range[1], n_kh)
    alpha_values = list(alpha_sweep) if alpha_sweep else [1.0]

    # Weights: just signal power × freq proximity
    edge_keys = list(meas_edges.keys())
    f0_sigma = 0.05 * f0_hz
    weights = np.zeros(N_meas)
    for ei, ek in enumerate(edge_keys):
        h = meas_edges[ek]['harmonics']
        df = meas_edges[ek]['f0_meas'] - f0_hz
        w_f = np.exp(-0.5 * (df / f0_sigma) ** 2) if f0_sigma > 0 else 1.0
        weights[ei] = (abs(h[0]) ** 2 + abs(h[1]) ** 2) * w_f
    weights /= max(weights.sum(), 1e-30)

    best = {'kh_list': None, 'alpha': 1.0, 'loss': np.inf,
            'Y_grey_list': None}
    import itertools as _it
    combos = list(_it.product(range(n_kh), repeat=N_sheets))
    loss_grid = np.full((len(alpha_values), len(combos)), np.nan)
    if verbose:
        print(f"  Sweep: {N_sheets} sheet(s) × {n_kh} κh values × "
              f"{len(alpha_values)} α = {len(alpha_values) * len(combos)} evals")

    for ai, alpha in enumerate(alpha_values):
        Q_art_scaled = {j: alpha * v for j, v in Q_art_work.items()}
        for ci, combo in enumerate(combos):
            Y_list = []
            for si, ki in enumerate(combo):
                kh = kh_values[ki]
                Y_s = (kh / mu) * sheets[si]['Ginv']
                Y_s = 0.5 * (Y_s + Y_s.T)
                Y_list.append(Y_s)
            P_dc = _solve_with_sheets(Y_list, Q_art_scaled, Q_ven_work,
                                       D_init, harmonic_idx=0)
            if P_dc is None:
                continue
            Q_sim_dc = _edge_Q_from_P(P_dc, 0)
            loss = 0.0
            for ei, ek in enumerate(edge_keys):
                q_sim = Q_sim_dc.get(ek, 0.0)
                if not np.isfinite(q_sim):
                    continue
                q_meas = meas_edges[ek]['harmonics'][0]
                loss += weights[ei] * abs(abs(q_sim) - abs(q_meas)) ** 2
            loss_grid[ai, ci] = loss
            if loss < best['loss']:
                best['loss'] = loss
                best['kh_list'] = [float(kh_values[k]) for k in combo]
                best['alpha'] = float(alpha)
                best['Y_grey_list'] = Y_list

    if verbose:
        kh_str = ", ".join(f"{k:.2e}" for k in (best['kh_list'] or []))
        print(f"  Best: κh = [{kh_str}], α = {best['alpha']:.2f}, "
              f"loss = {best['loss']:.4f}")

    # ── 7. Final full-harmonic solve at optimum ─────────────────────────
    Q_art_final = {j: best['alpha'] * v for j, v in Q_art_work.items()}
    Y_grey_list_star = best['Y_grey_list']
    node_P = {n: np.zeros(n_harmonics + 1, dtype=complex)
              for n in all_mod_nodes}
    for hi in range(n_harmonics + 1):
        P_h = _solve_with_sheets(Y_grey_list_star, Q_art_final, Q_ven_work,
                                  D_init, harmonic_idx=hi)
        if P_h is None:
            if verbose:
                print(f"  Final solve failed at harmonic {hi}")
            continue
        for n in all_mod_nodes:
            node_P[n][hi] = P_h[node_to_idx[n]]

    # Build sim result (subset — compatible with viewer's expectations)
    mean_Q = {}
    amp_Q = {}
    PI = {}
    RPSI = {}
    phase = {}
    WSS = {}
    dissipation = {}
    pulsatile_cost = {}
    storage_fraction = {}

    for u, v in base_edge_list:
        R_m, L_m = _get_edge_geometry(G_mod, u, v, radii_m=radii_dict)
        if R_m is None:
            continue
        Q_harms = np.zeros(n_harmonics + 1, dtype=complex)
        for hi in range(n_harmonics + 1):
            Y_diag, Y_off = _vessel_admittance(
                R_m, L_m, frequencies[hi], mu, RHO_BLOOD, D_init,
            )
            Q_harms[hi] = Y_diag * node_P[u][hi] + Y_off * node_P[v][hi]
        q0 = Q_harms[0].real
        q1 = abs(Q_harms[1])
        mean_Q[(u, v)] = q0
        amp_Q[(u, v)] = q1
        PI[(u, v)] = (2 * q1 / abs(q0)) if abs(q0) > 1e-15 else np.nan
        RPSI[(u, v)] = np.nan
        phase[(u, v)] = float(np.angle(Q_harms[1])) if q1 > 0 else np.nan
        WSS[(u, v)] = np.nan
        dissipation[(u, v)] = np.nan
        pulsatile_cost[(u, v)] = np.nan
        storage_fraction[(u, v)] = np.nan

    class _SimResult:
        pass
    result = _SimResult()
    result.f0_hz = f0_hz
    result.mean_Q = mean_Q
    result.amp_Q = amp_Q
    result.PI = PI
    result.RPSI = RPSI
    result.phase = phase
    result.WSS = WSS
    result.dissipation = dissipation
    result.pulsatile_cost = pulsatile_cost
    result.storage_fraction = storage_fraction
    result.node_pressures = node_P
    result.edge_flows = {k: np.array([mean_Q[k]] + [0j] * n_harmonics)
                         for k in mean_Q}

    # Residuals per edge (DC magnitude ratio)
    residuals = {}
    for ek in edge_keys:
        q_sim = mean_Q.get(ek, np.nan)
        q_meas = meas_edges[ek]['harmonics'][0].real
        if np.isfinite(q_sim) and abs(q_meas) > 1e-15:
            residuals[ek] = abs(abs(q_sim) - abs(q_meas)) / abs(q_meas)
        else:
            residuals[ek] = np.nan

    return {
        'D_opt': D_init,
        's': best['alpha'],
        'kh_opt': best['kh_list'],   # list, one per sheet
        'n_sheets': N_sheets,
        'sheet_nodes': [s['nodes'] for s in sheets],
        'sheet_blues': [s['blue'] for s in sheets],
        'loss': best['loss'],
        'loss_grid': loss_grid,
        'kh_values': kh_values,
        'alpha_values': alpha_values,
        'result': result,
        'residuals': residuals,
        'tile_id': tile_id,
        'Y_grey_list': Y_grey_list_star,
    }


def optimize_greyzone(
    G: nx.Graph,
    excluded_nodes: set,
    blue_red_map: Dict[int, List[int]],
    art_nodes: List[int],
    ven_nodes: List[int],
    v_ref: int,
    Q_art: Dict[int, np.ndarray],
    Q_ven: Dict[int, np.ndarray],
    n_harmonics: int = 3,
    f0_hz: float = 2.5,
    mu: float = MU_DEFAULT,
    D_init: float = 1e-3,
    radii_m: Optional[Dict[Tuple[int, int], float]] = None,
    q_min: float = 0.1,
    lambda_mag: float = 0.05,
    max_iter: int = 300,
    tile_id: Optional[int] = None,
    min_radius_px: float = 2.0,
    max_radius_px: float = 30.0,
    verbose: bool = True,
    # Legacy
    blue_node: Optional[int] = None,
    red_nodes: Optional[List[int]] = None,
) -> dict:
    """Grey-zone optimizer: replacement-edge approach.

    Replaces the excluded grey zone with direct edges from each blue
    entrance node to its connected red exit nodes. Each replacement
    edge has an unknown effective radius a_k.

    Parameters
    ----------
    blue_red_map : {blue_node: [red_node_1, red_node_2, ...]}
        Maps each blue entrance node to the red exit nodes reachable
        through it within the grey zone.
    """
    from .config import PX_SIZE_UM, FRAME_DT_S
    from scipy.optimize import minimize as sp_min

    # Legacy single-blue support
    if blue_red_map is None and blue_node is not None and red_nodes is not None:
        blue_red_map = {blue_node: red_nodes}

    # Flatten to ordered lists
    all_blue = sorted(blue_red_map.keys())
    all_red = []
    red_to_blue = {}  # red_node -> blue_node
    edge_pairs = []   # (blue, red) pairs in order
    for bn in all_blue:
        for rn in blue_red_map[bn]:
            if rn not in all_red:
                all_red.append(rn)
                red_to_blue[rn] = bn
                edge_pairs.append((bn, rn))
    K = len(edge_pairs)
    omega0 = 2.0 * np.pi * f0_hz

    # --- Build modified graph: remove grey zone, add replacement edges ---
    G_mod = G.copy()
    G_mod.remove_nodes_from(excluded_nodes)

    # Replacement-edge length = shortest path through the grey zone between
    # blue and red (summing edge 'length' attributes in px). Falls back to
    # Euclidean if no path exists or graph has no length data.
    px_to_m = PX_SIZE_UM * 1e-6

    # Build a subgraph containing only grey-zone nodes + all blue/red endpoints,
    # with edge weights = physical length in pixels.
    def _edge_length_px(u, v):
        d = G.edges[u, v]
        for key in ('length_true', 'length', 'path_length_px'):
            val = d.get(key)
            if val is not None and np.isfinite(val) and val > 0:
                return float(val)
        # Fallback: Euclidean between node positions
        ux = G.nodes[u].get('x', 0); uy = G.nodes[u].get('y', 0)
        vx = G.nodes[v].get('x', 0); vy = G.nodes[v].get('y', 0)
        return max(float(np.hypot(vx - ux, vy - uy)), 1.0)

    _grey_subnodes = set(excluded_nodes) | set(all_blue) | set(all_red)
    G_grey_sub = nx.Graph()
    G_grey_sub.add_nodes_from(_grey_subnodes)
    for u, v in G.edges():
        if u in _grey_subnodes and v in _grey_subnodes:
            # Include edges that touch the grey zone (skip pure-external edges)
            if (u in excluded_nodes) or (v in excluded_nodes):
                G_grey_sub.add_edge(u, v, length=_edge_length_px(u, v))

    lengths_px = []
    for bn, rn in edge_pairs:
        try:
            L_px = nx.shortest_path_length(G_grey_sub, source=bn, target=rn,
                                           weight='length')
        except Exception:
            # Fallback: Euclidean
            bx = G.nodes[bn].get('x', 0); by = G.nodes[bn].get('y', 0)
            rx = G.nodes[rn].get('x', 0); ry = G.nodes[rn].get('y', 0)
            L_px = max(float(np.hypot(rx - bx, ry - by)), 10.0)
        lengths_px.append(max(L_px, 10.0))

    lengths_m = [L * px_to_m for L in lengths_px]

    if verbose:
        from collections import Counter as _C
        _euc = [max(float(np.hypot(
            G.nodes[rn].get('x', 0) - G.nodes[bn].get('x', 0),
            G.nodes[rn].get('y', 0) - G.nodes[bn].get('y', 0))), 10.0)
                for bn, rn in edge_pairs]
        _ratio = [Lp / Le if Le > 0 else 1.0 for Lp, Le in zip(lengths_px, _euc)]
        print(f"  Replacement lengths: graph shortest-path through grey zone")
        print(f"    median = {np.median(lengths_px):.0f} px, "
              f"median path/euclidean ratio = {np.median(_ratio):.2f}")

    # Murray constraint: for each blue node, parent R³ = sum of daughter R³
    # Parent = the non-excluded, non-replacement edge with largest radius at blue node
    blue_parent_r3 = {}  # blue_node -> parent radius³ in pixels³
    blue_edge_indices = {}  # blue_node -> list of edge_pair indices
    for bn in all_blue:
        # Find parent vessel radius (largest non-excluded neighbor edge)
        max_r = 0.0
        for nb in G.neighbors(bn):
            if nb in excluded_nodes:
                continue
            if nb in set(all_red):
                continue  # skip replacement edge targets
            r = G.edges[bn, nb].get('radius', 0)
            if hasattr(r, 'item'):
                r = r.item()
            if r > max_r:
                max_r = r
        blue_parent_r3[bn] = max_r**3 if max_r > 0 else 125.0  # default 5³
        # Indices of edge_pairs belonging to this blue node
        blue_edge_indices[bn] = [ki for ki, (b, r) in enumerate(edge_pairs) if b == bn]

    if verbose:
        for bn in all_blue:
            r_parent = blue_parent_r3[bn]**(1/3)
            n_daughters = len(blue_edge_indices[bn])
            print(f"  Murray: blue {bn}, parent R={r_parent:.1f} px, "
                  f"{n_daughters} daughters")

    lambda_murray = 0.01  # Murray penalty weight (soft nudge)

    # Gauge node must be in G_mod (not excluded)
    if v_ref in excluded_nodes or v_ref not in G_mod:
        # Pick an interior node
        bc_set = set(art_nodes) | set(ven_nodes) | set(all_red) | set(all_blue)
        v_ref = next((n for n in G_mod.nodes() if n not in bc_set),
                     next(iter(G_mod.nodes())))

    if verbose:
        _eq = '=' * 60
        print(f"\n{_eq}")
        print("GREY-ZONE OPTIMIZER (replacement-edge approach)")
        print(f"{_eq}")
        for bn in all_blue:
            print(f"  Blue {bn} → {blue_red_map[bn]}")
        print(f"  Total replacement edges: K={K}")
        print(f"  Art: {art_nodes}, Ven: {ven_nodes}, gauge: {v_ref}")
        print(f"  D_init={D_init:.2e}, lambda_mag={lambda_mag}")

    # --- DC conservation: rescale venous to balance arterial ---
    Q_art_dc = sum(Q_art.get(j, np.zeros(1))[0].real for j in art_nodes)
    Q_ven_dc = sum(Q_ven.get(j, np.zeros(1))[0].real for j in ven_nodes)
    if abs(Q_ven_dc) > 1e-12 and abs(Q_art_dc) > 1e-12:
        corr = -Q_art_dc / Q_ven_dc
        for j in ven_nodes:
            if j in Q_ven:
                Q_ven[j] = Q_ven[j] * corr
        if verbose and abs(corr - 1.0) > 0.01:
            print(f"  DC conservation: venous rescaled by {corr:.3f}")

    # --- Extract measured harmonics (optionally tile-filtered) ---
    meas_edges = {}
    for u, v, d in G_mod.edges(data=True):
        piv_list = d.get('measurements_piv', [])
        if not piv_list:
            continue
        if tile_id is not None:
            tile_piv = [m for m in piv_list if m.get('tile_id') == tile_id]
            if not tile_piv:
                continue
            best = tile_piv[0]
        else:
            best = max(piv_list, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        f0_m = best.get('f0_hz', f0_hz)
        if Qt is None or len(Qt) < 20:
            continue
        from .harmonic import fit_harmonics
        Qt_arr = np.asarray(Qt, dtype=float)
        if np.nanmean(Qt_arr) < 0:
            Qt_arr = -Qt_arr
        hr = fit_harmonics(Qt_arr, f0_m, FRAME_DT_S,
                           K=n_harmonics, loss='huber', include_dc=True)
        q_dc = hr.get('a0', float(np.nanmean(Qt_arr)))
        if not np.isfinite(q_dc) or abs(q_dc) < q_min:
            continue
        harmonics = np.zeros(n_harmonics + 1, dtype=complex)
        harmonics[0] = q_dc
        for h in hr.get('harmonics', []):
            kk = h['k']
            if kk <= n_harmonics:
                harmonics[kk] = h['A'] - 1j * h['B']
        meas_edges[(u, v)] = {'harmonics': harmonics, 'f0_meas': f0_m}

    N_meas = len(meas_edges)
    edge_keys = list(meas_edges.keys())
    if verbose:
        print(f"  {N_meas} measured edges")

    # Weights: power × frequency proximity (no spatial proximity —
    # grey zone is a chokepoint so all downstream edges are sensitive)
    f0_sigma = 0.05 * f0_hz
    weights = np.zeros(N_meas)
    for ei, ek in enumerate(edge_keys):
        m = meas_edges[ek]
        h = m['harmonics']
        w_p = abs(h[0])**2 + abs(h[1])**2
        df = m['f0_meas'] - f0_hz
        w_f = np.exp(-0.5 * (df / f0_sigma)**2) if f0_sigma > 0 else 1.0
        # No proximity weighting — grey zone is a chokepoint, so ALL
        # downstream edges are equally sensitive to replacement-edge radii.
        weights[ei] = w_p * w_f
    weights /= max(weights.sum(), 1e-30)

    Q_meas_dc = np.array([abs(meas_edges[ek]['harmonics'][0].real)
                           for ek in edge_keys])

    # --- Build BC override (all boundary nodes) ---
    bc_all = {}
    bc_all.update(Q_art)
    bc_all.update(Q_ven)
    all_bc_nodes = list(art_nodes) + list(ven_nodes)

    # Network infrastructure for G_mod
    frequencies = [0.0] + [omega0 * k for k in range(1, n_harmonics + 1)]

    # --- Analytic adjoint optimizer (no s scaling) ---
    import warnings as _warn

    eval_count = [0]
    history = []

    # Precompute base edge list (excluding replacement edges)
    # Check both directions since G_mod.edges() may return either order
    ep_nodes = set()
    for bn, rn in edge_pairs:
        ep_nodes.add((bn, rn))
        ep_nodes.add((rn, bn))
    base_edge_list = [(u, v) for u, v in G_mod.edges()
                      if _get_edge_geometry(G_mod, u, v, radii_m=radii_m)[0] is not None
                      and (u, v) not in ep_nodes]
    if verbose:
        n_repl_in_base = sum(1 for u, v in G_mod.edges() if (u, v) in ep_nodes)
        print(f"  Base edges: {len(base_edge_list)}, "
              f"replacement edges filtered: {n_repl_in_base}")

    all_mod_nodes = list(G_mod.nodes())
    mod_node_to_idx = {n: i for i, n in enumerate(all_mod_nodes)}
    N_mod = len(all_mod_nodes)

    # Ensure gauge is valid
    if v_ref not in mod_node_to_idx:
        bc_set = set(art_nodes) | set(ven_nodes) | set(all_red) | set(all_blue)
        v_ref = next((n for n in all_mod_nodes if n not in bc_set),
                     all_mod_nodes[0])
    gauge_idx = mod_node_to_idx[v_ref]

    all_bc_nodes_list = list(art_nodes) + list(ven_nodes)
    frequencies = [0.0] + [omega0 * k for k in range(1, n_harmonics + 1)]

    def _vessel_admittance_and_derivs(R_m, L_m, omega, mu_v, rho_v, D_v):
        """Compute Y_diag, Y_off AND their derivatives w.r.t. R_m (radius).

        Returns (Y_diag, Y_off, dYdiag_dR, dYoff_dR).
        """
        r, ell, c = _per_length_params(R_m, mu_v, rho_v, D_v)
        # dr/dR = -32μ/(πR^5) = -4r/R.
        # dc/dR = 2πD·R   (since c = πR²D in the areal convention;
        # was 4πD·R under the prior radius convention).
        # Use finite differences below regardless — keeps the path
        # robust to any future convention changes.
        eps_R = R_m * 1e-6
        Yd_p, Yo_p = _vessel_admittance(R_m + eps_R, L_m, omega, mu_v, rho_v, D_v)
        Yd_m, Yo_m = _vessel_admittance(R_m - eps_R, L_m, omega, mu_v, rho_v, D_v)
        Yd, Yo = _vessel_admittance(R_m, L_m, omega, mu_v, rho_v, D_v)
        dYd_dR = (Yd_p - Yd_m) / (2 * eps_R)
        dYo_dR = (Yo_p - Yo_m) / (2 * eps_R)
        return Yd, Yo, dYd_dR, dYo_dR

    def objective_and_grad(theta):
        """Loss + gradient via adjoint method. theta = [a_1, ..., a_K] in pixels."""
        a_px = np.maximum(theta, 1.0)
        D = D_init  # fixed

        grad = np.zeros(K)
        L_total = 0.0

        for harm_idx, omega in enumerate(frequencies):
            # Build base admittance (non-replacement edges)
            Y_base = _assemble_laplacian(G_mod, omega, base_edge_list,
                                          mod_node_to_idx, mu, RHO_BLOOD, D,
                                          radii_m=radii_m)
            Y_lil = Y_base.tolil()

            # Add replacement edges and store derivatives
            repl_info = []
            for ki, (bn, rn) in enumerate(edge_pairs):
                a_m = a_px[ki] * px_to_m
                L_m = lengths_m[ki]
                Yd, Yo, dYd_dR, dYo_dR = _vessel_admittance_and_derivs(
                    a_m, L_m, omega, mu, RHO_BLOOD, D)
                ui = mod_node_to_idx[bn]
                vi = mod_node_to_idx[rn]
                Y_lil[ui, ui] += Yd
                Y_lil[vi, vi] += Yd
                Y_lil[ui, vi] += Yo
                Y_lil[vi, ui] += Yo
                repl_info.append((ui, vi, Yd, Yo, dYd_dR, dYo_dR))

            # Gauge
            Y_lil[gauge_idx, :] = 0
            Y_lil[:, gauge_idx] = 0
            Y_lil[gauge_idx, gauge_idx] = 1.0
            Y_csr = Y_lil.tocsr()

            # RHS
            Q_ext = np.zeros(N_mod, dtype=complex)
            for n in all_bc_nodes_list:
                if n in mod_node_to_idx:
                    bc = Q_art.get(n, Q_ven.get(n, np.zeros(n_harmonics+1)))
                    Q_ext[mod_node_to_idx[n]] = bc[harm_idx] * 1e-12
            Q_ext[gauge_idx] = 0.0

            # Forward solve
            with _warn.catch_warnings():
                _warn.simplefilter("ignore")
                P = spsolve(Y_csr, Q_ext)
            if not np.all(np.isfinite(P)):
                return 1e10, np.zeros(K)
            P[gauge_idx] = 0.0

            # Reconstruct edge flows + compute dL/dP
            dL_dP = np.zeros(N_mod, dtype=complex)

            for ei, ek in enumerate(edge_keys):
                eu, ev = ek
                if eu not in mod_node_to_idx or ev not in mod_node_to_idx:
                    continue
                Rm, Lm = _get_edge_geometry(G_mod, eu, ev, radii_m=radii_m)
                if Rm is None:
                    continue
                Yd_e, Yo_e = _vessel_admittance(Rm, Lm, omega, mu, RHO_BLOOD, D)
                ui_e = mod_node_to_idx[eu]
                vi_e = mod_node_to_idx[ev]
                Q_uv = (Yd_e * P[ui_e] + Yo_e * P[vi_e]) * 1e12

                Qm = meas_edges[ek]['harmonics']

                # Unnormalized absolute harmonic loss (DC + AC):
                #   L += w_e · |Q_pred[n] - Q_meas[n]|²
                # Matches refine-mode loss. Sign-flip pred if DC sign differs.
                if harm_idx == 0:
                    # Global sign from DC (set once, reused for AC via _sign)
                    if (Qm[0].real != 0 and Q_uv.real != 0 and
                            np.sign(Qm[0].real) != np.sign(Q_uv.real)):
                        _sign = -1.0
                    else:
                        _sign = 1.0
                    meas_edges[ek]['_sign'] = _sign
                else:
                    _sign = meas_edges[ek].get('_sign', 1.0)

                residual = _sign * Q_uv - Qm[harm_idx]
                L_total += weights[ei] * abs(residual)**2

                # dL/dQ_uv = 2 · w · conj(_sign · residual) = 2·w·_sign·conj(residual)
                dL_dQ = 2.0 * weights[ei] * _sign * np.conj(residual)

                # Q_uv = (Yd · P_u + Yo · P_v) · 1e12 → dQ/dP factor of 1e12
                dL_dP[ui_e] += dL_dQ * Yd_e * 1e12
                dL_dP[vi_e] += dL_dQ * Yo_e * 1e12

            dL_dP[gauge_idx] = 0.0

            # Adjoint solve: Y^T lambda = dL/dP (Y is symmetric → same matrix)
            with _warn.catch_warnings():
                _warn.simplefilter("ignore")
                lam = spsolve(Y_csr, dL_dP)
            if not np.all(np.isfinite(lam)):
                lam = np.zeros(N_mod, dtype=complex)
            lam[gauge_idx] = 0.0

            # Gradient: dL/d(a_k) = -lambda^T * dY/d(a_k) * P
            # dY/d(a_k) only has entries at (ui, ui), (vi, vi), (ui, vi), (vi, ui)
            for ki, (ui, vi, Yd, Yo, dYd_dR, dYo_dR) in enumerate(repl_info):
                # dY * P contribution:
                # dY[ui,ui] = dYd_dR, dY[vi,vi] = dYd_dR
                # dY[ui,vi] = dYo_dR, dY[vi,ui] = dYo_dR
                dYP_ui = dYd_dR * P[ui] + dYo_dR * P[vi]
                dYP_vi = dYo_dR * P[ui] + dYd_dR * P[vi]

                # grad[ki] -= Re(lambda^T * dY/dR * P) * dR/d(a_px)
                # dR/d(a_px) = px_to_m
                contrib = -(np.conj(lam[ui]) * dYP_ui +
                            np.conj(lam[vi]) * dYP_vi).real * px_to_m
                grad[ki] += contrib

        eval_count[0] += 1
        if verbose and eval_count[0] % 5 == 0:
            a_str = ", ".join(f"{a:.1f}" for a in a_px[:5])
            print(f"    eval {eval_count[0]}: L={L_total:.6f} D={D_init:.2e} "
                  f"a=[{a_str}...] |grad|={np.linalg.norm(grad):.2e}")
        history.append((eval_count[0], L_total))

        return L_total, grad

    # --- Optimize: theta = [log10(D), a_1, ..., a_K] ---
    a_init_val = 5.0
    theta0 = np.concatenate([[np.log10(D_init)], np.full(K, a_init_val)])
    bounds = [(-5.0, -1.0)] + [(1.0, 30.0)] * K

    # Wrapper that unpacks D from theta
    def _loss_with_D(theta):
        D_val = 10**theta[0]
        a_vals = np.maximum(theta[1:], 1.0)

        # Temporarily override D_init for the objective
        all_Q_pred = [{} for _ in range(n_harmonics + 1)]
        for harm_idx, omega in enumerate(frequencies):
            Y_b = _assemble_laplacian(G_mod, omega, base_edge_list,
                                       mod_node_to_idx, mu, RHO_BLOOD, D_val,
                                       radii_m=radii_m)
            Y_lil = Y_b.tolil()
            for ki, (bn, rn) in enumerate(edge_pairs):
                a_m = a_vals[ki] * px_to_m
                L_m = lengths_m[ki]
                Yd, Yo = _vessel_admittance(a_m, L_m, omega, mu, RHO_BLOOD, D_val)
                ui = mod_node_to_idx[bn]
                vi = mod_node_to_idx[rn]
                Y_lil[ui, ui] += Yd; Y_lil[vi, vi] += Yd
                Y_lil[ui, vi] += Yo; Y_lil[vi, ui] += Yo
            Y_lil[gauge_idx, :] = 0; Y_lil[:, gauge_idx] = 0
            Y_lil[gauge_idx, gauge_idx] = 1.0
            Y_csr = Y_lil.tocsr()

            Q_ext = np.zeros(N_mod, dtype=complex)
            for n in all_bc_nodes_list:
                if n in mod_node_to_idx:
                    bc = Q_art.get(n, Q_ven.get(n, np.zeros(n_harmonics+1)))
                    Q_ext[mod_node_to_idx[n]] = bc[harm_idx] * 1e-12
            Q_ext[gauge_idx] = 0.0

            with _warn.catch_warnings():
                _warn.simplefilter("ignore")
                P = spsolve(Y_csr, Q_ext)
            if not np.all(np.isfinite(P)):
                return 1e10
            P[gauge_idx] = 0.0

            for ei, ek in enumerate(edge_keys):
                eu, ev = ek
                if eu not in mod_node_to_idx or ev not in mod_node_to_idx:
                    continue
                Rm, Lm = _get_edge_geometry(G_mod, eu, ev, radii_m=radii_m)
                if Rm is None:
                    continue
                Yd, Yo = _vessel_admittance(Rm, Lm, omega, mu, RHO_BLOOD, D_val)
                all_Q_pred[harm_idx][ek] = (Yd * P[mod_node_to_idx[eu]] +
                                             Yo * P[mod_node_to_idx[ev]]) * 1e12

        # L2 complex ratio loss: |Q̂_n/Q̂_1 (pred) - Q̂_n/Q̂_1 (meas)|²
        # Pure waveform shape, sign-invariant, no DC dependence
        L_shape = 0.0
        for ei, ek in enumerate(edge_keys):
            Qm = meas_edges[ek]['harmonics']
            q1_p = all_Q_pred[1].get(ek, 0)
            q1_m = Qm[1]
            if abs(q1_p) < 1e-15 or abs(q1_m) < 1e-15:
                continue
            for n in range(2, n_harmonics + 1):
                qp = all_Q_pred[n].get(ek, 0)
                L_shape += weights[ei] * abs(qp / q1_p - Qm[n] / q1_m)**2

        # Murray penalty: (sum(a_k³) - R_parent³)² / R_parent⁶ for each blue node
        L_murray = 0.0
        for bn in all_blue:
            r3_parent = blue_parent_r3[bn]
            idxs = blue_edge_indices[bn]
            r3_sum = sum(a_vals[ki]**3 for ki in idxs)
            L_murray += ((r3_sum - r3_parent) / max(r3_parent, 1.0))**2

        L_total = L_shape + lambda_murray * L_murray

        eval_count[0] += 1
        if verbose and eval_count[0] % 5 == 0:
            a_str = ", ".join(f"{a:.1f}" for a in a_vals[:5])
            print(f"    eval {eval_count[0]}: L={L_total:.6f} "
                  f"(shape={L_shape:.4f}, murray={L_murray:.4f}) "
                  f"D={D_val:.2e} a=[{a_str}...]")
        history.append((eval_count[0], L_total))
        return L_total

    # Fixed-D L-BFGS-B with analytic adjoint gradients (Murray-free).
    # D is held at D_init; only K replacement radii are optimized.
    # Initial guess: parent-vessel radius at each blue, spread equally across
    # its K_b daughter replacement edges via Murray scaling R_each = R_par / K_b^(1/3)
    a0 = np.zeros(K)
    for ki, (bn, rn) in enumerate(edge_pairs):
        r_par3 = blue_parent_r3.get(bn, 125.0)
        n_daughters = max(1, len(blue_edge_indices.get(bn, [ki])))
        a0[ki] = max(min_radius_px,
                     min(max_radius_px,
                         (r_par3 / n_daughters) ** (1.0 / 3.0)))
    a_bounds = [(min_radius_px, max_radius_px)] * K

    if verbose:
        print(f"  Starting L-BFGS-B with adjoint gradients "
              f"(K={K} radii, D fixed at {D_init:.2e})")
        print(f"  Radius bounds: [{min_radius_px:.1f}, {max_radius_px:.1f}] px")
        L0, g0 = objective_and_grad(a0)
        print(f"  Initial loss: {L0:.6f}  |grad|={np.linalg.norm(g0):.3e}")

    from scipy.optimize import minimize as sp_min_lbfgs

    def _lbfgs_callback(xk):
        eval_count[0] += 1
        if verbose:
            L_v, _ = objective_and_grad(xk)
            a_str = ", ".join(f"{a:.1f}" for a in xk[:5])
            print(f"    iter {eval_count[0]}: L={L_v:.6f}  "
                  f"a=[{a_str}...]  range=[{xk.min():.1f}, {xk.max():.1f}]")

    opt = sp_min_lbfgs(objective_and_grad, a0,
                       method='L-BFGS-B', jac=True,
                       bounds=a_bounds,
                       callback=_lbfgs_callback,
                       options={'maxiter': max_iter,
                                'ftol': 1e-8, 'gtol': 1e-6,
                                'disp': False})

    D_opt = D_init  # fixed, not optimized
    a_opt = opt.x
    s_opt = 1.0  # no scaling


    # Final solve at optimum
    # Set replacement edges to optimal radii on G_mod
    for ki, (bn, rn) in enumerate(edge_pairs):
        a_m = max(a_opt[ki], 1.0) * px_to_m
        if G_mod.has_edge(bn, rn):
            G_mod.edges[bn, rn]['radius'] = a_opt[ki]
            G_mod.edges[bn, rn]['radius_px'] = a_opt[ki]

    rm_final = dict(radii_m) if radii_m else {}
    for ki, (bn, rn) in enumerate(edge_pairs):
        a_m = max(a_opt[ki], 1.0) * px_to_m
        rm_final[(bn, rn)] = a_m
        rm_final[(rn, bn)] = a_m

    bc_final = {}
    bc_final.update(Q_art)
    bc_final.update(Q_ven)

    result_final = solve_transmission_line(
        G_mod, D=D_opt, n_harmonics=n_harmonics, f0_hz=f0_hz,
        mu=mu, bc_harmonics_override=bc_final,
        boundary_nodes=all_bc_nodes_list,
        radii_m=rm_final,
        verbose=False,
    )

    if verbose:
        _eq = '=' * 60
        print(f"\n  Converged: {opt.message}")
        print(f"  D* = {D_opt:.4e}, s* = {s_opt:.3f}")
        print(f"  Loss = {opt.fun:.4f}")
        print(f"  Replacement edge radii (px):")
        for ki, (bn, rn) in enumerate(edge_pairs):
            print(f"    {bn}→{rn}: a={a_opt[ki]:.2f} px ({a_opt[ki]*PX_SIZE_UM:.1f} µm), "
                  f"L={lengths_px[ki]:.0f} px")
        print(f"{_eq}")

    # Per-edge residuals
    residuals = {}
    predictions = {}
    for ei, ek in enumerate(edge_keys):
        Q = result_final.edge_flows.get(ek, result_final.edge_flows.get((ek[1], ek[0])))
        if Q is None:
            continue
        predictions[ek] = Q * s_opt
        Q_m = meas_edges[ek]['harmonics']
        dc_p = abs(Q[0].real)
        dc_m = abs(Q_m[0].real)
        r = 0.0
        if dc_p > 1e-12 and dc_m > 1e-12:
            for n in range(1, n_harmonics + 1):
                r += abs(Q[n]/dc_p - Q_m[n]/dc_m)**2
        residuals[ek] = float(r)

    return {
        'D_opt': D_opt,
        's': s_opt,
        'a_radii_px': a_opt,
        'edge_pairs': edge_pairs,
        'red_nodes': all_red,
        'blue_red_map': blue_red_map,
        'mu': mu,
        'loss': float(opt.fun),
        'history': history,
        'fractions': {0: a_opt},  # legacy compat
        'predictions': predictions,
        'residuals': residuals,
        'meas_edges': {k: v['harmonics'] for k, v in meas_edges.items()},
        'result': result_final,
        'G_mod': G_mod,
        'lengths_px': lengths_px,
        'n_harmonics': n_harmonics,
    }


def optimize_greyzone_greens(
    G,
    red_nodes=None, art_nodes=None, ven_nodes=None,
    v_ref=None,
    Q_art=None, Q_ven=None,
    n_harmonics=3, f0_hz=2.5, mu=MU_DEFAULT,
    D_sweep=None, radii_m=None,
    q_min=0.1, lambda_mag=0.05, lambda_tik=1e-3,
    verbose=True,
    # Legacy aliases
    blue_nodes=None, arterial_nodes=None, venous_nodes=None,
    Q_total_harmonics=None, bc_arterial=None, bc_venous=None,
):
    """Grey-zone optimizer with overall scale s, shape+magnitude loss.

    Sweeps D. At each D: precomputes Green's functions, solves DC fractions
    (constrained), determines scale s (closed-form), solves AC fractions
    (stacked Tikhonov), evaluates shape+magnitude loss.
    """
    from .config import FRAME_DT_S
    from scipy.optimize import lsq_linear
    import warnings as _warn

    # Legacy aliases
    if red_nodes is None and blue_nodes is not None: red_nodes = blue_nodes
    if art_nodes is None and arterial_nodes is not None: art_nodes = arterial_nodes
    if ven_nodes is None and venous_nodes is not None: ven_nodes = venous_nodes
    if Q_art is None and bc_arterial is not None: Q_art = bc_arterial
    if Q_ven is None and bc_venous is not None: Q_ven = bc_venous

    if red_nodes is None: red_nodes = []
    if art_nodes is None: art_nodes = []
    if ven_nodes is None: ven_nodes = []
    if Q_art is None: Q_art = {}
    if Q_ven is None: Q_ven = {}
    if D_sweep is None: D_sweep = np.logspace(-4, -1, 25)

    omega0 = 2.0 * np.pi * f0_hz
    K = len(red_nodes)

    # --- Preprocessing: enforce DC conservation ---
    Q_art_dc = sum(Q_art.get(j, np.zeros(1))[0].real for j in art_nodes)
    Q_ven_dc = sum(Q_ven.get(j, np.zeros(1))[0].real for j in ven_nodes)
    if abs(Q_ven_dc) > 1e-12 and abs(Q_art_dc) > 1e-12:
        corr = -Q_art_dc / Q_ven_dc
        for j in ven_nodes:
            if j in Q_ven:
                Q_ven[j] = Q_ven[j] * corr
        if verbose and abs(corr - 1.0) > 0.01:
            print(f"  DC conservation: venous rescaled by {corr:.3f}")

    Q_gz_dc = Q_art_dc  # total DC through grey zone

    source_set = list(red_nodes) + list(art_nodes) + list(ven_nodes)

    if verbose:
        _eq = '=' * 60
        print(f"\n{_eq}")

        print(f"GREY-ZONE OPTIMIZER (shape + magnitude loss)")
        print(f"{_eq}")
        print(f"  Red nodes (K={K}), Art={len(art_nodes)}, Ven={len(ven_nodes)}, gauge={v_ref}")
        print(f"  Q_art_dc={Q_art_dc:.4f}, lambda_mag={lambda_mag}, lambda_tik={lambda_tik}")
        print(f"  D sweep: {len(D_sweep)} values [{D_sweep[0]:.1e}, {D_sweep[-1]:.1e}]")

    # --- Extract measured harmonics ---
    meas_edges = {}
    for u, v, d in G.edges(data=True):
        piv_list = d.get('measurements_piv', [])
        if not piv_list: continue
        best = max(piv_list, key=lambda m: m.get('snr_pulse', -np.inf))
        Qt = best.get('Q_t')
        f0_m = best.get('f0_hz', f0_hz)
        if Qt is None or len(Qt) < 20: continue
        from .harmonic import fit_harmonics
        Qt_arr = np.asarray(Qt, dtype=float)
        if np.nanmean(Qt_arr) < 0: Qt_arr = -Qt_arr
        hr = fit_harmonics(Qt_arr, f0_m, FRAME_DT_S, K=n_harmonics, loss='huber', include_dc=True)
        q_dc = hr.get('a0', float(np.nanmean(Qt_arr)))
        if not np.isfinite(q_dc) or abs(q_dc) < q_min: continue
        harmonics = np.zeros(n_harmonics + 1, dtype=complex)
        harmonics[0] = q_dc
        for h in hr.get('harmonics', []):
            kk = h['k']
            if kk <= n_harmonics: harmonics[kk] = h['A'] - 1j * h['B']
        meas_edges[(u, v)] = {'harmonics': harmonics, 'f0_meas': f0_m}

    N_meas = len(meas_edges)
    edge_keys = list(meas_edges.keys())
    if verbose: print(f"  {N_meas} measured edges (|Q_dc| >= {q_min})")
    if N_meas == 0: raise ValueError("No measured edges found")

    # --- Weights ---
    f0_sigma = 0.05 * f0_hz
    weights = np.zeros(N_meas)
    for ei, ek in enumerate(edge_keys):
        m = meas_edges[ek]; h = m['harmonics']
        w_p = abs(h[0])**2 + abs(h[1])**2
        df = m['f0_meas'] - f0_hz
        w_f = np.exp(-0.5 * (df / f0_sigma)**2) if f0_sigma > 0 else 1.0
        weights[ei] = w_p * w_f
    weights /= max(weights.sum(), 1e-30)

    # --- Network ---
    all_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    N_n = len(all_nodes)
    edge_list = [(u, v) for u, v in G.edges()
                 if _get_edge_geometry(G, u, v, radii_m=radii_m)[0] is not None]
    gauge_idx = node_to_idx.get(v_ref, next(
        (node_to_idx[n] for n in all_nodes if n not in set(source_set)), 0))
    frequencies = [0.0] + [omega0 * k for k in range(1, n_harmonics + 1)]

    # --- D sweep ---
    loss_curve = []
    best_loss = np.inf
    best_result = None

    for D_idx, D in enumerate(D_sweep):
        if verbose and D_idx % 5 == 0:
            print(f"  D={D:.2e} ({D_idx+1}/{len(D_sweep)})...", end='', flush=True)

        # Step 1: Green's functions
        greens = {s: {} for s in source_set}
        ok = True
        for hi, omega in enumerate(frequencies):
            Y = _assemble_laplacian(G, omega, edge_list, node_to_idx, mu, RHO_BLOOD, D, radii_m=radii_m)
            Yl = Y.tolil(); Yl[gauge_idx, :] = 0; Yl[:, gauge_idx] = 0; Yl[gauge_idx, gauge_idx] = 1.0
            Yc = Yl.tocsr()
            for s in source_set:
                if s not in node_to_idx: continue
                q = np.zeros(N_n, dtype=complex); q[node_to_idx[s]] = 1e-12; q[gauge_idx] = 0
                with _warn.catch_warnings():
                    _warn.simplefilter("ignore")
                    P = spsolve(Yc, q)
                if not np.all(np.isfinite(P)): greens[s][hi] = {}; ok = False; continue
                P[gauge_idx] = 0
                ef = {}
                for ek in edge_keys:
                    eu, ev = ek
                    Rm, Lm = _get_edge_geometry(G, eu, ev, radii_m=radii_m)
                    if Rm is None: continue
                    Yd, Yo = _vessel_admittance(Rm, Lm, omega, mu, RHO_BLOOD, D)
                    ef[ek] = (Yd * P[node_to_idx[eu]] + Yo * P[node_to_idx[ev]]) * 1e12
                greens[s][hi] = ef

        if not ok:
            loss_curve.append((D, np.inf))
            if verbose and D_idx % 5 == 0: print(" SINGULAR")
            continue

        # Step 2: Prescribed responses (at s=1)
        G_presc = {}
        for hi in range(n_harmonics + 1):
            gp = np.zeros(N_meas, dtype=complex)
            for ei, ek in enumerate(edge_keys):
                for j in art_nodes:
                    if j in greens and hi in greens[j]:
                        gp[ei] += Q_art.get(j, np.zeros(n_harmonics+1))[hi] * greens[j][hi].get(ek, 0)
                for j in ven_nodes:
                    if j in greens and hi in greens[j]:
                        gp[ei] += Q_ven.get(j, np.zeros(n_harmonics+1))[hi] * greens[j][hi].get(ek, 0)
            G_presc[hi] = gp

        Q_meas_dc = np.array([meas_edges[ek]['harmonics'][0].real for ek in edge_keys])

        # Step 3a: DC fractions
        if K > 0:
            A_dc = np.zeros((N_meas, K))
            for ki, bk in enumerate(red_nodes):
                if bk in greens and 0 in greens[bk]:
                    for ei, ek in enumerate(edge_keys):
                        A_dc[ei, ki] = np.sqrt(weights[ei]) * greens[bk][0].get(ek, 0)
            t_dc = np.sqrt(weights) * (Q_meas_dc - G_presc[0].real)
            if not np.all(np.isfinite(A_dc)) or not np.all(np.isfinite(t_dc)):
                g_dc = np.full(K, Q_gz_dc / max(K, 1))
            elif K == 1:
                g_dc = np.array([Q_gz_dc])
            else:
                aK = A_dc[:, -1]
                At = A_dc[:, :-1] - aK[:, np.newaxis]
                tt = t_dc - Q_gz_dc * aK
                try:
                    r = lsq_linear(At, tt, bounds=(0, Q_gz_dc))
                    gf = r.x; gK = max(Q_gz_dc - gf.sum(), 0)
                    g_dc = np.append(gf, gK)
                except Exception:
                    g_dc = np.full(K, Q_gz_dc / K)
        else:
            g_dc = np.array([])

        # H_0: predicted DC at s=1
        H_0 = G_presc[0].real.copy()
        for ki, bk in enumerate(red_nodes):
            if bk in greens and 0 in greens[bk]:
                for ei, ek in enumerate(edge_keys):
                    H_0[ei] += g_dc[ki] * greens[bk][0].get(ek, 0)

        # Step 3b: Scale s from log-magnitude
        valid = np.abs(H_0) > 1e-12
        if np.any(valid):
            lr = np.log(np.abs(Q_meas_dc[valid])) - np.log(np.abs(H_0[valid]))
            wv = weights[valid]
            ln_s = np.sum(wv * lr) / max(np.sum(wv), 1e-30)
            s = float(np.clip(np.exp(ln_s), 0.1, 10.0))
        else:
            s = 1.0

        # Step 3c: Stacked AC fractions
        if K > 0 and n_harmonics >= 1:
            abs_H0 = np.maximum(np.abs(H_0), 1e-15)
            abs_Qdc = np.maximum(np.abs(Q_meas_dc), 1e-15)
            nAC = n_harmonics
            A_st = np.zeros((nAC * N_meas, nAC * K), dtype=complex)
            t_st = np.zeros(nAC * N_meas, dtype=complex)
            for ni, n in enumerate(range(1, n_harmonics + 1)):
                r0 = ni * N_meas; c0 = ni * K
                Qn = np.array([meas_edges[ek]['harmonics'][n] for ek in edge_keys])
                for ei, ek in enumerate(edge_keys):
                    t_st[r0+ei] = np.sqrt(weights[ei]) * (Qn[ei]/abs_Qdc[ei] - G_presc[n][ei]/abs_H0[ei])
                    for ki, bk in enumerate(red_nodes):
                        if bk in greens and n in greens[bk]:
                            A_st[r0+ei, c0+ki] = np.sqrt(weights[ei]) * greens[bk][n].get(ek, 0) / abs_H0[ei]

            Ar = np.vstack([np.hstack([A_st.real, -A_st.imag]),
                            np.hstack([A_st.imag, A_st.real])])
            tr = np.concatenate([t_st.real, t_st.imag])
            if np.all(np.isfinite(Ar)) and np.all(np.isfinite(tr)):
                ATA = Ar.T @ Ar
                lam = lambda_tik * max(np.abs(np.diag(ATA)).max(), 1e-30)
                try: gv = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), Ar.T @ tr)
                except: gv = np.zeros(ATA.shape[0])
            else:
                gv = np.zeros(2 * nAC * K)
            g_ac = {}
            for ni, n in enumerate(range(1, n_harmonics + 1)):
                g_ac[n] = np.array([gv[ni*K+ki] + 1j*gv[nAC*K+ni*K+ki] for ki in range(K)])
        else:
            g_ac = {n: np.zeros(max(K,1), dtype=complex) for n in range(1, n_harmonics+1)}

        # Step 4: Loss
        L_shape = 0.0; L_mag = 0.0
        pred = {}
        for ei, ek in enumerate(edge_keys):
            Qp = np.zeros(n_harmonics+1, dtype=complex)
            Qp[0] = s * H_0[ei]
            for n in range(1, n_harmonics+1):
                q = G_presc[n][ei]
                for ki, bk in enumerate(red_nodes):
                    if bk in greens and n in greens[bk]:
                        q += g_ac[n][ki] * greens[bk][n].get(ek, 0)
                Qp[n] = s * q
            pred[ek] = Qp
            Qm = meas_edges[ek]['harmonics']
            if abs(H_0[ei]) > 1e-12:
                L_mag += weights[ei] * (np.log(abs(Qp[0].real)) - np.log(abs(Qm[0].real)))**2
            dp = abs(H_0[ei]); dm = abs(Qm[0].real)
            if dp > 1e-12 and dm > 1e-12:
                for n in range(1, n_harmonics+1):
                    sp = (G_presc[n][ei] + sum(g_ac[n][ki]*greens[bk][n].get(ek,0)
                          for ki,bk in enumerate(red_nodes) if bk in greens and n in greens[bk])) / dp
                    sm = Qm[n] / dm
                    L_shape += weights[ei] * abs(sp - sm)**2

        Lt = L_shape + lambda_mag * L_mag
        loss_curve.append((D, Lt))
        if Lt < best_loss:
            best_loss = Lt
            best_result = {'D': D, 's': s, 'g_dc': g_dc.copy(), 'g_ac': {n: g_ac[n].copy() for n in g_ac},
                           'predictions': dict(pred), 'L_shape': L_shape, 'L_mag': L_mag}
        if verbose and D_idx % 5 == 0:
            print(f" L={Lt:.4f} (shape={L_shape:.4f}, mag={lambda_mag*L_mag:.4f}) s={s:.3f}")

    # --- Output ---
    if best_result is None:
        best_result = {'D': D_sweep[0], 's': 1.0, 'g_dc': np.array([]), 'g_ac': {},
                       'predictions': {}, 'L_shape': np.inf, 'L_mag': np.inf}
    D_opt = best_result['D']; s_opt = best_result['s']
    if verbose:
        print(f"\nBest D={D_opt:.4e}, s={s_opt:.3f} (radius corr: {s_opt**0.25:.3f}x)")
        print(f"  Loss={best_loss:.6f} (shape={best_result['L_shape']:.4f}, mag={lambda_mag*best_result['L_mag']:.4f})")
        if K > 0:
            gd = best_result['g_dc']
            if len(gd) == K and gd.sum() > 0:
                fd = gd / gd.sum()
                top5 = sorted(range(K), key=lambda i: gd[i], reverse=True)[:5]
                for i in top5:
                    print(f"    Node {red_nodes[i]}: {gd[i]:.4f} nL/s ({fd[i]*100:.1f}%)")
            for n in range(1, n_harmonics+1):
                ga = best_result.get('g_ac', {}).get(n, np.zeros(1))
                if len(ga) > 0:
                    sigma = 1.0 - ga.sum()
                    print(f"  H{n} grey storage: |sigma|={abs(sigma):.3f}")

    # Residuals
    residuals = {}
    preds = best_result['predictions']
    for ei, ek in enumerate(edge_keys):
        if ek not in preds: continue
        Qp = preds[ek]; Qm = meas_edges[ek]['harmonics']
        dp = abs(Qp[0].real); dm = abs(Qm[0].real); r = 0.0
        if dp > 1e-12 and dm > 1e-12:
            for n in range(1, n_harmonics+1): r += abs(Qp[n]/dp - Qm[n]/dm)**2
        residuals[ek] = float(r)

    if verbose:
        rv = [v for v in residuals.values() if np.isfinite(v) and v > 0]
        if rv: print(f"  Residuals: med={np.median(rv):.6f}, max={max(rv):.6f}")
        print(f"{_eq}")

    fractions = {0: best_result['g_dc']}; fractions.update(best_result['g_ac'])
    return {'D_opt': D_opt, 's': s_opt, 'mu': mu, 'loss': best_loss, 'loss_curve': loss_curve,
            'fractions': fractions, 'predictions': best_result['predictions'],
            'residuals': residuals, 'meas_edges': {k: v['harmonics'] for k, v in meas_edges.items()},
            'red_nodes': red_nodes, 'n_harmonics': n_harmonics,
            'Q_art': Q_art, 'Q_ven': Q_ven, 'L_shape': best_result['L_shape'], 'L_mag': best_result['L_mag'],
            'blue_nodes': red_nodes, 'Q_total': np.zeros(n_harmonics+1)}


def plot_transmission_line_result(
    G: nx.Graph,
    result: TransmissionLineResult,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 10),
):
    """Plot network colored by predicted mean_Q and PI.

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph (for node positions / edge paths)
    result : TransmissionLineResult
    output_path : str, optional
        Save figure to this path. If None, plt.show().
    figsize : tuple
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import matplotlib.cm as cm

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for ax_idx, (field, title, cmap_name) in enumerate([
        ('mean_Q', 'Mean Flow |Q̄| (nL/s)', 'RdBu_r'),
        ('PI', 'Pulsatility Index', 'magma'),
        ('dissipation', 'Viscous Dissipation r·L·⟨Q²⟩ (W)', 'inferno'),
        ('pulsatile_cost', 'Pulsatile Cost ⟨Q²⟩/Q̄²', 'hot_r'),
    ]):
        ax = axes[ax_idx]
        field_dict = getattr(result, field)

        segments = []
        values = []

        for u, v in result.edge_flows.keys():
            val = field_dict.get((u, v))
            if val is None or not np.isfinite(val):
                continue

            # Get edge path for drawing
            data = G.edges[u, v]
            path = data.get('path')
            if path is not None and len(path) > 1:
                pts = np.array(path)
                # path is (row, col) = (y, x); plot as (x, y)
                xy = pts[:, ::-1] if pts.shape[1] == 2 else pts
                seg = np.column_stack([xy[:, 0], xy[:, 1]])
            else:
                # Fallback to node positions
                pos_u = G.nodes[u].get('pos', G.nodes[u].get('o'))
                pos_v = G.nodes[v].get('pos', G.nodes[v].get('o'))
                if pos_u is None or pos_v is None:
                    continue
                pos_u = np.array(pos_u)
                pos_v = np.array(pos_v)
                # pos is (row, col); plot as (x, y)
                seg = np.array([[pos_u[1], pos_u[0]],
                                [pos_v[1], pos_v[0]]])

            # LineCollection wants list of (N,2) arrays
            segments.append(seg.reshape(-1, 1, 2))
            values.append(val)

        if not segments:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        values = np.array(values)

        # Build LineCollection from segments
        # Each segment is (N_i, 1, 2); we need list of (N_i, 2)
        lines = []
        line_vals = []
        for seg, val in zip(segments, values):
            seg2d = seg.reshape(-1, 2)
            if len(seg2d) < 2:
                continue
            # Create line segments for this edge
            points = seg2d
            segs = np.concatenate([points[:-1, np.newaxis, :],
                                   points[1:, np.newaxis, :]], axis=1)
            lines.extend(segs)
            line_vals.extend([val] * len(segs))

        lines = np.array(lines)
        line_vals = np.array(line_vals)

        finite_vals = line_vals[np.isfinite(line_vals)]
        if field == 'mean_Q':
            # Diverging colormap centered on 0
            vmax = np.percentile(np.abs(finite_vals), 95)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        elif field == 'dissipation':
            # Log scale — spans many orders of magnitude
            from matplotlib.colors import LogNorm
            pos_vals = finite_vals[finite_vals > 0]
            vmin = np.percentile(pos_vals, 5)
            vmax = np.percentile(pos_vals, 95)
            norm = LogNorm(vmin=max(vmin, 1e-20), vmax=vmax, clip=True)
        elif field == 'pulsatile_cost':
            # Starts at 1 (steady), grows with pulsatility
            norm = Normalize(vmin=1.0,
                             vmax=min(np.percentile(finite_vals, 95), 20.0))
        else:
            # Sequential
            vmin = np.percentile(finite_vals, 5)
            vmax = np.percentile(finite_vals, 95)
            norm = Normalize(vmin=max(0, vmin), vmax=vmax)

        lc = LineCollection(lines, cmap=cmap_name, norm=norm, linewidths=1.5)
        lc.set_array(line_vals)
        ax.add_collection(lc)

        # Set limits from data
        all_x = np.concatenate([seg.reshape(-1, 2)[:, 0] for seg in segments])
        all_y = np.concatenate([seg.reshape(-1, 2)[:, 1] for seg in segments])
        margin = 50
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.max() + margin, all_y.min() - margin)  # Flip y

        # Mark boundary nodes
        for bn in result.boundary_nodes:
            pos = G.nodes[bn].get('pos', G.nodes[bn].get('o'))
            if pos is not None:
                pos = np.array(pos)
                ax.plot(pos[1], pos[0], 'r*', ms=12, zorder=5)

        ax.set_aspect('equal')
        ax.set_title(title)
        plt.colorbar(lc, ax=ax, shrink=0.8)

    fig.suptitle(
        f'Transmission Line Model  |  D={result.D:.1e} 1/Pa  |  '
        f'f0={result.f0_hz:.2f} Hz  |  {result.n_harmonics} harmonics  |  '
        f'µ={result.mu*1e3:.1f} mPa·s',
        fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  Saved figure to {output_path}")
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
# Radius adaptation — moved to adaptation.py, re-exported for compat
# =====================================================================

# Backward compatibility re-exports
from .adaptation import (  # noqa: E402, F401
    AdaptationParams,
    AdaptationResult,
    run_adaptation,
    plot_adaptation_result,
    solve_nutrients,
)
