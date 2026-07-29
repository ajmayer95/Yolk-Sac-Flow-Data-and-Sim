"""Linear one-chamber closed-loop hemodynamic solver on an arbitrary plexus.

Generalises the toy `toy_closed_loop.ipynb` notebook from 2-3 nodes to any
networkx plexus. Topology: a single heart chamber `H` in the middle;
each arterial boundary node of the plexus is connected to `H` by a
distributed RC conduit ("DA"), and each venous boundary node is
connected to `H` by another distributed RC conduit ("SV"). No valves —
the linear chamber model is valveless by construction.

AC solve per harmonic n = 1..N_max:
  L(n) · P(n) = H(n)

with L(n) the complex Laplacian (plexus + DA + SV edges + chamber
compliance iωC_h at H), and H(n) the clipped-cosine elastance source
i n ω₀ (V₀ − V_p0) Ẽ_n / E_0 at the heart node, using the stressed
volume V_stressed = V₀ − V_p0.

DC solve (Construction 3 — emergent mean flow): the chamber is split
into two ports (H_art, H_ven) and modeled as a DC pressure source
P̄_h = E_0·(V₀ − V_p0).  Dirichlet BCs P_H_art = P̄_h, P_H_ven = 0 are
imposed; the plexus+DA+SV resistive Laplacian is inverted; and the
emergent loop flow is

    Q̄ = P̄_h / R_loop_DC

where R_loop_DC is the total DC loop resistance (measured as the
current required to hold the prescribed port pressure difference).
Stiffer plexus ⇒ larger R_loop ⇒ smaller Q̄ automatically.  No Q̄
parameter to hand-tune.

Legacy 'imposed' DC mode (dc_mode='imposed', `Q_bar` set) rescales the
emergent solution to match a prescribed magnitude; the network still
sees conductance-weighted DC partitioning rather than equal-split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


MU_DEFAULT = 3.5e-3  # Pa·s (yolk-sac blood viscosity)

# Pixel → meter conversion for plexus edge geometry (1.7 µm/px on the
# 21-somite mosaic).  Imported lazily so this module doesn't pull in
# the full config on import.
def _px_size_m() -> float:
    try:
        from .config import PX_SIZE_UM
        return float(PX_SIZE_UM) * 1e-6
    except Exception:
        return 1.7e-6


# -----------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------


@dataclass
class LinearHeartParams:
    """One-chamber linear-loop parameters. All quantities are SI-compatible
    within one unit system; no hardcoded dimensions.
    """
    # Heart — physiologically order-of-magnitude SI defaults
    # (HH13–14 chick embryo, ~50 hr).  All in Pa, m³, m³/s.
    # Quantitative claims of this model concern dimensionless ratios
    # (impedance asymmetry, harmonic content); these defaults pick
    # a reasonable absolute regime so figures can be labeled in mmHg
    # / nL / nL/s, but the precise values are not calibrated.
    # Peristaltic two-chamber phasing (Choice A: same shape, time-shifted).
    # If non-zero, the AC architecture splits into two heart-port nodes
    # (H_art and H_ven), each with its own chamber capacitance C_h=1/E_0,
    # driven by the SAME elastance-shape forcing but with the venous side
    # lagging by τ = peristaltic_tau_frac · T.  This represents a
    # peristaltic activation wave traveling along the embryonic heart
    # tube, hitting the OFT first and the SV later.
    # When 0 (default), the single-chamber model is used (one heart node
    # shared between OFT and SV ports).
    peristaltic_tau_frac: float = 0.0    # τ / T, fraction of period (0..1)

    omega_0: float = 2 * np.pi * 2.5    # rad/s, HR ~150 bpm
    E_0: float = 1e12                   # Pa/m³ (operating-point elastance)
    V_0: float = 4e-10                  # m³  (~400 nL EDV)
    V_p0: float = 1e-10                 # m³  (~100 nL unstressed)
                                        # ⇒ V_stressed = 300 nL,
                                        # P̄_h = E_0·V_stressed = 300 Pa ≈ 2.25 mmHg
                                        # (boosted from 100 Pa to keep
                                        # Q̄ in HH14 range with the
                                        # higher-resistance conduits)
    E_amp: float = 2e11                 # Pa/m³ (|δE|/E_0 = 0.20)
    N_max: int = 20

    # Elastance waveform shape (a peak-normalized, zero-mean shape
    # function f̂(t) — amplitude is set separately by E_amp).
    #   'double_hill' (default) — Stergiopulos / Mynard double-Hill
    #       systolic pulse + diastolic relaxation.  Smooth (C^∞-ish),
    #       independently tunable rise and fall, asymmetric pulse.
    #       Fourier coefficients decay faster than 1/n² ⇒ cleaner
    #       harmonic truncation.  This is the standard cardiovascular
    #       elastance form (Suga & Sagawa 1974; Stergiopulos 1996).
    #       Tuning via `dh_alpha_S`, `dh_alpha_D`, `dh_m1`, `dh_m2`.
    #   'clipped_cosine' — physiological systolic-pulse profile
    #       (clipped cosine, demeaned + peak-normalized).  Rich
    #       harmonic content with 1/n² rolloff.
    #   'sinusoidal' — pure cos(ω_0 t).  Only H1.  Useful as a control.
    #   'clipped_cosine_full' (legacy) — full clipped cosine including
    #       DC.  Modulation depth ~1; violates linearization.
    elastance_shape: str = 'double_hill'

    # Double-Hill parameters (used when elastance_shape='double_hill').
    # Used for the OFT/DA port in two-chamber mode (= ventricular-like
    # contraction).  Defaults give "sharp rise + gradual fall" which
    # matches the empirical embryonic arterial waveform shape:
    #   • α_S = 0.10  ⇒ peak occurs early (~10% of cycle).
    #   • m1 = 30     ⇒ very sharp rising edge.
    #   • α_D = 0.50  ⇒ relaxation midpoint at mid-cycle.
    #   • m2 = 2      ⇒ gentle falling edge (gradual decay through
    #                   diastole, ramp-like).
    # In single-chamber mode this is the only set used.
    dh_alpha_S: float = 0.10      # early systolic peak time / T
    dh_alpha_D: float = 0.50      # relaxation midpoint / T
    dh_m1: float = 30.0           # sharp rising-edge Hill exponent
    dh_m2: float = 2.0            # gradual falling-edge Hill exponent
    dh_E_min_frac: float = 0.0    # E_min / E_max  (0 = fully relaxed)

    # AV channel — direct conduit between H_art (ventricle port) and
    # H_ven (atrium port) chamber nodes.  Closes the tube-heart
    # peristaltic loop: atrial pressure → AV → ventricular pressure
    # (and vice versa) without going around the entire plexus.
    # Modeled as a distributed-RC 2-port like DA / SV.
    #
    # Default targets HH 12–14 (~21 somite) AV canal — short tube
    # partly narrowed by cardiac cushions, no closed valve yet.
    #
    # **Critical constraint** (learned the hard way): AV_r must be
    # ≳ DA_r, otherwise the ventricle's path of least resistance
    # during contraction is BACKWARD through AV into the atrium
    # (regurgitation) rather than forward through OFT into the
    # plexus.  Symptoms: gold AV arrow only ever points V→A, atrium
    # ΔV >> ventricle ΔV.
    #
    # Default values:
    #   • AV_r = 1e13 (≈ DA_r) — AV opens enough to make atrium and
    #     ventricle exchange volume each cycle (clear bidirectional
    #     gold arrow), but stays high enough that ventricular
    #     ejection prefers OFT over AV regurgitation.  Geometry
    #     equivalent: r ≈ 25 µm, L ≈ 200 µm cushion-narrowed lumen.
    #   • AV_c = 1e-13 m³/Pa — cushion-like distensibility; gives a
    #     small AC phase lag between atrial and ventricular ports.
    # Tweaks:
    #   AV_r=5e12   → more open AV, but watch for V→A regurgitation
    #                  if AV_r drops below DA_r.
    #   AV_r=3e13   → cushion-tighter AV, clearer AV phase delay.
    #   AV_r→∞ AND AV_c→0 → disable AV entirely (decoupled chambers,
    #                  AC coupling only through the plexus).
    AV_r: float = 1e13
    AV_c: float = 1e-13

    # Two-chamber elastance-shape mode — Choice A vs Choice B.
    # CHOICE A (default, two_chamber_shapes=False): both chambers
    #   driven by the SAME elastance shape (the dh_* ventricular
    #   shape), with the SV port phase-shifted by τ (peristaltic
    #   activation wave traveling from atrium to ventricle).  Single
    #   waveform, well-parameterized.
    # CHOICE B (two_chamber_shapes=True): separate atrial dh_atrial_*
    #   shape on the SV port — independent contraction profiles per
    #   chamber.  More flexible but over-parameterized given the
    #   single-cell tube heart at HH13–14.
    two_chamber_shapes: bool = False
    dh_atrial_alpha_S: float = 0.30   # broader systolic peak time
    dh_atrial_alpha_D: float = 0.65   # later relaxation midpoint
    dh_atrial_m1: float = 1.0         # gentler rising-edge sharpness
    dh_atrial_m2: float = 6.0         # gentler falling-edge sharpness
    dh_atrial_E_min_frac: float = 0.0
    # Atrial-vs-ventricular amplitude ratio (atrium contracts weaker).
    # SV port forcing magnitude = E_amp · dh_atrial_amp_ratio.
    # Default raised from 0.5 → 1.0 so SV chamber forcing has the
    # full ventricular amplitude.  The conduit divider alone gives
    # SV ~5× DA pulsatility (R_DA/R_SV = 5 in light-filter regime),
    # so SV stays the more-pulsatile side without the atrial drive
    # being weaker than the ventricular drive.  Drop below 1.0 if
    # you want to model a measurably weaker atrial contraction.
    dh_atrial_amp_ratio: float = 1.0

    # DC flow mode:
    #   'emergent' — P̄_h = E_0·(V_0 − V_p0) set externally, then Q̄ is
    #                 solved from the DC loop (Construction 3).
    #   'imposed'  — use `Q_bar` directly at boundaries (legacy behavior).
    # 'emergent' is the physiologically honest choice: stiffer plexus
    # ⇒ lower Q̄ automatically.
    dc_mode: str = 'emergent'
    Q_bar: float = 0.1            # used only if dc_mode='imposed'

    # Conduit (DA and SV) per-edge parameters — distributed RC, in SI.
    # These are TOTAL values per conduit; assembly treats each as a
    # lumped 2-port with κ = √(i n ω₀ · r_tot · c_tot).
    #
    # Empirical asymmetry (Apr 2026):
    #   `r·c` and `r` are decoupled knobs:
    #     - `r·c` sets the FILTER SHAPE (κL² → phase delay,
    #        attenuation, harmonic mixing).  Counter-intuitively,
    #        large κL² *amplifies* high harmonics
    #        (Y_self/G = κ·coth κ grows with frequency at κL²≫1),
    #        making the waveform SHARPER, not smoother.  Small κL²
    #        ⇒ Y_self ≈ G (frequency-flat), output ≈ pressure-shaped.
    #     - `r` alone (with r·c held fixed) sets the DC admittance,
    #        and therefore the AC flow magnitude.
    #
    #   Assignment to match the user's measured embryonic waveforms:
    #     • DA: heavily distributed → κL² ≈ 3.5, |Y_self/G|≈1.65 at
    #           H1, ≈2.55 at H2.  The H2/H1 amplification produces a
    #           SHARP, spike-and-decay arterial-looking waveform AT
    #           LOW amplitude (small admittance to chamber pressure).
    #     • SV: nearly lumped → κL² ≈ 0.04, |Y_self/G|≈1 at all
    #           harmonics.  Output tracks chamber-side ΔP smoothly,
    #           giving a rounder, slower-rising venous-looking
    #           waveform AT LARGER amplitude (high admittance).
    #
    #   Magnitudes are tuned so |Y_SV|/|Y_DA| ≈ 2 — SV-DOMINANT
    #   pulsatility (venous amplitude ~2× arterial) consistent with
    #   embryonic single-chamber data.  This decouples shape (set by
    #   r·c) from amplitude (set by r alone).
    #
    #   Q̄ ≈ P̄_h / R_loop ≈ 100 / (R_DA/2 + R_SV/2) ≈ 120 nL/s.
    # Asymmetric (r, c) regime — decouples shape from amplitude.
    #   κL² = ω·r·c  controls the filter shape (small ⇒ flat, smooth;
    #         large ⇒ high-harmonic amplification, sharper peak).
    #   G   = 1/r    sets the DC admittance and AC amplitude.
    # By picking (r, c) asymmetrically (not just at constant r·c) we
    # can give one side BIG amplitude AND BROAD waveform, the other
    # side SMALL amplitude AND SHARP waveform — which matches the
    # empirical embryonic pattern:
    #   • SV (venous):   low r, moderate c → small κL² (light filter
    #         ⇒ transmits chamber shape directly = BROAD waveform),
    #         and high G = 1/r ⇒ high admittance = BIG amplitude.
    #   • DA (arterial): high r, moderate c → large κL² (heavy filter
    #         ⇒ H2 amplification = SHARP narrow peak), low G ⇒ low
    #         admittance = SMALL amplitude.
    # Both conduits in LIGHT-filter regime (κL² < 1) so the conduit
    # transfer function is roughly flat in frequency and each port
    # inherits its chamber-forcing shape directly.  The asymmetry is
    # then carried by:
    #   (a) DC admittance |Y| = 1/R: SV has higher G ⇒ SV-dominant
    #       AC amplitude regardless of harmonic content.
    #   (b) Different elastance shapes at each port (Choice B):
    #       sharp ventricular at DA, broad atrial at SV.
    #
    # **Ratio (Apr 2026, settled):** R_DA/R_SV = 5:1 — the validated
    # config that reproduces SV-broad-large + DA-sharp-small Q-asymmetry.
    # (Briefly tried 2:1 by raising SV_r to 5e12 to symmetrize chamber
    # ΔV; user wanted the SV pulsatility back, so reverted.)
    DA_r: float = 1e13            # Pa·s/m³ — very high resistance
    DA_c: float = 2e-15           # m³/Pa  — low compliance
                                  # κL²_DA ≈ 0.31 ⇒ light filter
                                  # |Y_DA(H1)| ≈ 1.0e-13
                                  # SHAPE: ≈ ventricular forcing
                                  # AMP:   small (1× reference)
    SV_r: float = 2e12            # Pa·s/m³ — moderate resistance
    SV_c: float = 1e-14           # m³/Pa  — low compliance
                                  # κL²_SV ≈ 0.31 ⇒ light filter
                                  # |Y_SV(H1)| ≈ 5.0e-13
                                  # SHAPE: ≈ atrial forcing
                                  # AMP:   ~5× DA (SV-dominant from
                                  #        R divider; validated config)
    # Net Q̄ ≈ P̄_h / R_loop ≈ 300 / (5e12 + 4e12_plexus + 1e13_AV)
    # ≈ 300/1.6e13 ≈ 19 nL/s, in HH14 CO range.

    # Venous-side compliance reservoir (Frank windkessel element).
    # Adds a parallel-to-ground capacitor at each sink (venous) BC
    # node, AC only.  Models the "venous reservoir" effect — bleeds
    # high-frequency content from the SV path while preserving DC and
    # H1 (since it's a single-pole low-pass, max ~2× H2/H1 reduction
    # relative to no-filter).  Default 0 = disabled.
    # Sweep range: 1e-15 (negligible) → 1e-13 (strong filter at H2).
    # Cutoff frequency f_c = 1/(2π·R_load·C_ven); want f_c near f₀.
    C_venous_BC: float = 0.0      # m³/Pa


# -----------------------------------------------------------------------
# Admittances
# -----------------------------------------------------------------------


def _edge_admittances(n: int, r_tot: float, c_tot: float,
                       omega_0: float) -> Tuple[complex, complex]:
    """Two-port admittance (Y_self, Y_trans) of a distributed RC edge.

    Implements the same coth / csch formulas as the notebook, with the
    same Taylor-expansion regularization near κ → 0.
    """
    G_e = 1.0 / r_tot
    if n == 0:
        return complex(G_e), complex(G_e)
    kappa = np.sqrt(1j * n * omega_0 * r_tot * c_tot + 0j)
    if abs(kappa) < 1e-6:
        Y_self = G_e * (1.0 + kappa * kappa / 3.0)
        Y_trans = G_e * (1.0 - kappa * kappa / 6.0)
    else:
        Y_self = G_e * kappa / np.tanh(kappa)
        Y_trans = G_e * kappa / np.sinh(kappa)
    return Y_self, Y_trans


def _plexus_edge_conductance(G: nx.Graph, u, v, mu: float) -> float:
    """Poiseuille DC conductance for a plexus edge.

    Graph stores R and L in PIXELS; we convert to METERS via PX_SIZE
    so the result is in SI units (m³ / Pa·s) consistent with the
    chamber and conduit parameters.
    """
    d = G.edges[u, v]
    R_px = d.get('radius')
    L_px = d.get('length', d.get('length_true'))
    if R_px is None or L_px is None or R_px <= 0 or L_px <= 0:
        return 0.0
    px = _px_size_m()
    R = float(R_px) * px        # meters
    L = float(L_px) * px        # meters
    return float(np.pi * R ** 4 / (8.0 * mu * L))


def _plexus_edge_compliance(G: nx.Graph, u, v, D) -> float:
    """Distributed-compliance lumped-to-node contribution for an edge:
    c_edge·L = π R² D · L per edge.  R, L converted from pixels to
    meters; D is AREAL distensibility in 1/Pa (ΔA/A = D·ΔP).  Result
    in m³/Pa (SI).

    `D` is either a scalar (uniform distensibility) or a callable
    `D(R_m) → 1/Pa` that returns per-vessel distensibility as a function
    of the vessel radius in meters.  The callable form lets synthetic
    runs apply size-dependent compliance scaling (e.g. larger vessels =
    more distensible).

    NOTE: pre-2026-05-18 this used the radius convention with a 2πR²D
    factor; switched to areal (πR²D) for consistency with vascular
    biomechanics literature.  D values under the new convention are
    2× the old ones for the same physics."""
    d = G.edges[u, v]
    R_px = d.get('radius')
    L_px = d.get('length', d.get('length_true'))
    if R_px is None or L_px is None or R_px <= 0 or L_px <= 0:
        return 0.0
    px = _px_size_m()
    R = float(R_px) * px        # meters
    L = float(L_px) * px        # meters
    D_eff = D(R) if callable(D) else D
    return float(np.pi * R * R * D_eff * L)


# -----------------------------------------------------------------------
# Elastance harmonics (clipped cosine)
# -----------------------------------------------------------------------


def _double_hill_shape(t_norm: np.ndarray,
                         alpha_S: float, alpha_D: float,
                         m1: float, m2: float,
                         E_min_frac: float = 0.0) -> np.ndarray:
    """Stergiopulos / Mynard double-Hill elastance shape on t_norm = t/T
    in [0, 1).  Returns the absolute (non-normalized) shape, with peak
    ≈ 1 and floor ≈ E_min_frac.

    g_1(t) = (t/(α_S T))^m1                  (rising-edge Hill)
    g_2(t) = (t/(α_D T))^m2                  (falling-edge Hill)
    e(t)   = g_1/(1+g_1) · 1/(1+g_2)
    E(t)   = (E_max - E_min)·e(t)/max(e) + E_min,  with E_max=1.
    """
    eps = 1e-12
    g1 = (t_norm / max(alpha_S, eps)) ** m1
    g2 = (t_norm / max(alpha_D, eps)) ** m2
    e = (g1 / (1.0 + g1)) * (1.0 / (1.0 + g2))
    e_peak = float(np.max(e)) if np.max(e) > eps else 1.0
    e_norm = e / e_peak                                 # peak ≈ 1
    return E_min_frac + (1.0 - E_min_frac) * e_norm


def elastance_harmonics(E_amp: float, N_max: int,
                          shape: str = 'clipped_cosine',
                          dh_alpha_S: float = 0.30,
                          dh_alpha_D: float = 0.45,
                          dh_m1: float = 1.32,
                          dh_m2: float = 27.4,
                          dh_E_min_frac: float = 0.0,
                          ) -> np.ndarray:
    """Fourier coefficients of the AC elastance forcing δE(t).

    Convention: real reconstruction is f(t) = F[0] + 2·Re Σ_{n≥1} F[n]·e^(inω₀t).
    One-sided array F[0..N_max].

    Shape and amplitude are decoupled:
      • The *shape function* f̂(t) is a periodic, zero-mean unit-amplitude
        waveform whose Fourier coefficients f̂_n encode the harmonic
        ratios (h_2/h_1, h_3/h_1, ...) — i.e. the spectral signature
        of the forcing.
      • `E_amp` is the peak-amplitude scaling applied AFTER the shape:
        δE(t) = E_amp · f̂(t).  All harmonics scale together, so
        harmonic *ratios* are invariant in E_amp.

    Available shapes (all zero-mean, peak |f̂(t)| = 1):
      'double_hill'     — Stergiopulos / Mynard standard cardiovascular
            elastance: smooth pulse with independently tunable rise
            (α_S, m1) and fall (α_D, m2), plus optional non-zero
            diastolic floor (E_min_frac).  Fourier coefficients
            decay faster than 1/n² ⇒ cleaner harmonic truncation.
            Refs: Suga & Sagawa 1974; Stergiopulos 1996; Mynard 2012.
      'clipped_cosine'  — clipped-cosine systolic-pulse, demeaned +
            normalized.  Rich harmonics with 1/n² rolloff, sharp
            corners at clipping points.
      'sinusoidal'      — pure cos(ω₀t).  Single H1 harmonic only.
            Useful as a control / null hypothesis.
      'clipped_cosine_full' (legacy) — full clipped-cosine waveform
            INCLUDING its DC component.  Modulation depth ~1.
    """
    F = np.zeros(N_max + 1, dtype=np.complex128)

    if shape == 'double_hill':
        # Numerically compute Fourier coefficients of the time-domain
        # double-Hill pulse, then demean + peak-normalize.  Use a fine
        # time grid so harmonics up to N_max are well-resolved (Nyquist).
        N_t = max(1024, 16 * (N_max + 1))
        t_norm = np.arange(N_t) / N_t                   # t/T in [0,1)
        e_t = _double_hill_shape(
            t_norm, dh_alpha_S, dh_alpha_D, dh_m1, dh_m2, dh_E_min_frac)
        # Demean + peak-normalize: f̂(t) = (e − ⟨e⟩) / max|e − ⟨e⟩|
        e_dc = float(e_t.mean())
        e_ac = e_t - e_dc
        peak = float(np.max(np.abs(e_ac)))
        if peak > 1e-30:
            f_hat = e_ac / peak
        else:
            f_hat = e_ac
        # Forward FFT (one-sided harmonics).  np.fft.rfft returns
        # F̃[k] = Σ_n f[n] e^(-i 2π k n/N) — we want F[k] = ⟨f e^(-iω_k t)⟩,
        # so divide by N to get the Fourier-series coefficient.
        Fk = np.fft.rfft(f_hat) / N_t
        n_keep = min(N_max + 1, Fk.size)
        F[:n_keep] = E_amp * Fk[:n_keep]
        F[0] = 0.0                                      # enforce demeaned

    elif shape == 'clipped_cosine':
        # Demean + normalize so peak |f̂(t)| = 1.
        # Raw clipped cosine: g(t) = max(cos ω₀t, 0).
        # Mean ⟨g⟩ = 1/π;  peak g = 1;  trough g = 0.
        # Demeaned: g(t) − 1/π   has peak (1 − 1/π) and trough (−1/π).
        # Normalize by (1 − 1/π) so peak = 1 (and trough = −1/π/(1−1/π)).
        norm = 1.0 - 1.0 / np.pi              # ≈ 0.6817
        # Raw clipped-cosine harmonics (relative to E_amp = 1):
        raw = np.zeros(N_max + 1, dtype=np.complex128)
        raw[0] = 1.0 / np.pi
        if N_max >= 1:
            raw[1] = 0.25
        for n in range(2, N_max + 1):
            if n % 2 == 1:
                raw[n] = 0.0
            else:
                k = n // 2
                raw[n] = ((-1) ** (k + 1)) / (np.pi * (4 * k * k - 1))
        # Demean (drop DC) and normalize so peak f̂(t) = 1.
        raw[0] = 0.0
        F = E_amp * (raw / norm)

    elif shape == 'sinusoidal':
        # Pure cosine: 2·Re[F[1] e^(iω₀t)] = 2·(E_amp/2)·cos(ω₀t)
        # = E_amp · cos(ω₀t).  Peak = E_amp, trough = −E_amp.
        if N_max >= 1:
            F[1] = E_amp * 0.5

    elif shape == 'clipped_cosine_full':
        # Full clipped cosine (legacy): NOT zero-mean.  Modulation
        # depth |δE|/E_0_implicit = π/4 ≈ 0.78 — violates linearization.
        F[0] = E_amp / np.pi
        if N_max >= 1:
            F[1] = E_amp * 0.25
        for n in range(2, N_max + 1):
            if n % 2 == 1:
                F[n] = 0.0
            else:
                k = n // 2
                F[n] = (E_amp * ((-1) ** (k + 1))
                         / (np.pi * (4 * k * k - 1)))

    else:
        raise ValueError(
            f"Unknown elastance shape: {shape!r} "
            f"(expected 'double_hill', 'clipped_cosine', "
            f"'sinusoidal', or 'clipped_cosine_full').")
    return F


# -----------------------------------------------------------------------
# Assembly + solve
# -----------------------------------------------------------------------


@dataclass
class LinearLoopResult:
    params: LinearHeartParams
    f0_hz: float
    heart_node: int                        # arterial heart port label
    source_nodes: List[int]
    sink_nodes: List[int]
    node_order: List                      # list of node labels
    P_harm: np.ndarray                    # (N_max+1, N_nodes) complex
    edge_flows: Dict                      # edge -> (N_max+1,) complex
    E_tilde: np.ndarray                   # (N_max+1,) complex
    loop_conduits: Dict                   # label -> list of edges (ghost DA/SV)
    heart_node_ven: object = None          # venous heart port (= heart_node in
                                           # single-chamber, distinct in two-chamber)
    # Emergent DC quantities (Construction 3)
    P_bar_h: float = 0.0                  # mean chamber pressure = E_0·(V_0−V_p0)
    R_loop_dc: float = 0.0                # total DC loop resistance
    Q_bar_emergent: float = 0.0           # emergent DC loop flow
    P_h_art_dc: float = 0.0               # DC pressure at arterial heart port
    P_h_ven_dc: float = 0.0               # DC pressure at venous heart port


def _assemble_extended_laplacian(n: int, G: nx.Graph,
                                   node_order: list,
                                   node_idx: dict,
                                   heart_node_art,
                                   heart_node_ven,
                                   sources: list, sinks: list,
                                   p: LinearHeartParams,
                                   mu: float, D_comp: float,
                                   ) -> Tuple[csr_matrix, np.ndarray]:
    """Build L(n), H(n) for harmonic n ≥ 1.

    `heart_node_art` and `heart_node_ven` may be the same label (single-
    chamber, default) or two distinct labels (two-chamber peristaltic).
    The two-chamber path adds chamber compliance C_h = 1/E_0 at EACH
    port independently (so the system has two pressure DOFs, both with
    their own forcing applied externally to H_n by the caller)."""
    N = len(node_order)
    L_mat = lil_matrix((N, N), dtype=np.complex128)
    H = np.zeros(N, dtype=np.complex128)
    omega = p.omega_0

    # Plexus edges: distributed RC via Poiseuille + per-length compliance
    for u, v in G.edges():
        G_e = _plexus_edge_conductance(G, u, v, mu)
        if G_e <= 0:
            continue
        # Treat plexus edges as lumped-compliance 2-port by assembling
        # the same distributed formula with r_tot = 1/G_e, c_tot from D.
        r_tot = 1.0 / G_e
        c_tot = _plexus_edge_compliance(G, u, v, D_comp)
        Y_self, Y_trans = _edge_admittances(n, r_tot, max(c_tot, 1e-30),
                                              omega)
        iu, iv = node_idx[u], node_idx[v]
        L_mat[iu, iu] += Y_self
        L_mat[iv, iv] += Y_self
        L_mat[iu, iv] -= Y_trans
        L_mat[iv, iu] -= Y_trans

    i_h_art = node_idx[heart_node_art]
    i_h_ven = node_idx[heart_node_ven]

    # Ghost DA conduits (H_art ↔ arterial source)
    for src in sources:
        Y_self, Y_trans = _edge_admittances(n, p.DA_r, p.DA_c, omega)
        i_s = node_idx[src]
        L_mat[i_h_art, i_h_art] += Y_self
        L_mat[i_s, i_s] += Y_self
        L_mat[i_h_art, i_s] -= Y_trans
        L_mat[i_s, i_h_art] -= Y_trans

    # Ghost SV conduits (venous sink ↔ H_ven)
    for snk in sinks:
        Y_self, Y_trans = _edge_admittances(n, p.SV_r, p.SV_c, omega)
        i_k = node_idx[snk]
        L_mat[i_h_ven, i_h_ven] += Y_self
        L_mat[i_k, i_k] += Y_self
        L_mat[i_h_ven, i_k] -= Y_trans
        L_mat[i_k, i_h_ven] -= Y_trans

    # Venous-side compliance reservoir (Frank windkessel).
    # Adds a parallel-to-ground capacitor at each sink BC node.  At
    # AC, contributes admittance Y_ven = jnω·C_venous_BC; at DC (n=0)
    # this is zero so no effect on the DC solve.  Bleeds H2/H3 from
    # the venous return path while preserving H1, producing the
    # asymmetric SV-low-pass behaviour empirically observed.
    C_ven = float(getattr(p, 'C_venous_BC', 0.0))
    if C_ven > 0.0 and n >= 1:
        Y_ven = 1j * n * omega * C_ven
        for snk in sinks:
            L_mat[node_idx[snk], node_idx[snk]] += Y_ven

    # Chamber compliance — added at each port (or once if single node)
    C_h = 1.0 / p.E_0
    L_mat[i_h_art, i_h_art] += 1j * n * omega * C_h
    if i_h_ven != i_h_art:
        L_mat[i_h_ven, i_h_ven] += 1j * n * omega * C_h

    # AV channel — direct conduit between H_art and H_ven.
    # Closes the peristaltic loop: atrial contraction → AV → ventricle.
    # Active only in two-chamber mode (two distinct port nodes).
    # Treats AV as a distributed-RC 2-port like DA/SV.  Set AV_r → ∞
    # or AV_c → 0 to disable.
    if (i_h_art != i_h_ven
            and getattr(p, 'AV_r', 0.0) > 0.0):
        Y_self_AV, Y_trans_AV = _edge_admittances(
            n, p.AV_r, max(p.AV_c, 1e-30), omega)
        L_mat[i_h_art, i_h_art] += Y_self_AV
        L_mat[i_h_ven, i_h_ven] += Y_self_AV
        L_mat[i_h_art, i_h_ven] -= Y_trans_AV
        L_mat[i_h_ven, i_h_art] -= Y_trans_AV

    return L_mat.tocsr(), H


def run_linear_one_chamber(G: nx.Graph,
                            params: Optional[LinearHeartParams] = None,
                            mu: float = MU_DEFAULT,
                            D_compliance: float = 1e-5,
                            verbose: bool = True,
                            ) -> LinearLoopResult:
    """Full-pipeline run of the linear one-chamber model on a plexus.

    Parameters
    ----------
    G : networkx.Graph
        Plexus graph with `boundary_type ∈ {'source', 'sink'}` on
        boundary nodes and `radius`, `length` on edges.
    params : LinearHeartParams
    mu : float
        Blood viscosity.
    D_compliance : float
        Plexus-edge wall distensibility (used to compute c_edge on plexus
        edges). Set 0 for purely resistive plexus.
    """
    p = params or LinearHeartParams()
    sources = sorted(n for n, d in G.nodes(data=True)
                      if d.get('boundary_type') == 'source')
    sinks = sorted(n for n, d in G.nodes(data=True)
                    if d.get('boundary_type') == 'sink')
    if not sources or not sinks:
        raise ValueError("Need at least one arterial source and one "
                          "venous sink boundary node.")

    # Extended node list: plexus + ghost heart node(s).
    # Two-chamber peristaltic mode if peristaltic_tau_frac != 0:
    # split into H_art and H_ven so each port can be driven by a
    # separately phase-shifted forcing.  Otherwise single chamber.
    two_chamber = abs(p.peristaltic_tau_frac) > 1e-12
    if two_chamber:
        heart_node_art = '__H_art__'
        heart_node_ven = '__H_ven__'
        node_order = list(G.nodes()) + [heart_node_art, heart_node_ven]
    else:
        heart_node_art = '__H__'
        heart_node_ven = heart_node_art      # alias to same node
        node_order = list(G.nodes()) + [heart_node_art]
    # Backwards-compat alias used by downstream code (single label).
    heart_node = heart_node_art
    node_idx = {n: i for i, n in enumerate(node_order)}
    N_nodes = len(node_order)

    # Elastance harmonics — ventricular (DA-port) shape (default for
    # both ports; in single-chamber and Choice-A modes, used at every
    # port).
    E_tilde = elastance_harmonics(
        p.E_amp, p.N_max, shape=p.elastance_shape,
        dh_alpha_S=p.dh_alpha_S, dh_alpha_D=p.dh_alpha_D,
        dh_m1=p.dh_m1, dh_m2=p.dh_m2,
        dh_E_min_frac=p.dh_E_min_frac)
    # Atrial (SV-port) shape — only used when two-chamber mode is
    # active and `two_chamber_shapes=True` (Choice B).
    use_atrial_shape = (two_chamber and
                         getattr(p, 'two_chamber_shapes', False))
    if use_atrial_shape:
        E_tilde_atrial = elastance_harmonics(
            p.E_amp * p.dh_atrial_amp_ratio, p.N_max,
            shape=p.elastance_shape,
            dh_alpha_S=p.dh_atrial_alpha_S,
            dh_alpha_D=p.dh_atrial_alpha_D,
            dh_m1=p.dh_atrial_m1,
            dh_m2=p.dh_atrial_m2,
            dh_E_min_frac=p.dh_atrial_E_min_frac)
    else:
        E_tilde_atrial = E_tilde   # same shape on both ports

    # ---- AC solve (single H node, chamber as capacitor + source) ----
    # Source amplitude uses (V_0 - V_p0): the pulsatile pressure contribution
    # is δE(t)·(V̄−V_p0), where V̄−V_p0 is the stressed operating volume.
    # When V_p0 = 0 this reduces to the previous V_0 formulation.
    V_stressed = p.V_0 - p.V_p0
    # Peristaltic time shift τ (seconds): SV chamber lags by this much
    # (negative τ_frac ⇒ SV LEADS DA, i.e. atrium-first).
    T_period = 2.0 * np.pi / p.omega_0
    tau_shift = p.peristaltic_tau_frac * T_period
    P_harm = np.zeros((p.N_max + 1, N_nodes), dtype=np.complex128)
    for n in range(1, p.N_max + 1):
        L_n, H_n = _assemble_extended_laplacian(
            n, G, node_order, node_idx,
            heart_node_art, heart_node_ven,
            sources, sinks, p, mu, D_compliance)
        # OFT/DA port: ventricular-like elastance (E_tilde).
        force_art = (1j * n * p.omega_0 * V_stressed * E_tilde[n]
                     / p.E_0)
        H_n[node_idx[heart_node_art]] = force_art
        if heart_node_ven != heart_node_art:
            # SV port: atrial-like elastance (E_tilde_atrial), with
            # phase shift τ.  In Choice A (two_chamber_shapes=False)
            # E_tilde_atrial == E_tilde, so this reduces to a pure
            # time shift.  In Choice B they differ in shape AND
            # amplitude (atrial amp = E_amp · dh_atrial_amp_ratio).
            force_ven_amp = (1j * n * p.omega_0 * V_stressed
                              * E_tilde_atrial[n] / p.E_0)
            H_n[node_idx[heart_node_ven]] = (
                force_ven_amp * np.exp(-1j * n * p.omega_0 * tau_shift))
        try:
            P_harm[n, :] = spsolve(L_n, H_n)
        except Exception as e:
            if verbose:
                print(f"  ⚠️  solve failed at n={n}: {e}")
            continue

    # ---- DC solve: split heart into H_art / H_ven ports ----
    # Construction 3: chamber acts as a DC pressure source P̄_h = E_0·(V̄−V_p0)
    # between its arterial and venous ports.  Q̄ emerges from Ohm's law
    # around the loop: Q̄ = P̄_h / R_loop where R_loop is the total DC
    # resistance traversed by a unit current entering H_art, passing
    # DA → plexus → SV, and exiting H_ven.
    #
    # We augment the plexus with two extra nodes (H_art, H_ven) only for
    # the DC solve.  The single AC heart_node in `node_order` retains a
    # summary DC pressure (the average of the two port pressures) for
    # convenience; edge DC reconstruction below uses the correct port
    # pressures explicitly.
    P_bar_h = p.E_0 * V_stressed                # mean chamber pressure
    # In two-chamber AC mode, H_art and H_ven are already in node_order;
    # in single-chamber mode we add a temporary H_ven slot just for DC.
    if two_chamber:
        N_dc = N_nodes
        i_h_art = node_idx[heart_node_art]
        i_h_ven = node_idx[heart_node_ven]
    else:
        N_dc = N_nodes + 1
        i_h_art = node_idx[heart_node]
        i_h_ven = N_nodes
    L0 = lil_matrix((N_dc, N_dc), dtype=np.float64)
    # Plexus resistive edges
    for u, v in G.edges():
        G_e = _plexus_edge_conductance(G, u, v, mu)
        if G_e <= 0:
            continue
        iu, iv = node_idx[u], node_idx[v]
        L0[iu, iu] += G_e; L0[iv, iv] += G_e
        L0[iu, iv] -= G_e; L0[iv, iu] -= G_e
    # DA conduits connect sources to H_art
    G_DA = 1.0 / p.DA_r
    for src in sources:
        i_s = node_idx[src]
        L0[i_h_art, i_h_art] += G_DA; L0[i_s, i_s] += G_DA
        L0[i_h_art, i_s] -= G_DA; L0[i_s, i_h_art] -= G_DA
    # SV conduits connect sinks to H_ven
    G_SV = 1.0 / p.SV_r
    for snk in sinks:
        i_k = node_idx[snk]
        L0[i_h_ven, i_h_ven] += G_SV; L0[i_k, i_k] += G_SV
        L0[i_h_ven, i_k] -= G_SV; L0[i_k, i_h_ven] -= G_SV

    # Dirichlet BCs: P_H_art = P_bar_h, P_H_ven = 0
    rhs = np.zeros(N_dc)
    L0_csr = L0.tolil()
    for i_pin, p_val in [(i_h_art, P_bar_h), (i_h_ven, 0.0)]:
        L0_csr[i_pin, :] = 0.0
        L0_csr[i_pin, i_pin] = 1.0
        rhs[i_pin] = p_val

    # If user specifically requested imposed DC, override P_bar_h later.
    Q_bar_emergent = 0.0
    R_loop = 0.0
    P_h_art_dc = P_bar_h
    P_h_ven_dc = 0.0
    try:
        P_dc_full = spsolve(L0_csr.tocsr(), rhs)
        # Emergent Q̄: sum of current leaving H_art through all DA conduits
        Q_bar_emergent = 0.0
        for src in sources:
            Q_bar_emergent += G_DA * (P_dc_full[i_h_art]
                                       - P_dc_full[node_idx[src]])
        if abs(P_bar_h) > 1e-30:
            R_loop = P_bar_h / max(Q_bar_emergent, 1e-30)
        # Write plexus DC pressures into P_harm[0, :].  In single-chamber
        # mode, the single AC heart_node gets the average of the two port
        # pressures (since the AC node is shared).  In two-chamber mode
        # the two heart-port DC pressures are already at i_h_art / i_h_ven.
        P_harm[0, :N_nodes] = P_dc_full[:N_nodes]
        if not two_chamber:
            P_harm[0, i_h_art] = 0.5 * (P_h_art_dc + P_h_ven_dc)

        if p.dc_mode == 'imposed':
            # Rescale DC pressures to match externally prescribed Q_bar
            # (keeps the emergent geometry but normalizes magnitude).
            if abs(Q_bar_emergent) > 1e-30:
                scale = p.Q_bar / Q_bar_emergent
                P_harm[0, :] *= scale
                P_h_art_dc *= scale
                P_h_ven_dc *= scale
                Q_bar_emergent = p.Q_bar
                R_loop = (P_h_art_dc - P_h_ven_dc) / max(p.Q_bar, 1e-30)

        if verbose:
            print(f"  DC Construction 3: P̄_h = E₀·(V₀−V_p0) = "
                  f"{P_bar_h:.4g}, R_loop = {R_loop:.4g}, "
                  f"Q̄_emergent = {Q_bar_emergent:.4g}  "
                  f"(mode={p.dc_mode})")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  DC solve failed: {e}; DC left at zero")

    # ---- Reconstruct per-edge harmonic flows ----
    # For distributed-RC edges, Q_u = Y_self·P_u − Y_trans·P_v gives
    # the current at the u-port specifically, not the mid-edge flow.
    # When an edge is stored with its "downstream" node first (e.g.,
    # a plexus edge next to arterial boundary stored (plexus, A)), the
    # u-port is the interior node and its harmonic spectrum has been
    # phase-shifted by the edge's own capacitive filter — giving a
    # fingerprint that doesn't match the boundary conduit.
    # Q_avg = (Y_self + Y_trans)/2 · (P_u − P_v) is symmetric under the
    # u↔v swap (only a sign flip, handled by downstream sign-flip-to-
    # positive-DC logic) and represents the mid-edge flow, yielding a
    # fingerprint consistent with the node-to-node KCL.  Matches
    # Q_stored at DC (where Y_self = Y_trans = G).
    edge_flows: Dict = {}
    for u, v in G.edges():
        G_e = _plexus_edge_conductance(G, u, v, mu)
        if G_e <= 0:
            continue
        r_tot = 1.0 / G_e
        c_tot = _plexus_edge_compliance(G, u, v, D_compliance)
        Q = np.zeros(p.N_max + 1, dtype=np.complex128)
        for n in range(p.N_max + 1):
            Y_self, Y_trans = _edge_admittances(
                n, r_tot, max(c_tot, 1e-30), p.omega_0)
            Q[n] = (0.5 * (Y_self + Y_trans)
                    * (P_harm[n, node_idx[u]] - P_harm[n, node_idx[v]]))
        edge_flows[(u, v)] = Q

    # DA / SV edges (ghost conduits): use the same Q_avg convention as
    # plexus edges so their fingerprints are comparable on equal footing.
    # At DC, the chamber is split into H_art / H_ven ports, so we
    # substitute the correct port pressure for each side at n=0.
    loop_conduits = {'DA': [], 'SV': []}
    for src in sources:
        Q = np.zeros(p.N_max + 1, dtype=np.complex128)
        for n in range(p.N_max + 1):
            Y_self, Y_trans = _edge_admittances(n, p.DA_r, p.DA_c,
                                                  p.omega_0)
            # DA conduit uses arterial heart-port pressure
            P_heart_n = (P_h_art_dc if n == 0
                         else P_harm[n, node_idx[heart_node_art]])
            Q[n] = (0.5 * (Y_self + Y_trans)
                    * (P_heart_n - P_harm[n, node_idx[src]]))
        edge_flows[(heart_node_art, src)] = Q
        loop_conduits['DA'].append((heart_node_art, src))
    for snk in sinks:
        Q = np.zeros(p.N_max + 1, dtype=np.complex128)
        for n in range(p.N_max + 1):
            Y_self, Y_trans = _edge_admittances(n, p.SV_r, p.SV_c,
                                                  p.omega_0)
            # SV conduit uses venous heart-port pressure (separate
            # node in two-chamber mode, same node in single-chamber)
            P_heart_n = (P_h_ven_dc if n == 0
                         else P_harm[n, node_idx[heart_node_ven]])
            Q[n] = (0.5 * (Y_self + Y_trans)
                    * (P_harm[n, node_idx[snk]] - P_heart_n))
        edge_flows[(snk, heart_node_ven)] = Q
        loop_conduits['SV'].append((snk, heart_node_ven))

    # AV channel: atrium (heart_node_ven) → ventricle (heart_node_art).
    # Positive Q ⇒ flow from atrium toward ventricle.  At DC, the
    # Construction-3 split pins H_art and H_ven independently, so
    # AV's DC flow is set by the (P_h_ven_dc - P_h_art_dc) gauge.
    if (two_chamber and getattr(p, 'AV_c', 0.0) > 0.0
            and getattr(p, 'AV_r', 0.0) > 0.0):
        Q = np.zeros(p.N_max + 1, dtype=np.complex128)
        for n in range(p.N_max + 1):
            Y_self, Y_trans = _edge_admittances(
                n, p.AV_r, max(p.AV_c, 1e-30), p.omega_0)
            P_atr_n = (P_h_ven_dc if n == 0
                        else P_harm[n, node_idx[heart_node_ven]])
            P_ven_n = (P_h_art_dc if n == 0
                        else P_harm[n, node_idx[heart_node_art]])
            Q[n] = (0.5 * (Y_self + Y_trans)
                    * (P_atr_n - P_ven_n))
        edge_flows[(heart_node_ven, heart_node_art)] = Q
        loop_conduits['AV'] = [(heart_node_ven, heart_node_art)]

    if verbose:
        n_h = 2 if two_chamber else 1
        mode = (f"two-chamber peristaltic (τ/T = "
                f"{p.peristaltic_tau_frac:.3f})"
                if two_chamber else "single-chamber")
        print(f"  LinearOneChamber: {len(G.nodes())} plexus nodes "
              f"+ {n_h} H; {len(sources)} arterial, {len(sinks)} venous; "
              f"N_max={p.N_max} harmonics solved [{mode}].")

    return LinearLoopResult(
        params=p,
        f0_hz=p.omega_0 / (2 * np.pi),
        heart_node=heart_node_art,
        heart_node_ven=heart_node_ven,
        source_nodes=list(sources),
        sink_nodes=list(sinks),
        node_order=node_order,
        P_harm=P_harm,
        edge_flows=edge_flows,
        E_tilde=E_tilde,
        loop_conduits=loop_conduits,
        P_bar_h=float(P_bar_h),
        R_loop_dc=float(R_loop),
        Q_bar_emergent=float(Q_bar_emergent),
        P_h_art_dc=float(P_h_art_dc),
        P_h_ven_dc=float(P_h_ven_dc),
    )
