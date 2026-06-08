"""Read-only mosaic viewer — exploration only, no editing or compute.

Designed for two audiences:

  1. **Intern onboarding** — they can see the mosaic, color edges by any
     measured field, click an edge to inspect its Q(t) and metadata.  No
     buttons that change the graph; no compute that takes minutes.

  2. **Publication / data sharing** — a single self-contained Qt+napari
     app a reviewer can launch on a gpickle to explore the underlying
     measurements without learning the analysis pipeline.

Run from CLI:

    python -m pertile.viewer.mosaic_readonly_app <graph.gpickle> \
        [--tiff <stitched.tif>] [--initial-field mean_Q]

Or from Python:

    from pertile.viewer.mosaic_readonly_app import ReadOnlyMosaicViewer
    v = ReadOnlyMosaicViewer(graph_path, tiff_path=None,
                              initial_field='mean_Q')
    v.run()
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


# ── Defaults / constants ───────────────────────────────────────────────
# Default fields to expose in the dropdown (subset of what PIV stores).
# Order matters — first ones are most useful.
DEFAULT_FIELDS: List[str] = [
    'mean_Q', 'amp_Q', 'PI', 'phase', 'snr_pulse', 'snr_f0',
    'f0_hz', 'v_max',
    # geometry from edge attrs
    'radius', 'length',
    # viscous dissipation Φ = r·L·⟨Q²⟩ (W).  Available for both
    # measured (precomputed from PIV Q_t + geometry) and sim (from
    # `result.dissipation`).  Sim path goes via SIM_FIELD_MAP.
    'dissipation',
]
# Categorical field handled specially (discrete per-class colors):
CATEGORICAL_FIELDS = ['quality_tier', 'harmonic_class']
# Per-edge simulation outputs (in-memory only, written by
# `_run_simulation`, never persisted to the gpickle).  Stored on edges
# under the keys below; the Source toggle in the View tab decides
# whether the dropdown fields ('mean_Q', 'amp_Q', 'PI', 'phase') resolve
# to measured or sim values.  Q magnitudes are stored as |.| so they
# match the measured `mean_Q ≥ 0` convention (memory/methods.md).
# ── Field composition (4-selector model) ─────────────────────────────
# The View tab composes the colormap field from four selectors:
#   Source    — Measured / Simulated   (existing radio)
#   Quantity  — Q / P                  (P only valid in Simulated)
#   Property  — Magnitude / Phase / Resolution / Mean / PI / Dissipation
#               / Pressure drop / Radius / Length / Harmonic class
#   Harmonic  — DC / H1 / H2 / H3      (only for harmonic-keyed properties)
# Each property is tagged with the list of (source, quantity) tuples
# for which it is valid; `_combo_valid` enforces the matrix.
PROPERTY_DEFS = [
    # (key,           display label,          is_harmonic_keyed, valid_in)
    ('magnitude',     'Magnitude |·|',        True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P')}),
    ('phase',         'Phase ∠·',             True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P')}),
    ('resolution',    'Resolution (Z = amp/SE)', True,
        {('measured', 'Q')}),
    # `mean` dropped — redundant with `magnitude @ DC` for measured Q
    # and sim Q (both anchored to |Q̄| ≥ 0).  Sim P loses access to the
    # signed gauge DC, which is fine — pressure_drop carries the
    # meaningful absolute differential and gauge polarity isn't a daily
    # diagnostic.
    ('PI',            'Pulsatility index',    False,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P')}),
    # Heart-rate fundamental.  Measured-Q reads the top-level `f0_hz`
    # attr set during analysis (== best measurement's f0); sim reads
    # the solver f0 it ran with.  Useful for spotting tiles where the
    # fit locked onto an aliased peak or a slightly different rate.
    ('frequency',     'Fundamental f₀ (Hz)',  False,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P')}),
    # Per-vessel "is this signal-rich?" — Var(fit)/Var(resid) from the
    # K=3 harmonic fit on the kymograph Q(t).  Biased AGAINST near-
    # steady vessels (DC contributes 0 to Var(fit)) — high for arteries,
    # low for clean venous flow.  Distinct from "is this reliable?",
    # which is answered by Z_DC (available as Resolution @ DC).
    ('total_snr',     'Total SNR (Var(fit)/Var(resid))', False,
        {('measured', 'Q')}),
    ('dissipation',   'Dissipation Φ',        False,
        {('measured', 'Q'), ('sim', 'Q')}),
    ('drop',          'Pressure drop',        False,
        {('sim', 'P')}),
    ('radius',        'Radius (px)',          False,
        {('measured', 'Q'), ('measured', 'P'),
         ('sim', 'Q'), ('sim', 'P')}),
    ('length',        'Length (px)',          False,
        {('measured', 'Q'), ('measured', 'P'),
         ('sim', 'Q'), ('sim', 'P')}),
    ('harmonic_class','Resolved-harmonic class', False,
        {('measured', 'Q')}),
]
HARMONIC_KEYS = ['DC', 'H1', 'H2', 'H3']
HARMONIC_LABELS = {'DC': 'DC', 'H1': 'H₁', 'H2': 'H₂', 'H3': 'H₃'}
# Phase at DC is undefined; harmonic-class is already H1-resolved → skip.
def _safe_float(val) -> Optional[float]:
    """Coerce a candidate value to float, returning None for missing /
    non-numeric / non-finite inputs.  Used by the field resolver."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _combo_valid(source: str, quantity: str, prop_key: str,
                  harmonic: str) -> bool:
    pdef = next((p for p in PROPERTY_DEFS if p[0] == prop_key), None)
    if pdef is None:
        return False
    _, _, is_harmonic, valid_in = pdef
    if (source, quantity) not in valid_in:
        return False
    # Phase at DC is mathematically undefined; PI ("pulsatile / mean")
    # measures non-DC content, so attaching it to the DC slot is
    # conceptually empty.  Grey both at DC regardless of whether they
    # are harmonic-keyed in the property table.
    if harmonic == 'DC' and prop_key in ('phase', 'PI'):
        return False
    return True


SIM_FIELD_MAP: Dict[str, str] = {
    'mean_Q':          '_sim_tmp_mean_Q',
    'amp_Q':           '_sim_tmp_amp_Q',
    'PI':              '_sim_tmp_PI',
    'phase':           '_sim_tmp_phase_Q',
    'dissipation':     '_sim_tmp_dissipation', # from result.dissipation
    'pressure_mean':   '_sim_tmp_pressure_mean', # sim-only, P_mid,DC (Pa)
    'pressure_amp':    '_sim_tmp_pressure_amp',  # sim-only, |P_mid,H1| (Pa)
    'pressure_phase':  '_sim_tmp_phase_P',       # sim-only
    'pressure_drop':   '_sim_tmp_pressure_drop', # sim-only, |P_u_DC − P_v_DC| (Pa)
    # Geometry overrides — populated only when "Uniform conductance" is
    # checked at sim time, so radius/length fields show the override
    # value in sim mode.  Heterogeneous sim → keys absent → fall through
    # to the measured top-level attrs.
    'radius':          '_sim_tmp_radius',
    'length':          '_sim_tmp_length',
}
# Fields the comparison plot should treat as pressure (P-mode).
SIM_PRESSURE_FIELDS = {'pressure_mean', 'pressure_amp',
                       'pressure_phase'}
# Storage-only keys (not in any dropdown).
SIM_INTERNAL_KEYS: List[str] = [
    '_sim_tmp_harmonics',     # complex Q coeffs [DC, H1, H2, H3] per edge
    '_sim_tmp_p_harmonics',   # complex midpoint-P coeffs per edge
    '_sim_tmp_f0_hz',         # solver f0 for that edge
    # `_sim_tmp_radius` / `_sim_tmp_length` are listed in SIM_FIELD_MAP
    # (dropdown override), so they're already cleaned via that path.
]
# Discrete colour map for the resolved-harmonics class.
HARMONIC_CLASS_COLORS = {
    0: '#666666',   # DC only — no resolved pulsatile component
    1: '#1f77b4',   # DC + H1
    2: '#ffbf00',   # DC + H1 + H2
    3: '#d62728',   # DC + H1 + H2 + H3
}

# Raw video frame dimensions used to size tile boundaries on the mosaic.
# This is the imaging-pipeline default; if your videos have different
# native dimensions, pass them in or set them as class attrs.
TILE_RAW_HEIGHT = 704
TILE_RAW_WIDTH = 640

QUALITY_RANK = {'A': 3, 'B': 2, 'C': 1, 'X': 0}
TIER_COLORS = {'A': '#00ff00', 'B': '#ffff00', 'C': '#ff8800', 'X': '#888888'}
# Per-harmonic SNR threshold for "harmonic is real, not noise".
# Per-harmonic detection significance Z = |Q̂|/σ is the underlying
# reliability surface.  We cache four layers (Z_DC, Z_H1, Z_H2, Z_H3)
# per edge and threshold at *display time* — no precomputed tiers.
# Categories are just level sets of these layers and live in the
# colormap slider, never in the cache.
#
# DC and AC have different null distributions, so when you want a
# common threshold axis convert each layer to its p-value before
# comparing:
#   AC (Hk): Â/σ ~ Rayleigh        → P(>z) = exp(-z²/2)
#   DC:      |a₀|·√N/σ ~ folded-normal → P(>z) = erfc(z/√2)
HARMONIC_SNR_THRESHOLD = 3.0  # legacy figure-annotation reference; not used by gates.
# Cache-schema version for the `_h_*` per-edge attributes.  Bump
# whenever the meaning of cached values changes (e.g. σ std→MAD).
# An edge whose `_h_cache_ver` is missing or less than this constant
# is re-computed at startup.
HARMONIC_CACHE_VERSION = 2


def _harmonic_snrs(Q_t: np.ndarray, f0: float,
                    dt: float = 1.0/250) -> Optional[Dict[str, float]]:
    """Per-harmonic Z-statistics: Z = Â/SE.  σ from MAD (robust to
    motion-spike outliers), SE_amp = σ·√(2/N), SE_dc = σ/√N.

    Z is the raw ratio (Â, not the debiased amplitude) so the Rayleigh
    null interpretation applies: under H₀, Z ~ Rayleigh(σ=1) with
    E[Z] ≈ 1.25 and P(Z > z) = exp(−z²/2).  Threshold tiers:
        Z ≥ 4 excellent, ≥ 3 good, ≥ 2 marginal, < 2 unresolved.

    Returns dict {DC, H1, H2, H3, sigma, r2} or None on failure.
    `sigma` is the MAD-based estimate (not std).
    """
    try:
        from .harmonic import fit_harmonics  # type: ignore
    except ImportError:
        try:
            from ..analysis.harmonic import fit_harmonics
        except ImportError:
            return None
    try:
        hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                            loss='huber', include_dc=True)
    except Exception:
        return None
    resid = np.asarray(hr.get('resid', []), dtype=float)
    resid_ok = resid[np.isfinite(resid)] if resid.size else None
    if resid_ok is None or resid_ok.size == 0:
        return None
    med = float(np.median(resid_ok))
    sigma = 1.4826 * float(np.median(np.abs(resid_ok - med)))
    if not (np.isfinite(sigma) and sigma > 0):
        # MAD can collapse to 0 if >50% of residuals are exactly equal —
        # fall back to std so SE remains well-defined.
        sigma = float(np.std(resid_ok))
    N = len(Q_t)
    se_dc = sigma / np.sqrt(N) if N > 0 else np.inf
    se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else np.inf
    z_dc = abs(float(hr['a0'])) / max(se_dc, 1e-30)
    z = {1: 0.0, 2: 0.0, 3: 0.0}
    for h in hr['harmonics']:
        k = h.get('k')
        if k in z:
            z[k] = float(h['amp']) / max(se_amp, 1e-30)
    return {
        'DC': z_dc, 'H1': z[1], 'H2': z[2], 'H3': z[3],
        'sigma': sigma, 'r2': float(hr.get('r2', float('nan'))),
    }


def _harmonic_fit_full(Q_t: np.ndarray, f0: float,
                        dt: float = 1.0/250) -> Optional[Dict[str, float]]:
    """Like _harmonic_snrs but returns amplitudes + phases too.  σ from
    MAD (robust).  Adds `total_snr` = Var(fit)/Var(resid) — the
    per-vessel "how clean overall" scalar that's invariant to whether
    the waveform is H1-only or harmonic-rich.

    Returns dict with keys:
        amp_DC, Z_DC
        amp_H{k}, phase_H{k}, Z_H{k}   for k=1,2,3
        sigma, r2, total_snr
    """
    try:
        from ..analysis.harmonic import fit_harmonics
    except ImportError:
        return None
    try:
        hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                            loss='huber', include_dc=True)
    except Exception:
        return None
    resid = np.asarray(hr.get('resid', []), dtype=float)
    resid_ok = resid[np.isfinite(resid)] if resid.size else None
    if resid_ok is None or resid_ok.size == 0:
        return None
    med = float(np.median(resid_ok))
    sigma = 1.4826 * float(np.median(np.abs(resid_ok - med)))
    if not (np.isfinite(sigma) and sigma > 0):
        sigma = float(np.std(resid_ok))
    # Var(fit)/Var(resid) uses ordinary variance for both — robust σ
    # only enters via the SE denominators below.
    signal_fit = np.asarray(hr.get('signal', []), dtype=float)
    sig_ok = signal_fit[np.isfinite(signal_fit)] if signal_fit.size else None
    total_snr = float('nan')
    if sig_ok is not None and sig_ok.size and resid_ok.size:
        var_r = float(np.var(resid_ok))
        if var_r > 1e-30:
            total_snr = float(np.var(sig_ok) / var_r)
    N = len(Q_t)
    se_dc = sigma / np.sqrt(N) if N > 0 else np.inf
    se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else np.inf
    a0 = float(hr['a0'])
    out = {
        'amp_DC': a0,
        'Z_DC': abs(a0) / max(se_dc, 1e-30),
        'sigma': sigma,
        'r2': float(hr.get('r2', float('nan'))),
        'total_snr': total_snr,
    }
    for k in (1, 2, 3):
        out[f'amp_H{k}'] = 0.0
        out[f'phase_H{k}'] = 0.0
        out[f'Z_H{k}'] = 0.0
    for h in hr['harmonics']:
        k = h.get('k')
        if k in (1, 2, 3):
            out[f'amp_H{k}'] = float(h['amp'])
            out[f'phase_H{k}'] = float(h['phi'])
            out[f'Z_H{k}'] = float(h['amp']) / max(se_amp, 1e-30)
    return out


def _harmonic_class(snrs: Dict[str, float],
                     threshold: float = HARMONIC_SNR_THRESHOLD) -> int:
    """Highest k such that every Hk' for k' ≤ k passes the threshold.
    Returns 0 if even H1 fails (signal is essentially DC + noise)."""
    if snrs is None:
        return 0
    n = 0
    for k in (1, 2, 3):
        if snrs.get(f'H{k}', 0.0) >= threshold:
            n = k
        else:
            break
    return n


def _harmonic_class_label(n: int) -> str:
    """Human-readable resolved-harmonics label."""
    if n == 0: return 'DC only'
    if n == 1: return 'DC + H1'
    if n == 2: return 'DC + H1 + H2'
    return 'DC + H1 + H2 + H3'
# Color for edges with no usable measurement (missing PIV, gated/flagged,
# or a field value that's NaN/inf).  Dim enough to read as "no data,
# present for topology only" but visible against the dark mosaic image.
NO_DATA_COLOR = '#555555'


def _measurement_usable(m: Optional[dict]) -> bool:
    """True if a PIV measurement passed PIV's own gates (tier ≠ X) and
    has a valid fit.  This is what the user thinks of as a 'real
    measurement' vs 'garbage'."""
    if m is None:
        return False
    if not m.get('fit_success', True):
        return False
    if m.get('quality_tier', 'X') == 'X':
        return False
    return True


# ── Helpers ────────────────────────────────────────────────────────────
def _best_measurement(piv_list: Optional[list]) -> Optional[dict]:
    """Pick the highest-tier, highest-SNR measurement for one edge."""
    if not piv_list:
        return None
    candidates = [m for m in piv_list if m.get('fit_success', True)]
    if not candidates:
        return None
    return max(candidates,
               key=lambda m: (QUALITY_RANK.get(m.get('quality_tier', 'X'), 0),
                              m.get('snr_f0', 0) or 0))


def _resolve_edge_field(G, u, v, field: str,
                        source: str = 'measured') -> Optional[float]:
    """Return the scalar value of `field` for edge (u, v), or None.

    `source` selects which underlying data path is used.  For
    `source='sim'` the field name is mapped via SIM_FIELD_MAP to the
    in-memory `_sim_tmp_*` keys; geometry / categorical / non-mappable
    fields fall through to the measured-side logic regardless of source.
    """
    d = G.edges[u, v]
    if source == 'sim' and field in SIM_FIELD_MAP:
        val = d.get(SIM_FIELD_MAP[field])
        if val is not None:
            try:
                return float(val) if np.isfinite(val) else None
            except (TypeError, ValueError):
                return None
        # Fall through to the measured branch if the sim override is
        # absent (e.g. radius/length in heterogeneous sim mode).
    # Geometry, precomputed harmonic_class — top-level edge attrs.
    if field in ('radius', 'length', 'harmonic_class'):
        val = d.get(field)
        if val is None:
            return None
        try:
            return float(val) if np.isfinite(val) else None
        except (TypeError, ValueError):
            return None
    # Measured dissipation — precomputed once at load time and stored
    # in-memory as `_meas_dissipation` so the colormap lookup is O(1).
    if field == 'dissipation':
        val = d.get('_meas_dissipation')
        if val is None:
            return None
        try:
            return float(val) if np.isfinite(val) else None
        except (TypeError, ValueError):
            return None
    # All other fields come from the best PIV measurement
    m = _best_measurement(d.get('measurements_piv'))
    if m is None:
        return None
    if field in m:
        val = m[field]
        try:
            v_ = float(val)
            return v_ if np.isfinite(v_) else None
        except (TypeError, ValueError):
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# Viewer
# ══════════════════════════════════════════════════════════════════════
class ReadOnlyMosaicViewer:
    """napari-based read-only mosaic viewer with field coloring + edge
    inspection.  No buttons that modify state."""

    def __init__(
        self,
        graph_path: Path,
        tiff_path: Optional[Path] = None,
        tile_positions: Optional[Path] = None,
        video_dir: Optional[Path] = None,
        video_pattern: Optional[str] = None,
        initial_field: str = 'mean_Q',
        cache_harmonic_class: bool = True,
        force_recompute_harmonic_class: bool = False,
    ):
        self.graph_path = Path(graph_path)
        self.tiff_path = Path(tiff_path) if tiff_path else None
        self.tile_positions_path = Path(tile_positions) if tile_positions else None
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_pattern = video_pattern
        self.cache_harmonic_class = bool(cache_harmonic_class)
        self.force_recompute_harmonic_class = bool(force_recompute_harmonic_class)
        self.G = self._load_graph()
        self.mosaic = self._load_tiff() if self.tiff_path else None
        self.tiles = self._load_tile_positions() if self.tile_positions_path else {}
        self.edge_list: List[Tuple[int, int]] = list(self.G.edges())
        # Currently-overlaid video tile (napari layer + tile_id) — only
        # one at a time; clicking a new edge replaces the previous overlay.
        self._video_layer = None
        self._video_tile_id: Optional[int] = None
        # Precompute midpoints for nearest-edge lookup.
        self._edge_midpoints = np.array([
            [0.5 * (self.G.nodes[u]['x'] + self.G.nodes[v]['x']),
             0.5 * (self.G.nodes[u]['y'] + self.G.nodes[v]['y'])]
            for (u, v) in self.edge_list
        ])
        # Precompute per-edge `harmonic_class` (0/1/2/3 = number of
        # statistically resolved cardiac harmonics).  Stored on the
        # in-memory graph only — never written back to the gpickle.
        # Cheap enough to do once at startup (~30 s for ~6k edges).
        self._precompute_harmonic_classes()
        # Precompute measured viscous dissipation per edge from PIV
        # Q_t + geometry.  Stored as `_meas_dissipation` (W) on edge
        # attrs, in-memory only.
        self._precompute_measured_dissipation()
        # ── Four-selector field model ──────────────────────────────
        # Source already added below as `self.current_source`.
        # Quantity: 'Q' or 'P'.  P only valid in Simulated source.
        # Property: see PROPERTY_DEFS.  Some are harmonic-keyed.
        # Harmonic: 'DC' / 'H1' / 'H2' / 'H3'.  Ignored for non-keyed
        # properties.
        self.current_quantity: str = 'Q'
        self.current_property: str = 'magnitude'
        self.current_harmonic: str = 'DC'
        # Backwards-compat: `current_field` is now a synthesized string
        # so legacy checks like `'phase' in field` and
        # `field in SIM_PRESSURE_FIELDS` keep working.  See the
        # `current_field` @property below.  `self.fields` is no longer
        # needed (the selectors are static) but is kept as an empty
        # list to avoid breaking anything that iterates it.
        self.fields: List[str] = []
        # initial_field may still set the starting Property if it
        # matches a known key.
        if (initial_field
                and any(p[0] == initial_field for p in PROPERTY_DEFS)):
            self.current_property = initial_field
        self.log_scale = False
        self.show_nodes = False
        self._edges_layer = None
        self._nodes_layer = None
        self._click_marker_layer = None
        self._tile_boundaries_layer = None
        self._tile_labels_layer = None
        self.show_tile_boundaries = False
        self.show_tile_labels = False
        # Network-wide reliability filtering is intentionally out of the
        # UI for now — the cached per-harmonic Z layers (`_h_Z_DC`,
        # `_h_Z_HN`) are the right basis for a future continuous,
        # threshold-at-display-time filter.  See discussion thread for
        # the design (four-layer Z field, no precomputed tiers).
        # edge (u,v) → {'amp_DC', 'Z_DC', 'amp_H1', 'phase_H1', 'Z_H1', ...}
        # Populated for the currently-loaded tile only (either by the
        # "Browse tiles" video-load path, or eagerly by the tile filter
        # combo).  Used both by the time-series inspect panel for
        # cross-edge-comparable phase data, and by the tile filter to
        # restrict measured-Q colormap fields to one tile's values.
        self._tile_harmonic_cache: Dict[Tuple[int, int], dict] = {}
        # Tile filter state — None ⇒ "All tiles", colour by best
        # measurement across tiles (default).  When set to an int,
        # measured-Q fields read per-tile values from
        # `_tile_harmonic_cache`; edges with no measurement on the
        # filtered tile render as grey.  Doesn't affect sim or geometry.
        self.current_tile_filter: Optional[int] = None
        # Percentile threshold on per-edge total_snr (Var(fit)/Var(resid)).
        # Hides the bottom X % of edges by network-wide `_h_total_snr`.
        # The percentile is always computed from the graph-wide cached
        # best-of-edge values — independent of whether a tile filter is
        # active.  0 = no filtering, default.
        self.total_snr_pct_filter: float = 0.0
        # Cached per-refresh threshold so we compute the percentile
        # once per call to `_refresh_edges` instead of per edge.
        self._cached_total_snr_threshold: Optional[float] = None
        # Edges queued for the side-by-side comparison plot.  Each
        # entry is (u, v); duplicates ignored.  Cleared via the
        # "Clear comparison" button or when the user explicitly resets.
        self._comparison_edges: List[Tuple[int, int]] = []
        self._comparison_marker_layer = None
        self._comparison_edge_overlay_layer = None
        # Boundary nodes — arterial sources ('source') and venous sinks
        # ('sink').  Identified from the `boundary_type` node attribute
        # set by the analysis pipeline; both have a complex
        # `bc_harmonics` array (DC + H1 + H2 + H3).
        self._source_nodes: List[int] = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'source']
        self._sink_nodes: List[int] = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'sink']
        self._boundary_nodes = self._source_nodes + self._sink_nodes
        self._source_layer = None
        self._sink_layer = None
        print(f"  boundary: {len(self._source_nodes)} sources (A), "
              f"{len(self._sink_nodes)} sinks (V)")
        # Simulation state (forward transmission-line solve).  Results
        # live IN-MEMORY only — never persisted to the gpickle.  Keys
        # use the `_sim_tmp_` prefix so they don't collide with the
        # editing-app's stored `_sim` fields.
        self._sim_active: bool = False
        self._sim_last_D: float = float('nan')
        self._sim_last_f0_hz: float = float('nan')
        # Data source for the View tab field colormap + the comparison
        # plot.  'measured' uses PIV results from `measurements_piv`;
        # 'sim' uses the in-memory `_sim_tmp_*` outputs.  The Source
        # radio in the View tab drives this; 'sim' is disabled until
        # `_sim_active`.
        self.current_source: str = 'measured'
        # Mosaic height — used to invert node y so points align with the
        # image layer (mosaic uses display-y = row, graph stores math-y).
        # Falls back to max graph-y if no TIFF is loaded.
        if self.mosaic is not None:
            self.mosaic_height = int(self.mosaic.shape[0])
        else:
            ys = [self.G.nodes[n].get('y', 0) for n in self.G.nodes()]
            self.mosaic_height = int(max(ys) + 1) if ys else 0
        # Per-node "best" measured value for the currently-selected field;
        # populated by `_refresh_edges` (which already iterates edges) so
        # nodes can be colored by the mean of their incident edges.
        self._node_field_values: Dict[int, float] = {}

    # ── data loading ───────────────────────────────────────────────────
    # ── Tile-filtered measured-Q resolver ─────────────────────────────
    def _resolve_tile_filtered_value(self, u, v, d: dict,
                                       prop: str) -> Optional[float]:
        """Read the per-tile equivalent of the requested measured-Q
        property from `_tile_harmonic_cache` (populated for whichever
        tile is currently selected as the filter).  Returns None when
        the edge has no measurement on that tile, when the property
        isn't well-defined per-tile (dissipation has no clean per-tile
        equivalent), or when the harmonic fit failed for that
        measurement.
        """
        # Geometry doesn't change per tile.
        if prop == 'radius':
            return _safe_float(d.get('radius'))
        if prop == 'length':
            return _safe_float(d.get('length'))
        # Frequency — pulled directly from the matching tile's
        # measurement (the per-tile harmonic cache reused this same
        # f0 during the fit, but doesn't store it back, so it's
        # cheaper to read the source).
        if prop == 'frequency':
            tile_id = self.current_tile_filter
            for mm in (d.get('measurements_piv') or []):
                if mm.get('tile_id') == tile_id:
                    return _safe_float(mm.get('f0_hz'))
            return None

        entry = self._tile_harmonic_cache.get((u, v))
        if entry is None:
            return None  # no measurement on the filtered tile for this edge

        if prop == 'PI':
            a0 = abs(float(entry.get('amp_DC', 0.0) or 0.0))
            amp1 = float(entry.get('amp_H1', 0.0) or 0.0)
            if a0 < 1e-15:
                return None
            return 2.0 * amp1 / a0
        if prop == 'total_snr':
            return _safe_float(entry.get('total_snr'))
        if prop == 'harmonic_class':
            snrs = {'DC': float(entry.get('Z_DC', 0.0) or 0.0),
                    'H1': float(entry.get('Z_H1', 0.0) or 0.0),
                    'H2': float(entry.get('Z_H2', 0.0) or 0.0),
                    'H3': float(entry.get('Z_H3', 0.0) or 0.0)}
            return _safe_float(_harmonic_class(snrs))
        if prop == 'dissipation':
            # `_meas_dissipation` is a graph-level aggregate from PIV
            # processing; no clean per-tile equivalent exists.  Return
            # None so the field reads as "not available at this tile".
            return None

        # Harmonic-keyed properties — keys in the per-tile cache match
        # the layout produced by `_harmonic_fit_full`.
        k_idx = HARMONIC_KEYS.index(self.current_harmonic)
        if prop == 'magnitude':
            if k_idx == 0:
                v_ = entry.get('amp_DC')
                return abs(_safe_float(v_)) if v_ is not None else None
            return _safe_float(entry.get(f'amp_H{k_idx}'))
        if prop == 'phase':
            if k_idx == 0:
                return None  # phase at DC undefined
            return _safe_float(entry.get(f'phase_H{k_idx}'))
        if prop == 'resolution':
            if k_idx == 0:
                return _safe_float(entry.get('Z_DC'))
            return _safe_float(entry.get(f'Z_H{k_idx}'))
        return None

    # ── Field resolver (4-selector model) ─────────────────────────────
    def _resolve_field_value(self, u, v) -> Optional[float]:
        """Return the per-edge scalar for the current selector state,
        or None.  Centralised lookup used by both the colormap and the
        node-averaging pass.

        Routing matrix:
          aggregate properties  → top-level edge attr lookup
          harmonic-keyed (measured) → cached `_h_amp/phase/Z_H{k}` etc.
          harmonic-keyed (sim, Q)   → derived from `_sim_tmp_harmonics`
          harmonic-keyed (sim, P)   → derived from `_sim_tmp_p_harmonics`
        Invalid combos return None (rendered grey).
        """
        if not self.G.has_edge(u, v):
            return None
        if not _combo_valid(self.current_source, self.current_quantity,
                             self.current_property,
                             self.current_harmonic):
            return None
        d = self.G.edges[u, v]
        prop = self.current_property
        src = self.current_source
        qty = self.current_quantity

        # Tile-filter override: when a tile is selected, every measured-Q
        # value is taken from THAT tile's harmonic fit instead of the
        # best-across-tiles graph attrs.  Edges with no measurement on
        # the filtered tile fall through to None (rendered grey).
        if (src == 'measured' and qty == 'Q'
                and self.current_tile_filter is not None):
            return self._resolve_tile_filtered_value(u, v, d, prop)

        # ── Aggregate (non-harmonic-keyed) properties ──
        # 'mean' was dropped — use Magnitude @ DC for the |Q̄| / |P̄|
        # equivalent.
        if prop == 'PI':
            if src == 'measured' and qty == 'Q':
                m = _best_measurement(d.get('measurements_piv'))
                if m is None:
                    return None
                try:
                    val = float(m.get('PI', float('nan')))
                except (TypeError, ValueError):
                    return None
                return val if np.isfinite(val) else None
            if src == 'sim' and qty == 'Q':
                return _safe_float(d.get('_sim_tmp_PI'))
            if src == 'sim' and qty == 'P':
                p = d.get('_sim_tmp_p_harmonics')
                if p is None or len(p) < 2:
                    return None
                p_dc = abs(float(p[0].real))
                if not np.isfinite(p_dc) or p_dc < 1e-12:
                    return None
                amp1 = float(abs(p[1]))
                if not np.isfinite(amp1):
                    return None
                return 2.0 * amp1 / p_dc
            return None
        if prop == 'dissipation':
            if src == 'measured':
                return _safe_float(d.get('_meas_dissipation'))
            if src == 'sim':
                return _safe_float(d.get('_sim_tmp_dissipation'))
            return None
        if prop == 'drop':
            return _safe_float(d.get('_sim_tmp_pressure_drop'))
        if prop == 'radius':
            # sim-mode override takes precedence (uniform-R / uniform-C).
            if src == 'sim':
                v_ = d.get('_sim_tmp_radius')
                if v_ is not None:
                    return _safe_float(v_)
            return _safe_float(d.get('radius'))
        if prop == 'length':
            if src == 'sim':
                v_ = d.get('_sim_tmp_length')
                if v_ is not None:
                    return _safe_float(v_)
            return _safe_float(d.get('length'))
        if prop == 'harmonic_class':
            return _safe_float(d.get('harmonic_class'))
        if prop == 'total_snr':
            # Cached during _precompute_harmonic_classes; only present
            # for measured Q edges that had a usable Q_t to fit.
            return _safe_float(d.get('_h_total_snr'))
        if prop == 'frequency':
            if src == 'sim':
                return _safe_float(d.get('_sim_tmp_f0_hz'))
            # Measured: top-level edge attr is set during analysis from
            # the best measurement; fall back to scanning the per-tile
            # measurement list if it isn't there.
            v_ = d.get('f0_hz')
            if v_ is None:
                m = _best_measurement(d.get('measurements_piv'))
                if m is not None:
                    v_ = m.get('f0_hz')
            return _safe_float(v_)

        # ── Harmonic-keyed (magnitude / phase / resolution) ──
        k_idx = HARMONIC_KEYS.index(self.current_harmonic)
        if prop == 'magnitude':
            if src == 'measured' and qty == 'Q':
                if k_idx == 0:
                    v_ = d.get('_h_amp_DC')
                    if v_ is None:
                        m = _best_measurement(d.get('measurements_piv'))
                        if m is None:
                            return None
                        try:
                            return abs(float(m.get('mean_Q',
                                                    float('nan'))))
                        except (TypeError, ValueError):
                            return None
                    try:
                        return abs(float(v_))
                    except (TypeError, ValueError):
                        return None
                return _safe_float(d.get(f'_h_amp_H{k_idx}'))
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                if k_idx == 0:
                    return abs(float(h[0].real))
                return float(abs(h[k_idx]))
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                if k_idx == 0:
                    return abs(float(h[0].real))
                return float(abs(h[k_idx]))
            return None
        if prop == 'phase':
            if k_idx == 0:
                return None  # phase at DC undefined
            if src == 'measured' and qty == 'Q':
                return _safe_float(d.get(f'_h_phase_H{k_idx}'))
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                return float(np.angle(h[k_idx]))
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                return float(np.angle(h[k_idx]))
            return None
        if prop == 'resolution':
            if src == 'measured' and qty == 'Q':
                if k_idx == 0:
                    return _safe_float(d.get('_h_Z_DC'))
                return _safe_float(d.get(f'_h_Z_H{k_idx}'))
            return None
        return None

    # ── Synthesized "current_field" name ──────────────────────────────
    @property
    def current_field(self) -> str:
        """Synthesized canonical key from the 4 selectors.

        Format: ``<quantity>.<property>[@<harmonic>]`` (e.g.
        ``Q.magnitude@H2`` / ``P.mean`` / ``Q.PI``).  Legacy code paths
        check substrings like ``'phase' in field`` — these stay valid
        because the property key appears verbatim in the string.
        """
        parts = [self.current_quantity, self.current_property]
        prop = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        if prop is not None and prop[2]:
            return '.'.join(parts) + '@' + self.current_harmonic
        return '.'.join(parts)

    @current_field.setter
    def current_field(self, _val):
        """Setter kept as a no-op for legacy compatibility.  The actual
        selection is owned by the four selector state variables."""
        return

    def _current_field_label(self) -> str:
        """Human-readable label for the colorbar."""
        pdef = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        prop_lbl = pdef[1] if pdef else self.current_property
        qsym = self.current_quantity
        suffix = (f"  ({qsym}, {HARMONIC_LABELS[self.current_harmonic]})"
                  if (pdef is not None and pdef[2])
                  else f"  ({qsym})")
        return prop_lbl + suffix

    def _load_graph(self):
        print(f"Loading graph: {self.graph_path}")
        with open(self.graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _load_tiff(self):
        try:
            import tifffile
            print(f"Loading TIFF: {self.tiff_path}")
            img = tifffile.imread(self.tiff_path)
            print(f"  shape={img.shape}, dtype={img.dtype}")
            return img
        except Exception as e:
            print(f"  Could not load TIFF ({e}); proceeding without it.")
            return None

    def _load_tile_positions(self) -> Dict[int, dict]:
        """Parse tile_positions_manual.json → {tile_id: {translate_x,
        translate_y, scale_x, scale_y}}.  Also computes the global tile
        offset (min translate_x/y across all tiles) so video tiles can
        be placed in image-row coordinates: row = translate_y − offset_y.
        """
        try:
            with open(self.tile_positions_path) as f:
                tp = json.load(f)
        except Exception as e:
            print(f"  Could not load tile positions ({e}); video overlay disabled.")
            return {}
        tiles: Dict[int, dict] = {}
        for vid_str, entry in tp.get('tiles', {}).items():
            try:
                vid = int(vid_str)
            except ValueError:
                continue
            tiles[vid] = {
                'translate_x': float(entry.get('translate_x', 0.0)),
                'translate_y': float(entry.get('translate_y', 0.0)),
                'scale_x':     float(entry.get('scale_x', 1.0)),
                'scale_y':     float(entry.get('scale_y', 1.0)),
            }
        # Global tile offsets — match the editing viewer's convention
        # (`_core.py`: `top_left_y - self._tile_offset_y`).  Without
        # this shift, negative translate_y values would push videos
        # outside the mosaic image.
        if tiles:
            self._tile_offset_x = min(t['translate_x'] for t in tiles.values())
            self._tile_offset_y = min(t['translate_y'] for t in tiles.values())
        else:
            self._tile_offset_x = 0.0
            self._tile_offset_y = 0.0
        print(f"  Loaded positions for {len(tiles)} tiles "
              f"from {self.tile_positions_path.name} "
              f"(offset={self._tile_offset_x:.0f}, {self._tile_offset_y:.0f})")
        return tiles

    def _get_video_path_for(self, tile_id: int) -> Optional[Path]:
        """Resolve the directory or file containing the raw frames for `tile_id`."""
        if self.video_dir is None or self.video_pattern is None:
            return None
        name = self.video_pattern.format(vid=tile_id)
        candidate = self.video_dir / name
        if candidate.exists():
            return candidate
        # Some configs use `loc{vid}_C...` directories; fall back to a glob.
        matches = list(self.video_dir.glob(f"*loc{tile_id}_*"))
        if matches:
            return matches[0]
        return None

    def _remove_video_overlay(self):
        if self._video_layer is not None:
            # Stop the dims animation BEFORE removing the layer so
            # napari's AnimationThread doesn't keep a dangling reference
            # to the now-destroyed QtDimSliderWidget — the dangling
            # reference causes a TypeError on next .play() when it tries
            # to disconnect from the dead Qt object.
            try:
                self.viewer.window._qt_viewer.dims.stop()
            except Exception:
                pass
            try:
                self.viewer.layers.remove(self._video_layer)
            except Exception:
                pass
            self._video_layer = None
            self._video_tile_id = None
        # Per-tile harmonic cache is tile-local; without a tile, drop it.
        self._tile_harmonic_cache.clear()

    def _populate_tile_harmonic_cache(self, tile_id: int):
        """Compute per-tile harmonic fits for every edge that has a
        measurement on `tile_id`.  Uses THAT tile's measurement (not
        the best-quality one) so the f0 is consistent everywhere in
        the view."""
        self._tile_harmonic_cache.clear()
        n = 0
        for u, v, d in self.G.edges(data=True):
            meas = d.get('measurements_piv') or []
            m_tile = next((mm for mm in meas
                           if mm.get('tile_id') == tile_id), None)
            if m_tile is None or m_tile.get('Q_t') is None:
                continue
            if not _measurement_usable(m_tile):
                continue
            Q_t = np.asarray(m_tile['Q_t'], dtype=float)
            if Q_t.size < 30:
                continue
            full = _harmonic_fit_full(
                Q_t, f0=float(m_tile.get('f0_hz', 2.5)), dt=1.0/250)
            if full is None:
                continue
            self._tile_harmonic_cache[(u, v)] = full
            n += 1
        print(f"  Computed per-tile harmonics for {n} edges in tile "
              f"{tile_id} (f0 anchored at tile-{tile_id}'s value).")

    def _load_video_overlay(self, tile_id: int):
        """Load the per-tile video and add it as a napari 3D Image layer
        translated/scaled onto the mosaic so the embryo lines up with
        the graph.  Replaces any prior overlay (only one at a time)."""
        if tile_id == self._video_tile_id:
            return  # already showing
        self._remove_video_overlay()
        if tile_id not in self.tiles:
            print(f"  No tile_positions entry for tile {tile_id}; "
                  f"can't place the video on the mosaic.")
            return
        video_path = self._get_video_path_for(tile_id)
        if video_path is None:
            print(f"  No video found for tile {tile_id} "
                  f"(looked under {self.video_dir} with pattern "
                  f"{self.video_pattern!r}).")
            return
        from pertile.io.tiff import load_tiff_stack, cut_before_fade
        print(f"  Loading video for tile {tile_id}: {video_path.name}")
        stack = load_tiff_stack(video_path, max_frames=600)
        stack, cut_frame = cut_before_fade(stack, verbose=False)
        # Normalise to uint8 with 1%/99% contrast — keeps GPU upload cheap
        # and gives a sensible default contrast.
        if stack.dtype != np.uint8:
            sample = stack[::max(1, stack.shape[0] // 20)]
            lo, hi = np.percentile(sample, (1.0, 99.0))
            if hi <= lo:
                hi = lo + 1.0
            stack = np.clip(
                (stack.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                0, 255).astype(np.uint8)
        # Compute the effective scale factor between the loaded video's
        # frame size and the canonical raw frame size.  Full-res
        # acquisitions are TILE_RAW_HEIGHT × TILE_RAW_WIDTH; bundles
        # that ship pre-downsampled videos will arrive smaller.  This
        # lets a single overlay path handle both the full-res and the
        # 2×-downsampled bundles correctly — without this detection
        # the bundle videos sit in the top-left quarter of their
        # intended footprint because the viewer's internal `[::2,::2]`
        # below would compound the bundle's downsample.
        H_in, W_in = stack.shape[1], stack.shape[2]
        ds_factor_y = TILE_RAW_HEIGHT / float(H_in) if H_in else 1.0
        ds_factor_x = TILE_RAW_WIDTH / float(W_in) if W_in else 1.0
        # Only run the internal GPU-memory downsample if the input is
        # close to full resolution (within ~30%).  Below that we trust
        # the bundle's downsampling and skip the slicing — the loaded
        # video is already small enough.
        if ds_factor_x < 1.5 and ds_factor_y < 1.5:
            stack = np.ascontiguousarray(stack[:, ::2, ::2])
            ds_factor_y *= 2.0
            ds_factor_x *= 2.0
        T, H, W = stack.shape
        # tile_width/height come from the mosaic_graph (we don't have a
        # mosaic-image reference if the user didn't pass --tiff).  Use
        # scale_x/scale_y as a multiplicative correction relative to the
        # raw video pixels.  The placement convention matches
        # `_tile_mgmt._load_tile_video`.
        tile_entry = self.tiles[tile_id]
        scale_y = tile_entry['scale_y'] * ds_factor_y
        scale_x = tile_entry['scale_x'] * ds_factor_x
        print(f"  video {H_in}×{W_in} → display scale "
              f"({ds_factor_y:.2f}, {ds_factor_x:.2f})  "
              f"× tile_pos ({tile_entry['scale_y']:.3f}, "
              f"{tile_entry['scale_x']:.3f})")
        # Tile translate values are in a "relative-to-tile-1" frame and
        # can be negative (e.g. −1147 at stage 21).  Convert to image-
        # row coords by subtracting the global min offset, matching the
        # editing viewer's convention (`_core.py: top_left_y -
        # self._tile_offset_y`).  This is a different transformation
        # than the `mosaic_height − y` flip applied to graph nodes/edges
        # because the two pipelines established their conventions
        # independently.
        translate_y_display = (tile_entry['translate_y']
                                - self._tile_offset_y)
        translate_x = tile_entry['translate_x'] - self._tile_offset_x
        self._video_layer = self.viewer.add_image(
            stack,
            name=f"Video tile {tile_id}",
            colormap='gray',
            contrast_limits=(0, 255),
            interpolation2d='nearest',
            translate=(0, translate_y_display, translate_x),
            scale=(1, scale_y, scale_x),
            opacity=0.85,
            blending='translucent',
        )
        self._video_tile_id = tile_id
        # Populate per-tile harmonic cache (used by the time-series
        # inspect panel for cross-edge-comparable phase at the loaded
        # tile's f0).  Fast: typically 100–500 edges × ~5 ms each.
        self._populate_tile_harmonic_cache(tile_id)
        # Start at frame 0 and try to autoplay.  napari's AnimationThread
        # can raise TypeError on the first .play() after a layer swap
        # (it tries to disconnect signals from a destroyed previous
        # slider); the disconnect failure is non-fatal — the new
        # playback still starts on a retry.
        self.viewer.dims.set_point(0, 0)
        for _attempt in range(2):
            try:
                self.viewer.window._qt_viewer.dims.play(axis=0, fps=30)
                break
            except TypeError:
                # Force-stop and retry once.
                try:
                    self.viewer.window._qt_viewer.dims.stop()
                except Exception:
                    pass
            except Exception:
                break
        cut_note = f" (truncated at frame {cut_frame})" if cut_frame else ""
        print(f"  → {T} frames at "
              f"({translate_y_display:.0f}, {translate_x:.0f}), "
              f"scale=({scale_y:.3f}, {scale_x:.3f}){cut_note}")

    def _precompute_measured_dissipation(self):
        """Compute viscous dissipation per edge from PIV Q_t + geometry.

        Φ = r · L · ⟨Q²⟩  (Watts)
        with r = 8μ / (πR⁴), L from edge attr `length`, R from `radius`,
        and ⟨Q²⟩ = mean(Q_t²) using the best PIV measurement.  Result
        stored as `_meas_dissipation` on each edge (in-memory only).
        Edges without a usable measurement / geometry get nan.
        """
        from ..analysis.config import PX_SIZE_UM
        from ..analysis.transmission_line import MU_DEFAULT
        px_to_m = PX_SIZE_UM * 1e-6
        nl_to_m3 = 1e-12  # 1 nL = 1e-12 m³  → Q [nL/s] × 1e-12 = Q [m³/s]
        n = 0
        for u, v, d in self.G.edges(data=True):
            if '_meas_dissipation' in d:
                continue
            m = _best_measurement(d.get('measurements_piv'))
            R_px = d.get('radius')
            L_px = d.get('length')
            if (m is None or m.get('Q_t') is None
                    or R_px is None or L_px is None):
                d['_meas_dissipation'] = float('nan')
                continue
            try:
                R_px_f = float(R_px); L_px_f = float(L_px)
            except (TypeError, ValueError):
                d['_meas_dissipation'] = float('nan')
                continue
            if not (np.isfinite(R_px_f) and R_px_f > 0
                    and np.isfinite(L_px_f) and L_px_f > 0):
                d['_meas_dissipation'] = float('nan')
                continue
            R_m = R_px_f * px_to_m
            L_m = L_px_f * px_to_m
            r = 8.0 * MU_DEFAULT / (np.pi * R_m ** 4)
            Q_t = np.asarray(m['Q_t'], dtype=float) * nl_to_m3
            Q2 = float(np.nanmean(Q_t ** 2))
            d['_meas_dissipation'] = r * L_m * Q2
            n += 1
        print(f"  Precomputed dissipation for {n} edges "
              f"(Φ in W; range typical 1e-15 to 1e-12)")

    def _precompute_harmonic_classes(self):
        """Run one harmonic fit per edge (DC + H1 + H2 + H3) and cache
        everything needed for the per-harmonic field views:

          • `harmonic_class`   — int 0/1/2/3 or None
          • `_h_amp_DC`, `_h_Z_DC`
          • `_h_amp_H1`, `_h_phase_H1`, `_h_Z_H1`
          • `_h_amp_H2`, `_h_phase_H2`, `_h_Z_H2`
          • `_h_amp_H3`, `_h_phase_H3`, `_h_Z_H3`
          • `_h_r2`, `_h_sigma`

        The `_h_` prefix marks them as viewer-cached derived attributes.
        Cached back to the gpickle if `cache_harmonic_class` is True so
        relaunches are instant.

        Skip rules:
          - If `harmonic_class` is already present on every edge and
            the user didn't pass `--force-recompute-harmonic-class`,
            do nothing.
          - If `--force-recompute-harmonic-class`, wipe existing cache
            first and recompute from scratch.
        """
        import time
        CACHED_KEYS = (
            'harmonic_class',
            '_h_amp_DC', '_h_Z_DC',
            '_h_amp_H1', '_h_phase_H1', '_h_Z_H1',
            '_h_amp_H2', '_h_phase_H2', '_h_Z_H2',
            '_h_amp_H3', '_h_phase_H3', '_h_Z_H3',
            '_h_r2', '_h_sigma',
            '_h_total_snr', '_h_cache_ver',
        )
        if self.force_recompute_harmonic_class:
            for _, _, d in self.G.edges(data=True):
                for k in CACHED_KEYS:
                    d.pop(k, None)
        # Skip the recompute only if EVERY edge has the full `_h_*` cache
        # AT THE CURRENT SCHEMA VERSION.  Bumping HARMONIC_CACHE_VERSION
        # invalidates everything stored under the old convention (e.g.
        # σ definition changed std → MAD) without needing the user to
        # pass --force-recompute-harmonic-class.
        n_with_class = 0
        n_with_h1_amp = 0
        n_with_cur_ver = 0
        for _, _, d in self.G.edges(data=True):
            if 'harmonic_class' in d:
                n_with_class += 1
            if '_h_amp_H1' in d:
                n_with_h1_amp += 1
            if d.get('_h_cache_ver', 0) >= HARMONIC_CACHE_VERSION:
                n_with_cur_ver += 1
        n_edges = self.G.number_of_edges()
        if (n_with_class == n_edges
                and n_with_h1_amp == n_edges
                and n_with_cur_ver == n_edges):
            print(f"  harmonic_class + `_h_*` cache present on all "
                  f"{n_with_class} edges at version "
                  f"{HARMONIC_CACHE_VERSION}; skipping precompute.")
            return
        if n_with_class == n_edges and n_with_cur_ver < n_edges:
            print(f"  harmonic cache present but at older schema "
                  f"version (have {n_with_cur_ver}/{n_edges} at v"
                  f"{HARMONIC_CACHE_VERSION}); recomputing with new "
                  f"MAD-σ + total_snr convention.")
            # Wipe stale entries so the loop below actually re-fits
            # them (otherwise the `_h_amp_H1 in d` skip would fire).
            for _, _, d in self.G.edges(data=True):
                if d.get('_h_cache_ver', 0) < HARMONIC_CACHE_VERSION:
                    for k in CACHED_KEYS:
                        d.pop(k, None)
        elif n_with_class == n_edges:
            print(f"  harmonic_class present on all {n_with_class} edges "
                  f"but `_h_*` cache missing on {n_edges - n_with_h1_amp}; "
                  f"running full precompute …")
        n_total = self.G.number_of_edges()
        n_resolved = 0
        n_failed = 0
        t0 = time.time()
        print(f"  Precomputing harmonic_class for {n_total} edges ...")
        for i, (u, v, d) in enumerate(self.G.edges(data=True)):
            # Skip only if BOTH the class label AND the extended
            # `_h_*` cache are present — otherwise we re-fit so the
            # 4-selector resolver can read amp/phase/Z at H₁/H₂/H₃.
            if 'harmonic_class' in d and '_h_amp_H1' in d:
                continue
            m = _best_measurement(d.get('measurements_piv'))
            if m is None or not _measurement_usable(m) \
                    or m.get('Q_t') is None:
                d['harmonic_class'] = None
                continue
            Q_t = np.asarray(m['Q_t'], dtype=float)
            if Q_t.size < 30:
                d['harmonic_class'] = None
                continue
            full = _harmonic_fit_full(
                Q_t, f0=float(m.get('f0_hz', 2.5)), dt=1.0/250)
            if full is None:
                d['harmonic_class'] = None
                n_failed += 1
                continue
            # Pack Z dict for the classification helper.
            snrs = {'DC': full['Z_DC'], 'H1': full['Z_H1'],
                    'H2': full['Z_H2'], 'H3': full['Z_H3']}
            d['harmonic_class'] = int(_harmonic_class(snrs))
            d['_h_amp_DC'] = full['amp_DC']
            d['_h_Z_DC'] = full['Z_DC']
            for k in (1, 2, 3):
                d[f'_h_amp_H{k}'] = full[f'amp_H{k}']
                d[f'_h_phase_H{k}'] = full[f'phase_H{k}']
                d[f'_h_Z_H{k}'] = full[f'Z_H{k}']
            d['_h_r2'] = full['r2']
            d['_h_sigma'] = full['sigma']
            d['_h_total_snr'] = full.get('total_snr', float('nan'))
            d['_h_cache_ver'] = HARMONIC_CACHE_VERSION
            n_resolved += 1
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (n_total - i - 1) / max(rate, 1e-3)
                print(f"    {i + 1}/{n_total} edges  "
                      f"({rate:.0f}/s, ~{eta:.0f}s remaining)")
        elapsed = time.time() - t0
        # Tally distribution
        counts = {0: 0, 1: 0, 2: 0, 3: 0, None: 0}
        for _, _, d in self.G.edges(data=True):
            counts[d.get('harmonic_class')] = (
                counts.get(d.get('harmonic_class'), 0) + 1)
        print(f"  Done in {elapsed:.1f} s.  "
              f"Class distribution: "
              f"DC-only={counts[0]}, +H1={counts[1]}, "
              f"+H1+H2={counts[2]}, +H1+H2+H3={counts[3]}, "
              f"no-data={counts[None]}")
        # Persist back to the gpickle so the next launch skips compute.
        if self.cache_harmonic_class:
            self._save_graph_with_harmonic_class()

    def _save_graph_with_harmonic_class(self):
        """Atomically write self.G back to self.graph_path so the
        harmonic_class attributes survive across viewer launches.

        Atomic = pickle to a temp file in the same directory, then
        os.replace() onto the target.  This guarantees a partial write
        can never leave the gpickle in a corrupt state.

        Only `harmonic_class` is ever added by this viewer; every other
        attribute on G is left as it was loaded.  No edges are added or
        removed.
        """
        import os, pickle, tempfile
        target = Path(self.graph_path).resolve()
        if not target.parent.is_dir():
            print(f"  Save skipped: parent dir {target.parent} not found.")
            return
        try:
            fd, tmp = tempfile.mkstemp(
                suffix='.harm_save.tmp', dir=str(target.parent),
                prefix='.' + target.stem + '.')
            try:
                with os.fdopen(fd, 'wb') as f:
                    pickle.dump(self.G, f,
                                 protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, target)
                size_mb = target.stat().st_size / (1024 ** 2)
                print(f"  Cached harmonic_class to {target.name} "
                      f"({size_mb:.0f} MB).  "
                      f"Disable with --no-cache-harmonic-class.")
            except Exception:
                try: os.remove(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  Warning: could not write cache: {e}  "
                  f"(compute will repeat next launch)")

    def _discover_fields(self) -> List[str]:
        """Keep only the fields actually present in this graph's data."""
        found: List[str] = []
        # Top-level edge attributes (geometry + precomputed harmonic_class).
        for f in ('radius', 'length', 'harmonic_class'):
            for _, _, d in self.G.edges(data=True):
                if f in d:
                    found.append(f); break
        # Dissipation lives under `_meas_dissipation` (precomputed at
        # load time); surface it as the user-facing `dissipation` field.
        for _, _, d in self.G.edges(data=True):
            if '_meas_dissipation' in d:
                found.append('dissipation'); break
        # PIV-side fields
        for f in DEFAULT_FIELDS:
            if f in found:
                continue
            for _, _, d in self.G.edges(data=True):
                piv = d.get('measurements_piv') or []
                if any(f in m for m in piv):
                    found.append(f); break
        # Categorical PIV fields (quality_tier).  harmonic_class is also
        # categorical but lives top-level, so it's already in `found`.
        for f in CATEGORICAL_FIELDS:
            if f in found:
                continue
            for _, _, d in self.G.edges(data=True):
                piv = d.get('measurements_piv') or []
                if any(f in m for m in piv):
                    found.append(f); break
        # When a sim has been run, expose the sim-only fields — these
        # have no measured equivalent.  Other sim outputs (mean_Q,
        # amp_Q, PI, phase) share names with measured and route via
        # the Source toggle, not the dropdown.
        if getattr(self, '_sim_active', False):
            sim_only_checks = [
                ('pressure_mean',  '_sim_tmp_pressure_mean'),
                ('pressure_amp',   '_sim_tmp_pressure_amp'),
                ('pressure_phase', '_sim_tmp_phase_P'),
                ('pressure_drop',  '_sim_tmp_pressure_drop'),
            ]
            for display_name, edge_key in sim_only_checks:
                for _, _, d in self.G.edges(data=True):
                    if edge_key in d:
                        found.append(display_name); break
        return found

    # ── napari + Qt UI ─────────────────────────────────────────────────
    def _setup_viewer(self):
        import napari
        title = f"Mosaic (read-only) — {self.graph_path.name}"
        self.viewer = napari.Viewer(title=title)

        # Background image (if available)
        if self.mosaic is not None:
            self.viewer.add_image(self.mosaic, name='Mosaic', opacity=0.75)

        # Edges layer is built lazily by _refresh_edges (which removes the
        # previous one and re-adds a fresh one).  This sidesteps a napari
        # quirk where Shapes layers don't re-render colour after a bulk
        # .data + .edge_color update on an initially-empty layer.
        self._edges_layer = None
        # Nodes layer (hidden by default)
        self._nodes_layer = self.viewer.add_points(
            np.empty((0, 2)), size=4, face_color='white',
            edge_color='black', name='Nodes', visible=False,
        )
        # Comparison-set marker layer — large halo rings, per-point
        # coloured to match the comparison plot.  Sized larger than the
        # cyan click marker so both stay visible when the user clicks an
        # edge that's already in the comparison set.
        try:
            self._comparison_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=36, face_color='transparent',
                border_color='#1f77b4', border_width=0.35,
                name='Comparison set', symbol='ring', opacity=0.95,
            )
        except TypeError:
            self._comparison_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=36, face_color='transparent',
                edge_color='#1f77b4', edge_width=0.35,
                name='Comparison set', symbol='ring', opacity=0.95,
            )

        # Click-highlight marker.  napari ≥ 0.5 renamed edge_* → border_*
        # AND made border_width relative to point size by default (0-1).
        # Pass a relative width (~20% of size) which works on both old
        # and new napari API names; if your napari is too old to accept
        # `border_width`, swap it for `edge_width=0.2`.
        try:
            self._click_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=24, face_color='transparent',
                border_color='cyan', border_width=0.2,
                name='Selected', symbol='ring',
            )
        except TypeError:
            self._click_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=24, face_color='transparent',
                edge_color='cyan', edge_width=0.2,
                name='Selected', symbol='ring',
            )

        # Boundary-node badges — red dot + "A" at arterial sources, blue
        # dot + "V" at venous sinks.  Always visible above the edges.
        def _to_display(n):
            return [self.mosaic_height - self.G.nodes[n]['y'],
                    self.G.nodes[n]['x']]
        if self._source_nodes:
            src_pts = np.array([_to_display(n) for n in self._source_nodes])
            self._source_layer = self.viewer.add_points(
                src_pts, size=18, face_color='#d62728',
                border_color='white',
                name='Arterial boundary (A)',
                text={'string': ['A'] * len(self._source_nodes),
                      'size': 12, 'color': 'white',
                      'anchor': 'center'},
            )
        if self._sink_nodes:
            snk_pts = np.array([_to_display(n) for n in self._sink_nodes])
            self._sink_layer = self.viewer.add_points(
                snk_pts, size=18, face_color='#1f77b4',
                border_color='white',
                name='Venous boundary (V)',
                text={'string': ['V'] * len(self._sink_nodes),
                      'size': 12, 'color': 'white',
                      'anchor': 'center'},
            )

        # Click handler attached after every layer rebuild via
        # `_attach_edge_click`.
        # Use the viewer-wide mouse-drag callback so it survives layer
        # rebuilds: it looks up the click position and asks
        # _inspect_nearest_edge to do hit testing against our
        # precomputed midpoints.  Boundary-node hits are intercepted
        # first so the user can inspect BC harmonics.
        @self.viewer.mouse_drag_callbacks.append
        def _on_click(viewer, event):
            if event.type != 'mouse_press' or len(event.position) < 2:
                return
            row, col = event.position[-2], event.position[-1]
            if self._inspect_nearest_boundary_node(col, row):
                return
            self._inspect_nearest_edge(col, row)

    def _setup_panel(self):
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
            QCheckBox, QPushButton, QGroupBox, QTextEdit, QTabWidget,
            QDoubleSpinBox, QScrollArea,
        )
        tabs = QTabWidget()
        view_panel = QWidget(); view_layout = QVBoxLayout(view_panel)
        ts_panel = QWidget();   ts_layout = QVBoxLayout(ts_panel)
        sim_panel = QWidget();  sim_layout = QVBoxLayout(sim_panel)
        inf_panel = QWidget();  inf_layout = QVBoxLayout(inf_panel)

        # ══════ Tab 1: VIEW ═══════════════════════════════════════════
        layout = view_layout    # alias for the existing widget code

        # ── Data source: Measured (PIV) vs Simulated (solver output) ──
        from qtpy.QtWidgets import QRadioButton, QButtonGroup
        src_group = QGroupBox("Data source")
        src_lay = QHBoxLayout(src_group)
        self._src_radio_measured = QRadioButton("Measured")
        self._src_radio_measured.setChecked(self.current_source == 'measured')
        self._src_radio_sim = QRadioButton("Simulated")
        self._src_radio_sim.setChecked(self.current_source == 'sim')
        self._src_radio_sim.setEnabled(self._sim_active)
        if not self._sim_active:
            self._src_radio_sim.setToolTip(
                "Run a simulation in the Simulate tab to enable.")
        self._src_button_group = QButtonGroup(src_group)
        self._src_button_group.addButton(self._src_radio_measured)
        self._src_button_group.addButton(self._src_radio_sim)
        self._src_radio_measured.toggled.connect(self._on_source_change)
        self._src_radio_sim.toggled.connect(self._on_source_change)
        src_lay.addWidget(self._src_radio_measured)
        src_lay.addWidget(self._src_radio_sim)
        src_lay.addStretch(1)
        layout.addWidget(src_group)

        # Field selector — four small combos compose the colormap.
        fld_group = QGroupBox("Color edges by")
        fld_lay = QVBoxLayout(fld_group)

        # Quantity row (Q / P; P disabled in Measured source)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel('Quantity:'))
        self._quantity_combo = QComboBox()
        self._quantity_combo.addItem('Q (flow)',     userData='Q')
        self._quantity_combo.addItem('P (pressure)', userData='P')
        self._quantity_combo.setCurrentIndex(
            0 if self.current_quantity == 'Q' else 1)
        self._quantity_combo.currentIndexChanged.connect(
            self._on_quantity_change)
        q_row.addWidget(self._quantity_combo); q_row.addStretch(1)
        fld_lay.addLayout(q_row)

        # Property row
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel('Property:'))
        self._property_combo = QComboBox()
        for key, label, _is_harm, _valid_in in PROPERTY_DEFS:
            self._property_combo.addItem(label, userData=key)
            if key == self.current_property:
                self._property_combo.setCurrentIndex(
                    self._property_combo.count() - 1)
        self._property_combo.currentIndexChanged.connect(
            self._on_property_change)
        p_row.addWidget(self._property_combo); p_row.addStretch(1)
        fld_lay.addLayout(p_row)

        # Harmonic row (only enabled for harmonic-keyed properties)
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel('Harmonic:'))
        self._harmonic_combo = QComboBox()
        for hk in HARMONIC_KEYS:
            self._harmonic_combo.addItem(HARMONIC_LABELS[hk], userData=hk)
            if hk == self.current_harmonic:
                self._harmonic_combo.setCurrentIndex(
                    self._harmonic_combo.count() - 1)
        self._harmonic_combo.currentIndexChanged.connect(
            self._on_harmonic_change)
        h_row.addWidget(self._harmonic_combo); h_row.addStretch(1)
        fld_lay.addLayout(h_row)

        # Sync enable-state of the combos (Quantity, Harmonic) and the
        # item flags inside Property to the current Source + selection.
        self._sync_field_combos()

        self._log_chk = QCheckBox("Log scale")
        self._log_chk.setToolTip("Useful for mean_Q, amp_Q (orders of magnitude).")
        self._log_chk.toggled.connect(self._on_log_toggle)
        fld_lay.addWidget(self._log_chk)

        # Colour-bound percentiles — adapt the colormap range to the
        # data on every refresh.  Defaults: 2 / 98 (same as the old
        # "Auto" behaviour).  Lower lo / higher hi → looser clip;
        # tighter percentiles around 50 → more aggressive contrast.
        # Phase fields ignore these and stay fixed at [−π, π].
        bounds_row = QWidget()
        b_lay = QHBoxLayout(bounds_row); b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.addWidget(QLabel('lo pct:'))
        self._pct_lo_spin = QDoubleSpinBox()
        self._pct_lo_spin.setRange(0.0, 100.0)
        self._pct_lo_spin.setDecimals(1)
        self._pct_lo_spin.setSingleStep(0.5)
        self._pct_lo_spin.setValue(2.0)
        self._pct_lo_spin.setToolTip(
            "Lower percentile of finite per-edge values, mapped to the\n"
            "low end of the colormap.  Values below get clipped.")
        self._pct_lo_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._pct_lo_spin)
        b_lay.addWidget(QLabel(' hi pct:'))
        self._pct_hi_spin = QDoubleSpinBox()
        self._pct_hi_spin.setRange(0.0, 100.0)
        self._pct_hi_spin.setDecimals(1)
        self._pct_hi_spin.setSingleStep(0.5)
        self._pct_hi_spin.setValue(98.0)
        self._pct_hi_spin.setToolTip(
            "Upper percentile of finite per-edge values, mapped to the\n"
            "high end of the colormap.  Values above get clipped.")
        self._pct_hi_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._pct_hi_spin)
        b_lay.addStretch(1)
        fld_lay.addWidget(bounds_row)
        # Colour bar toggle — overlays a floating colorbar on top of the
        # napari canvas (right side, vertically centred).  No embedded
        # canvas in the side panel.
        self._cbar_chk = QCheckBox("Show colour bar (overlaid on canvas)")
        self._cbar_chk.setChecked(True)
        self._cbar_chk.toggled.connect(self._on_cbar_toggle)
        fld_lay.addWidget(self._cbar_chk)
        # The overlay QLabel lives in `self._cbar_overlay`; created /
        # replaced by `_refresh_cbar()`.
        self._cbar_overlay = None
        self._cbar_orig_resize = None
        # Tile filter — restricts measured-Q fields to one tile's
        # measurement instead of best-across-tiles.  Only meaningful
        # when tile positions are loaded; hidden otherwise.
        self._tile_filter_combo = None
        if self.tiles:
            tilef_row = QWidget()
            tilef_lay = QHBoxLayout(tilef_row)
            tilef_lay.setContentsMargins(0, 0, 0, 0)
            tilef_lay.addWidget(QLabel("Tile filter:"))
            self._tile_filter_combo = QComboBox()
            self._tile_filter_combo.addItem(
                "All tiles (best measurement)", userData=None)
            for tid in sorted(self.tiles.keys()):
                self._tile_filter_combo.addItem(
                    f"tile {tid}", userData=int(tid))
            self._tile_filter_combo.setToolTip(
                "When set to a specific tile, every measured-Q field\n"
                "shows that tile's measurement only (not the best\n"
                "across tiles).  Edges with no measurement on the\n"
                "selected tile render as 'no data' grey.\n"
                "Doesn't affect Sim fields or geometry.")
            self._tile_filter_combo.currentIndexChanged.connect(
                self._on_tile_filter_change)
            tilef_lay.addWidget(self._tile_filter_combo)
            tilef_lay.addStretch(1)
            fld_lay.addWidget(tilef_row)
        layout.addWidget(fld_group)

        # Percentile filter on total_snr — gates measured-Q fields by
        # network-wide percentile of Var(fit)/Var(resid).  Percentile is
        # computed against the GRAPH-WIDE cached values (best-of-edge),
        # so the threshold is stable regardless of whether a tile
        # filter is on.  0 = no filtering, default.
        from qtpy.QtWidgets import QDoubleSpinBox
        tspct_row = QWidget()
        tspct_lay = QHBoxLayout(tspct_row)
        tspct_lay.setContentsMargins(0, 0, 0, 0)
        tspct_lay.addWidget(QLabel('Hide bottom % by total_snr:'))
        self._tspct_filter_spin = QDoubleSpinBox()
        self._tspct_filter_spin.setRange(0.0, 99.0)
        self._tspct_filter_spin.setDecimals(1)
        self._tspct_filter_spin.setSingleStep(1.0)
        self._tspct_filter_spin.setValue(0.0)
        self._tspct_filter_spin.setToolTip(
            "Hide edges in the bottom X % of total_snr "
            "(Var(fit)/Var(resid))\n"
            "across the WHOLE network.  Threshold is always computed\n"
            "from graph-wide best-of-edge values — independent of the\n"
            "Tile filter setting.  Only affects measured-Q fields.\n"
            "0 = no filtering (default).")
        self._tspct_filter_spin.valueChanged.connect(
            self._on_total_snr_pct_change)
        tspct_lay.addWidget(self._tspct_filter_spin)
        tspct_lay.addStretch(1)
        fld_lay.addWidget(tspct_row)

        # Tile browser — load any tile's video by picking from a dropdown,
        # independent of edge selection.  Only shown if positions + video
        # paths are configured.
        if bool(self.tiles) and self.video_dir is not None:
            tile_group = QGroupBox("Browse tiles")
            tile_lay = QVBoxLayout(tile_group)
            self._tile_combo = QComboBox()
            # Tile entries: each item carries the int tile ID as
            # userData so the handler can use currentData() directly
            # instead of parsing the label text.  Tiles whose video
            # file isn't present in `video_dir` (typical for
            # partial-coverage bundles like the intern packages) get
            # " (no video)" appended and are dim-disabled.
            self._tile_combo.addItem("(pick a tile)", userData=None)
            n_missing = 0
            for tid in sorted(self.tiles.keys()):
                vid_path = self._get_video_path_for(int(tid))
                label = (f"{tid}"
                          if vid_path is not None
                          else f"{tid}  (no video)")
                self._tile_combo.addItem(label, userData=int(tid))
                if vid_path is None:
                    n_missing += 1
                    idx = self._tile_combo.count() - 1
                    item = self._tile_combo.model().item(idx)
                    if item is not None:
                        from qtpy.QtCore import Qt as _Qt
                        item.setFlags(item.flags()
                                       & ~_Qt.ItemIsEnabled
                                       & ~_Qt.ItemIsSelectable)
            tip = ("Pick a tile ID then click 'Load this tile' to "
                    "overlay\nits raw 250-fps video on the mosaic.")
            if n_missing:
                tip += (f"\n\n{n_missing} tile(s) are greyed out "
                         f"because their video files\nare not present "
                         f"in this bundle.")
            self._tile_combo.setToolTip(tip)
            tile_lay.addWidget(self._tile_combo)
            self._tile_load_btn = QPushButton('Load this tile video')
            self._tile_load_btn.clicked.connect(self._load_video_from_dropdown)
            tile_lay.addWidget(self._tile_load_btn)
            layout.addWidget(tile_group)

        # Display toggles
        d_group = QGroupBox("Display")
        d_lay = QVBoxLayout(d_group)
        self._nodes_chk = QCheckBox("Show nodes")
        self._nodes_chk.toggled.connect(self._on_nodes_toggle)
        d_lay.addWidget(self._nodes_chk)
        if bool(self.tiles):
            self._tile_boundaries_chk = QCheckBox("Show tile boundaries")
            self._tile_boundaries_chk.toggled.connect(
                self._on_tile_boundaries_toggle)
            d_lay.addWidget(self._tile_boundaries_chk)
            self._tile_labels_chk = QCheckBox("Show tile numbers")
            self._tile_labels_chk.toggled.connect(
                self._on_tile_labels_toggle)
            d_lay.addWidget(self._tile_labels_chk)
        layout.addWidget(d_group)

        layout.addStretch()

        # ══════ Tab 2: TIME SERIES ════════════════════════════════════
        layout = ts_layout
        # Edge inspection panel (now lives in the time-series tab).
        inspect_group = QGroupBox("Selected edge")
        inspect_lay = QVBoxLayout(inspect_group)
        self._info_box = QTextEdit()
        self._info_box.setReadOnly(True)
        self._info_box.setMaximumHeight(220)
        self._info_box.setPlaceholderText(
            "Click an edge in the viewer to inspect it.")
        inspect_lay.addWidget(self._info_box)
        self._plot_qt_btn = QPushButton("Plot Q(t) for selected")
        self._plot_qt_btn.setEnabled(False)
        self._plot_qt_btn.clicked.connect(self._plot_selected_qt)
        inspect_lay.addWidget(self._plot_qt_btn)
        # Run Farneback optical flow on the selected edge from the
        # full-resolution tile video.  Read-only: no graph mutation,
        # no measurement persistence — just shows Q(t) + spectrum in
        # a matplotlib popup.
        # OF button intentionally hidden in this build.  Distributed
        # bundles ship spatially-downsampled videos, so absolute Q
        # magnitudes coming out of the Farneback path would be subtly
        # wrong (per-pixel µm size shifts with the downsample).  The
        # button is still wired below so re-enabling is a one-line
        # change — just uncomment the addWidget call.
        self._run_of_btn = QPushButton("Run optical flow on selected")
        self._run_of_btn.setEnabled(False)
        self._run_of_btn.setToolTip(
            "Reload the full-res tile video, run Farneback OF on the\n"
            "selected vessel, and pop up Q(t) + spectrum.  No data\n"
            "is written back to the graph.  Requires --tile-positions\n"
            "and --video-dir; ~5-10s per call.")
        self._run_of_btn.clicked.connect(self._on_run_optical_flow_clicked)
        # inspect_lay.addWidget(self._run_of_btn)   # ← uncomment to restore
        # Clear the curved OF-region overlay (created by Run OF; persists
        # across mosaic interactions until cleared).  Disabled when no
        # overlay is active.  Also hidden as a no-op alongside the OF
        # button so it doesn't sit there dangling.
        self._clear_of_region_btn = QPushButton("Clear OF region overlay")
        self._clear_of_region_btn.setEnabled(False)
        self._clear_of_region_btn.setToolTip(
            "Remove the curved orange ribbon that highlights the last\n"
            "OF integration region on the mosaic.")
        self._clear_of_region_btn.clicked.connect(
            self._tear_down_of_region_overlay)
        # inspect_lay.addWidget(self._clear_of_region_btn)  # ← restore with OF

        # Comparison set — accumulate edges across multiple clicks and
        # overlay their Q(t)s on a single figure.  Useful for comparing
        # nearby vessels' waveforms.
        self._add_to_comp_btn = QPushButton("Add to comparison set")
        self._add_to_comp_btn.setEnabled(False)
        self._add_to_comp_btn.setToolTip(
            "Add the currently-selected edge to a comparison set.\n"
            "Use 'Plot comparison' to see all of them overlaid.")
        self._add_to_comp_btn.clicked.connect(self._add_to_comparison)
        inspect_lay.addWidget(self._add_to_comp_btn)
        self._plot_comp_btn = QPushButton("Plot comparison (0)")
        self._plot_comp_btn.setEnabled(False)
        self._plot_comp_btn.clicked.connect(self._plot_comparison)
        inspect_lay.addWidget(self._plot_comp_btn)
        self._clear_comp_btn = QPushButton("Clear comparison")
        self._clear_comp_btn.setEnabled(False)
        self._clear_comp_btn.clicked.connect(self._clear_comparison)
        inspect_lay.addWidget(self._clear_comp_btn)
        # Video overlay controls — only shown if positions + video paths
        # were given on the CLI.  Otherwise hide them rather than confuse
        # the user.
        video_enabled = bool(self.tiles) and self.video_dir is not None
        self._auto_video_chk = QCheckBox("Auto-load video on edge click")
        self._auto_video_chk.setEnabled(video_enabled)
        self._auto_video_chk.setToolTip(
            "When ON, clicking an edge automatically loads + overlays\n"
            "the raw video for that tile.  Loading takes ~5-10s per\n"
            "tile; the previous overlay is removed each click.")
        inspect_lay.addWidget(self._auto_video_chk)
        self._video_btn = QPushButton("Load video for selected tile")
        self._video_btn.setEnabled(False)
        self._video_btn.clicked.connect(self._load_video_for_selected)
        inspect_lay.addWidget(self._video_btn)
        self._unload_video_btn = QPushButton("Remove video overlay")
        self._unload_video_btn.setEnabled(False)
        self._unload_video_btn.clicked.connect(
            lambda: (self._remove_video_overlay(),
                     self._unload_video_btn.setEnabled(False)))
        inspect_lay.addWidget(self._unload_video_btn)
        if not video_enabled:
            note = QLabel(
                "<small><i>Video overlay disabled — pass <code>"
                "--tile-positions</code> + <code>--video-dir</code> + "
                "<code>--video-pattern</code> (or <code>--config</code>) "
                "to enable.</i></small>")
            note.setWordWrap(True)
            inspect_lay.addWidget(note)
        layout.addWidget(inspect_group)
        layout.addStretch()

        # ══════ Tab 3: SIMULATE ═══════════════════════════════════════
        layout = sim_layout
        # Built by `_build_simulate_tab` so the logic stays separable.
        self._build_simulate_tab(layout)
        layout.addStretch()

        # ══════ Tab 4: INFERENCE ══════════════════════════════════════
        layout = inf_layout
        # Built by `_build_inference_tab`.  The inference adapter lives
        # in its own contiguous code block so the underlying inversion
        # algorithm can be swapped out without disturbing the rest of
        # the viewer.
        self._build_inference_tab(layout)
        layout.addStretch()

        # Info label (View tab footer)
        info = QLabel(
            f"<small>{self.G.number_of_edges()} edges, "
            f"{self.G.number_of_nodes()} nodes<br>"
            f"<b>Read-only viewer</b> — no graph edits, no PIV re-runs."
            f"</small>")
        info.setWordWrap(True)
        view_layout.addWidget(info)

        # Assemble the tabbed panel
        tabs.addTab(view_panel, "View")
        tabs.addTab(ts_panel, "Time series")
        tabs.addTab(sim_panel, "Simulate")
        tabs.addTab(inf_panel, "Inference")
        self.viewer.window.add_dock_widget(tabs, name="Controls", area='right')
        self._selected_edge: Optional[Tuple[int, int]] = None
        # Build the initial colour bar to reflect current field.
        self._refresh_cbar()

    # ── core: refresh edge rendering ───────────────────────────────────
    def _refresh_edges(self):
        # Always draw every edge in the network.  Edges with a usable
        # PIV measurement get coloured by the selected field; edges with
        # no measurement or with measurement-quality flagged garbage
        # (gated, flag-on-delta, fit failed) get a single dim-grey
        # colour so the topology stays visible without pretending we
        # have data we don't.
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        # total_snr percentile filter — applies only to measured-Q
        # fields.  Threshold computed ONCE from the network-wide
        # `_h_total_snr` distribution; per-edge gate below.  Re-derive
        # if the percentile changed (handler clears the cache).
        ts_filter_active = (
            float(self.total_snr_pct_filter) > 0.0
            and self.current_source == 'measured'
            and self.current_quantity == 'Q'
        )
        ts_threshold: Optional[float] = None
        if ts_filter_active:
            if self._cached_total_snr_threshold is None:
                all_tsnr = []
                for _, _, dd in self.G.edges(data=True):
                    v_ = dd.get('_h_total_snr')
                    if v_ is not None and np.isfinite(v_):
                        all_tsnr.append(float(v_))
                if all_tsnr:
                    self._cached_total_snr_threshold = float(
                        np.percentile(np.asarray(all_tsnr),
                                       self.total_snr_pct_filter))
                else:
                    self._cached_total_snr_threshold = -np.inf
            ts_threshold = self._cached_total_snr_threshold
        EDGE_WIDTH_MIN = 0.5
        EDGE_WIDTH_MAX = 22.0
        lines: List[np.ndarray] = []
        vals: List[Optional[float]] = []   # None ⇒ no usable measurement
        tiers: List[Optional[str]] = []
        widths: List[float] = []
        H_mosaic = self.mosaic_height
        for (u, v) in self.edge_list:
            d = self.G.edges[u, v]
            m = _best_measurement(d.get('measurements_piv'))
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            # napari shapes layer wants (row, col) = (display_y, x);
            # display_y = mosaic_height − graph_y.
            lines.append(np.array([[H_mosaic - y1, x1],
                                    [H_mosaic - y2, x2]]))
            # Edge width ≈ vessel radius (NOT diameter — keeps neighbour
            # edges from overlapping at large radii while still being
            # visually proportional).
            r_px = float(d.get('radius', 1.0))
            widths.append(float(np.clip(r_px,
                                         EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)))
            usable = _measurement_usable(m)
            if is_categorical:
                # Two categorical fields, different sources:
                #   quality_tier   — per-measurement (tier letter)
                #   harmonic_class — top-level edge attr (int 0-3)
                # Apply the same total_snr percentile gate so the
                # filter is consistent across continuous + categorical
                # views of measured-Q data.
                if (ts_filter_active and ts_threshold is not None):
                    edge_tsnr = d.get('_h_total_snr')
                    gate_passes = (edge_tsnr is not None
                                    and np.isfinite(edge_tsnr)
                                    and float(edge_tsnr) >= ts_threshold)
                else:
                    gate_passes = True
                if not gate_passes:
                    tiers.append(None)
                elif self.current_property == 'harmonic_class':
                    tiers.append(d.get('harmonic_class'))
                elif 'quality_tier' in field:
                    tiers.append(m.get('quality_tier') if usable else None)
                else:
                    tiers.append(None)
                vals.append(None)
            else:
                # 4-selector resolver — already handles measured/sim
                # routing and harmonic indexing.  Returns None for
                # invalid combos or missing data.
                val = self._resolve_field_value(u, v)
                # total_snr percentile gate.  Always checks the
                # graph-wide cached `_h_total_snr` (best-of-edge), so
                # the filter is stable across tile-filter changes.
                if (ts_filter_active and val is not None
                        and ts_threshold is not None):
                    edge_tsnr = d.get('_h_total_snr')
                    if (edge_tsnr is None
                            or not np.isfinite(edge_tsnr)
                            or float(edge_tsnr) < ts_threshold):
                        val = None
                vals.append(val if (val is not None
                                    and np.isfinite(val)) else None)
                tiers.append(None)

        # Compute colors.  Edges with `vals[i] is None` (no usable
        # measurement) get NO_DATA_COLOR regardless of the chosen field.
        if is_categorical:
            cat_map = (HARMONIC_CLASS_COLORS
                       if field == 'harmonic_class' else TIER_COLORS)
            colors = [
                cat_map.get(t, NO_DATA_COLOR) if t is not None
                else NO_DATA_COLOR
                for t in tiers]
        else:
            finite_arr = np.array(
                [v for v in vals if v is not None], dtype=float)
            if len(finite_arr) == 0:
                colors = [NO_DATA_COLOR] * len(lines)
            else:
                if self.log_scale:
                    pos = finite_arr[finite_arr > 0]
                    if len(pos) == 0:
                        finite_eff = finite_arr
                    else:
                        finite_eff = np.log10(pos)
                else:
                    finite_eff = finite_arr
                # Bounds: percentile clips of the per-edge value
                # distribution.  Spinbox defaults are 2 / 98.
                p_lo = float(getattr(self, '_pct_lo_spin', None).value()
                             if getattr(self, '_pct_lo_spin', None)
                             else 2.0)
                p_hi = float(getattr(self, '_pct_hi_spin', None).value()
                             if getattr(self, '_pct_hi_spin', None)
                             else 98.0)
                # Guard against inverted spinbox state (user dragged
                # lo above hi): swap so the percentiles stay sane.
                if p_lo > p_hi:
                    p_lo, p_hi = p_hi, p_lo
                vlo, vhi = np.percentile(finite_eff, [p_lo, p_hi])
                if vlo == vhi:
                    vhi = vlo + 1e-9
                cmap = (plt.get_cmap('hsv') if 'phase' in field
                        else plt.get_cmap('viridis'))
                if 'phase' in field:
                    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
                    self._last_vmin, self._last_vmax = -np.pi, np.pi
                else:
                    norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                    self._last_vmin, self._last_vmax = float(vlo), float(vhi)
                colors = []
                for v_ in vals:
                    if v_ is None:
                        colors.append(NO_DATA_COLOR)
                    else:
                        v_use = (np.log10(max(v_, 1e-30))
                                 if self.log_scale and v_ > 0 else v_)
                        colors.append(mcolors.to_hex(cmap(norm(v_use))))

        # Rebuild the edges layer from scratch.  Replacing it cleanly
        # avoids a napari Shapes-layer rendering bug where the .data +
        # .edge_color assignments on an existing layer don't always
        # repaint correctly.
        if self._edges_layer is not None:
            try:
                self.viewer.layers.remove(self._edges_layer)
            except (KeyError, ValueError):
                pass
            self._edges_layer = None
        if lines:
            self._edges_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color=colors,
                edge_width=widths,
                name='Edges', opacity=0.7,
            )
        n_usable = sum(1 for v in vals if v is not None)
        print(f"  Drew {len(lines)} edges  "
              f"({n_usable} coloured by {field}, "
              f"{len(lines) - n_usable} grey: no usable measurement)")
        # Status message via window title
        title = (f"Mosaic (read-only) — {self.graph_path.name}  |  "
                 f"field={field}  |  n_drawn={len(lines)}")
        self.viewer.title = title
        # Refresh the colour-bar overlay to match the new bounds /
        # colormap.  (Old `_cbar_canvas` attribute is gone; the live
        # state is in `_cbar_chk` + `_cbar_overlay`.)
        if getattr(self, '_cbar_chk', None) is not None:
            self._refresh_cbar()

    def _refresh_nodes(self):
        if self._nodes_layer is None:
            return
        if not self.show_nodes:
            self._nodes_layer.visible = False
            return
        # Collect per-node positions (with y-flip to align with the mosaic
        # image) and per-node field values (mean of incident edges with
        # usable measurements).  Nodes with no incident usable edge get
        # NO_DATA_COLOR — same convention as the edges layer.
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        node_field: Dict[int, list] = {n: [] for n in self.G.nodes()}
        for u, v in self.edge_list:
            d = self.G.edges[u, v]
            m = _best_measurement(d.get('measurements_piv'))
            if is_categorical:
                if not _measurement_usable(m):
                    continue
                if self.current_property == 'harmonic_class':
                    hc = d.get('harmonic_class')
                    if hc is not None:
                        node_field[u].append(int(hc))
                        node_field[v].append(int(hc))
                else:   # quality_tier
                    tier = m.get('quality_tier', 'X')
                    node_field[u].append(QUALITY_RANK.get(tier, 0))
                    node_field[v].append(QUALITY_RANK.get(tier, 0))
            else:
                val = self._resolve_field_value(u, v)
                if val is not None and np.isfinite(val):
                    node_field[u].append(val)
                    node_field[v].append(val)
        # Build points + colors in node-order
        nodes = list(self.G.nodes())
        pts = np.array([[self.mosaic_height - self.G.nodes[n]['y'],
                         self.G.nodes[n]['x']] for n in nodes])
        vals = np.array([np.mean(node_field[n]) if node_field[n] else np.nan
                         for n in nodes], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0 or is_categorical:
            face = [NO_DATA_COLOR] * len(nodes)
        else:
            arr = vals.copy()
            if self.log_scale:
                pos = finite[finite > 0]
                if len(pos):
                    finite_eff = np.log10(pos)
                    arr = np.where((arr > 0) & np.isfinite(arr),
                                   np.log10(np.maximum(arr, 1e-30)),
                                   np.nan)
                else:
                    finite_eff = finite
            else:
                finite_eff = finite
            vlo, vhi = np.percentile(finite_eff, [2, 98])
            if vlo == vhi:
                vhi = vlo + 1e-9
            cmap = (plt.get_cmap('hsv') if 'phase' in field
                    else plt.get_cmap('viridis'))
            if 'phase' in field:
                norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
            else:
                norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
            face = []
            for v_ in arr:
                if not np.isfinite(v_):
                    face.append(NO_DATA_COLOR)
                else:
                    face.append(mcolors.to_hex(cmap(norm(v_))))
        self._nodes_layer.data = pts
        if len(pts):
            self._nodes_layer.face_color = face
        self._nodes_layer.visible = True

    # ── Simulate tab + colour-bar UI ───────────────────────────────────
    def _build_simulate_tab(self, layout):
        """Forward transmission-line solver controls.

        MVP scope: BCs come from the stored `bc_harmonics` on boundary
        nodes (no custom override yet).  D and f₀ are user-adjustable.
        Results are written to the in-memory graph under the
        `_sim_tmp_*` namespace and added to the View-tab field dropdown
        when present.  Never persisted to the gpickle.
        """
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QRadioButton, QButtonGroup, QDoubleSpinBox, QCheckBox,
            QComboBox as _QComboBox)

        # ── Detect default f₀ from the graph ──
        self._sim_detected_f0 = self._detect_sim_f0_hz()

        # ── BC source ──
        bc_group = QGroupBox("Boundary conditions")
        bc_lay = QVBoxLayout(bc_group)
        self._sim_bc_radio_measured = QRadioButton(
            "Use measured BCs (from boundary `bc_harmonics`)")
        self._sim_bc_radio_measured.setChecked(True)
        self._sim_bc_radio_custom = QRadioButton(
            "Custom BCs (specify waveform below)")
        bc_group_buttons = QButtonGroup(bc_group)
        bc_group_buttons.addButton(self._sim_bc_radio_measured)
        bc_group_buttons.addButton(self._sim_bc_radio_custom)
        self._sim_bc_radio_measured.toggled.connect(
            self._on_bc_source_change)
        self._sim_bc_radio_custom.toggled.connect(
            self._on_bc_source_change)
        bc_lay.addWidget(self._sim_bc_radio_measured)
        bc_lay.addWidget(self._sim_bc_radio_custom)
        # Quick summary of available boundary nodes.
        bc_info_lines = [
            f"Sources (A): {len(self._source_nodes)}",
            f"Sinks (V):   {len(self._sink_nodes)}",
        ]
        bc_lay.addWidget(QLabel("<small><tt>"
                                 + "<br>".join(bc_info_lines)
                                 + "</tt></small>"))

        # ── Custom waveform sub-panel ──
        # Twelve spinboxes: amp fraction (|H_k|/|Q̄|) + phase (degrees)
        # for H1, H2, H3 on EACH side (A and V) independently.  DC
        # magnitude is set automatically from the Inlet/outlet flux +
        # N_src / N_snk and the sign convention (+ on A, − on V into
        # the network).  Equal-split mode is the default — every A
        # boundary gets the same A waveform and every V boundary gets
        # the same V waveform.  Live preview reflects both at once.
        self._sim_custom_group = QGroupBox(
            "Custom waveform shape  (independent A and V)")
        cust_lay = QVBoxLayout(self._sim_custom_group)
        _cust_help_lbl = QLabel(
            "<small>Specify the A and V waveforms separately.  Each "
            "row gives the amp fraction (|H<sub>k</sub>|/|Q̄|) and "
            "phase (degrees) per harmonic for that side.  The chosen "
            "shape is applied to ALL boundaries of that type — every "
            "A gets the A spec, every V gets the V spec.  DC magnitudes "
            "come from flux / N.  Preview shows both with the sign "
            "chosen so DC reads positive on each side.</small>")
        _cust_help_lbl.setWordWrap(True)
        cust_lay.addWidget(_cust_help_lbl)
        # Twelve spinboxes — (side, harmonic, kind) → spinbox.
        # Header row labels the columns.
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(QLabel("<b></b>"))
        hdr_row.addWidget(QLabel("<b>A amp</b>"))
        hdr_row.addWidget(QLabel("<b>A phase°</b>"))
        hdr_row.addWidget(QLabel("<b>V amp</b>"))
        hdr_row.addWidget(QLabel("<b>V phase°</b>"))
        hdr_row.addStretch(1)
        cust_lay.addLayout(hdr_row)
        self._sim_custom_spins = {}   # (side, harmonic, kind) → spinbox
        # Sensible defaults: H1 ≈ 0.5 (PI ~ 1), H2 ≈ 0.15, H3 ≈ 0.05.
        # Same defaults for both sides; the user changes V to taste.
        _default_amps = {1: 0.5, 2: 0.15, 3: 0.05}
        for k in (1, 2, 3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"H{k}:"))
            for side in ('A', 'V'):
                amp = QDoubleSpinBox()
                amp.setDecimals(3); amp.setRange(0.0, 5.0)
                amp.setSingleStep(0.05)
                amp.setValue(_default_amps[k])
                amp.valueChanged.connect(self._refresh_custom_preview)
                row.addWidget(amp)
                phase = QDoubleSpinBox()
                phase.setDecimals(1); phase.setRange(-360.0, 360.0)
                phase.setSingleStep(5.0); phase.setValue(0.0)
                phase.valueChanged.connect(self._refresh_custom_preview)
                row.addWidget(phase)
                self._sim_custom_spins[(side, k, 'amp')] = amp
                self._sim_custom_spins[(side, k, 'phase')] = phase
            row.addStretch(1)
            cust_lay.addLayout(row)
        # Preview canvas (small matplotlib figure)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigureCanvas)
        self._sim_custom_fig = Figure(figsize=(4.0, 1.7), dpi=90,
                                       tight_layout=True)
        self._sim_custom_canvas = _FigureCanvas(self._sim_custom_fig)
        self._sim_custom_canvas.setMinimumHeight(150)
        cust_lay.addWidget(self._sim_custom_canvas)
        # Initial enable state — spinboxes disabled (measured is the
        # default BC source), preview canvas always visible.
        for sp in self._sim_custom_spins.values():
            sp.setEnabled(False)
        bc_lay.addWidget(self._sim_custom_group)
        layout.addWidget(bc_group)
        # Draw the initial preview after widgets exist (will reflect
        # measured BCs since that's the default selection).  Connect
        # the flux + equal-split widgets later — those signals are
        # wired below once the corresponding controls are built.

        # ── Parameters ──
        param_group = QGroupBox("Parameters")
        param_lay = QVBoxLayout(param_group)
        # D (distensibility) — areal convention.  Split into mantissa
        # and exponent so the user can pick orders of magnitude
        # explicitly: D = mantissa × 10^exponent.  Default 1.0e-3.
        d_row = QHBoxLayout()
        d_row.addWidget(QLabel("D (1/Pa, areal):"))
        self._sim_D_mantissa = QDoubleSpinBox()
        self._sim_D_mantissa.setDecimals(2)
        self._sim_D_mantissa.setRange(1.0, 9.99)
        self._sim_D_mantissa.setSingleStep(0.1)
        self._sim_D_mantissa.setValue(1.0)
        self._sim_D_mantissa.setToolTip(
            "Mantissa for D in scientific notation (1.00 to 9.99).")
        d_row.addWidget(self._sim_D_mantissa)
        d_row.addWidget(QLabel("× 10^"))
        from qtpy.QtWidgets import QSpinBox
        self._sim_D_exponent = QSpinBox()
        self._sim_D_exponent.setRange(-7, -1)
        self._sim_D_exponent.setSingleStep(1)
        self._sim_D_exponent.setValue(-3)
        self._sim_D_exponent.setToolTip(
            "Exponent for D in scientific notation (−7 to −1).\n"
            "Areal convention: typical HH-stage yolk-sac D ≈ 1.3e-3.")
        d_row.addWidget(self._sim_D_exponent)
        d_row.addStretch(1)
        param_lay.addLayout(d_row)
        # f0 (read-only label, auto-detected).
        f0_row = QHBoxLayout()
        f0_row.addWidget(QLabel("f₀ (Hz):"))
        f0_val = (f"{self._sim_detected_f0:.3f}"
                  if np.isfinite(self._sim_detected_f0) else "n/a")
        self._sim_f0_lbl = QLabel(f"<tt>{f0_val}</tt>  "
                                   "<small>(auto from graph)</small>")
        f0_row.addWidget(self._sim_f0_lbl)
        f0_row.addStretch(1)
        param_lay.addLayout(f0_row)
        # Inlet/outlet flux: scale BCs so total source DC = this value.
        flux_row = QHBoxLayout()
        flux_row.addWidget(QLabel("Inlet/outlet flux (nL/s):"))
        self._sim_flux_spin = QDoubleSpinBox()
        self._sim_flux_spin.setDecimals(3)
        self._sim_flux_spin.setRange(0.001, 1000.0)
        self._sim_flux_spin.setSingleStep(0.1)
        self._sim_flux_spin.setValue(1.0)
        self._sim_flux_spin.setToolTip(
            "Total inflow (Σ source DC) is scaled to this value.  All\n"
            "boundary harmonics are scaled by the same factor, so the\n"
            "AC content's relative shape is preserved.  The solver's\n"
            "conservation step then enforces Σ sinks = Σ sources.")
        flux_row.addWidget(self._sim_flux_spin)
        flux_row.addStretch(1)
        param_lay.addLayout(flux_row)
        # V-side BC handling: lets the user switch from Q-BC at V
        # (default, measured Q_V drives the network) to pressure
        # matching at V, with two flavours.  Both P-match modes turn
        # the measured Q_V into a validation target rather than an
        # input — what the user trusts physically is constrained, what
        # they trust empirically can be checked independently.
        v_bc_row = QHBoxLayout()
        v_bc_row.addWidget(QLabel("V-side BC:"))
        self._sim_v_bc_combo = _QComboBox()
        self._sim_v_bc_combo.addItems([
            "Q at V (use measured)",
            "P-match V at DC + AC rescaled by predicted DC × measured PI",
        ])
        self._sim_v_bc_combo.setToolTip(
            "Q at V: prescribe measured Q at every V (default).\n\n"
            "P-match DC + AC rescaled: two-pass.\n"
            "  Pass 1: P=0 at every V at DC → predicts Q̄_V_k per V.\n"
            "  Pass 2: prescribe Q at every V at every AC harmonic n,\n"
            "    where |Q_n_V_k| = (|Q_n^measured|/|Q̄_V_k^measured|)\n"
            "                       · |Q̄_V_k^predicted|,\n"
            "    and phase = phase(measured Q_n_V_k).\n"
            "  The mean-flow-normalised measured amplitudes become AC\n"
            "  constraints, scaled by the model's DC prediction.")
        v_bc_row.addWidget(self._sim_v_bc_combo)
        v_bc_row.addStretch(1)
        param_lay.addLayout(v_bc_row)

        # DC distribution + per-type AC averaging:
        # - Unchecked: preserve each boundary's measured ratios.
        # - Checked: every A gets the SAME waveform (DC = flux/N_src
        #   plus the complex mean of measured AC across all A's); same
        #   for V (with sign flip).  The A↔V phase difference (e.g.
        #   cardiac antiphase) is preserved across types.
        self._sim_equal_split_chk = QCheckBox(
            "Uniform per-type BCs  "
            "(one waveform per A's, one per V's)")
        self._sim_equal_split_chk.setChecked(True)
        self._sim_equal_split_chk.setToolTip(
            "Checked (default):  every A gets the SAME waveform and\n"
            "every V gets the SAME waveform.  In Custom mode that's\n"
            "the A / V spec from the spinboxes; in Measured mode it's\n"
            "the per-type complex average of stored bc_harmonics.\n"
            "DC is exactly ±flux/N per side, so mass balance is exact\n"
            "by construction and the solver rebalance is a no-op.\n\n"
            "Unchecked:  each boundary keeps its measured per-node\n"
            "ratios (Measured mode only; meaningless for Custom\n"
            "because the Custom spec is already per-type).  The\n"
            "solver's DC-conservation step may rescale slightly so\n"
            "Σ = 0; applied total flux can drift from the target.")
        param_lay.addWidget(self._sim_equal_split_chk)
        # Live BC preview reacts to flux + equal-split changes too, so
        # the user always sees the actual scaled waveform.
        self._sim_flux_spin.valueChanged.connect(
            self._refresh_custom_preview)
        self._sim_equal_split_chk.toggled.connect(
            self._refresh_custom_preview)
        # Draw the initial preview now that all the inputs exist.
        self._refresh_custom_preview()
        # Equalize the inflow stem conductance per A node.  For each
        # arterial source, walk from A through degree-2 nodes to the
        # first branch point — that's the "inflow stem".  Override
        # every stem edge's radius so each A's stem has the same total
        # Poiseuille resistance (median across A's).  Lengths are kept;
        # only radii change.  Useful when stem lengths differ
        # (e.g. one A has a single edge to first branch, another has
        # two edges) and you want symmetric inflow pressure drops.
        # Equalize A-stem conductance — UI removed for now.  The checkbox
        # object stays parentless (so `.isChecked()` still works in the
        # solver path and returns False), but isn't added to any layout.
        # To restore: re-add `param_lay.addWidget(self._sim_equalize_A_chk)`.
        self._sim_equalize_A_chk = QCheckBox(
            "Equalize A-stem conductance  (A → first-branch point)")
        self._sim_equalize_A_chk.setChecked(False)
        # Geometry override mode: measured, uniform radius only, or
        # uniform conductance (R AND L constant).  All three keep the
        # original graph intact; only the solve-time copy is altered.
        geom_row = QHBoxLayout()
        geom_row.addWidget(QLabel("Geometry:"))
        self._sim_geom_combo = _QComboBox()
        self._sim_geom_combo.addItems([
            "Measured (default)",
            "Uniform radius  (R const, L from measured)",
            "Uniform conductance  (R, L both constant)",
        ])
        self._sim_geom_combo.setToolTip(
            "Measured: use the graph's stored radius/length per edge.\n"
            "Uniform radius: override every R to 5 px; conductance is\n"
            "  then determined by per-edge length alone.\n"
            "Uniform conductance: override both R (5 px) and L (50 px),\n"
            "  so flow distribution is purely topological.\n"
            "Original graph is unchanged; only the solve sees overrides.")
        geom_row.addWidget(self._sim_geom_combo)
        geom_row.addStretch(1)
        param_lay.addLayout(geom_row)
        layout.addWidget(param_group)

        # ── Actions ──
        actions_row = QHBoxLayout()
        self._sim_run_btn = QPushButton("Run simulation")
        self._sim_run_btn.clicked.connect(self._run_simulation)
        actions_row.addWidget(self._sim_run_btn)
        self._sim_clear_btn = QPushButton("Clear simulation")
        self._sim_clear_btn.setEnabled(False)
        self._sim_clear_btn.clicked.connect(self._clear_simulation)
        actions_row.addWidget(self._sim_clear_btn)
        layout.addLayout(actions_row)

        # ── Status ──
        self._sim_status_lbl = QLabel(
            "<small><i>No simulation has been run yet.  Results live "
            "in memory under `_sim_tmp_*` fields and disappear when "
            "you close the viewer.</i></small>")
        self._sim_status_lbl.setWordWrap(True)
        layout.addWidget(self._sim_status_lbl)
        layout.addStretch(1)

        # ── No-boundary guard ──
        # If the graph carries no source / sink designations (e.g. a
        # generic mosaic without A/V assignment), every control on
        # this tab is meaningless — the solver has nothing to drive
        # the flow with.  Disable the whole tab's widgets and show
        # a single explanatory label instead of half-working spinboxes.
        if not self._source_nodes and not self._sink_nodes:
            for w in (bc_group, param_group, actions_row,
                      self._sim_status_lbl):
                _w = (w if hasattr(w, 'setEnabled')
                       else None)  # actions_row is a layout
                if _w is not None:
                    _w.setEnabled(False)
            # Disable layout-only children too (actions_row).
            for i in range(actions_row.count()):
                item = actions_row.itemAt(i).widget()
                if item is not None:
                    item.setEnabled(False)
            # Replace the bottom status with a clear message.
            self._sim_status_lbl.setText(
                "<small><b>Simulation disabled.</b>  This graph has no "
                "boundary nodes designated as sources (A) or sinks (V), "
                "so there's nothing to drive a flow simulation.  Add "
                "<code>boundary_type</code> annotations (\"source\" / "
                "\"sink\") to the relevant nodes in the gpickle to "
                "enable this tab.</small>")
            self._sim_status_lbl.setEnabled(True)  # leave the note readable

    # ══════════════════════════════════════════════════════════════════
    # Inference tab — self-contained adapter region
    # ──────────────────────────────────────────────────────────────────
    # All inference plumbing (tile carve, inversion call, diagnostic
    # render) lives in the methods below.  To swap out the underlying
    # inversion algorithm, edit only:
    #   • `_inference_get_carve_edges`  (which edges form the carve)
    #   • `_inference_run`              (the per-tile call returning a
    #                                    dict of summary scalars)
    #   • `_inference_render_diagnostic` (the matplotlib popup figure)
    # The UI builder + dispatch glue below depend only on those
    # adapter methods, so the View / Time-series / Simulate tabs
    # don't need to change when the inversion model changes.
    # ══════════════════════════════════════════════════════════════════

    def _build_inference_tab(self, layout):
        """UI: tile picker, carve overlay toggle, run button, results
        text box, render-diagnostic button.  Status label at the foot."""
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout,
            QComboBox, QCheckBox, QTextEdit, QSpinBox, QDoubleSpinBox,
        )
        # ── Tile picker ──
        pick_group = QGroupBox("Tile")
        pick_lay = QVBoxLayout(pick_group)
        # Available tile IDs are derived from `measurements_piv` on
        # graph edges.  Build once at panel-construction time.
        self._inference_tile_ids = self._inference_available_tiles()
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Tile ID:"))
        self._inf_tile_combo = QComboBox()
        for tid in self._inference_tile_ids:
            self._inf_tile_combo.addItem(str(tid), userData=int(tid))
        if not self._inference_tile_ids:
            self._inf_tile_combo.addItem("(none — no PIV tiles found)")
            self._inf_tile_combo.setEnabled(False)
        self._inf_tile_combo.currentIndexChanged.connect(
            self._on_inference_tile_change)
        pick_row.addWidget(self._inf_tile_combo)
        pick_row.addStretch(1)
        pick_lay.addLayout(pick_row)
        self._inf_show_carve_chk = QCheckBox(
            "Highlight carve on mosaic")
        self._inf_show_carve_chk.setChecked(False)
        self._inf_show_carve_chk.toggled.connect(
            self._on_inference_show_carve_toggle)
        pick_lay.addWidget(self._inf_show_carve_chk)
        layout.addWidget(pick_group)

        # ── Parameters ──
        param_group = QGroupBox("Inversion parameters")
        param_lay = QVBoxLayout(param_group)
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("Harmonics:"))
        self._inf_h1_chk = QCheckBox("H₁")
        self._inf_h1_chk.setChecked(True); self._inf_h1_chk.setEnabled(False)
        self._inf_h1_chk.setToolTip("H₁ is always included.")
        self._inf_h2_chk = QCheckBox("H₂")
        # Default off — H2 is useful for the dispersion analysis on
        # the well-resolved subset, but on H2-noisy tiles (e.g. tile
        # 26) including it triggers the warm-start fallback machinery
        # and ends up at the same D̂ as the H1-only fit anyway, at
        # ~10× the wall-clock cost.  Re-enable explicitly when you
        # need the dual-harmonic comparison.
        self._inf_h2_chk.setChecked(False)
        self._inf_h2_chk.setToolTip(
            "Off by default — including H2 only helps on tiles where\n"
            "H2 is well-resolved (clean 2× phase scaling).  On H2-noisy\n"
            "tiles it fights H1 and the warm-start fallback ends at the\n"
            "same D̂ as H1-only.  Turn on for the dispersion analysis.")
        h_row.addWidget(self._inf_h1_chk)
        h_row.addWidget(self._inf_h2_chk)
        h_row.addStretch(1)
        param_lay.addLayout(h_row)
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("FGLS outer iterations:"))
        self._inf_n_outer_spin = QSpinBox()
        self._inf_n_outer_spin.setRange(1, 10)
        self._inf_n_outer_spin.setValue(2)
        self._inf_n_outer_spin.setToolTip(
            "Outer FGLS refits of the per-channel noise model.\n"
            "Production canonical = 2 (one inner LM + one FGLS refit).")
        n_row.addWidget(self._inf_n_outer_spin)
        n_row.addStretch(1)
        param_lay.addLayout(n_row)
        # Heteroscedastic noise model with KCL-floored b_n
        self._inf_hetero_chk = QCheckBox(
            "Heteroscedastic noise: σ² = a² + (b·|Q|)²  "
            "(FGLS-fit, b floored at KCL)")
        self._inf_hetero_chk.setChecked(True)
        self._inf_hetero_chk.setToolTip(
            "When ON, FGLS fits both a (additive floor) and b\n"
            "(multiplicative coefficient) per channel from the\n"
            "squared inversion residuals.  b is floored at the\n"
            "KCL measurement-noise values below — so the inversion\n"
            "can never claim noise smaller than independently\n"
            "measured at the data level.\n\n"
            "When OFF, σ is a per-channel constant refit as RMS\n"
            "residual each FGLS pass (legacy additive-only model).")
        param_lay.addWidget(self._inf_hetero_chk)
        # KCL floor spinboxes (one per channel)
        floor_row = QHBoxLayout()
        floor_row.addWidget(QLabel("b floor (KCL):"))
        self._inf_b_floor_spins = {}
        for ch, default in (('dc', 0.29), ('h1', 0.0), ('h2', 0.0)):
            floor_row.addWidget(QLabel(f"  {ch.upper()}"))
            sp = QDoubleSpinBox()
            sp.setDecimals(3); sp.setRange(0.0, 5.0)
            sp.setSingleStep(0.01); sp.setValue(default)
            sp.setToolTip(
                "Lower bound on the multiplicative coefficient b\n"
                "for this channel.  FGLS-fitted b is clipped to this\n"
                "value (the inversion can't claim less noise than\n"
                "KCL says the measurement has).")
            floor_row.addWidget(sp)
            self._inf_b_floor_spins[ch] = sp
        floor_row.addStretch(1)
        param_lay.addLayout(floor_row)
        layout.addWidget(param_group)

        # ── Actions ──
        act_row = QHBoxLayout()
        self._inf_run_btn = QPushButton("Run inversion")
        self._inf_run_btn.clicked.connect(self._on_inference_run_clicked)
        act_row.addWidget(self._inf_run_btn)
        self._inf_diag_btn = QPushButton("Render 6-panel diagnostic")
        self._inf_diag_btn.setEnabled(False)
        self._inf_diag_btn.clicked.connect(
            self._on_inference_diagnostic_clicked)
        act_row.addWidget(self._inf_diag_btn)
        layout.addLayout(act_row)

        # ── Results ──
        self._inf_results_box = QTextEdit()
        self._inf_results_box.setReadOnly(True)
        self._inf_results_box.setMaximumHeight(220)
        self._inf_results_box.setPlaceholderText(
            "Pick a tile and click 'Run inversion'.")
        layout.addWidget(self._inf_results_box)
        # Status
        self._inf_status_lbl = QLabel(
            "<small><i>Inference results live in memory; nothing is "
            "written back to the gpickle.</i></small>")
        self._inf_status_lbl.setWordWrap(True)
        layout.addWidget(self._inf_status_lbl)

        # Carve overlay layer slots (created lazily on first toggle).
        self._inference_carve_layer = None        # edge Shapes layer
        self._inference_carve_node_layer = None   # boundary-node Points layer
        self._inference_last_result = None
        # OF region overlay on the mosaic — curved orange ribbon showing
        # the actual integration region for the last OF run.  Persists
        # until the next OF run replaces it or "Clear OF region" is hit.
        self._of_region_layer = None
        # No initial carve render — the checkbox defaults to off so
        # the launch view stays uncluttered.  The first toggle-on
        # triggers `_refresh_inference_carve_overlay` lazily.

    # ── Adapter ──────────────────────────────────────────────────────
    # These three methods are the ONLY entry points into the inversion
    # algorithm.  Swap them out (or replace the whole region) to change
    # what "Run inversion" means.
    def _inference_available_tiles(self) -> List[int]:
        """Return a sorted list of tile IDs the inversion can run on.
        Default rule: any tile that appears as `tile_id` on at least
        one usable PIV measurement."""
        seen = set()
        for _, _, d in self.G.edges(data=True):
            for m in (d.get('measurements_piv') or []):
                tid = m.get('tile_id')
                if tid is None:
                    continue
                try:
                    seen.add(int(tid))
                except (TypeError, ValueError):
                    pass
        return sorted(seen)

    def _inference_get_carve_topology(self, tile_id: int) -> dict:
        """Return carve geometry for the given tile as a dict with keys
        `edges` (list of (u,v)) and `boundary_nodes` (list of node ids).
        Wraps `scripts.inspect_tile.build_tile_problem`."""
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _scripts = _Path(self.graph_path).resolve().parent
            # Walk up to find the project root with a `scripts/` folder.
            for _ancestor in [_scripts] + list(_scripts.parents):
                if (_ancestor / 'scripts' / 'inspect_tile.py').exists():
                    _root = _ancestor
                    break
            else:
                from pertile.viewer import mosaic_readonly_app as _self_mod
                _root = _Path(_self_mod.__file__).resolve().parents[2]
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))
            if str(_root / 'scripts') not in _sys.path:
                _sys.path.insert(0, str(_root / 'scripts'))
            from inspect_tile import build_tile_problem
            prob = build_tile_problem(self.G, int(tile_id))
        except Exception as e:
            print(f"  inference: carve extraction failed ({type(e).__name__}: {e})")
            return {'edges': [], 'boundary_nodes': []}
        return {
            'edges': [tuple(e) for e in prob.get('edges_in', [])],
            'boundary_nodes': list(prob.get('boundary_nodes', [])),
        }

    def _inference_run(self, tile_id: int, harmonics, n_outer: int,
                        use_heteroscedastic: bool = True,
                        b_floor_nL=None,
                        warmstart_threshold: float = 0.5):
        """Run the inversion on one tile with the profile-likelihood
        warm-start hybrid.

        Flow:
          1. Initial LM at D_init = 1.3e-3 (production_fit default).
          2. Profile-likelihood scan with the LM's FGLS noise model.
          3. If |D̂_LM − D̂_profile| / D̂_profile > `warmstart_threshold`
             (default 0.5), re-run LM warm-started at D̂_profile.  This
             refits both P_b and (a, b) at the global basin, so the
             noise model is self-consistent with the final D̂.
          4. Keep whichever result has lower χ²; attach the profile
             scan to the chosen result so the diagnostic can render
             without recomputing.

        Returns a result dict shaped like scripts/production_fit.py's
        canonical return, plus the keys:
          profile_D, profile_chi2, profile_D_hat,
          warmstart_triggered (bool), warmstart_rel_gap (float),
          warmstart_chi2_before / _D_before / _did_not_improve
          (when triggered).
        """
        import sys as _sys
        from pathlib import Path as _Path
        from pertile.viewer import mosaic_readonly_app as _self_mod
        _root = _Path(_self_mod.__file__).resolve().parents[2]
        for p in (str(_root), str(_root / 'scripts')):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        from production_fit import production_fit, nL_per_m3
        from inspect_tile import profile_likelihood

        def _call(D_init=None):
            kw = dict(
                harmonics=tuple(harmonics),
                n_outer=int(n_outer),
                use_heteroscedastic=bool(use_heteroscedastic),
                b_floor_nL=(dict(b_floor_nL)
                             if b_floor_nL is not None else None),
                verbose=False,
            )
            if D_init is not None:
                kw['D_init'] = float(D_init)
            return production_fit(self.G, int(tile_id), **kw)

        # ── Step 1: initial LM at the default D_init ──
        res = _call()
        if res.get('error'):
            return res

        # ── Step 2: profile-likelihood scan with the LM's noise ──
        a_DC_SI = res['a_DC_fit'] / nL_per_m3
        a_H1_SI = res['a_H1_fit'] / nL_per_m3
        b_DC = float(res.get('b_DC_fit') or 0.0)
        b_H1 = float(res.get('b_H1_fit') or 0.0)
        noise_dict = {
            'dc': (a_DC_SI, b_DC),
            'h1': (a_H1_SI, b_H1),
        }
        prof_D, prof_chi2 = profile_likelihood(res['prob'], noise_dict)
        finite_prof = np.isfinite(prof_chi2)
        if not finite_prof.any():
            # Profile failed — return the LM result with markers so
            # the result panel can still report something useful.
            res['profile_D'] = prof_D
            res['profile_chi2'] = prof_chi2
            res['profile_D_hat'] = float('nan')
            res['warmstart_triggered'] = False
            res['warmstart_rel_gap'] = float('nan')
            return res
        i_min = int(np.nanargmin(prof_chi2))
        D_profile_hat = float(prof_D[i_min])
        rel_gap = (abs(res['D_hat'] - D_profile_hat)
                   / max(D_profile_hat, 1e-30))

        # ── Step 3: trigger check ──
        if rel_gap <= warmstart_threshold:
            res['profile_D'] = prof_D
            res['profile_chi2'] = prof_chi2
            res['profile_D_hat'] = D_profile_hat
            res['warmstart_triggered'] = False
            res['warmstart_rel_gap'] = float(rel_gap)
            return res

        # ── Step 4: warm-start LM at the profile's global basin ──
        res2 = _call(D_init=D_profile_hat)
        if res2.get('error'):
            res['profile_D'] = prof_D
            res['profile_chi2'] = prof_chi2
            res['profile_D_hat'] = D_profile_hat
            res['warmstart_triggered'] = True
            res['warmstart_rel_gap'] = float(rel_gap)
            res['warmstart_failed'] = res2['error']
            return res

        # ── Step 5: pick the lower-χ² result ──
        if res2['chi2'] < res['chi2']:
            # Adopt res2.  IMPORTANT: re-run the profile scan with
            # res2's (a, b) so the trajectory (which uses res2's
            # internal chi²) and the profile share the same noise
            # scale.  Otherwise the trajectory plots ~200-unit chi²
            # above the profile curve (different σ → different
            # absolute chi²).
            noise_dict2 = {
                'dc': (res2['a_DC_fit'] / nL_per_m3,
                       float(res2.get('b_DC_fit') or 0.0)),
                'h1': (res2['a_H1_fit'] / nL_per_m3,
                       float(res2.get('b_H1_fit') or 0.0)),
            }
            prof_D2, prof_chi2_2 = profile_likelihood(
                res2['prob'], noise_dict2)
            finite2 = np.isfinite(prof_chi2_2)
            if finite2.any():
                D_profile_hat2 = float(
                    prof_D2[int(np.nanargmin(prof_chi2_2))])
                res2['profile_D'] = prof_D2
                res2['profile_chi2'] = prof_chi2_2
                res2['profile_D_hat'] = D_profile_hat2
            else:
                # Profile-2 failed — fall back to the LM-#1 profile
                # (trajectory may be visually offset, but other panels
                # still work).
                res2['profile_D'] = prof_D
                res2['profile_chi2'] = prof_chi2
                res2['profile_D_hat'] = D_profile_hat
            res2['warmstart_triggered'] = True
            res2['warmstart_rel_gap'] = float(rel_gap)
            res2['warmstart_chi2_before'] = float(res['chi2'])
            res2['warmstart_D_before'] = float(res['D_hat'])
            res2['warmstart_did_not_improve'] = False
            return res2
        # Warm-start didn't improve χ² — that's the "H2 fighting H1"
        # signature.  Auto-run an H1-only fit so the user can see the
        # H1-trusted D̂ alongside the H1+H2 result; for the tile's
        # production-map entry the H1-only value is the safer choice.
        res['profile_D'] = prof_D
        res['profile_chi2'] = prof_chi2
        res['profile_D_hat'] = D_profile_hat
        res['warmstart_triggered'] = True
        res['warmstart_rel_gap'] = float(rel_gap)
        res['warmstart_chi2_before'] = float(res['chi2'])
        res['warmstart_chi2_after'] = float(res2['chi2'])
        res['warmstart_D_after'] = float(res2['D_hat'])
        res['warmstart_did_not_improve'] = True
        if 2 in harmonics:
            h1_only = production_fit(
                self.G, int(tile_id),
                harmonics=(1,),
                n_outer=int(n_outer),
                use_heteroscedastic=bool(use_heteroscedastic),
                b_floor_nL=(dict(b_floor_nL)
                             if b_floor_nL is not None else None),
                verbose=False,
            )
            if not h1_only.get('error'):
                # Keep just the headline numbers — we don't need the
                # full T / prob payload again.
                res['h1_only_fit'] = dict(
                    D_hat=float(h1_only['D_hat']),
                    sigma_D=float(h1_only['sigma_D']),
                    rel_sigma_D=float(h1_only['rel_sigma_D']),
                    chi2=float(h1_only['chi2']),
                    iters=int(h1_only['iters']),
                    a_DC_fit=float(h1_only['a_DC_fit']),
                    a_H1_fit=float(h1_only['a_H1_fit']),
                    b_DC_fit=(float(h1_only['b_DC_fit'])
                              if h1_only.get('b_DC_fit') is not None
                              else None),
                    b_H1_fit=(float(h1_only['b_H1_fit'])
                              if h1_only.get('b_H1_fit') is not None
                              else None),
                )
        return res

    def _inference_render_diagnostic(self, tile_id, result):
        """Render the canonical 6-panel inspect_tile diagnostic for
        a result dict produced by `_inference_run`.  Pops up a
        matplotlib window."""
        import sys as _sys
        from pathlib import Path as _Path
        from pertile.viewer import mosaic_readonly_app as _self_mod
        _root = _Path(_self_mod.__file__).resolve().parents[2]
        for p in (str(_root), str(_root / 'scripts')):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        import inspect_tile
        from production_fit import nL_per_m3
        prob = result['prob']
        sig_dc = result['sigma_dc_final']
        sig_h_dict = result['sigma_h_final']
        r_dc_w = result['r_dc'] / sig_dc[prob['valid_dc']]
        r_h1_w = result['r_h'][1] / sig_h_dict[1][prob['valid_h1']]
        chi2_total = float(np.sum(r_dc_w ** 2)
                           + np.sum(np.abs(r_h1_w) ** 2))
        if 2 in result['harmonics']:
            r_h2_w = (result['r_h'][2]
                      / sig_h_dict[2][:len(result['r_h'][2])])
            chi2_total += float(np.sum(np.abs(r_h2_w) ** 2))
        n_rows = int(prob['valid_dc'].sum()) + 2 * int(prob['valid_h1'].sum())
        if 2 in result['harmonics']:
            n_rows += 2 * len(result['r_h'][2])
        n_params = (1 + (len(prob['boundary_nodes']) - 1)
                    + 2 * len(prob['boundary_nodes']) * len(result['harmonics']))
        dof = max(n_rows - n_params, 1)
        full = dict(P_DC=result['P_DC'], P_H1=result['P_H'][1],
                    chi2_total=chi2_total, dof=dof,
                    r_dc=r_dc_w, r_h1=r_h1_w)
        # Reuse the profile scan computed by `_inference_run` (which
        # also drives the warm-start trigger) so we don't pay for a
        # second grid evaluation.  Fall back to computing it now if
        # the result dict doesn't carry it (e.g. older runs).
        a_DC_SI = result['a_DC_fit'] / nL_per_m3
        a_H1_SI = result['a_H1_fit'] / nL_per_m3
        b_DC = float(result.get('b_DC_fit') or 0.0)
        b_H1 = float(result.get('b_H1_fit') or 0.0)
        noise_dict = {
            'dc': (a_DC_SI, b_DC),
            'h1': (a_H1_SI, b_H1),
        }
        prof_D = result.get('profile_D')
        prof_chi2 = result.get('profile_chi2')
        if prof_D is None or prof_chi2 is None:
            prof_D, prof_chi2 = inspect_tile.profile_likelihood(
                prob, noise_dict)
        # Render to a temp PNG then display the PNG in a Qt dialog.
        # Going through `plt.show()` doesn't work because napari's Qt
        # event loop is already running — matplotlib's blocking show
        # would try to start a second one and the figure renders as a
        # black box.  PNG → QPixmap → QDialog sidesteps that entirely.
        import tempfile
        with tempfile.NamedTemporaryFile(
                suffix='.png', delete=False) as _tmp:
            _tmp_path = _tmp.name
        inspect_tile.render(
            int(tile_id), prob, full, prof_D, prof_chi2,
            noise_dict,       # per-channel (a, b) for DC and H1
            _tmp_path, show=False,
            lm_trajectory=result.get('lm_history'))
        self._show_diagnostic_png(_tmp_path, int(tile_id), result)

    def _show_diagnostic_png(self, png_path: str, tile_id: int, result: dict):
        """Pop up a Qt dialog displaying a saved diagnostic PNG.

        The dialog is non-modal, has a fixed window title with the
        tile_id and headline D̂, and is retained on `self` so Python's
        GC doesn't close it the instant this method returns."""
        from qtpy.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QScrollArea)
        from qtpy.QtGui import QPixmap
        from qtpy.QtCore import Qt
        dlg = QDialog(self.viewer.window._qt_window)
        dlg.setWindowTitle(
            f"Inference diagnostic — tile {tile_id}  "
            f"D̂ = {result['D_hat']:.3e},  "
            f"σ_D/D̂ = {result['rel_sigma_D']:.1%}")
        # Layout: scroll area → label → pixmap
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        lbl = QLabel()
        pixmap = QPixmap(png_path)
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl)
        lay.addWidget(scroll)
        # Match the rendered figure size (default render dpi=180 →
        # 15×9 inches → ~2700×1620 px) but cap to 90% of screen.
        try:
            screen = self.viewer.window._qt_window.screen().availableGeometry()
            target_w = min(int(pixmap.width()),
                           int(0.9 * screen.width()))
            target_h = min(int(pixmap.height() + 50),
                           int(0.9 * screen.height()))
        except Exception:
            target_w, target_h = 1200, 800
        dlg.resize(target_w, target_h)
        # Keep a reference so the dialog stays alive after this method
        # returns.  Reuse a single list slot so old dialogs are GC'd
        # naturally if the user closes them.
        if not hasattr(self, '_inf_diag_dialogs'):
            self._inf_diag_dialogs = []
        # Prune any that have already been closed.
        self._inf_diag_dialogs = [
            d for d in self._inf_diag_dialogs if d.isVisible()]
        self._inf_diag_dialogs.append(dlg)
        dlg.show()

    # ── UI glue ──────────────────────────────────────────────────────
    def _on_inference_tile_change(self, _idx=None):
        # Stale: previous result was for a different tile.
        self._inf_diag_btn.setEnabled(False)
        self._inf_results_box.setPlainText("")
        self._inference_last_result = None
        if self._inf_show_carve_chk.isChecked():
            self._refresh_inference_carve_overlay()

    def _on_inference_show_carve_toggle(self, checked: bool):
        if not checked:
            self._tear_down_inference_carve_overlay()
        else:
            self._refresh_inference_carve_overlay()

    def _refresh_inference_carve_overlay(self):
        """Update the carve highlight on the mosaic — both the carved
        edges (thick yellow lines) and the carve boundary nodes (filled
        orange dots) — to reflect the currently-selected tile."""
        self._tear_down_inference_carve_overlay()
        tile_id = self._inf_tile_combo.currentData()
        if tile_id is None:
            return
        topo = self._inference_get_carve_topology(int(tile_id))
        edges = topo['edges']
        bnodes = topo['boundary_nodes']
        H = self.mosaic_height
        # ── Edge overlay (thick yellow lines) ──
        lines = []
        for (u, v) in edges:
            if not self.G.has_edge(u, v):
                continue
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
        if lines:
            self._inference_carve_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color='#ffd400',  # bright yellow
                edge_width=8.0,
                opacity=0.85,
                name=f'Inference carve (tile {tile_id})',
            )
        # ── Boundary-node overlay (filled dots) ──
        pts = []
        for n in bnodes:
            try:
                x = self.G.nodes[n]['x']
                y = self.G.nodes[n]['y']
            except KeyError:
                continue
            pts.append([H - y, x])
        if pts:
            self._inference_carve_node_layer = self.viewer.add_points(
                np.asarray(pts), size=18,
                face_color='#ff6600',  # orange
                border_color='white',
                opacity=0.95,
                name=f'Carve boundary nodes (tile {tile_id})',
            )

    def _tear_down_inference_carve_overlay(self):
        for attr in ('_inference_carve_layer',
                     '_inference_carve_node_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)

    def _on_inference_run_clicked(self):
        from qtpy.QtWidgets import QApplication
        import time
        tile_id = self._inf_tile_combo.currentData()
        if tile_id is None:
            return
        harmonics = [1]
        if self._inf_h2_chk.isChecked():
            harmonics.append(2)
        n_outer = int(self._inf_n_outer_spin.value())
        use_hetero = bool(self._inf_hetero_chk.isChecked())
        b_floor_nL = {ch: float(sp.value())
                      for ch, sp in self._inf_b_floor_spins.items()}
        self._inf_run_btn.setEnabled(False)
        self._inf_diag_btn.setEnabled(False)
        noise_tag = ("heteroscedastic + KCL floor"
                     if use_hetero else "additive only")
        self._inf_status_lbl.setText(
            f"<small>Running inversion on tile {tile_id} "
            f"(harmonics={harmonics}, n_outer={n_outer}, "
            f"{noise_tag}) …</small>")
        QApplication.processEvents()
        t0 = time.time()
        try:
            res = self._inference_run(
                int(tile_id), harmonics, n_outer,
                use_heteroscedastic=use_hetero,
                b_floor_nL=b_floor_nL)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Inversion failed: "
                f"{type(e).__name__}: {e}</font></small>")
            self._inf_run_btn.setEnabled(True)
            return
        elapsed = time.time() - t0
        if res.get('error'):
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>{res['error']}</font></small>")
            self._inf_run_btn.setEnabled(True)
            return
        # Format result summary.
        lines = [
            f"Tile {tile_id}  ·  harmonics={harmonics}  ·  n_outer={n_outer}",
            f"  D̂        = {res['D_hat']:.3e}  1/Pa",
            f"  σ_D       = {res['sigma_D']:.3e}",
            f"  σ_D / D̂   = {res['rel_sigma_D']:.1%}",
            f"  χ²        = {res['chi2']:.2f}",
            f"  iters     = {res['iters']}",
        ]
        # Noise-model summary.  Heteroscedastic prints (a, b) per
        # channel and tags whether the KCL floor was binding.
        is_hetero = bool(res.get('use_heteroscedastic'))
        if is_hetero:
            lines.append("  Noise model: σ² = a² + (b·|Q|)²  "
                          "(FGLS, b floored at KCL)")
            floored = res.get('noise_floored') or {}
            for ch_disp, ch_key in (('DC', 'dc'), ('H1', 'h1'),
                                      ('H2', 'h2')):
                a_v = res.get(f'a_{ch_disp}_fit')
                b_v = res.get(f'b_{ch_disp}_fit')
                if a_v is None or b_v is None:
                    continue
                tag = (' [FLOORED]' if floored.get(ch_key)
                       else '')
                lines.append(
                    f"    {ch_disp}: a = {a_v:.4f} nL/s,  "
                    f"b = {b_v:.3f}{tag}")
        else:
            lines.append("  Noise model: additive constant (legacy)")
            lines.append(
                f"    a_DC = {res['a_DC_fit']:.4f} nL/s")
            lines.append(
                f"    a_H1 = {res['a_H1_fit']:.4f} nL/s")
            if 2 in harmonics:
                lines.append(
                    f"    a_H2 = {res['a_H2_fit']:.4f} nL/s")
        n_edges = len(res['prob']['edges_in'])
        n_valid_dc = int(res['prob']['valid_dc'].sum())
        n_valid_h1 = int(res['prob']['valid_h1'].sum())
        lines.append(
            f"  edges     = {n_edges}  ({n_valid_dc} valid DC, "
            f"{n_valid_h1} valid H1)")

        # Profile-likelihood sanity check + warm-start status.
        D_profile_hat = res.get('profile_D_hat')
        if D_profile_hat is not None and np.isfinite(D_profile_hat):
            rel_gap = res.get('warmstart_rel_gap', float('nan'))
            triggered = bool(res.get('warmstart_triggered'))
            lines.append("")
            lines.append(
                f"  Profile min: D = {D_profile_hat:.3e}  "
                f"(|ΔD|/D_profile = {rel_gap:.1%})")
            if triggered:
                if res.get('warmstart_did_not_improve'):
                    lines.append(
                        f"  Warm-start fired but did NOT improve χ² —")
                    lines.append(
                        f"    signature of H2 fighting H1 "
                        f"(both basins reach same χ²)")
                    h1_only = res.get('h1_only_fit')
                    if h1_only is not None:
                        lines.append("")
                        lines.append(
                            "  ── H1-only sanity fit "
                            "(auto-run; trust this for tile-level D̂) ──")
                        lines.append(
                            f"    D̂        = {h1_only['D_hat']:.3e}  1/Pa")
                        lines.append(
                            f"    σ_D / D̂   = "
                            f"{h1_only['rel_sigma_D']:.1%}")
                        lines.append(
                            f"    χ²        = {h1_only['chi2']:.2f}  "
                            f"(iters = {h1_only['iters']})")
                else:
                    lines.append(
                        f"  ★ WARM-STARTED from D_profile  "
                        f"(χ² {res['warmstart_chi2_before']:.2f} → "
                        f"{res['chi2']:.2f},  "
                        f"D {res['warmstart_D_before']:.3e} → "
                        f"{res['D_hat']:.3e})")
        self._inf_results_box.setPlainText("\n".join(lines))
        self._inf_status_lbl.setText(
            f"<small>OK · {elapsed:.1f}s · "
            f"D̂ = {res['D_hat']:.3e} 1/Pa.  Click "
            f"<b>Render 6-panel diagnostic</b> for the full inspector "
            f"figure.</small>")
        self._inference_last_result = (int(tile_id), res)
        self._inf_run_btn.setEnabled(True)
        self._inf_diag_btn.setEnabled(True)
        # Auto-enable the carve overlay on the first successful run so
        # the user immediately sees which edges/nodes feed the inversion.
        if not self._inf_show_carve_chk.isChecked():
            self._inf_show_carve_chk.setChecked(True)

    def _on_inference_diagnostic_clicked(self):
        from qtpy.QtWidgets import QApplication
        if self._inference_last_result is None:
            return
        tile_id, res = self._inference_last_result
        self._inf_status_lbl.setText(
            "<small>Rendering 6-panel diagnostic …</small>")
        QApplication.processEvents()
        try:
            self._inference_render_diagnostic(tile_id, res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Diagnostic render failed: "
                f"{type(e).__name__}: {e}</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>Diagnostic for tile {tile_id} rendered.</small>")

    # ══════════════════════════════════════════════════════════════════
    # End of inference adapter region.
    # ══════════════════════════════════════════════════════════════════

    def _detect_sim_f0_hz(self) -> float:
        """Pick a sensible default heart rate for the solver.  Priority:
        1. `G.graph['sim_f0_hz']` if set by the analysis pipeline.
        2. Median of `G.graph['tile_f0_piv']` values.
        3. Median of `G.graph['tile_f0s']` (legacy kymograph).
        4. Fallback constant 2.5 Hz.
        """
        f0 = self.G.graph.get('sim_f0_hz')
        if f0 is not None:
            try:
                f0 = float(f0)
                if np.isfinite(f0) and f0 > 0:
                    return f0
            except (TypeError, ValueError):
                pass
        for key in ('tile_f0_piv', 'tile_f0s'):
            d = self.G.graph.get(key) or {}
            vals = []
            for v in d.values():
                try:
                    vv = float(v)
                    if np.isfinite(vv) and vv > 1.0:
                        vals.append(vv)
                except (TypeError, ValueError):
                    pass
            if vals:
                return float(np.median(vals))
        return 2.5

    def _find_inflow_stem(self, a_node) -> List[Tuple[int, int]]:
        """Walk from a boundary node through degree-2 nodes to the
        first node whose degree ≠ 2 (branch point or dead end).
        Returns the ordered list of edges traversed."""
        edges: List[Tuple[int, int]] = []
        prev = None
        curr = a_node
        while True:
            if curr != a_node and self.G.degree(curr) > 2:
                break  # branch point
            nbrs = [n for n in self.G.neighbors(curr) if n != prev]
            if not nbrs:
                break  # dead end
            nxt = nbrs[0]
            edges.append((curr, nxt))
            prev = curr
            curr = nxt
            # Cap to avoid pathological infinite loops on self-loops.
            if len(edges) > 64:
                break
        return edges

    def _stem_poiseuille_resistance(self, stem_edges) -> float:
        """Sum Σ 8μ L_e / (π R_e⁴) along a stem (Pa·s/m³)."""
        from ..analysis.config import PX_SIZE_UM
        from ..analysis.transmission_line import MU_DEFAULT
        px_to_m = PX_SIZE_UM * 1e-6
        total = 0.0
        for (u, v) in stem_edges:
            d = self.G.edges[u, v]
            try:
                R_px = float(d.get('radius'))
                L_px = float(d.get('length'))
            except (TypeError, ValueError):
                continue
            if not (R_px > 0 and L_px > 0):
                continue
            R_m = R_px * px_to_m
            L_m = L_px * px_to_m
            total += 8.0 * MU_DEFAULT * L_m / (np.pi * R_m ** 4)
        return total

    def _run_simulation(self):
        """Call `solve_transmission_line` with the stored boundary
        `bc_harmonics` (picked up automatically when no override is
        passed) and write per-edge results to the in-memory graph."""
        import time
        from qtpy.QtWidgets import QApplication
        if not self._source_nodes and not self._sink_nodes:
            self._sim_status_lbl.setText(
                "<small><font color='#c33'>No boundary nodes on this "
                "graph (need `boundary_type` set).  Cannot run.</font>"
                "</small>")
            return
        D = (float(self._sim_D_mantissa.value())
             * (10.0 ** int(self._sim_D_exponent.value())))
        f0 = (float(self._sim_detected_f0)
              if np.isfinite(self._sim_detected_f0) else 2.5)
        target_flux = float(self._sim_flux_spin.value())
        # Geometry override mode — combo index ⇒ behaviour:
        #   0 → measured            (no override)
        #   1 → uniform radius      (R const, L preserved)
        #   2 → uniform conductance (R AND L const)
        geom_mode_idx = int(self._sim_geom_combo.currentIndex())
        override_R = geom_mode_idx in (1, 2)
        override_L = geom_mode_idx == 2
        geom_tag = (
            "" if geom_mode_idx == 0
            else ", uniform R" if geom_mode_idx == 1
            else ", uniform R+L")
        split_tag = (", equal-split" if self._sim_equal_split_chk.isChecked()
                     else "")
        eqA_tag = (", A-stem eq" if self._sim_equalize_A_chk.isChecked()
                   else "")
        split_tag += eqA_tag
        _v_bc_idx = int(self._sim_v_bc_combo.currentIndex())
        v_bc_tag = ("" if _v_bc_idx == 0
                    else ", P-match V@DC + AC rescaled")
        bc_src_tag = (", CUSTOM BCs"
                      if self._sim_bc_radio_custom.isChecked() else "")
        self._sim_run_btn.setEnabled(False)
        self._sim_status_lbl.setText(
            f"<small>Running solve at D = {D:.3g}, f₀ = {f0:.3f} Hz "
            f"flux = {target_flux:.3g} nL/s"
            f"{bc_src_tag}{geom_tag}{split_tag}{v_bc_tag} …</small>")
        QApplication.processEvents()

        # Build bc_harmonics_override.  Stored `bc_harmonics` on this
        # graph carries +outflow at sinks; the solver's "Q into network"
        # convention is +inflow everywhere, so we flip sinks by −1.
        # Two distribution modes:
        #   measured-ratio (default): scale all by `target_flux / |Σ src
        #     DC|` so the relative per-node imbalance is preserved.  The
        #     solver's DC-conservation step may then rescale slightly.
        #   equal-split: set DC = ±target_flux / N per type, keep AC
        #     harmonics measured.  Mass balance exact → no rebalance.
        equal_split = bool(self._sim_equal_split_chk.isChecked())
        custom_bc = bool(self._sim_bc_radio_custom.isChecked())
        bc_signed: Dict[int, np.ndarray] = {}
        sum_src_dc = 0.0
        for n in (self._source_nodes + self._sink_nodes):
            bch = self.G.nodes[n].get('bc_harmonics')
            if bch is None:
                continue
            s = -1.0 if n in self._sink_nodes else 1.0
            arr = s * np.asarray(bch, dtype=complex)
            bc_signed[n] = arr
            if n in self._source_nodes and np.isfinite(arr[0].real):
                sum_src_dc += float(arr[0].real)

        bc_override: Dict[int, np.ndarray] = {}
        if custom_bc:
            # Build BCs from the user-specified waveforms.  A and V
            # specs are now independent — each side has its own amps +
            # phases.  Every A boundary gets the A spec, every V
            # boundary gets the V spec (with sign convention applied so
            # V's Q-into-network is negative-DC).
            (amps_A, phases_A), (amps_V, phases_V) = \
                self._read_custom_waveform()
            n_src = max(1, sum(1 for n in self._source_nodes
                               if n in bc_signed
                               or self.G.nodes[n].get('bc_harmonics')
                                   is not None))
            n_snk = max(1, sum(1 for n in self._sink_nodes
                               if n in bc_signed
                               or self.G.nodes[n].get('bc_harmonics')
                                   is not None))
            Qbar_A = +target_flux / n_src
            Qbar_V = -target_flux / n_snk  # canonical sign flip
            ac_A = [amps_A[i] * Qbar_A * np.exp(1j * phases_A[i])
                    for i in range(3)]
            ac_V = [amps_V[i] * Qbar_V * np.exp(1j * phases_V[i])
                    for i in range(3)]
            A_arr = np.array([Qbar_A] + ac_A, dtype=complex)
            V_arr = np.array([Qbar_V] + ac_V, dtype=complex)
            for n in self._source_nodes:
                bc_override[n] = A_arr.copy()
            for n in self._sink_nodes:
                bc_override[n] = V_arr.copy()
        elif equal_split:
            # Each boundary inside a type (A or V) gets THE SAME
            # waveform.  AC harmonics are the per-type complex average
            # of the measured BCs, so the A↔V phase difference (e.g.
            # ventricular ejection vs atrial suction) is preserved
            # across types but per-node spread within a type is gone.
            src_arrs = [bc_signed[n] for n in self._source_nodes
                        if n in bc_signed]
            snk_arrs = [bc_signed[n] for n in self._sink_nodes
                        if n in bc_signed]
            n_src = max(1, len(src_arrs))
            n_snk = max(1, len(snk_arrs))
            src_avg = (np.mean(np.stack(src_arrs, axis=0), axis=0)
                       if src_arrs else None)
            snk_avg = (np.mean(np.stack(snk_arrs, axis=0), axis=0)
                       if snk_arrs else None)
            if src_avg is not None:
                src_avg = src_avg.copy()
                src_avg[0] = +target_flux / n_src
            if snk_avg is not None:
                snk_avg = snk_avg.copy()
                snk_avg[0] = -target_flux / n_snk
            for n in bc_signed:
                if n in self._source_nodes and src_avg is not None:
                    bc_override[n] = src_avg.copy()
                elif n in self._sink_nodes and snk_avg is not None:
                    bc_override[n] = snk_avg.copy()
        else:
            if abs(sum_src_dc) > 1e-12:
                flux_scale = target_flux / abs(sum_src_dc)
            else:
                flux_scale = 1.0
            for n, arr in bc_signed.items():
                bc_override[n] = flux_scale * arr

        # Geometry overrides: save the keys we touch, override for the
        # duration of the solve, restore in `finally` so the mutation
        # is invisible to the rest of the viewer state.
        R_UNIFORM_PX, L_UNIFORM_PX = 5.0, 50.0
        radius_keys = ('radius', 'radius_px', 'radius_px_true')
        length_keys = ('length', 'length_true', 'path_length_px')
        touched_keys = []
        if override_R:
            touched_keys.extend(radius_keys)
        if override_L:
            touched_keys.extend(length_keys)
        saved_geom: Dict[Tuple[int, int], dict] = {}
        if touched_keys:
            for u, v, d in self.G.edges(data=True):
                saved_geom[(u, v)] = {k: d.get(k) for k in touched_keys}
                if override_R:
                    for k in radius_keys:
                        d[k] = R_UNIFORM_PX
                if override_L:
                    for k in length_keys:
                        d[k] = L_UNIFORM_PX

        # A-stem conductance equalization.  For each A node, walk to
        # the first branch and set every edge in that stem to a
        # per-stem-uniform radius such that Σ 8μL/(πR⁴) = target_R,
        # where target_R is the median across all A stems.  Lengths
        # preserved.  Applied AFTER any uniform-R/L override so the
        # stems get the correct per-stem radius rather than the
        # uniform value.
        equalize_A = bool(self._sim_equalize_A_chk.isChecked())
        if equalize_A and self._source_nodes:
            from ..analysis.config import PX_SIZE_UM
            from ..analysis.transmission_line import MU_DEFAULT
            px_to_m = PX_SIZE_UM * 1e-6
            stems = {a: self._find_inflow_stem(a)
                     for a in self._source_nodes}
            resistances = {a: self._stem_poiseuille_resistance(e)
                           for a, e in stems.items()}
            finite_R = [r for r in resistances.values()
                        if r > 0 and np.isfinite(r)]
            if finite_R:
                target_R = float(np.median(finite_R))
                for a, edges in stems.items():
                    if not edges:
                        continue
                    L_total_m = 0.0
                    for (u, v) in edges:
                        L_px = self.G.edges[u, v].get('length')
                        if L_px is None:
                            continue
                        try:
                            L_total_m += float(L_px) * px_to_m
                        except (TypeError, ValueError):
                            continue
                    if L_total_m <= 0 or target_R <= 0:
                        continue
                    R_new_m = (8.0 * MU_DEFAULT * L_total_m
                                / (np.pi * target_R)) ** 0.25
                    R_new_px = R_new_m / px_to_m
                    for (u, v) in edges:
                        key = (u, v) if self.G.has_edge(u, v) else (v, u)
                        if key not in saved_geom:
                            saved_geom[key] = {}
                        ed = self.G.edges[u, v]
                        for rk in radius_keys:
                            if rk not in saved_geom[key]:
                                saved_geom[key][rk] = ed.get(rk)
                            ed[rk] = R_new_px

        # V-side BC mode (combo index → behaviour):
        #   0 → Q at V (measured)            — standard Q-BC everywhere
        #   1 → P-match V at DC + AC rescaled — Option B.  Pass 1 sets
        #        P=0 at every V to predict Q̄_V from physics.  Pass 2
        #        prescribes per-V AC = (|measured_n|/|measured_dc|)
        #        × Q̄_V_predicted, with measured phases — i.e. the
        #        mean-flow-normalised measured amplitudes used as
        #        constraints, scaled by the model's DC prediction.
        v_bc_mode = int(self._sim_v_bc_combo.currentIndex())

        def _q_into_node(edge_flows_dict, node):
            """Sum Q on incident edges, signed so result = Q flowing
            INTO `node`.  edge_flows is keyed (u, v) with +Q along
            u→v."""
            n_harmonics = 3
            acc = np.zeros(n_harmonics + 1, dtype=complex)
            for nbr in self.G.neighbors(node):
                if (nbr, node) in edge_flows_dict:
                    acc += np.asarray(edge_flows_dict[(nbr, node)],
                                       dtype=complex)
                elif (node, nbr) in edge_flows_dict:
                    acc -= np.asarray(edge_flows_dict[(node, nbr)],
                                       dtype=complex)
            return acc

        try:
            # Lazy import — keeps viewer startup cheap and avoids
            # pulling in the heavy adaptation module circular chain.
            from ..analysis.transmission_line import solve_transmission_line
            t0 = time.time()
            if v_bc_mode == 0:
                # Standard: prescribe Q at every boundary.
                result = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=bc_override,
                    verbose=False,
                )
            else:
                # Option B: P-match V at DC, rescale per-V AC by the
                # predicted Q̄_V × measured per-V PI/phase.
                # Pass 1: predict Q̄_V per V.
                a_only_override = {
                    n: arr for n, arr in bc_override.items()
                    if n in self._source_nodes
                }
                result_pass1 = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=a_only_override,
                    sink_pressure_bc=0.0,
                    verbose=False,
                )
                # Build pass-2 override: A's unchanged, V's get DC =
                # predicted (signed into-network) and AC = ratio ×
                # predicted DC, with the ratio taken from each V's
                # MEASURED bc_harmonics so the per-V PI and phase
                # are the data-side constraint.
                pass2_override: Dict[int, np.ndarray] = dict(a_only_override)
                for sn in self._sink_nodes:
                    measured = self.G.nodes[sn].get('bc_harmonics')
                    if measured is None:
                        continue
                    measured = np.asarray(measured, dtype=complex)
                    q_into_v = _q_into_node(result_pass1.edge_flows, sn)
                    # Solver's "Q at V" convention is +inflow into the
                    # network at V = −outflow_from_network_at_V.
                    # `q_into_v` is the predicted Q flowing INTO V
                    # from the network (= outflow_from_network), so
                    # the solver-convention Q_V is −q_into_v.
                    Q_v_dc_new = -q_into_v[0].real
                    new_arr = np.zeros_like(measured)
                    new_arr[0] = Q_v_dc_new
                    # Ratio Q_n / Q_dc is convention-independent (any
                    # uniform sign flip cancels).  Multiplying by the
                    # new DC rescales the AC magnitude while keeping
                    # the measured per-V phase intact.
                    if abs(measured[0]) > 1e-12 and abs(Q_v_dc_new) > 1e-30:
                        for k in range(1, len(measured)):
                            new_arr[k] = (measured[k] / measured[0]
                                           ) * Q_v_dc_new
                    pass2_override[sn] = new_arr
                result = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=pass2_override,
                    verbose=False,
                )
            elapsed = time.time() - t0
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._sim_status_lbl.setText(
                f"<small><font color='#c33'>Solver failed: "
                f"{type(e).__name__}: {e}</font></small>")
            self._sim_run_btn.setEnabled(True)
            return
        finally:
            # Always restore the graph's original geometry so the
            # mutation is invisible to the rest of the viewer.
            if saved_geom:
                for (u, v), saved in saved_geom.items():
                    for k, original in saved.items():
                        if original is None:
                            self.G.edges[u, v].pop(k, None)
                        else:
                            self.G.edges[u, v][k] = original

        # Write per-edge sim results to the in-memory graph.  Use the
        # `_sim_tmp_` namespace so we don't collide with editing-app
        # outputs (`mean_Q_sim`, `PI_sim`, …) already stored on the
        # graph.
        #
        # Q fields are stored as magnitudes:
        #   mean_Q  →  |mean_Q|       (matches measured: mean_Q ≥ 0
        #                              by convention — see methods.md)
        #   amp_Q   →  |H₁(Q)|        (already a magnitude in the result)
        #
        # Two phases are exposed so the user can see both flow and
        # pressure timing per harmonic.  `result.phase` is the Q-phase
        # with the sign canonicalisation already applied by the solver
        # (flips when flow direction would otherwise be negative);
        # `result.pressure_phase` is the midpoint-pressure H₁ phase.
        n = 0
        for (u, v) in result.edge_flows.keys():
            if not self.G.has_edge(u, v):
                continue
            d = self.G.edges[u, v]
            mq = result.mean_Q.get((u, v), float('nan'))
            aq = result.amp_Q.get((u, v), float('nan'))
            d['_sim_tmp_mean_Q'] = (float(abs(mq)) if np.isfinite(mq)
                                    else float('nan'))
            d['_sim_tmp_amp_Q'] = (float(abs(aq)) if np.isfinite(aq)
                                    else float('nan'))
            d['_sim_tmp_PI'] = float(
                result.PI.get((u, v), float('nan')))
            d['_sim_tmp_phase_Q'] = float(
                result.phase.get((u, v), float('nan')))
            d['_sim_tmp_phase_P'] = float(
                result.pressure_phase.get((u, v), float('nan')))
            d['_sim_tmp_dissipation'] = float(
                result.dissipation.get((u, v), float('nan')))
            # Geometry overrides — written only when the corresponding
            # override mode was active, so the sim-mode View tab can
            # show the constant value.  Heterogeneous sim → keys
            # absent → fall through to the measured top-level attrs.
            if override_R:
                d['_sim_tmp_radius'] = R_UNIFORM_PX
            if override_L:
                d['_sim_tmp_length'] = L_UNIFORM_PX
            # Persist the full complex harmonic array (DC + H₁ + H₂ + H₃)
            # so `_plot_comparison` can reconstruct a synthetic Q(t)
            # when the user is viewing a sim field.  Sign-canonicalise
            # so mean_Q ≥ 0 (matches measured `mean_Q` convention from
            # methods.md), shifting all harmonic phases by π if needed.
            harm = result.edge_flows.get((u, v))
            if harm is not None and len(harm) >= 1:
                arr = np.asarray(harm, dtype=complex).copy()
                if np.isfinite(arr[0].real) and arr[0].real < 0:
                    arr = -arr
                d['_sim_tmp_harmonics'] = arr
                d['_sim_tmp_f0_hz'] = float(f0)
            # Midpoint pressure harmonics + summary scalars
            # (sim-only fields, available in the View tab when the
            # Simulated source is active).  Pressure PI is undefined
            # near the gauge node where |P_DC| → 0; we record nan
            # there rather than picking an arbitrary reference.
            p_u = result.node_pressures.get(u)
            p_v = result.node_pressures.get(v)
            if p_u is not None and p_v is not None:
                p_u_arr = np.asarray(p_u, dtype=complex)
                p_v_arr = np.asarray(p_v, dtype=complex)
                p_mid = 0.5 * (p_u_arr + p_v_arr)
                d['_sim_tmp_p_harmonics'] = p_mid
                p_dc = float(p_mid[0].real) if len(p_mid) > 0 else float('nan')
                amp_p = (float(abs(p_mid[1]))
                         if len(p_mid) > 1 else float('nan'))
                # Signed mean midpoint pressure (Pa).  Kept signed so
                # the gauge polarity is visible in the colormap; users
                # who only want magnitude can toggle log scale off and
                # read the sign directly.
                d['_sim_tmp_pressure_mean'] = p_dc
                d['_sim_tmp_pressure_amp'] = amp_p
                # Pressure drop across the edge: |P_u_DC − P_v_DC| (Pa).
                # Magnitude so the colormap reads unambiguously; sign
                # would depend on edge orientation (undirected graph).
                if len(p_u_arr) > 0 and len(p_v_arr) > 0:
                    d['_sim_tmp_pressure_drop'] = float(
                        abs(p_u_arr[0].real - p_v_arr[0].real))
            n += 1
        self._sim_active = True
        self._sim_last_D = D
        self._sim_last_f0_hz = f0
        # Enable the Simulated source radio in the View tab.
        if getattr(self, '_src_radio_sim', None) is not None:
            self._src_radio_sim.setEnabled(True)
            self._src_radio_sim.setToolTip("")
        # Re-sync the field combos so the P quantity and sim-only
        # properties become enabled (or stay enabled).
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        self._sim_status_lbl.setText(
            f"<small>OK · {elapsed:.1f}s · {n} edges<br>"
            f"D = {D:.3g}, f₀ = {f0:.3f} Hz<br>"
            f"Switch the View-tab Source to <b>Simulated</b> to view "
            f"sim fields; P quantity becomes available.</small>")
        self._sim_run_btn.setEnabled(True)
        self._sim_clear_btn.setEnabled(True)

    def _clear_simulation(self):
        """Strip `_sim_tmp_*` keys from every edge, fall back to the
        measured source, and refresh the View-tab UI."""
        sim_keys = list(SIM_FIELD_MAP.values()) + SIM_INTERNAL_KEYS
        removed = 0
        for _, _, d in self.G.edges(data=True):
            for k in sim_keys:
                if k in d:
                    del d[k]
                    removed += 1
        self._sim_active = False
        # Force the View tab back to the measured source.
        self.current_source = 'measured'
        if getattr(self, '_src_radio_measured', None) is not None:
            self._src_radio_measured.blockSignals(True)
            self._src_radio_sim.blockSignals(True)
            try:
                self._src_radio_measured.setChecked(True)
                self._src_radio_sim.setEnabled(False)
            finally:
                self._src_radio_measured.blockSignals(False)
                self._src_radio_sim.blockSignals(False)
        # Resync the field combos (P quantity + sim-only properties
        # disappear / get greyed out) and re-render.
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        self._sim_clear_btn.setEnabled(False)
        self._sim_status_lbl.setText(
            f"<small>Cleared {removed} sim values from "
            f"{self.G.number_of_edges()} edges.</small>")

    def _on_bounds_change(self, _val=None):
        # Either percentile spinbox changed → re-render with the
        # new clip range.
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_total_snr_pct_change(self, val: float):
        """Update the network-wide total_snr percentile filter.  The
        threshold itself is re-derived from cached `_h_total_snr`
        values on every refresh — set independently of any tile filter
        so it stays stable when you swap tiles."""
        self.total_snr_pct_filter = float(val)
        # Force re-derivation of the cached threshold next refresh.
        self._cached_total_snr_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_tile_filter_change(self, _idx=None):
        """Switch between best-measurement coloring and per-tile-only
        coloring.  When a tile is selected, repopulates
        `_tile_harmonic_cache` for that tile (~5 ms × edges with a
        measurement on this tile) so the resolver reads per-tile values
        immediately.  Always re-populating keeps the cache consistent
        even if the video-load path wrote a different tile's data
        since the last filter change.
        """
        if self._tile_filter_combo is None:
            return
        new_filter = self._tile_filter_combo.currentData()
        new_filter = int(new_filter) if new_filter is not None else None
        if new_filter == self.current_tile_filter:
            return
        self.current_tile_filter = new_filter
        if new_filter is not None:
            self._populate_tile_harmonic_cache(new_filter)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_cbar_toggle(self, checked: bool):
        if not bool(checked) and self._cbar_overlay is not None:
            self._cbar_overlay.setParent(None)
            self._cbar_overlay.deleteLater()
            self._cbar_overlay = None
        else:
            self._refresh_cbar()

    def _refresh_cbar(self):
        """Render the colormap legend as a floating QLabel/QPixmap on the
        napari canvas (right side, vertically centred).  Same approach
        as the editing viewer's `_update_colorbar_layer`."""
        # Always tear down the previous overlay (if any) before drawing.
        old = self._cbar_overlay
        if old is not None:
            old.setParent(None)
            old.deleteLater()
            self._cbar_overlay = None
        if not getattr(self, '_cbar_chk', None) \
                or not self._cbar_chk.isChecked():
            return
        from matplotlib.colors import Normalize, LogNorm
        from qtpy.QtWidgets import QLabel
        from qtpy.QtGui import QPixmap, QImage
        from qtpy.QtCore import Qt
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        vmin = getattr(self, '_last_vmin', None)
        vmax = getattr(self, '_last_vmax', None)
        if is_categorical or vmin is None or vmax is None:
            return  # nothing meaningful to draw
        is_phase = (self.current_property == 'phase')
        use_log = bool(self.log_scale) and not is_phase
        # Compose colorbar label from the selectors so it always reads
        # cleanly: "Magnitude |·|  (Q, H₁)", "Pulsatility index  (P)", …
        label = self._current_field_label()
        # Build the matplotlib figure (Agg, transparent bg, white text).
        dpi = 100
        if is_phase:
            # Color wheel for phase.
            fig_sz = 2.8
            fig = plt.figure(figsize=(fig_sz, fig_sz), dpi=dpi)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='polar')
            n = 512
            theta = np.linspace(0, 2 * np.pi, n + 1)
            r_in, r_out = 0.45, 1.0
            C = np.tile(np.linspace(0, 2 * np.pi, n), (1, 1))
            ax.pcolormesh(theta, [r_in, r_out], C,
                           cmap=plt.get_cmap('hsv'),
                           norm=Normalize(vmin=0, vmax=2*np.pi),
                           shading='auto')
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.text(0, 0, label, ha='center', va='center',
                    color='white', fontsize=11, transform=ax.transData)
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.patch.set_alpha(0)
        else:
            fig_h_in, fig_w_in = 5.0, 1.8
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            cmap = plt.get_cmap('viridis')
            if use_log and vmin > 0:
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax)
            cbar.set_label(label, color='white', fontsize=14, labelpad=10)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                           labelsize=11, width=2, length=5)
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.patch.set_alpha(0)
            fig.subplots_adjust(left=0.05, right=0.35, top=0.97, bottom=0.03)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        # Parent to the vispy canvas widget so it floats over the image.
        qt_viewer = self.viewer.window._qt_viewer
        canvas_widget = qt_viewer.canvas.native
        overlay = QLabel(canvas_widget)
        overlay.setPixmap(pixmap)
        overlay.setFixedSize(w, h)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.setStyleSheet("background: transparent;")
        overlay.raise_()
        # Position: right edge, vertically centred.
        def _reposition():
            pw = canvas_widget.width()
            ph = canvas_widget.height()
            x = max(0, pw - w - 20)
            y = max(0, (ph - h) // 2)
            overlay.move(x, y)
        _reposition()
        overlay.show()
        # Re-position on canvas resize (preserve any prior handler).
        if self._cbar_orig_resize is None:
            self._cbar_orig_resize = canvas_widget.resizeEvent
        def _on_resize(event, _orig=self._cbar_orig_resize):
            _orig(event)
            cur = self._cbar_overlay
            if cur is not None and cur.isVisible():
                _reposition()
        canvas_widget.resizeEvent = _on_resize
        self._cbar_overlay = overlay

    # ── tile boundaries + labels (optional overlay) ────────────────────
    def _refresh_tile_boundaries(self):
        # Remove existing layer (None on first call) and re-add if toggled on.
        if self._tile_boundaries_layer is not None:
            try:
                self.viewer.layers.remove(self._tile_boundaries_layer)
            except (KeyError, ValueError):
                pass
            self._tile_boundaries_layer = None
        if not self.show_tile_boundaries or not self.tiles:
            return
        rects: List[np.ndarray] = []
        for tid in sorted(self.tiles.keys()):
            t = self.tiles[tid]
            r0 = t['translate_y'] - self._tile_offset_y
            c0 = t['translate_x'] - self._tile_offset_x
            r1 = r0 + TILE_RAW_HEIGHT * t['scale_y']
            c1 = c0 + TILE_RAW_WIDTH * t['scale_x']
            rects.append(np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]]))
        self._tile_boundaries_layer = self.viewer.add_shapes(
            rects, shape_type='polygon',
            edge_color='yellow', edge_width=2.0,
            face_color='transparent',
            name='Tile boundaries', opacity=0.7,
        )

    def _refresh_tile_labels(self):
        if self._tile_labels_layer is not None:
            try:
                self.viewer.layers.remove(self._tile_labels_layer)
            except (KeyError, ValueError):
                pass
            self._tile_labels_layer = None
        if not self.show_tile_labels or not self.tiles:
            return
        pts: List[List[float]] = []
        labels: List[str] = []
        for tid in sorted(self.tiles.keys()):
            t = self.tiles[tid]
            r0 = t['translate_y'] - self._tile_offset_y
            c0 = t['translate_x'] - self._tile_offset_x
            r_mid = r0 + 0.5 * TILE_RAW_HEIGHT * t['scale_y']
            c_mid = c0 + 0.5 * TILE_RAW_WIDTH * t['scale_x']
            pts.append([r_mid, c_mid])
            labels.append(str(tid))
        self._tile_labels_layer = self.viewer.add_points(
            np.array(pts), size=0, name='Tile numbers',
            text={'string': labels, 'size': 14,
                   'color': 'yellow', 'anchor': 'center'},
        )

    # ── click handler ──────────────────────────────────────────────────
    def _inspect_nearest_boundary_node(
            self, x: float, y: float, max_px: float = 18.0) -> bool:
        """If the click is within `max_px` of a boundary node, display
        its BC harmonics in the info panel and return True.  Otherwise
        return False so the caller can fall through to edge hit-test.

        `x`, `y` are in graph coordinates (x = G.nodes[n]['x'], y is
        the click-row converted to graph y via mosaic_height − row).
        """
        if not self._boundary_nodes:
            return False
        y_graph = self.mosaic_height - y
        best_n, best_d = None, np.inf
        for n in self._boundary_nodes:
            dx = self.G.nodes[n]['x'] - x
            dy = self.G.nodes[n]['y'] - y_graph
            d = float(np.hypot(dx, dy))
            if d < best_d:
                best_n, best_d = n, d
        if best_n is None or best_d > max_px:
            return False
        # Disable edge-only buttons (BC click clears edge selection).
        self._selected_edge = None
        try:
            self._plot_qt_btn.setEnabled(False)
            self._add_to_comp_btn.setEnabled(False)
            self._video_btn.setEnabled(False)
            self._run_of_btn.setEnabled(False)
        except AttributeError:
            pass
        d = self.G.nodes[best_n]
        btype = d.get('boundary_type', '?')
        letter = 'A' if btype == 'source' else 'V' if btype == 'sink' else '?'
        side = ('arterial source' if btype == 'source'
                else 'venous sink' if btype == 'sink' else btype)
        info: List[str] = []
        info.append(f"Boundary node {best_n}  [{letter}]  ({side})")
        info.append(f"  position: ({d.get('x', float('nan')):.0f}, "
                    f"{d.get('y', float('nan')):.0f})")
        bch = d.get('bc_harmonics')
        if bch is None:
            info.append("\n  No bc_harmonics stored on this node.")
        else:
            bch = np.asarray(bch)
            # Apply canonical sign convention so phases at sources and
            # sinks report the same impedance phase, not differ by π:
            #   stored Q at sources = +inflow
            #   stored Q at sinks   = +outflow (network sees −Q)
            # → multiply by −1 at sinks to get Q_signed (always "into
            #   the network"), then read magnitude / phase off that.
            sign = -1.0 if btype == 'sink' else 1.0
            bch_signed = bch * sign
            sign_note = " (Q_signed = −Q_outflow)" if sign < 0 else ""
            info.append(f"\n  bc_harmonics (length {len(bch)}): "
                        f"DC + H₁ + H₂ + H₃{sign_note}")
            labels = ['DC', 'H₁', 'H₂', 'H₃']
            for i, c in enumerate(bch_signed[:4]):
                lab = labels[i] if i < len(labels) else f'H{i}'
                if i == 0:
                    info.append(f"    {lab}:  Q̄ = {float(np.real(c)):+.4g}  "
                                f"(imag = {float(np.imag(c)):+.3g})")
                else:
                    mag = float(np.abs(c))
                    ph = float(np.angle(c))
                    info.append(f"    {lab}:  |a| = {mag:.4g},  "
                                f"∠ = {np.degrees(ph):+.1f}°")
        self._info_box.setPlainText("\n".join(info))
        # Move the click marker to the boundary node so the user sees
        # which one they hit.
        self._click_marker_layer.data = np.array(
            [[self.mosaic_height - d['y'], d['x']]])
        return True

    def _inspect_nearest_edge(self, x: float, y: float, max_px: float = 25.0):
        # `y` here is the display row from napari's click event.  The
        # graph stores y in math-convention (image displayed with
        # mosaic_height − y).  Convert click-row back to graph-y so we
        # can compare against precomputed midpoints.
        if len(self._edge_midpoints) == 0:
            return
        y_graph = self.mosaic_height - y
        d2 = ((self._edge_midpoints[:, 0] - x) ** 2 +
              (self._edge_midpoints[:, 1] - y_graph) ** 2)
        i = int(np.argmin(d2))
        dist = float(np.sqrt(d2[i]))
        if dist > max_px:
            self._info_box.setPlainText(
                f"No edge within {max_px} px of click "
                f"(closest was {dist:.1f} px away).")
            self._plot_qt_btn.setEnabled(False)
            self._click_marker_layer.data = np.empty((0, 2))
            return
        u, v = self.edge_list[i]
        self._selected_edge = (u, v)
        self._plot_qt_btn.setEnabled(True)
        self._add_to_comp_btn.setEnabled(True)
        # OF button requires a tile + video; gate it on having both.
        self._run_of_btn.setEnabled(
            bool(self.tiles) and self.video_dir is not None)
        # Place click marker at (mosaic_height − midpoint_y, midpoint_x).
        self._click_marker_layer.data = np.array(
            [[self.mosaic_height - self._edge_midpoints[i, 1],
              self._edge_midpoints[i, 0]]])
        # Enable / drive the video buttons
        m_best = _best_measurement(
            self.G.edges[u, v].get('measurements_piv'))
        self._selected_tile_id = (m_best.get('tile_id')
                                  if m_best is not None else None)
        can_video = (self._selected_tile_id is not None
                     and bool(self.tiles)
                     and self.video_dir is not None)
        self._video_btn.setEnabled(can_video)
        if can_video and getattr(self, '_auto_video_chk', None) \
                and self._auto_video_chk.isChecked():
            self._load_video_overlay(int(self._selected_tile_id))
            self._unload_video_btn.setEnabled(True)
        # Build readable summary — show ALL measurements for this edge,
        # ordered best-quality-first.  Each measurement has a small
        # marker indicating whether PIV's internal gates accepted it
        # ("✓" usable / "✗" rejected) but we don't surface the tier
        # vocabulary anywhere visible.
        d = self.G.edges[u, v]
        meas = d.get('measurements_piv') or []
        meas_ranked = sorted(
            meas,
            key=lambda mm: (QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                            mm.get('snr_f0', 0) or 0,
                            mm.get('snr_pulse', 0) or 0),
            reverse=True)
        info: List[str] = []
        info.append(f"Edge ({u}, {v})")
        info.append(f"  endpoints: ({self.G.nodes[u]['x']:.0f}, "
                    f"{self.G.nodes[u]['y']:.0f}) → "
                    f"({self.G.nodes[v]['x']:.0f}, "
                    f"{self.G.nodes[v]['y']:.0f})")
        info.append(f"  radius: {d.get('radius', float('nan')):.1f} px")
        info.append(f"  length: {d.get('length', float('nan')):.1f} px")
        info.append(f"  measurements: {len(meas)} "
                    f"(ranked by quality, best first)")
        display_keys = ('mean_Q', 'amp_Q', 'PI', 'phase', 'snr_pulse',
                        'snr_f0', 'f0_hz', 'v_max')
        for idx, mm in enumerate(meas_ranked):
            mark = '✓' if _measurement_usable(mm) else '✗'
            tile_id = mm.get('tile_id')
            info.append(f"\n  {mark} measurement {idx + 1} "
                        f"(tile {tile_id}):")
            for k in display_keys:
                if k in mm:
                    val = mm[k]
                    try:
                        info.append(f"    {k}: {float(val):.4g}")
                    except (TypeError, ValueError):
                        info.append(f"    {k}: {val}")
        # Per-harmonic Z-statistics + classification for the best
        # measurement.  Computed on demand (one harmonic fit per click,
        # cheap).  Shows the user which harmonics rise above the noise.
        best = meas_ranked[0] if meas_ranked else None
        if best is not None and best.get('Q_t') is not None:
            snrs = _harmonic_snrs(
                np.asarray(best['Q_t'], dtype=float),
                f0=float(best.get('f0_hz', 2.5)),
                dt=1.0/250)
            if snrs is not None:
                klass = _harmonic_class(snrs)
                info.append("\n  Per-harmonic SNR (Z = amp / SE_amp):")
                for k in ('DC', 'H1', 'H2', 'H3'):
                    pass_mark = '✓' if snrs[k] >= HARMONIC_SNR_THRESHOLD else '·'
                    info.append(f"    {pass_mark} Z({k}) = {snrs[k]:.1f}")
                info.append(f"  Resolved: {_harmonic_class_label(klass)} "
                            f"(threshold Z > {HARMONIC_SNR_THRESHOLD:.0f})")
        self._info_box.setPlainText("\n".join(info))

    # ── comparison set management ──────────────────────────────────────
    def _add_to_comparison(self):
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        if (u, v) in self._comparison_edges:
            return  # already in set
        self._comparison_edges.append((u, v))
        self._refresh_comparison_markers()
        self._refresh_comparison_buttons()

    def _clear_comparison(self):
        self._comparison_edges.clear()
        self._refresh_comparison_markers()
        self._refresh_comparison_buttons()

    def _refresh_comparison_markers(self):
        # Always tear down the previous edge overlay — napari Shapes
        # layers don't always repaint cleanly on .data/.edge_color reassign
        # (same bug worked around in the main edges layer rebuild).
        if self._comparison_edge_overlay_layer is not None:
            try:
                self.viewer.layers.remove(
                    self._comparison_edge_overlay_layer)
            except (KeyError, ValueError):
                pass
            self._comparison_edge_overlay_layer = None

        if self._comparison_marker_layer is None:
            return
        if not self._comparison_edges:
            self._comparison_marker_layer.data = np.empty((0, 2))
            return
        # Match the comparison plot's per-edge colors:
        #   index 0 → matplotlib C0 (blue, Edge A)
        #   index 1 → matplotlib C1 (orange, Edge B)
        # Any extras beyond the two used by the plot fade to grey.
        comp_colors = ['#1f77b4', '#ff7f0e']
        H = self.mosaic_height
        pts, marker_colors = [], []
        lines, line_colors, line_widths = [], [], []
        for k, (u, v) in enumerate(self._comparison_edges):
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            color = (comp_colors[k] if k < len(comp_colors)
                     else '#888888')
            # Mid-point dot for the small marker layer.
            pts.append([H - 0.5 * (y1 + y2), 0.5 * (x1 + x2)])
            marker_colors.append(color)
            # Thick bright line overlaying the actual edge — the
            # primary "this edge is in the comparison set" signal,
            # visible on top of any field colouring.
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
            line_colors.append(color)
            # Width is generous; the underlying edge_width caps at
            # EDGE_WIDTH_MAX=22 so 16 px sits well above any vessel
            # while not swamping the image.
            line_widths.append(16.0)

        # Small filled dot at each edge midpoint — secondary cue, easy
        # to see when the edge itself is short or off-screen.
        self._comparison_marker_layer.data = np.asarray(pts)
        self._comparison_marker_layer.size = 14
        try:
            self._comparison_marker_layer.border_color = marker_colors
            self._comparison_marker_layer.face_color = marker_colors
        except (AttributeError, TypeError):
            try:
                self._comparison_marker_layer.edge_color = marker_colors
                self._comparison_marker_layer.face_color = marker_colors
            except (AttributeError, TypeError):
                pass

        # Bold edge overlay — the headline marker.
        if lines:
            self._comparison_edge_overlay_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color=line_colors,
                edge_width=line_widths,
                name='Comparison edges',
                opacity=0.85,
            )

    def _refresh_comparison_buttons(self):
        n = len(self._comparison_edges)
        self._plot_comp_btn.setText(f"Plot comparison ({n})")
        self._plot_comp_btn.setEnabled(n >= 2)
        self._clear_comp_btn.setEnabled(n > 0)

    def _build_sim_measurement(self, u, v, n_frames: int = 750,
                                quantity: str = 'Q'):
        """Reconstruct a measurement-shaped dict from the stored
        per-edge sim harmonics for edge (u, v).

        `quantity = 'Q'` pulls from `_sim_tmp_harmonics` (flow); `'P'`
        pulls from `_sim_tmp_p_harmonics` (midpoint pressure).  The
        returned dict has the same keys the comparison plot reads from
        `measurements_piv` (Q_t, f0_hz, mean_Q, amp_Q, PI, tile_id),
        so it can flow through the existing pipeline unchanged; the
        Q_t slot carries P(t) values in pressure mode and the caller
        relabels axes accordingly.

        Returns None when the edge has no sim data or f0 is missing.
        """
        if not self.G.has_edge(u, v):
            return None
        d = self.G.edges[u, v]
        harm_key = ('_sim_tmp_p_harmonics' if quantity == 'P'
                    else '_sim_tmp_harmonics')
        harm = d.get(harm_key)
        if harm is None:
            return None
        f0 = d.get('_sim_tmp_f0_hz')
        if f0 is None:
            f0 = self._sim_last_f0_hz
        try:
            f0 = float(f0)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(f0) and f0 > 0):
            return None
        arr = np.asarray(harm, dtype=complex)
        dt = 1.0 / 250.0
        t = np.arange(n_frames) * dt
        omega = 2 * np.pi * f0
        # Real-time reconstruction: f(t) = Σ_k Re{ c_k · exp(i k ω t) }
        #                                = Σ_k Re(c_k) cos(kωt) − Im(c_k) sin(kωt)
        sig = np.full(n_frames, arr[0].real, dtype=float)
        for k in range(1, len(arr)):
            sig += (arr[k].real * np.cos(k * omega * t)
                    - arr[k].imag * np.sin(k * omega * t))
        mean_val = float(arr[0].real)
        amp_val = float(abs(arr[1])) if len(arr) > 1 else float('nan')
        PI = (2.0 * amp_val / abs(mean_val)
              if abs(mean_val) > 1e-12 else float('nan'))
        return {
            'Q_t': sig,        # carries P(t) when quantity='P'
            'f0_hz': f0,
            'mean_Q': mean_val,
            'amp_Q': amp_val,
            'PI': PI,
            'tile_id': None,
            'quality_tier': 'A',
            'snr_pulse': 99.0,
            'snr_f0': 99.0,
            'fit_success': True,
        }

    def _plot_comparison(self):
        """Two-edge comparison figure focused on harmonic attenuation.

        Layout (three rows):
          Row 1 — two raw Q(t) panels (one per edge), shared y-axis;
                  amplitude attenuation reads visually.
          Row 2 — single-period overlay of bold harmonic fits (no
                  individual cycles); tiny corner note with per-edge Z
                  values for H₁, H₂.
          Row 3 — attenuation–phase scatter, one marker per harmonic.
                  x = Δφ (degrees, B − A); y = pulse-relative
                  attenuation. Fixed axes (−90°…+90°, 0…1.2) for
                  cross-call consistency. Reference lines at y=1, x=0.

        Pulse-relative attenuation (per harmonic k):

            atten_k = (|H_k^B| / Q̄_B) / (|H_k^A| / Q̄_A)

        strips out the trivial mean-flow rescaling and isolates the
        per-harmonic damping of the pulsatile component.  Q̄ is taken
        from the harmonic-fit DC term (a0).

        H₂ marker is filled if Z_{H₂} ≥ 3 on BOTH edges (resolved); else
        drawn as an open circle at 50 % alpha with the failing Z value
        tagged next to it.
        """
        if len(self._comparison_edges) < 2:
            return
        from ..analysis.harmonic import fit_harmonics
        dt = 1.0 / 250.0
        Z_THRESH_H2 = 3.0
        # Source selector: 'sim' uses reconstructed Q(t) (or P(t) for
        # pressure-mode fields) from the stored per-edge harmonics;
        # 'measured' uses PIV Q_t.
        src = getattr(self, 'current_source', 'measured')
        # Pressure mode: when the Quantity selector is P (sim only),
        # the comparison plot reconstructs midpoint P(t) instead of
        # Q(t) and relabels axes accordingly.
        quantity = ('P' if (src == 'sim'
                            and self.current_quantity == 'P')
                    else 'Q')
        # Pick measurement per edge (prefer the loaded-tile measurement
        # in measured mode; sim mode has one synthetic measurement per
        # edge).
        per_edge_data = []
        for (u, v) in self._comparison_edges:
            if not self.G.has_edge(u, v):
                continue
            if src == 'sim':
                m = self._build_sim_measurement(u, v, quantity=quantity)
                if m is None:
                    continue
                per_edge_data.append((u, v, m))
                continue
            meas = self.G.edges[u, v].get('measurements_piv') or []
            if not meas:
                continue
            # Focus-tile rule (same as the colormap + Plot Q(t)):
            # filter > loaded video > best-of-edge fallback.
            tid = (self.current_tile_filter
                    if self.current_tile_filter is not None
                    else self._video_tile_id)
            m = None
            if tid is not None:
                m = next((mm for mm in meas
                          if mm.get('tile_id') == tid), None)
            if m is None:
                m = _best_measurement(meas)
            if m is None or m.get('Q_t') is None:
                continue
            per_edge_data.append((u, v, m))
        if len(per_edge_data) < 2:
            return
        per_edge_data = per_edge_data[:2]

        # ── Regime classification + A/B ordering ───────────────────────
        # Per-edge topo_av_score (0 = arterial side, 1 = venous side,
        # 0.5 = interior).  Pulse-source proximity ordering rule:
        #   arterial-dominated regime (both score < 0.5):
        #     Edge A = smaller graph_dist_art (closer to arterial source)
        #     pulse propagates A → B (same as DC flow)
        #   venous-dominated regime (both score > 0.5):
        #     Edge A = smaller graph_dist_ven (closer to venous sink,
        #              which is the *local* pulse source)
        #     pulse propagates A → B (opposite to DC flow)
        #   cross-regime: ordering is arbitrary; flag in the suptitle
        #              and render row-3 markers as open circles.
        def _edge_topo(u, v):
            d = self.G.edges[u, v]
            return (float(d.get('topo_av_score', np.nan)),
                    float(d.get('graph_dist_art', np.nan)),
                    float(d.get('graph_dist_ven', np.nan)))
        topo = [_edge_topo(u, v) for (u, v, _) in per_edge_data]
        scores = [t[0] for t in topo]
        if all(np.isfinite(scores)) and max(scores) < 0.5:
            regime = 'arterial'
        elif all(np.isfinite(scores)) and min(scores) > 0.5:
            regime = 'venous'
        elif any(not np.isfinite(s) for s in scores):
            regime = 'unknown'
        else:
            regime = 'cross'
        swapped = False
        if regime == 'arterial':
            # smaller graph_dist_art = closer to A source = upstream in pulse-prop
            if topo[1][1] < topo[0][1]:
                per_edge_data = [per_edge_data[1], per_edge_data[0]]
                topo = [topo[1], topo[0]]
                swapped = True
        elif regime == 'venous':
            # smaller graph_dist_ven = closer to V sink (= local pulse source)
            if topo[1][2] < topo[0][2]:
                per_edge_data = [per_edge_data[1], per_edge_data[0]]
                topo = [topo[1], topo[0]]
                swapped = True
        # cross / unknown: leave the user-supplied order
        if swapped:
            # Keep the in-canvas overlays in sync with the final A/B
            # assignment: the user clicked in one order, the regime
            # ordering may have swapped them, so the rings/edge highlight
            # colors must follow suit (A → blue, B → orange).
            new_order = [(u, v) for (u, v, _) in per_edge_data]
            remaining = [e for e in self._comparison_edges
                          if e not in new_order]
            self._comparison_edges = new_order + remaining
            self._refresh_comparison_markers()

        def _wrap(p):
            return (float(p) + np.pi) % (2 * np.pi) - np.pi

        edge_fits = []
        edge_colors = ['C0', 'C1']
        edge_letters = ['A', 'B']
        for (u, v, m) in per_edge_data:
            Q_t = np.asarray(m['Q_t'], dtype=float)
            T = len(Q_t)
            f0 = float(m.get('f0_hz', 2.5))
            try:
                hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                                    loss='huber', include_dc=True)
                a0 = float(hr['a0'])
                harms = {h['k']: h for h in hr['harmonics']}
                h1, h2, h3 = harms.get(1), harms.get(2), harms.get(3)
                phi1 = (float(np.arctan2(-h1['B'], h1['A']))
                        if h1 is not None else float('nan'))
                phi2 = (float(np.arctan2(-h2['B'], h2['A']))
                        if h2 is not None else float('nan'))
                phi3 = (float(np.arctan2(-h3['B'], h3['A']))
                        if h3 is not None else float('nan'))
                amp1 = float(h1['amp']) if h1 is not None else float('nan')
                amp2 = float(h2['amp']) if h2 is not None else float('nan')
                amp3 = float(h3['amp']) if h3 is not None else float('nan')
                resid = np.asarray(hr.get('resid', []), dtype=float)
                sigma = float(np.std(resid)) if resid.size else float('inf')
                se_amp = (sigma * np.sqrt(2.0 / T)
                          if T > 0 else float('inf'))
                z1 = amp1 / max(se_amp, 1e-30) if np.isfinite(amp1) else np.nan
                z2 = amp2 / max(se_amp, 1e-30) if np.isfinite(amp2) else np.nan
                z3 = amp3 / max(se_amp, 1e-30) if np.isfinite(amp3) else np.nan
            except Exception:
                a0 = 0.0
                amp1 = amp2 = amp3 = phi1 = phi2 = phi3 = np.nan
                z1 = z2 = z3 = np.nan
            edge_fits.append(dict(
                u=u, v=v, m=m, Q_t=Q_t, T=T, f0=f0, a0=a0,
                amp1=amp1, amp2=amp2, amp3=amp3,
                phi1=phi1, phi2=phi2, phi3=phi3,
                z1=z1, z2=z2, z3=z3))

        # Quantity-aware axis / title labels for Q vs P modes.
        unit = 'Pa' if quantity == 'P' else 'nL/s'
        sym = quantity  # 'Q' or 'P'
        sym_bar = f'{sym}̄'  # combining macron → Q̄ / P̄
        mean_label = f'mean_{sym}'

        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.55, wspace=0.30,
            left=0.09, right=0.97, top=0.91, bottom=0.07)

        # ── Top: raw signal, shared y across both panels ──
        ax_raw_A = fig.add_subplot(gs[0, 0])
        ax_raw_B = fig.add_subplot(gs[0, 1], sharey=ax_raw_A)
        for k, ed in enumerate(edge_fits):
            ax = (ax_raw_A, ax_raw_B)[k]
            t_axis = np.arange(ed['T']) * dt
            ax.plot(t_axis, ed['Q_t'], color=edge_colors[k],
                    lw=0.7, alpha=0.9)
            ax.set_title(
                f"Edge {edge_letters[k]}: ({ed['u']}, {ed['v']}) "
                f"tile {ed['m'].get('tile_id')}\n"
                f"{mean_label}={ed['m'].get('mean_Q', float('nan')):.3f}  "
                f"PI={ed['m'].get('PI', float('nan')):.2f}  "
                f"f₀={ed['f0']:.2f} Hz",
                fontsize=9, color=edge_colors[k])
            ax.grid(alpha=0.3)
            ax.set_xlabel('time (s)')
            if k == 0:
                ax.set_ylabel(f'{sym} ({unit})')
        ymins = [np.nanmin(ed['Q_t']) for ed in edge_fits]
        ymaxs = [np.nanmax(ed['Q_t']) for ed in edge_fits]
        ax_raw_A.set_ylim(min(ymins), max(ymaxs))

        # ── Middle: single-period fold, mean-normalized harmonic fits ──
        # Plots (Q − Q̄) / Q̄ for each edge: removes the absolute-amplitude
        # comparison (already in row 1) and exposes the pulsatile-fraction
        # waveform.  Peak heights here ≈ pulsatility index contributions.
        ax_fold = fig.add_subplot(gs[1, :])
        N_fold = 200
        phase_grid = np.linspace(0.0, 1.0, N_fold, endpoint=False)
        for k, ed in enumerate(edge_fits):
            f0 = ed['f0']
            qbar = ed['a0']
            if not (np.isfinite(f0) and f0 > 0
                    and np.isfinite(qbar) and abs(qbar) > 0):
                continue
            color = edge_colors[k]
            T_period = 1.0 / f0
            omega = 2 * np.pi * f0
            t_one = phase_grid * T_period
            y_fit = np.zeros(N_fold)
            for kh, amp_k, phi_k in [
                    (1, ed['amp1'], ed['phi1']),
                    (2, ed['amp2'], ed['phi2']),
                    (3, ed['amp3'], ed['phi3'])]:
                if np.isfinite(amp_k) and np.isfinite(phi_k):
                    y_fit += amp_k * np.cos(kh * omega * t_one + phi_k)
            ax_fold.plot(phase_grid, y_fit / abs(qbar),
                         color=color, lw=2.0,
                         label=f"Edge {edge_letters[k]}")
        ax_fold.axhline(0.0, color='#888', lw=0.5, ls=':')
        ax_fold.set_xlim(0, 1)
        ax_fold.set_xlabel('cycle fraction')
        ax_fold.set_ylabel(f'({sym} − {sym_bar}) / {sym_bar}')
        ax_fold.set_title(
            'Mean-normalized single-period harmonic fit', fontsize=10)
        ax_fold.legend(fontsize=8, loc='upper right')
        ax_fold.grid(alpha=0.3)
        # Tiny corner note with Z values for resolution context.
        zA1, zB1 = edge_fits[0].get('z1', np.nan), edge_fits[1].get('z1', np.nan)
        zA2, zB2 = edge_fits[0].get('z2', np.nan), edge_fits[1].get('z2', np.nan)
        def _zs(z):
            return f"{z:.1f}" if np.isfinite(z) else "n/a"
        ax_fold.text(
            0.015, 0.97,
            f"H₁: Z_A={_zs(zA1)}, Z_B={_zs(zB1)}\n"
            f"H₂: Z_A={_zs(zA2)}, Z_B={_zs(zB2)}",
            ha='left', va='top', fontsize=8, family='monospace',
            color='#555', transform=ax_fold.transAxes)

        # ── Bottom row: per-harmonic transfer function (H₁, H₂) ──
        # Left  — amplitude pulsatile fraction |H_k|/Q̄ for each edge.
        # Right — phase shift Δφ_k between A and B.
        # Both panels overlay a faint √k diffusive prediction so
        # observed vs predicted is a one-glance read.  H₃ is dropped
        # since (a) it's the noisiest and (b) two harmonics already
        # over-determine the diffusive scaling.
        edA, edB = edge_fits[0], edge_fits[1]
        qbarA = (edA['a0'] if np.isfinite(edA['a0']) and abs(edA['a0']) > 0
                 else float(edA['m'].get('mean_Q', np.nan)))
        qbarB = (edB['a0'] if np.isfinite(edB['a0']) and abs(edB['a0']) > 0
                 else float(edB['m'].get('mean_Q', np.nan)))
        cross_regime = (regime == 'cross')

        def _pulse_frac(amp, q):
            if not (np.isfinite(amp) and np.isfinite(q)) or q == 0:
                return np.nan
            return amp / abs(q)

        ks = [1, 2]
        amps_A = [edA['amp1'], edA['amp2']]
        amps_B = [edB['amp1'], edB['amp2']]
        phis_A = [edA['phi1'], edA['phi2']]
        phis_B = [edB['phi1'], edB['phi2']]
        zs_A   = [edA.get('z1', np.nan), edA.get('z2', np.nan)]
        zs_B   = [edB.get('z1', np.nan), edB.get('z2', np.nan)]

        pA = [_pulse_frac(amps_A[i], qbarA) for i in range(2)]
        pB = [_pulse_frac(amps_B[i], qbarB) for i in range(2)]

        ax_atten = fig.add_subplot(gs[2, 0])
        ax_phase = fig.add_subplot(gs[2, 1])

        # ── Attenuation panel ───────────────────────────────────────
        # Stems show |H_k|/Q̄ per edge.  Atten annotation in the gap.
        x_off = 0.16
        STEM_LW = 6
        all_p = [p for p in pA + pB if np.isfinite(p) and p > 0]
        for i, k in enumerate(ks):
            zA, zB = zs_A[i], zs_B[i]
            resolved = (np.isfinite(zA) and np.isfinite(zB)
                        and zA >= Z_THRESH_H2 and zB >= Z_THRESH_H2)
            solid = resolved and not cross_regime
            xA, xB = k - x_off, k + x_off
            if np.isfinite(pA[i]) and pA[i] > 0:
                ax_atten.vlines(xA, 1e-4, pA[i], colors='C0',
                                lw=STEM_LW,
                                alpha=0.95 if solid else 0.4)
                ax_atten.plot(xA, pA[i], 'o', ms=12,
                              mfc='C0' if solid else 'none',
                              mec='C0', mew=2.0,
                              alpha=0.95 if solid else 0.7)
            if np.isfinite(pB[i]) and pB[i] > 0:
                ax_atten.vlines(xB, 1e-4, pB[i], colors='C1',
                                lw=STEM_LW,
                                alpha=0.95 if solid else 0.4)
                ax_atten.plot(xB, pB[i], 'o', ms=12,
                              mfc='C1' if solid else 'none',
                              mec='C1', mew=2.0,
                              alpha=0.95 if solid else 0.7)
            # Atten label between the stems.
            if (np.isfinite(pA[i]) and pA[i] > 0
                    and np.isfinite(pB[i]) and pB[i] > 0):
                atten = pB[i] / pA[i]
                ax_atten.annotate(
                    '', xy=(xB, pB[i]), xytext=(xA, pA[i]),
                    arrowprops=dict(arrowstyle='->', color='#444',
                                    lw=1.2,
                                    connectionstyle='arc3,rad=-0.2'))
                ax_atten.text(
                    k, np.sqrt(pA[i] * pB[i]),
                    f"×{atten:.2f}",
                    ha='center', va='center', fontsize=9,
                    color='#222',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#bbb', alpha=0.9))

        ax_atten.plot([], [], 'o', color='C0', ms=10, label='Edge A')
        ax_atten.plot([], [], 'o', color='C1', ms=10, label='Edge B')
        ax_atten.legend(fontsize=7, loc='upper right')

        ax_atten.set_yscale('log')
        if all_p:
            ax_atten.set_ylim(max(1e-3, 0.3 * min(all_p)),
                              max(all_p) * 1.6)
        else:
            ax_atten.set_ylim(1e-3, 1.0)
        ax_atten.set_xlim(0.5, 2.5)
        ax_atten.set_xticks([1, 2])
        ax_atten.set_xticklabels(['H₁', 'H₂'])
        ax_atten.set_xlabel('harmonic')
        ax_atten.set_ylabel(
            f'|H$_k$| / {sym_bar}  (pulsatile fraction)')
        atten_title = 'Amplitude per harmonic'
        if cross_regime:
            atten_title += ' [cross-regime: hollow]'
        ax_atten.set_title(atten_title, fontsize=10)
        ax_atten.grid(alpha=0.3, which='both', axis='y')

        # ── Phase panel ─────────────────────────────────────────────
        dphis = []
        for i, k in enumerate(ks):
            if np.isfinite(phis_A[i]) and np.isfinite(phis_B[i]):
                dphis.append(np.degrees(_wrap(phis_B[i] - phis_A[i])))
            else:
                dphis.append(np.nan)
        for i, k in enumerate(ks):
            zA, zB = zs_A[i], zs_B[i]
            resolved = (np.isfinite(zA) and np.isfinite(zB)
                        and zA >= Z_THRESH_H2 and zB >= Z_THRESH_H2)
            solid = resolved and not cross_regime
            if np.isfinite(dphis[i]):
                ax_phase.vlines(k, 0, dphis[i],
                                colors='#444', lw=STEM_LW * 0.6,
                                alpha=0.9 if solid else 0.4)
                ax_phase.plot(k, dphis[i], 'o', ms=13,
                              mfc='#444' if solid else 'none',
                              mec='#444', mew=2.0,
                              alpha=0.95 if solid else 0.7)
                ax_phase.text(k + 0.08, dphis[i],
                              f"{dphis[i]:+.0f}°",
                              ha='left', va='center', fontsize=9,
                              color='#222')
        ax_phase.axhline(0.0, color='#888', lw=0.7, ls='--', alpha=0.7)
        # Symmetric y-range around 0 so positive and negative shifts read.
        if any(np.isfinite(d) for d in dphis):
            absmax = max(abs(d) for d in dphis if np.isfinite(d))
            ymax = max(30.0, absmax * 1.25)
            ax_phase.set_ylim(-ymax, ymax)
        else:
            ax_phase.set_ylim(-90, 90)
        ax_phase.set_xlim(0.5, 2.5)
        ax_phase.set_xticks([1, 2])
        ax_phase.set_xticklabels(['H₁', 'H₂'])
        ax_phase.set_xlabel('harmonic')
        ax_phase.set_ylabel('Δφ  (degrees,  B − A)')
        ax_phase.set_title('Phase shift per harmonic', fontsize=10)
        ax_phase.grid(alpha=0.3, axis='y')

        # Title (bold) + subtitle (smaller, italic) for regime context.
        if regime == 'arterial':
            subtitle = ("arterial-dominated pair; pulse propagates "
                        "A → B (DC flow A → B)")
        elif regime == 'venous':
            subtitle = ("venous-dominated pair; pulse propagates A → B "
                        "(DC flow is reversed: B → A)")
        elif regime == 'cross':
            subtitle = ("CROSS-REGIME pair — attenuation does not "
                        "directly apply (markers hollow)")
        else:
            subtitle = ("regime unknown — topo_av_score missing on one "
                        "or both edges")
        main_title = (
            "Edge A vs B — pressure attenuation"
            if quantity == 'P'
            else "Edge A vs B — harmonic attenuation")
        if self._video_tile_id is not None:
            main_title += f"  (tile {self._video_tile_id})"
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=0.985)
        fig.text(0.5, 0.953, subtitle,
                 ha='center', va='top', fontsize=10, style='italic',
                 color='#444')
        plt.show()

    # ── Optical-flow re-analysis (read-only) ──────────────────────────
    def _on_run_optical_flow_clicked(self):
        """Run the unified Farneback OF (`compute_of_profile_and_Q`) on
        the currently-selected edge and pop the editing viewer's full
        6-panel diagnostic — exactly the same figure as Analyze Mode in
        `mosaic_app`, with no graph writes and no napari arrow overlay.

        Strategy: build the `of_result` dict in the shape
        `_analyze_of_display` expects, alias `self.mosaic_graph` to
        `self.G` so the editing-viewer method can read AV refs without
        modification, then call the method as an unbound function.

        This deliberately replaces the earlier `compute_of_qt_local`
        path — that one only returned `Q_t` + `v_mean`, missing the
        velocity field, vessel mask, radial profile, and ROI bounds
        the figure needs."""
        from qtpy.QtWidgets import QApplication
        import time as _time
        if self._selected_edge is None:
            self._info_box.setPlainText(
                "Click an edge first, then press 'Run optical flow'.")
            return
        u, v = self._selected_edge
        # Focus-tile rule: View-tab tile filter wins over the
        # click-derived selected tile, so OF runs on whichever tile
        # the rest of the UI is currently focused on.
        tile_id = (self.current_tile_filter
                    if self.current_tile_filter is not None
                    else self._selected_tile_id)
        if tile_id is None:
            self._info_box.setPlainText(
                f"Edge ({u}, {v}) has no PIV measurement on file, so "
                f"there's no recorded tile to use.\n"
                f"Load a tile video first ('Browse tiles' or the "
                f"per-edge button), or set the Tile filter in the View "
                f"tab to choose a tile explicitly.")
            return
        if not self.tiles or self.video_dir is None:
            self._info_box.setPlainText(
                "Optical flow needs --tile-positions and --video-dir "
                "on the CLI (or the corresponding entries in the "
                "config bundle).")
            return
        video_path = self._get_video_path_for(int(tile_id))
        if video_path is None:
            self._info_box.setPlainText(
                f"No video found for tile {tile_id} under "
                f"{self.video_dir}.")
            return

        # Lazy imports so launch stays light.
        try:
            from ..analysis.flow import get_chain_coords
            from ..analysis.of_profile import compute_of_profile_and_Q
            from ..analysis.config import PX_SIZE_UM, FRAME_DT_S
            from ..io.tiff import load_tiff_stack, cut_before_fade
        except Exception as e:
            self._info_box.setPlainText(
                f"Optical-flow imports failed: {type(e).__name__}: {e}")
            return

        self._run_of_btn.setEnabled(False)
        self._info_box.setPlainText(
            f"Running optical flow on edge ({u}, {v}), tile {tile_id} "
            f"…  (~5–10s)")
        QApplication.processEvents()

        try:
            t0 = _time.time()
            # 1. Centerline + radius from the graph (mosaic Cartesian).
            coords_xy, mean_radius_px, _was_rev = get_chain_coords(
                self.G, [(u, v)], margin_px=8)
            if coords_xy is None or len(coords_xy) < 5:
                self._info_box.setPlainText(
                    f"Vessel ({u}, {v}) too short for OF "
                    f"(N={0 if coords_xy is None else len(coords_xy)}).")
                self._run_of_btn.setEnabled(True)
                return

            # 2. Load full-res video, percentile-normalise to uint8.
            print(f"\n  ── Optical-flow re-analysis (read-only) ──")
            print(f"  edge ({u}, {v}), tile {tile_id}, "
                  f"video {video_path.name}")
            stack = load_tiff_stack(video_path, max_frames=900)
            stack, _cut_at = cut_before_fade(stack, verbose=False)
            if stack.dtype != np.uint8:
                sample = stack[::max(1, stack.shape[0] // 20)]
                lo, hi = np.percentile(sample, (1.0, 99.0))
                if hi <= lo:
                    hi = lo + 1.0
                stack = np.clip(
                    (stack.astype(np.float32) - lo)
                    * (255.0 / (hi - lo)),
                    0, 255).astype(np.uint8)
            T, H, W = stack.shape
            print(f"  {T} frames, {H}×{W}, "
                  f"radius (mosaic) = {mean_radius_px:.1f} px")

            # If the loaded video is pre-downsampled (bundles ship
            # 352×320 instead of the canonical 704×640), the per-axis
            # tile_positions scales need to be rescaled by the same
            # factor so the converted centerline + radius land inside
            # the actual video pixels.  Detect the downsampling and
            # apply.
            ds_y = float(H) / float(TILE_RAW_HEIGHT) if H else 1.0
            ds_x = float(W) / float(TILE_RAW_WIDTH) if W else 1.0

            # 3. Convert mosaic Cartesian coords → tile pixel (row, col).
            #    Read-only viewer stores tiles as dicts, not TileInfo —
            #    can't reuse _core._mosaic_to_tile_coords directly, but
            #    the math is equivalent: divide mosaic-pixel displacements
            #    by per-axis mosaic-px-per-tile-px scale.
            tile_entry = self.tiles[int(tile_id)]
            # scale_x/scale_y assume full-res frames.  Multiply by the
            # downsample factor so coords convert into the actual video
            # we just loaded.  (Full-res ⇒ ds_x = ds_y = 1.0 ⇒ no-op.)
            sx = float(tile_entry['scale_x']) / ds_x
            sy = float(tile_entry['scale_y']) / ds_y
            trans_y_disp = float(
                tile_entry['translate_y'] - self._tile_offset_y)
            trans_x = float(
                tile_entry['translate_x'] - self._tile_offset_x)
            x_graph = coords_xy[:, 0]
            y_graph = coords_xy[:, 1]
            display_row = self.mosaic_height - y_graph
            j_tile = (x_graph - trans_x) / sx
            i_tile = (display_row - trans_y_disp) / sy
            centerline_rc = np.column_stack([i_tile, j_tile])
            # Anisotropic radius correction — matches the editing
            # viewer's _mosaic_radius_to_tile when given vessel_angle.
            vessel_angle = float(np.arctan2(
                coords_xy[-1, 1] - coords_xy[0, 1],
                coords_xy[-1, 0] - coords_xy[0, 0]))
            c_perp = abs(np.sin(vessel_angle))
            s_perp = abs(np.cos(vessel_angle))
            r_tile = mean_radius_px * float(np.sqrt(
                (c_perp / sx) ** 2 + (s_perp / sy) ** 2))
            print(f"  centerline tile range: rows "
                  f"[{i_tile.min():.0f}, {i_tile.max():.0f}], cols "
                  f"[{j_tile.min():.0f}, {j_tile.max():.0f}]")
            print(f"  radius (tile)  = {r_tile:.1f} px  "
                  f"(θ = {np.degrees(vessel_angle):.0f}°)")

            # 4. Run the unified OF pipeline — same call as the editing
            # viewer's Analyze Mode.
            edge_data = self.G.edges[u, v]
            f0 = edge_data.get('f0_hz', None)
            fb_ws = max(3, min(9, int(r_tile / 4))) | 1
            unified = compute_of_profile_and_Q(
                stack, centerline_rc, r_tile,
                bg_percentile=10,
                fb_winsize=fb_ws, fb_levels=1,
                fb_poly_n=3, fb_poly_sigma=0.7,
                dt=1, n_bins=30, fit_fraction=0.75,
                refine_centerline=True,
                use_flow_tangent=True,
                tangent_smooth_sigma=5.0,
                f0_hz=f0, frame_dt_s=FRAME_DT_S,
            )
            elapsed = _time.time() - t0
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"Optical flow failed: {type(e).__name__}: {e}")
            self._run_of_btn.setEnabled(True)
            return

        # 5. Build the of_result dict in the shape _analyze_of_display
        #    expects — mirrors the editing viewer's adapter at
        #    pertile/viewer/mosaic/_kirchhoff.py:2870-2937.
        #
        # If the bundle ships pre-downsampled videos (e.g. ds_x = 0.5
        # for a 2× spatial downsample) one displayed pixel covers
        # (1 / ds) × PX_SIZE_UM real micrometers, so use the effective
        # pixel size for the unit conversion.  At ds = 1 (full-res)
        # this collapses to the original formula.
        ds_avg = 0.5 * (ds_x + ds_y) if (ds_x and ds_y) else 1.0
        PX_SIZE_EFF = float(PX_SIZE_UM) / ds_avg
        conv = PX_SIZE_EFF ** 3 / FRAME_DT_S * 1e-6  # px²/frame → nL/s
        v_conv = PX_SIZE_EFF / FRAME_DT_S            # px/frame  → µm/s
        R_um = r_tile * PX_SIZE_EFF
        Q_plug_nLs = unified['Q_plug_mean'] * conv
        Q_pois_nLs = unified['Q_pois_mean'] * conv
        v0_fix_ums = unified['v0_fixed'] * v_conv
        v0_free_ums = unified.get('v0_free', np.nan) * v_conv
        R_fit_free = unified.get('R_fit_free', r_tile)
        Q_pois_free_nLs = (unified.get('v0_free', np.nan)
                            * np.pi * R_fit_free ** 2 / 2.0 * conv)
        v_mean_ums = Q_plug_nLs * 1e6 / (np.pi * R_um ** 2)
        vf = unified.get('vfield_mean')
        roi_offset = unified.get('roi_offset', (0, 0))
        r0_roi, c0_roi = int(roi_offset[0]), int(roi_offset[1])
        if vf is not None:
            roi_h, roi_w = vf.shape[:2]
        else:
            roi_h, roi_w = H, W
            r0_roi = c0_roi = 0
        rr_in = unified['rr_in']
        cc_in = unified['cc_in']
        mask_roi = np.zeros((roi_h, roi_w), dtype=bool)
        if len(rr_in) > 0:
            mask_roi[rr_in, cc_in] = True
        Q_pois_orig_nLs = unified.get(
            'Q_pois_orig_mean', np.nan) * conv
        v0_orig_ums = unified.get('v0_orig', np.nan) * v_conv

        of_result = dict(
            Q_mean=Q_plug_nLs,
            Q_poiseuille=Q_pois_nLs,
            Q_pois_orig=Q_pois_orig_nLs,
            Q_pois_free=Q_pois_free_nLs,
            v_mean=v_mean_ums,
            v0_fit=v0_fix_ums,
            v0_free=v0_free_ums,
            v0_orig=v0_orig_ums,
            R_stored_px=r_tile,
            R_flow_px=R_fit_free,
            R_flow_um=R_fit_free * PX_SIZE_UM,
            delta_px=unified.get('cl_shift', 0.0),
            best_side='center',
            fit_left=None, fit_right=None,
            c_floor=np.nan,
            profile_r=unified['bin_centers'],
            profile_v=np.abs(unified['profile_v']) * v_conv,
            profile_v_orig=np.abs(unified.get('profile_v_orig',
                np.full_like(unified['profile_v'], np.nan))) * v_conv,
            profile_r_local=None, profile_v_local=None,
            Q_t_of_mean=unified['Q_plug_t'] * conv,
            Q_t_of_pois=unified['Q_pois_t'] * conv,
            Q_t_of_local=None,
            Q_local=np.nan,
            v_field_mean=(vf * v_conv if vf is not None else None),
            roi_bounds=(r0_roi, r0_roi + roi_h, c0_roi, c0_roi + roi_w),
            vessel_mask_roi=mask_roi,
            wide_mask_roi=mask_roi,
            cl_shift=unified.get('cl_shift', 0.0),
            cl_used=unified.get('cl_used'),
            v0_deconv=unified.get('v0_deconv', np.nan) * v_conv,
            R_deconv=unified.get('R_deconv', r_tile),
            sigma_of_fit=unified.get('sigma_of_fit', np.nan),
        )

        # 6. Render curved ROI overlay on the mosaic — shows that the
        #    OF integration region follows the real centerline, not the
        #    straight graph edge.  Persists until next OF run / clear.
        try:
            self._show_of_region_overlay(
                of_result, int(tile_id), u, v,
                sx, sy, trans_x, trans_y_disp)
        except Exception as e:
            print(f"  (ROI overlay failed: {type(e).__name__}: {e})")

        # 7. Info-box summary (matches what shows in the editing viewer's
        #    console, condensed).
        self._info_box.setPlainText(
            f"Optical flow — edge ({u}, {v}), tile {tile_id}\n"
            f"  Q_pois (ridge CL) = {abs(Q_pois_nLs):.4f} nL/s\n"
            f"  Q_plug (ridge CL) = {abs(Q_plug_nLs):.4f} nL/s\n"
            f"  v̄                 = {abs(v_mean_ums):.0f} µm/s\n"
            f"  R (tile)          = {r_tile:.2f} px  "
            f"({R_um:.2f} µm)\n"
            f"  CL shift          = {unified.get('cl_shift', 0.0):+.1f} px\n"
            f"  computed in {elapsed:.1f}s   "
            f"(read-only: graph not modified)")

        # 7. Render the pared-down 4-panel diagnostic.  Uses the same
        #    `of_result` shape the editing viewer's _analyze_of_display
        #    expects (so we could fall back to that if needed), but
        #    drops the AV-fingerprint and RPSI-annotated panels —
        #    keeps only the data the read-only daily workflow needs.
        try:
            f0_hz = edge_data.get('f0_hz')
            if f0_hz is None or not np.isfinite(float(f0_hz)):
                f0_hz = 2.5  # sane yolk-sac fallback
            self._render_of_diagnostic(
                u, v, int(tile_id), of_result,
                float(f0_hz), PX_SIZE_UM, FRAME_DT_S,
                r_tile)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                self._info_box.toPlainText()
                + f"\n\nDiagnostic figure failed: "
                f"{type(e).__name__}: {e}")
        self._run_of_btn.setEnabled(True)

    def _render_of_diagnostic(self, u, v, tile_id, of_result,
                               f0_hz, PX_SIZE_UM, FRAME_DT_S,
                               r_tile_px):
        """4-panel optical-flow diagnostic:
          (0,0) radial velocity profile + Poiseuille fit
          (0,1) mean velocity field heatmap + quiver + vessel contour
          (1,0) cycle-averaged Q(t) over one period (no RPSI markers)
          (1,1) harmonic amplitudes vs SE noise floor — shows where Z = amp/SE
                comes from (the same statistic used for harmonic-class gating).

        Uses pertile.viewer.mosaic.show_figure for the Qt embedding —
        same path the editing viewer uses, no PNG roundtrip."""
        from matplotlib.gridspec import GridSpec
        from scipy.ndimage import uniform_filter1d
        from ..analysis.harmonic import fit_harmonics
        from .mosaic import show_figure

        Q_t_pois = np.asarray(of_result['Q_t_of_pois'], dtype=float)
        # Sign-flip so mean is positive (preserves waveform shape).
        _s_pois = -1.0 if np.nanmean(Q_t_pois) < 0 else 1.0
        Q_t_pois_s = Q_t_pois * _s_pois
        pr  = of_result['profile_r']
        pv  = of_result['profile_v']
        R_s = of_result['R_stored_px']
        v0_fix = of_result['v0_fit']
        v_mean = of_result['v_mean']
        vfield = of_result['v_field_mean']
        roi_bounds = of_result['roi_bounds']
        vessel_mask_roi = of_result['vessel_mask_roi']

        # Single up-front harmonic fit on the full Q_t — used by *both*
        # the cycle-averaged reconstruction overlay and the SNR panel,
        # and to compute the harmonic_class tag for the suptitle.  Keeps
        # the two bottom panels consistent.
        #
        # Two methodology choices vs the older `_harmonic_class`
        # convention cached on the graph:
        #   (a) σ from MAD, not std — robust to motion-spike outliers,
        #       consistent with the Huber loss used in the fit.
        #   (b) Numerator debiased: |Ĥ|_db = √max(0, Â² − 2·SE²).  Â is
        #       the magnitude of a 2-vector (A, B), so under null its
        #       expectation is √(π/2)·σ_a ≈ 1.25·SE, NOT 0.  Subtracting
        #       2·SE² makes the displayed amplitude ≈ 0 for a noise-only
        #       harmonic instead of a small positive bias.
        # The *tiering* (Rayleigh tail thresholds 2 / 3 / 4) is on the
        # raw Â/SE — that's what's Rayleigh-distributed under null.
        # We keep the cached `harmonic_class` (on the old std-based
        # threshold-3 convention) untouched so this panel doesn't fight
        # the categorical field used elsewhere.
        hr = None
        snrs_raw = {'DC': 0.0, 'H1': 0.0, 'H2': 0.0, 'H3': 0.0}
        sigma_resid = np.nan      # MAD-based, used for SE
        sigma_resid_std = np.nan  # plain std, for variance-explained
        total_snr = np.nan        # Var(fit) / Var(resid)
        try:
            hr = fit_harmonics(Q_t_pois_s, FRAME_DT_S,
                                f0_hz, K=3, loss='huber',
                                include_dc=True)
            resid = np.asarray(hr.get('resid', []), dtype=float)
            resid_ok = resid[np.isfinite(resid)] if resid.size else None
            if resid_ok is not None and resid_ok.size:
                # σ from MAD (1.4826·median|r−median(r)|) — robust.
                med = float(np.median(resid_ok))
                sigma_resid = 1.4826 * float(
                    np.median(np.abs(resid_ok - med)))
                sigma_resid_std = float(np.std(resid_ok))
            # Total harmonic SNR: Var(fit) / Var(resid).
            signal_fit = np.asarray(hr.get('signal', []), dtype=float)
            if (signal_fit.size and resid_ok is not None
                    and resid_ok.size and sigma_resid_std > 0):
                fit_ok = signal_fit[np.isfinite(signal_fit)]
                if fit_ok.size:
                    total_snr = float(
                        np.var(fit_ok) / max(np.var(resid_ok), 1e-30))
            if np.isfinite(sigma_resid) and sigma_resid > 0:
                N_full = int(np.isfinite(Q_t_pois_s).sum())
                se_amp_full = sigma_resid * np.sqrt(2.0 / max(N_full, 1))
                se_dc_full = sigma_resid / np.sqrt(max(N_full, 1))
                a0_full = float(hr.get('a0', np.nan))
                snrs_raw['DC'] = (abs(a0_full) / max(se_dc_full, 1e-30)
                                   if np.isfinite(a0_full) else 0.0)
                _harms = {h['k']: h for h in hr.get('harmonics', [])}
                for k in (1, 2, 3):
                    if k in _harms:
                        snrs_raw[f'H{k}'] = (float(_harms[k]['amp'])
                                              / max(se_amp_full, 1e-30))
        except Exception:
            hr = None
        # Binary class (suptitle) — keep on the cached convention so the
        # figure agrees with the categorical field used elsewhere.
        h_class = _harmonic_class(snrs_raw)
        h_class_label = _harmonic_class_label(h_class)

        # Rayleigh-scale tiers (per-harmonic QC, distinct from the
        # binary class).  Z = Â/SE under null is Rayleigh(σ=1) with
        # E[Z] ≈ 1.25,  P(Z>z) = exp(−z²/2).
        def _z_tier(z):
            if not np.isfinite(z):
                return ('unresolved', '#888888')
            if z >= 4.0:   return ('excellent', '#2ca02c')
            if z >= 3.0:   return ('good',      '#1f77b4')
            if z >= 2.0:   return ('marginal',  '#ff7f0e')
            return            ('unresolved', '#cc0000')

        fig = plt.figure(figsize=(15, 9.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.33, wspace=0.27)

        # ── (0,0) Radial velocity profile ──
        ax = fig.add_subplot(gs[0, 0])
        if pr is not None and pv is not None:
            valid = np.isfinite(pv)
            left  = valid & (pr < 0)
            right = valid & (pr >= 0)
            if left.any():
                ax.scatter(pr[left], pv[left],
                           c='steelblue', s=25, marker='o',
                           zorder=5, label='left of CL')
            if right.any():
                ax.scatter(pr[right], pv[right],
                           c='coral', s=25, marker='s',
                           zorder=5, label='right of CL')
            # Poiseuille fit (fixed R)
            if np.isfinite(v0_fix) and np.isfinite(R_s) and R_s > 0:
                r_smooth = np.linspace(-R_s, R_s, 200)
                v_fix = abs(v0_fix) * np.clip(
                    1.0 - (r_smooth / R_s) ** 2, 0, None)
                ax.plot(r_smooth, v_fix, color='indigo',
                        lw=2.0, alpha=0.85,
                        label=f'Poiseuille  v₀ = {abs(v0_fix):.0f} µm/s')
            # Plug-flow mean v̄ line
            v_mean_abs = abs(v_mean)
            if np.isfinite(v_mean_abs) and np.isfinite(R_s):
                ax.hlines(v_mean_abs, -R_s, R_s, colors='darkorange',
                          linestyles='-', lw=1.4, alpha=0.7,
                          label=f'Plug v̄ = {v_mean_abs:.0f} µm/s')
            if np.isfinite(R_s):
                ax.axvline(R_s, color='red', ls='--', lw=0.7, alpha=0.5)
                ax.axvline(-R_s, color='red', ls='--', lw=0.7, alpha=0.5,
                           label='±R_stored')
            ax.axvline(0, color='black', ls='-', lw=0.4, alpha=0.3)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'no profile data',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray')
        ax.set_xlabel('signed distance from centerline (tile px)')
        ax.set_ylabel('|v_axial| (µm/s)')
        ax.set_title(f'Radial velocity profile — edge ({u}, {v})')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

        # ── (0,1) Mean velocity field + quiver ──
        ax = fig.add_subplot(gs[0, 1])
        r0, r1, c0, c1 = roi_bounds
        roi_h, roi_w = r1 - r0, c1 - c0
        if vfield is not None and vfield.size > 0:
            speed = np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2)
            speed_display = speed.copy()
            outside = ~(vessel_mask_roi | of_result['wide_mask_roi'])
            speed_display[outside] *= 0.2
            im = ax.imshow(speed_display, cmap='hot',
                            origin='upper', aspect='equal')
            plt.colorbar(im, ax=ax, label='|v| (µm/s)', shrink=0.8)
            # Quiver (subsampled, clipped to vessel mask so the arrows
            # annotate the actual integration region, not the dim halo).
            step = max(3, int(R_s / 1.5))
            yr, xr = np.mgrid[0:roi_h:step, 0:roi_w:step]
            vy = vfield[::step, ::step, 1]
            vx = vfield[::step, ::step, 0]
            mask_sub = vessel_mask_roi[::step, ::step]
            if mask_sub.shape == xr.shape and mask_sub.any():
                yr_q = yr[mask_sub]; xr_q = xr[mask_sub]
                vx_q = vx[mask_sub]; vy_q = vy[mask_sub]
            else:
                yr_q, xr_q, vx_q, vy_q = yr, xr, vx, vy
            ax.quiver(xr_q, yr_q, vx_q, vy_q,
                       color='cyan', alpha=0.7,
                       scale_units='xy', angles='xy',
                       width=0.004, headwidth=3.5)
            # Vessel boundary
            ax.contour(vessel_mask_roi.astype(float),
                        levels=[0.5], colors='white',
                        linewidths=0.8, linestyles='--')
        else:
            ax.text(0.5, 0.5, 'no velocity field',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray')
        ax.set_title(f'Mean velocity field')
        ax.set_xlabel('col (ROI px)'); ax.set_ylabel('row (ROI px)')

        # ── (1,0) Cycle-averaged Q(t) ──
        ax = fig.add_subplot(gs[1, 0])
        period = max(1, int(round(1.0 / (f0_hz * FRAME_DT_S))))
        n_full = (len(Q_t_pois_s) // period) * period
        n_cycles = n_full // period if period > 0 else 0
        if n_cycles >= 2 and period >= 10:
            folded = Q_t_pois_s[:n_full].reshape(n_cycles, period)
            q_cyc = np.nanmean(folded, axis=0)
            q_cyc = uniform_filter1d(q_cyc, max(3, period // 30))
            # Align so diastole (min) sits at 20% into the cycle —
            # purely cosmetic, matches editing viewer's panel.
            i_dia = int(np.argmin(q_cyc))
            target_frac = 0.2
            shift = int(target_frac * period) - i_dia
            q_disp = np.roll(q_cyc, shift)
            t_cycle = np.arange(period) * FRAME_DT_S * 1000.0  # ms
            q_mean_cyc = float(np.mean(q_disp))
            # Also overlay every-cycle traces faintly for context.
            for i in range(min(n_cycles, 30)):
                ax.plot(t_cycle, np.roll(folded[i], shift),
                        color='#bbb', lw=0.4, alpha=0.35)
            ax.plot(t_cycle, q_disp, color='k', lw=2.0,
                    label=f'Cycle-averaged Q(t)  (n = {n_cycles})')
            # Harmonic-fit reconstruction over the same window — uses the
            # SAME `hr` as the SNR panel so the two are consistent.
            # Recompute the time-axis fit at one period of t_cycle so we
            # don't need to fold the full-data signal.
            if hr is not None and np.isfinite(hr.get('a0', np.nan)):
                t_sec = t_cycle / 1000.0  # ms → s
                a0_h = float(hr.get('a0', np.nan))
                y_dc = np.full_like(t_sec, a0_h)
                y_cum = y_dc.copy()
                _harms_disp = {h['k']: h for h in hr.get('harmonics', [])}
                # Align reconstruction phase to the cycle-averaged trace.
                # The folded waveform's argmin defines t=0 of the cycle;
                # we rolled q_cyc by `shift` to put min at target_frac.
                # Shift the reconstruction phase by the same fraction.
                phi_align = 2 * np.pi * (target_frac)
                for k in (1, 2, 3):
                    h = _harms_disp.get(k)
                    if h is None:
                        continue
                    omega_k = 2 * np.pi * k * f0_hz
                    A = float(h['A']); B = float(h['B'])
                    y_cum = y_cum + (
                        A * np.cos(omega_k * t_sec - k * phi_align)
                        + B * np.sin(omega_k * t_sec - k * phi_align))
                ax.plot(t_cycle, y_cum, color='C0',
                        lw=1.6, ls='--', alpha=0.95,
                        label=f'DC+H₁+H₂+H₃ fit  (PI={2.0*float(_harms_disp[1]["amp"])/max(abs(a0_h),1e-12):.2f})'
                              if 1 in _harms_disp else 'DC+H… fit')
            ax.axhline(q_mean_cyc, color='gray', ls='--', lw=1.0,
                        alpha=0.7,
                        label=f'Q̄ = {q_mean_cyc:.3f} nL/s')
            ax.set_xlim(t_cycle[0], t_cycle[-1])
            # Honest pulsatility scale — anchor y at 0.
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5,
                    f'too few cycles for averaging '
                    f'(n_cycles = {n_cycles}, period = {period})',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray')
        ax.set_xlabel('time within cycle (ms)')
        ax.set_ylabel('Q (nL/s)')
        ax.set_title(f'Cycle-averaged waveform   '
                      f'(f₀ = {f0_hz:.2f} Hz)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

        # ── (1,1) Harmonic SNR — where the Z statistics come from ──
        ax = fig.add_subplot(gs[1, 1])
        Q_dm = Q_t_pois_s - np.nanmean(Q_t_pois_s)
        Q_dm = Q_dm[np.isfinite(Q_dm)]
        N = len(Q_dm)
        if N > 20 and f0_hz > 0:
            # FFT magnitude spectrum (one-sided, single-bin amplitude).
            freqs = np.fft.rfftfreq(N, d=FRAME_DT_S)
            mag = (2.0 / N) * np.abs(np.fft.rfft(Q_dm))
            ax.semilogy(freqs, np.maximum(mag, 1e-6),
                         color='#888', lw=0.7,
                         label='|FFT(Q − Q̄)|')
            if hr is not None and np.isfinite(sigma_resid) and sigma_resid > 0:
                se_amp = sigma_resid * np.sqrt(2.0 / N)
                se_dc  = sigma_resid / np.sqrt(N)
                # Rayleigh-scale reference lines.  Under H_null, Â/SE
                # is Rayleigh(σ=1), so:
                #   E[Z|null]   ≈ 1.25            ← grey, "noise mean"
                #   Z = 2       ≈ P=14%   marginal
                #   Z = 3       ≈ P=1.1%  good
                #   Z = 4       ≈ P=0.03% excellent
                ax.axhline(1.25 * se_amp, color='#888', ls=':',
                            lw=0.8, alpha=0.7,
                            label='E[Z|null] ≈ 1.25  (Rayleigh)')
                ax.axhline(2.0 * se_amp, color='#ff7f0e', ls=':',
                            lw=0.9, alpha=0.7,
                            label='Z = 2  marginal')
                ax.axhline(3.0 * se_amp, color='#1f77b4', ls='--',
                            lw=1.0, alpha=0.85,
                            label='Z = 3  good')
                ax.axhline(4.0 * se_amp, color='#2ca02c', ls='--',
                            lw=1.0, alpha=0.85,
                            label='Z = 4  excellent')
                # SE_amp baseline (the unit, σ·√(2/N) with σ from MAD).
                ax.axhline(se_amp, color='#cc0000', ls='-',
                            lw=0.9, alpha=0.5,
                            label=f'SE_amp (MAD) = {se_amp:.3g}')

                a0 = float(hr.get('a0', np.nan))
                z_dc_raw = (abs(a0) / max(se_dc, 1e-30)
                             if np.isfinite(a0) else 0.0)
                # DC debias: |â₀|_db = √max(0, â₀² − SE_dc²).  (No factor
                # of 2 — DC is a single scalar, not a 2-vector magnitude.)
                a0_db = float(np.sqrt(max(0.0, a0 * a0 - se_dc * se_dc)))
                _tier_dc, _col_dc = _z_tier(z_dc_raw)
                harms = {h['k']: h for h in hr.get('harmonics', [])}
                # Plot DC slightly inside the axis (not on the y-spine),
                # extend x-min left of 0 so the star sits clearly visible.
                f_dc_marker = 0.08 * f0_hz
                ax.scatter([f_dc_marker], [max(a0_db, 1e-6)],
                            marker='*', s=240, c=_col_dc,
                            edgecolor='k', lw=0.7, zorder=6,
                            label=f'DC  Z = {z_dc_raw:.1f}  ({_tier_dc})')
                ax.vlines(f_dc_marker, se_dc, max(a0_db, se_dc),
                           colors=_col_dc, lw=1.6, alpha=0.6)
                for k in (1, 2, 3):
                    h = harms.get(k)
                    if h is None:
                        continue
                    amp_k = float(h['amp'])
                    f_k = k * f0_hz
                    z_k_raw = amp_k / max(se_amp, 1e-30)
                    # Debiased amplitude: |Ĥ|_db = √max(0, Â² − 2·SE²).
                    # Under null this is exactly 0 (after the clip),
                    # so a noise-only harmonic plots at the floor.
                    amp_k_db = float(np.sqrt(max(
                        0.0, amp_k * amp_k - 2.0 * se_amp * se_amp)))
                    _tier_k, _col_k = _z_tier(z_k_raw)
                    ax.scatter([f_k], [max(amp_k_db, 1e-6)],
                                marker='o', s=140, c=_col_k,
                                edgecolor='k', lw=0.7, zorder=6,
                                label=f'H{k}  Z = {z_k_raw:.1f}  '
                                       f'({_tier_k})')
                    ax.vlines(f_k,
                               min(se_amp, max(amp_k_db, 1e-6)),
                               max(se_amp, max(amp_k_db, 1e-6)),
                               colors=_col_k, lw=1.6, alpha=0.7)
                    ax.axvline(f_k, color=_col_k,
                                ls='--', lw=0.5, alpha=0.25)
            ax.set_xlim(-0.3, min(4.5 * f0_hz, freqs[-1]))
            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('amplitude (nL/s)  —  log scale')
            ax.set_title(
                'Harmonic SNR   '
                '(Z = Â/SE,  SE = σ_MAD·√(2/N);  '
                'markers = debiased |Ĥ|)')
            # Legend OUTSIDE the data area — currently the upper-right
            # parks it directly on the H1 peak.  Put it below the panel
            # in a single horizontal row so the spectrum is unobscured.
            ax.legend(fontsize=6.5, loc='upper center',
                       bbox_to_anchor=(0.5, -0.18),
                       ncol=5, frameon=True, framealpha=0.92)
            ax.grid(alpha=0.3, which='both')
        else:
            ax.text(0.5, 0.5, 'too few samples for SNR',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray')

        # Suptitle includes harmonic class so the SNR panel's threshold
        # decision translates directly to the categorical output used by
        # the rest of the pipeline.  total_snr = Var(fit) / Var(resid) is
        # the per-vessel "how clean is this signal overall" — high
        # regardless of whether the waveform is H1-only or harmonic-rich,
        # which avoids penalising clean arterial vessels for being
        # high-pass attenuated.
        tsnr_str = (f',  Var(fit)/Var(resid) = {total_snr:.1f}'
                     if np.isfinite(total_snr) else '')
        fig.suptitle(
            f'OF analysis — edge ({u}, {v}), tile {tile_id}   '
            f'Q_pois = {abs(np.nanmean(Q_t_pois_s)):.4f} nL/s,  '
            f'R = {r_tile_px:.1f} px ({r_tile_px * PX_SIZE_UM:.1f} µm)   '
            f'•  class {h_class}: {h_class_label}{tsnr_str}',
            fontsize=12)
        # Reserve room at the bottom for the SNR-panel legend.
        fig.subplots_adjust(bottom=0.13, top=0.92)
        show_figure(fig)

    def _show_of_region_overlay(self, of_result, tile_id, u, v,
                                  sx, sy, trans_x, trans_y_disp):
        """Render the OF integration mask as a curved orange ribbon on
        the mosaic.  Replaces any previous OF overlay.

        Pipeline:
          1.  Extract mask boundary contour(s) in ROI-pixel coords
              via skimage.measure.find_contours.
          2.  ROI → tile pixel (add roi_offset).
          3.  Tile pixel → mosaic display (multiply by sx/sy, add tile
              translates).  Inverse of the forward transform used to
              build centerline_rc for compute_of_profile_and_Q.
          4.  Add as a napari Shapes layer (polygon, orange fill +
              outline).  Mosaic display coords are (row, col) with
              row 0 at the top — matches every other Shapes layer in
              this viewer.

        Persists until the next OF run replaces it or
        _tear_down_of_region_overlay clears it explicitly."""
        try:
            from skimage.measure import find_contours
        except ImportError:
            print("  (skimage missing — skipping ROI overlay)")
            return
        mask = of_result.get('vessel_mask_roi')
        if mask is None or not mask.any():
            return
        roi_bounds = of_result.get('roi_bounds')
        if roi_bounds is None:
            return
        r0_roi, _r1_roi, c0_roi, _c1_roi = roi_bounds
        # find_contours returns a list of (N, 2) arrays in (row, col)
        # ROI coords.  Each ring becomes one polygon — keep them all so
        # vessels with internal holes (rare) draw correctly.
        contours = find_contours(mask.astype(float), level=0.5)
        if not contours:
            return
        polys = []
        H_mosaic = self.mosaic_height
        for c in contours:
            if len(c) < 3:
                continue
            i_roi = c[:, 0]
            j_roi = c[:, 1]
            # ROI → tile pixel
            i_tile = i_roi + r0_roi
            j_tile = j_roi + c0_roi
            # Tile pixel → mosaic display (inverse of the j_tile / i_tile
            # formulae used when building centerline_rc).  display_row
            # is the napari row index (row 0 at the top).
            display_row = i_tile * sy + trans_y_disp
            display_col = j_tile * sx + trans_x
            # napari Shapes wants (row, col) — same as display.  No
            # mosaic-height flip needed here because we never went
            # through the Cartesian-y representation.
            poly = np.column_stack([display_row, display_col])
            polys.append(poly)
        # Replace any prior overlay before adding the new one.
        self._tear_down_of_region_overlay(suppress_button=True)
        if not polys:
            return
        _prev_active = self.viewer.layers.selection.active
        self._of_region_layer = self.viewer.add_shapes(
            polys,
            shape_type='polygon',
            edge_color='#ff8800',
            face_color='#ff8800',
            edge_width=2.5,
            opacity=0.32,
            name=f'OF region — edge ({u},{v}), tile {tile_id}',
        )
        # Don't let the overlay capture mouse events — clicks must still
        # reach the edges layer for selection / inspection.
        try:
            self._of_region_layer.mouse_pan = False
            self._of_region_layer.mouse_zoom = False
        except Exception:
            pass
        if _prev_active is not None:
            try:
                self.viewer.layers.selection.active = _prev_active
            except Exception:
                pass
        try:
            self._clear_of_region_btn.setEnabled(True)
        except AttributeError:
            pass

    def _tear_down_of_region_overlay(self, suppress_button: bool = False):
        """Remove the OF region Shapes layer if present.  `suppress_button`
        skips the Clear-button disable (used internally by the show path
        when replacing one overlay with another)."""
        lyr = getattr(self, '_of_region_layer', None)
        if lyr is not None:
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            self._of_region_layer = None
        if not suppress_button:
            try:
                self._clear_of_region_btn.setEnabled(False)
            except AttributeError:
                pass

    def _plot_selected_qt(self):
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        meas = self.G.edges[u, v].get('measurements_piv') or []
        if not meas:
            return
        # Rank by quality so colour order is deterministic.
        meas_ranked = sorted(
            meas,
            key=lambda mm: (QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                            mm.get('snr_f0', 0) or 0,
                            mm.get('snr_pulse', 0) or 0),
            reverse=True)
        # Focus-tile rule: prefer the View-tab tile filter if set
        # (matches what's currently coloured on the network), else the
        # loaded video tile, else fall back to all measurements.
        filter_note = None
        focus_tid = (self.current_tile_filter
                      if self.current_tile_filter is not None
                      else self._video_tile_id)
        if focus_tid is not None:
            filtered = [m for m in meas_ranked
                        if m.get('tile_id') == focus_tid]
            source = ('tile filter'
                       if self.current_tile_filter is not None
                       else 'currently loaded video')
            if filtered:
                meas_ranked = filtered
                filter_note = (f"showing only tile {focus_tid} "
                                f"({source})")
            else:
                filter_note = (f"this edge has no measurement from "
                                f"tile {focus_tid} ({source}) — "
                                f"showing all instead")

        # Three-row figure:
        #   top = raw Q(t) for every measurement
        #   mid = harmonic decomposition of the best measurement
        #   bot = magnitude spectrum |FFT(Q_t)| (best measurement)
        # The top two share a time axis; the bottom has its own freq axis.
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85],
                              hspace=0.35)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_spec = fig.add_subplot(gs[2, 0])

        # ── Top: raw Q(t) traces for every measurement ──
        for m in meas_ranked:
            Q_t = m.get('Q_t')
            if Q_t is None:
                continue
            Q_t = np.asarray(Q_t, dtype=float)
            t = np.arange(len(Q_t)) / 250.0  # 250 fps
            usable = _measurement_usable(m)
            mark = '✓' if usable else '✗'
            ax_top.plot(
                t, Q_t,
                lw=0.8 if usable else 0.6,
                ls='-' if usable else '--',
                alpha=0.85 if usable else 0.5,
                label=(f"{mark} tile {m.get('tile_id')}  "
                       f"mean_Q={m.get('mean_Q', float('nan')):.3f}  "
                       f"PI={m.get('PI', float('nan')):.2f}"))
        ax_top.set_ylabel('Q (nL/s)')
        title_top = (f'Edge ({u}, {v}) — raw Q(t) across '
                     f'{len(meas_ranked)} of {len(meas)} measurements')
        if filter_note:
            title_top += f'\n({filter_note})'
        ax_top.set_title(title_top)
        ax_top.legend(fontsize=8, loc='best')
        ax_top.grid(alpha=0.3)

        # ── Bottom: harmonic decomposition of the best measurement ──
        # Refit harmonics on the raw Q_t so we get H1 *and* H2 even
        # though the measurement only stores H1 fields directly.
        best = next((m for m in meas_ranked
                     if _measurement_usable(m) and m.get('Q_t') is not None),
                    None)
        if best is not None:
            from ..analysis.harmonic import fit_harmonics
            Q_t = np.asarray(best['Q_t'], dtype=float)
            T = len(Q_t)
            t = np.arange(T) / 250.0
            f0 = float(best.get('f0_hz', 2.5))
            try:
                hr = fit_harmonics(Q_t, frame_dt=1.0/250, f0=f0,
                                    K=3, loss='huber', include_dc=True)
                a0 = float(hr['a0'])
                harms = {h['k']: h for h in hr['harmonics']}
                omega = 2 * np.pi * f0
                # Cumulative reconstructions
                y_dc = np.full(T, a0)
                h1 = harms.get(1)
                y_h1 = (y_dc + h1['A'] * np.cos(omega * t)
                              + h1['B'] * np.sin(omega * t)) if h1 else y_dc
                h2 = harms.get(2)
                y_h2 = (y_h1
                        + h2['A'] * np.cos(2 * omega * t)
                        + h2['B'] * np.sin(2 * omega * t)) if h2 else y_h1
                h3 = harms.get(3)
                y_h3 = (y_h2
                        + h3['A'] * np.cos(3 * omega * t)
                        + h3['B'] * np.sin(3 * omega * t)) if h3 else y_h2
                # Relative phase: phi_k − k·phi_1 (wrapped to [-π, π]).
                # This is what determines waveform shape — when Δφ_rel=0
                # higher harmonics reinforce H1's peak; when ±π they
                # cancel at the peak.
                def _wrap(p):
                    return (float(p) + np.pi) % (2 * np.pi) - np.pi
                # Per-harmonic Z-statistics from the same fit's residuals.
                resid = np.asarray(hr.get('resid', []), dtype=float)
                sigma = float(np.std(resid)) if resid.size else float('inf')
                N = len(Q_t)
                se_dc = sigma / max(np.sqrt(N), 1e-30)
                se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else float('inf')
                z_dc = abs(a0) / max(se_dc, 1e-30)
                z1 = (h1['amp'] / max(se_amp, 1e-30)) if h1 else 0.0
                z2 = (h2['amp'] / max(se_amp, 1e-30)) if h2 else 0.0
                z3 = (h3['amp'] / max(se_amp, 1e-30)) if h3 else 0.0
                snrs = {'DC': z_dc, 'H1': z1, 'H2': z2, 'H3': z3}
                klass = _harmonic_class(snrs)
                # Raw Q(t) shown faintly for noise context; a Butterworth
                # low-pass filtered version (cutoff = 4·f0, just above
                # H3) is overlaid darker so you can see the underlying
                # cardiac shape without the high-frequency junk.
                ax_mid.plot(t, Q_t, color='#bbb', lw=0.5, alpha=0.45,
                            label='raw Q(t)')
                try:
                    from scipy.signal import butter, filtfilt
                    nyq = 0.5 / dt   # 125 Hz at 250 fps
                    cutoff = min(4.0 * f0, 0.9 * nyq)
                    b, a_lp = butter(N=4, Wn=cutoff / nyq, btype='low')
                    Q_lp = filtfilt(b, a_lp, Q_t)
                    ax_mid.plot(t, Q_lp, color='#444', lw=0.9, alpha=0.85,
                                label=f'Q(t) low-passed @ {cutoff:.1f} Hz')
                except Exception:
                    pass
                ax_mid.plot(t, y_dc, color='C2', lw=1.2, ls=':',
                            label=f'DC = {a0:.3f}  (Z={z_dc:.1f})')
                if h1 is not None:
                    ax_mid.plot(t, y_h1, color='C0', lw=1.5,
                                label=f"+ H1 (amp={h1['amp']:.3f}, "
                                      f"PI={2*h1['amp']/max(abs(a0),1e-12):.2f}, "
                                      f"φ={h1['phi']:+.2f}, Z={z1:.1f})")
                if h2 is not None:
                    dphi2 = _wrap(h2['phi'] - 2 * (h1['phi'] if h1 else 0))
                    ax_mid.plot(t, y_h2, color='C3', lw=1.4, ls='--',
                                label=f"+ H2 (amp={h2['amp']:.3f}, "
                                      f"H2/H1={h2['amp']/max(h1['amp'],1e-12):.2f}, "
                                      f"Δφ_rel={dphi2:+.2f}, Z={z2:.1f})")
                if h3 is not None:
                    dphi3 = _wrap(h3['phi'] - 3 * (h1['phi'] if h1 else 0))
                    ax_mid.plot(t, y_h3, color='C1', lw=1.4, ls='-.',
                                label=f"+ H3 (amp={h3['amp']:.3f}, "
                                      f"H3/H1={h3['amp']/max(h1['amp'],1e-12):.2f}, "
                                      f"Δφ_rel={dphi3:+.2f}, Z={z3:.1f})")
                ax_mid.set_title(
                    f"Harmonic decomposition — tile {best.get('tile_id')}  "
                    f"(f₀ = {f0:.2f} Hz,  R² = "
                    f"{hr.get('r2', float('nan')):.2f})")
            except Exception as e:
                ax_mid.text(0.5, 0.5, f'fit_harmonics failed: {e}',
                            ha='center', va='center',
                            transform=ax_mid.transAxes)

            # ── Spectrum panel ──
            # FFT magnitude of the (de-meaned) Q_t to reveal what
            # frequency content is in the signal beyond the fitted
            # harmonics — camera shake, electrical noise, sub-harmonics.
            dt = 1.0 / 250
            Q_dm = Q_t - np.mean(Q_t)
            freqs = np.fft.rfftfreq(T, d=dt)
            mag = (2.0 / T) * np.abs(np.fft.rfft(Q_dm))
            ax_spec.semilogy(freqs, np.maximum(mag, 1e-6),
                              color='#444', lw=0.7)
            # Mark the fitted-harmonic frequencies
            for k in (1, 2, 3):
                ax_spec.axvline(k * f0, color='C0' if k == 1
                                          else 'C3' if k == 2 else 'C1',
                                ls='--', lw=0.9, alpha=0.7,
                                label=f'k·f₀ = {k*f0:.2f} Hz' if k <= 3 else None)
            # Cut off display just past the 4th harmonic — everything
            # higher is noise / aliasing for cardiac-rate signals.
            ax_spec.set_xlim(0, min(4.5 * f0, freqs[-1]))
            ax_spec.set_xlabel('frequency (Hz)')
            ax_spec.set_ylabel('|FFT(Q)|  (nL/s)')
            ax_spec.set_title('Magnitude spectrum  '
                              '(content outside the H1-H3 dashed lines = noise / aliasing)')
            ax_spec.legend(fontsize=8, loc='upper right')
            ax_spec.grid(alpha=0.3, which='both')
        else:
            ax_mid.text(0.5, 0.5,
                        'No usable measurement with Q_t — cannot decompose.',
                        ha='center', va='center',
                        transform=ax_mid.transAxes)
            ax_spec.text(0.5, 0.5, 'No spectrum to show.',
                          ha='center', va='center',
                          transform=ax_spec.transAxes)
        ax_mid.set_xlabel('time (s)'); ax_mid.set_ylabel('Q (nL/s)')
        ax_mid.legend(fontsize=8, loc='best')
        ax_mid.grid(alpha=0.3)
        plt.show()

    # ── callbacks ──────────────────────────────────────────────────────
    def _sync_field_combos(self):
        """Sync enable / disable state of the Quantity, Property, and
        Harmonic combos to the current Source + selection.  Greys out
        invalid combinations (e.g. P in Measured, Pressure drop in
        Q-mode, Phase at DC) so the user can never pick a bad combo.

        If the current selection becomes invalid after a Source change,
        falls back to the first valid Property / Quantity it finds.
        """
        if not hasattr(self, '_quantity_combo'):
            return
        from qtpy.QtCore import Qt
        # Quantity: P only enabled when Sim source is active.
        sim_active = bool(getattr(self, '_sim_active', False))
        qmodel = self._quantity_combo.model()
        q_p_item = qmodel.item(1)  # 'P' entry
        if q_p_item is not None:
            flags = q_p_item.flags()
            if sim_active and self.current_source == 'sim':
                q_p_item.setFlags(flags | Qt.ItemIsEnabled
                                    | Qt.ItemIsSelectable)
            else:
                q_p_item.setFlags(flags & ~Qt.ItemIsEnabled
                                    & ~Qt.ItemIsSelectable)
                if self.current_quantity == 'P':
                    self.current_quantity = 'Q'
                    self._quantity_combo.blockSignals(True)
                    self._quantity_combo.setCurrentIndex(0)
                    self._quantity_combo.blockSignals(False)
        # Property: enable rows matching (source, quantity, harmonic)
        # validity — delegate to _combo_valid so DC-incompatible
        # properties (phase, PI) grey out symmetrically.
        pmodel = self._property_combo.model()
        first_valid_idx = -1
        for i, (key, _lbl, _h, _valid_in) in enumerate(PROPERTY_DEFS):
            item = pmodel.item(i)
            if item is None:
                continue
            ok = _combo_valid(self.current_source, self.current_quantity,
                               key, self.current_harmonic)
            flags = item.flags()
            if ok:
                item.setFlags(flags | Qt.ItemIsEnabled
                                | Qt.ItemIsSelectable)
                if first_valid_idx < 0:
                    first_valid_idx = i
            else:
                item.setFlags(flags & ~Qt.ItemIsEnabled
                                & ~Qt.ItemIsSelectable)
        # If the currently-selected Property is invalid, fall back.
        if not _combo_valid(self.current_source, self.current_quantity,
                             self.current_property, self.current_harmonic):
            if first_valid_idx >= 0:
                self.current_property = PROPERTY_DEFS[first_valid_idx][0]
                self._property_combo.blockSignals(True)
                self._property_combo.setCurrentIndex(first_valid_idx)
                self._property_combo.blockSignals(False)
        # Harmonic combo: enabled only for harmonic-keyed properties.
        pdef = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        harmonic_keyed = bool(pdef and pdef[2])
        self._harmonic_combo.setEnabled(harmonic_keyed)
        # Disable DC entry when Property = phase (undefined).
        hmodel = self._harmonic_combo.model()
        dc_item = hmodel.item(0)
        if dc_item is not None:
            flags = dc_item.flags()
            if self.current_property == 'phase':
                dc_item.setFlags(flags & ~Qt.ItemIsEnabled
                                    & ~Qt.ItemIsSelectable)
                if self.current_harmonic == 'DC':
                    self.current_harmonic = 'H1'
                    self._harmonic_combo.blockSignals(True)
                    self._harmonic_combo.setCurrentIndex(1)
                    self._harmonic_combo.blockSignals(False)
            else:
                dc_item.setFlags(flags | Qt.ItemIsEnabled
                                    | Qt.ItemIsSelectable)

    def _on_quantity_change(self, _idx=None):
        new_q = self._quantity_combo.currentData()
        if new_q is None or new_q == self.current_quantity:
            return
        self.current_quantity = new_q
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_property_change(self, _idx=None):
        new_p = self._property_combo.currentData()
        if new_p is None or new_p == self.current_property:
            return
        self.current_property = new_p
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_harmonic_change(self, _idx=None):
        new_h = self._harmonic_combo.currentData()
        if new_h is None or new_h == self.current_harmonic:
            return
        self.current_harmonic = new_h
        # Harmonic gates property availability (phase/PI invalid at DC).
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_log_toggle(self, checked: bool):
        self.log_scale = bool(checked)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_source_change(self, _checked=None):
        if not self._src_radio_measured.isChecked() \
                and not self._src_radio_sim.isChecked():
            return  # transient state during exclusive-group switch
        new_src = 'sim' if self._src_radio_sim.isChecked() else 'measured'
        if new_src == self.current_source:
            return
        self.current_source = new_src
        # Re-sync enablement; the selection may need to fall back.
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    # ── Custom-BC helpers ────────────────────────────────────────────
    def _on_bc_source_change(self, _checked=None):
        """Toggle the Custom-waveform spinboxes based on BC source.
        The preview canvas stays live in both modes — it always shows
        whatever waveform would actually be applied at solve time."""
        if not hasattr(self, '_sim_custom_group'):
            return
        is_custom = bool(self._sim_bc_radio_custom.isChecked())
        # Disable the spinboxes individually, but leave the group (and
        # preview canvas) enabled so the user can still see the
        # measured waveform.
        for sp in getattr(self, '_sim_custom_spins', {}).values():
            sp.setEnabled(is_custom)
        # Re-render the preview for the new mode.
        self._refresh_custom_preview()

    def _read_custom_waveform(self):
        """Return per-side (amps, phases) tuples for the A and V custom
        waveform specs.  Amplitudes are |H_k|/|Q̄| (length 3, k=1..3);
        phases are in radians (length 3).
        Returns: ((amps_A, phases_A), (amps_V, phases_V)).
        """
        out = {}
        for side in ('A', 'V'):
            amps = []
            phases = []
            for k in (1, 2, 3):
                a = float(
                    self._sim_custom_spins[(side, k, 'amp')].value())
                p = np.radians(float(
                    self._sim_custom_spins[(side, k, 'phase')].value()))
                amps.append(a)
                phases.append(p)
            out[side] = (amps, phases)
        return out['A'], out['V']

    def _refresh_custom_preview(self):
        """Refresh the BC waveform preview.

        Shows whichever waveform would actually be applied at solve
        time — custom (from the spinboxes) or measured (per-type
        complex average of stored bc_harmonics, with the canonical
        −1 sign-flip applied to sinks).  Y-axis is absolute Q(t) in
        nL/s so the A↔V magnitude asymmetry reads honestly.
        """
        if not hasattr(self, '_sim_custom_fig'):
            return
        f0 = (float(self._sim_detected_f0)
              if np.isfinite(self._sim_detected_f0) else 2.5)
        T_period = 1.0 / f0
        t = np.linspace(0.0, 2.0 * T_period, 400)
        omega = 2.0 * np.pi * f0
        target_flux = float(self._sim_flux_spin.value())
        is_custom = self._sim_bc_radio_custom.isChecked()
        n_src = max(1, len(self._source_nodes))
        n_snk = max(1, len(self._sink_nodes))

        self._sim_custom_fig.clear()
        ax = self._sim_custom_fig.add_subplot(111)

        if is_custom:
            (amps_A, phases_A), (amps_V, phases_V) = \
                self._read_custom_waveform()
            Qbar_A = +target_flux / n_src
            Qbar_V = -target_flux / n_snk
            y_A = np.full_like(t, Qbar_A)
            y_V = np.full_like(t, Qbar_V)
            for k_idx, (a, p) in enumerate(
                    zip(amps_A, phases_A), start=1):
                y_A += a * Qbar_A * np.cos(k_idx * omega * t + p)
            for k_idx, (a, p) in enumerate(
                    zip(amps_V, phases_V), start=1):
                y_V += a * Qbar_V * np.cos(k_idx * omega * t + p)
            PI_A = 2.0 * abs(amps_A[0]) if amps_A[0] else 0.0
            PI_V = 2.0 * abs(amps_V[0]) if amps_V[0] else 0.0
            title = (f"CUSTOM  Q̄_A = {abs(Qbar_A):.3f}  "
                     f"(PI_A ≈ {PI_A:.2f}),  "
                     f"Q̄_V = {abs(Qbar_V):.3f} nL/s  "
                     f"(PI_V ≈ {PI_V:.2f})")
        else:
            # Measured: per-type complex average of stored bc_harmonics,
            # with canonical sign-flip on sinks (Q into network).
            src_arrs = [np.asarray(self.G.nodes[n].get('bc_harmonics'),
                                     dtype=complex)
                        for n in self._source_nodes
                        if self.G.nodes[n].get('bc_harmonics') is not None]
            snk_arrs = [-np.asarray(self.G.nodes[n].get('bc_harmonics'),
                                      dtype=complex)
                        for n in self._sink_nodes
                        if self.G.nodes[n].get('bc_harmonics') is not None]
            if not src_arrs or not snk_arrs:
                ax.text(0.5, 0.5,
                        'No measured bc_harmonics on this graph.',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=10, color='#888')
                ax.set_xticks([]); ax.set_yticks([])
                self._sim_custom_canvas.draw_idle()
                return
            src_avg = np.mean(np.stack(src_arrs, axis=0), axis=0).copy()
            snk_avg = np.mean(np.stack(snk_arrs, axis=0), axis=0).copy()
            # Apply the same flux scaling the solver path uses.
            if self._sim_equal_split_chk.isChecked():
                src_avg[0] = +target_flux / len(src_arrs)
                snk_avg[0] = -target_flux / len(snk_arrs)
                mode_label = 'equal-split'
            else:
                sum_src = sum(arr[0].real for arr in src_arrs)
                if abs(sum_src) > 1e-12:
                    scale = target_flux / abs(sum_src)
                    src_avg = scale * src_avg
                    snk_avg = scale * snk_avg
                mode_label = 'measured-ratio'
            # Reconstruct Q(t) for both types.
            y_A = np.full_like(t, src_avg[0].real, dtype=float)
            y_V = np.full_like(t, snk_avg[0].real, dtype=float)
            kmax = min(len(src_avg), 4)
            for k in range(1, kmax):
                y_A += (src_avg[k].real * np.cos(k * omega * t)
                        - src_avg[k].imag * np.sin(k * omega * t))
                y_V += (snk_avg[k].real * np.cos(k * omega * t)
                        - snk_avg[k].imag * np.sin(k * omega * t))
            title = (f"MEASURED (per-type avg, {mode_label})   "
                     f"Q̄_A = {abs(src_avg[0].real):.3f},  "
                     f"Q̄_V = {abs(snk_avg[0].real):.3f} nL/s")

        # Display convention: flip each trace so its DC reads positive.
        # "Positive" = the sign that yields DC > 0 for that side, so
        # both A and V sit on the same axis and the magnitudes are
        # directly comparable.  Solver-internal sign convention (sinks
        # store -Q_outflow) is unaffected — this is purely cosmetic.
        s_A = -1.0 if float(np.nanmean(y_A)) < 0 else 1.0
        s_V = -1.0 if float(np.nanmean(y_V)) < 0 else 1.0
        y_A = y_A * s_A
        y_V = y_V * s_V

        ax.plot(t * 1000.0, y_A, color='C0', lw=1.5, label='A')
        ax.plot(t * 1000.0, y_V, color='C1', lw=1.5, label='V')
        ax.axhline(0, color='#888', lw=0.5)
        ax.set_xlabel('t (ms)', fontsize=8)
        ax.set_ylabel('Q(t) (nL/s)', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        self._sim_custom_canvas.draw_idle()

    def _on_nodes_toggle(self, checked: bool):
        self.show_nodes = bool(checked)
        self._refresh_nodes()

    def _on_tile_boundaries_toggle(self, checked: bool):
        self.show_tile_boundaries = bool(checked)
        self._refresh_tile_boundaries()

    def _on_tile_labels_toggle(self, checked: bool):
        self.show_tile_labels = bool(checked)
        self._refresh_tile_labels()

    def _load_video_for_selected(self):
        tid = getattr(self, '_selected_tile_id', None)
        if tid is None:
            return
        self._load_video_overlay(int(tid))
        self._unload_video_btn.setEnabled(self._video_layer is not None)

    def _load_video_from_dropdown(self):
        if not hasattr(self, '_tile_combo'):
            return
        # currentData() returns the int tile ID stored as userData;
        # falls back to None for the "(pick a tile)" placeholder and
        # for greyed-out entries (which can't be selected anyway).
        tid = self._tile_combo.currentData()
        if tid is None:
            return
        self._load_video_overlay(int(tid))
        self._unload_video_btn.setEnabled(self._video_layer is not None)

    # ── public entry ───────────────────────────────────────────────────
    def run(self):
        self._setup_viewer()
        self._setup_panel()
        self._refresh_edges()
        self._refresh_nodes()
        # Initial colour-bar draw — needs `_last_vmin`/`_last_vmax`
        # populated by the first `_refresh_edges()` above, so do it here
        # rather than at the end of `_setup_panel`.
        self._refresh_cbar()
        import napari
        napari.run()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Read-only mosaic viewer (intern + publication use).")
    parser.add_argument('graph', nargs='?', default=None,
                        help="Path to mosaic_graph_analyzed.gpickle. "
                             "Optional if --config is given.")
    parser.add_argument('--config', default=None,
                        help="Path to a batch_analyze_config.json (the "
                             "same one the analysis pipeline uses).  "
                             "Convenience: pulls graph, tiff, tile "
                             "positions, video dir, and video pattern "
                             "from one file.  Individual --tiff / "
                             "--tile-positions / --video-dir / "
                             "--video-pattern args override.")
    parser.add_argument('--tiff', default=None,
                        help="Optional stitched mosaic TIFF for background.")
    parser.add_argument('--tile-positions', default=None,
                        help="Path to tile_positions_manual.json.  "
                             "Required for video overlay.")
    parser.add_argument('--video-dir', default=None,
                        help="Directory containing per-tile videos. "
                             "Required for video overlay.")
    parser.add_argument('--video-pattern', default=None,
                        help="Filename pattern for tile videos, e.g. "
                             "'10x 250fps loc{vid}_C001H001S0001'. "
                             "Required for video overlay.")
    parser.add_argument('--initial-field', default='mean_Q',
                        help="Field to colour edges by on startup. "
                             "Default: mean_Q.")
    parser.add_argument('--no-cache-harmonic-class', action='store_true',
                        help="Don't write the precomputed harmonic_class "
                             "back to the gpickle.  Default is to cache "
                             "(adds an edge attr; touches no other data).")
    parser.add_argument('--force-recompute-harmonic-class',
                        action='store_true',
                        help="Wipe any existing harmonic_class values and "
                             "recompute from scratch.  Useful if PIV was "
                             "re-run and the cached values are stale.")
    args = parser.parse_args()

    # Resolve paths via config if provided; individual args win.
    # Paths in the config can be RELATIVE to the config file's directory
    # — that makes the data folder portable (intern only changes the
    # config path, not its contents).
    cfg: dict = {}
    cfg_dir: Optional[Path] = None
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg_dir = cfg_path.parent
        with open(cfg_path) as f:
            cfg = json.load(f)

    def _resolve(cli_val, cfg_key):
        v = cli_val if cli_val is not None else cfg.get(cfg_key)
        if v is None:
            return None
        p = Path(v)
        if p.is_absolute() or cfg_dir is None:
            return p
        return (cfg_dir / p).resolve()

    graph = _resolve(args.graph, 'mosaic_graph')
    if not graph:
        parser.error("graph path required (positional arg or "
                     "config['mosaic_graph']).")
    tiff = _resolve(args.tiff, 'mosaic_tiff')
    tile_positions = _resolve(args.tile_positions, 'tile_positions')
    video_dir = _resolve(args.video_dir, 'video_dir')
    # video_pattern is a filename pattern (not a path) — pass through.
    video_pattern = args.video_pattern or cfg.get('video_pattern')

    viewer = ReadOnlyMosaicViewer(
        Path(graph),
        tiff_path=Path(tiff) if tiff else None,
        tile_positions=Path(tile_positions) if tile_positions else None,
        video_dir=Path(video_dir) if video_dir else None,
        video_pattern=video_pattern,
        initial_field=args.initial_field,
        cache_harmonic_class=not args.no_cache_harmonic_class,
        force_recompute_harmonic_class=args.force_recompute_harmonic_class,
    )
    viewer.run()


if __name__ == '__main__':
    main()
