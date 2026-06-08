"""
Analysis configuration - convenience exports from the base config system.

This module provides direct access to commonly-used analysis parameters
from the centralized config system (pertile.config).

The authoritative config is in configs/default.json, loaded via pertile.config.
This module provides convenient constant exports for backwards compatibility.
"""
from __future__ import annotations
from typing import List

from pertile.config import get_default_config

# Load config once
_cfg = get_default_config()

# Helper to safely get nested config values
def _get(section: str, key: str, default):
    """Get a config value from section.key, with default fallback."""
    sect = getattr(_cfg, section, None)
    if sect is None:
        return default
    return getattr(sect, key, default)

# =============================================================================
# Calibration
# =============================================================================
PX_SIZE_UM: float = _get('calibration', 'px_size_um', 1.7)
"""Pixel size in micrometers (1.7 µm/px for 10x objective)."""

FRAME_DT_S: float = _get('calibration', 'frame_dt_s', 0.004)
"""Time between frames in seconds (1/fps)."""

FPS: int = _get('calibration', 'fps', 250)
"""Frames per second."""

DOC_Z_C_UM: float = _get('calibration', 'doc_z_c_um', 13.5)
"""Depth-of-correlation half-depth in micrometers (Olsen & Adrian model).
Computed from optics: z_c = δz_corr/2 where δz_corr follows Olsen & Adrian (2000).
For 10x/NA0.3 air objective, 0.5 µm beads, λ=605 nm: z_c ≈ 13.5 µm (DOC ≈ 27 µm).
Set to 0 to disable DOC correction (use focal-plane Poiseuille)."""

DOC_MODEL: str = _get('calibration', 'doc_model', 'squared_lorentzian')
"""DOC weighting function W(z) model.
'squared_lorentzian': 1/(1+u²)² — Olsen & Adrian (2000), derived for PIV cross-correlation.
'lorentzian':         1/(1+u²)   — broader tails, intermediate depth averaging.
'gaussian':           exp(-u²/2) — narrowest effective DOC, least depth averaging."""

# =============================================================================
# Optics (hardware constraints for synthetic model and DOC computation)
# =============================================================================
OPTICS_NA: float = _get('optics', 'NA', 0.3)
"""Numerical aperture of the objective."""

OPTICS_WAVELENGTH_UM: float = _get('optics', 'wavelength_um', 0.605)
"""Emission wavelength in micrometers (FluoSpheres F-8812, orange-red fluorescence)."""

OPTICS_BEAD_DIAMETER_UM: float = _get('optics', 'bead_diameter_um', 0.5)
"""Tracer bead diameter in micrometers."""

OPTICS_N_MEDIUM: float = _get('optics', 'n_medium', 1.0)
"""Refractive index of immersion medium (1.0 for air objective)."""

OPTICS_MAGNIFICATION: float = _get('optics', 'magnification', 10.0)
"""Objective magnification."""

# =============================================================================
# Heart Rate Detection
# =============================================================================
FMIN_HZ: float = _get('heart_rate', 'fmin_hz', 1.0)
"""Lower frequency bound for heart rate detection (Hz)."""

FMAX_HZ: float = _get('heart_rate', 'fmax_hz', 3.5)
"""Upper frequency bound for heart rate detection (Hz)."""

N_HARMONICS: int = _get('heart_rate', 'n_harmonics', 3)
"""Number of harmonics to fit in harmonic regression."""

# =============================================================================
# Kymograph / GST Parameters
# =============================================================================
GST_WINDOWS: List[int] = _get('kymograph', 'gst_windows', [7])
"""Window sizes for GST computation. Single window [7] for speed.
History: [5,7,9,11,13,15,17] → [5,9,13] → [7]. Multi-window picks best
coherence per pixel but benchmarks show single window gives similar accuracy
at 3-5× speedup. Window 7 balances spatial resolution and noise averaging."""

COHERENCE_THRESHOLD: float = _get('kymograph', 'coherence_threshold', 0.75)
"""Minimum coherence for velocity estimation (higher = more selective)."""

BAND_RADIUS_PX: int = _get('kymograph', 'band_radius_px', 1)
"""Width of sampling band perpendicular to centerline (+-band_radius pixels)."""

OFFSET_STEP_PX: int = _get('kymograph', 'offset_step_px', 2)
"""Spacing between radial offset sampling positions (pixels).
With band_radius=2 (5px width), step=3 gives ~40% overlap between bands.
Step=1 (original) gives 80% overlap and highly correlated measurements."""

MARGIN_PX: int = _get('kymograph', 'margin_px', 8)
"""Margin to trim at vessel ends to exclude junction nodes (pixels)."""

MIN_PATH_LENGTH_PX: int = _get('kymograph', 'min_path_length_px', 8)
"""Minimum centerline path length required for reliable kymograph analysis (pixels).
Vessels shorter than this will be marked as unmeasured and treated as grey for
super-node merging in Kirchhoff checking."""

# =============================================================================
# Temporal Detrending
# =============================================================================
DETREND_WINDOWS: List[int] = [0]
"""Window sizes to try for temporal detrending (0 = no detrending, disabled by default)."""

DEFAULT_DETREND_WINDOW: int = 0
"""Default detrending window size (0 = disabled)."""

# =============================================================================
# Quality Thresholds
# =============================================================================
COV_THRESHOLD: float = _get('quality', 'min_coverage', 0.25)
"""Minimum coverage for weighted median velocity."""

MIN_VALID_FRACTION: float = _get('quality', 'min_valid_fraction', 0.20)
"""Minimum fraction of valid samples required."""

V_MAX_UM_S: float = _get('quality', 'v_max_um_s', 5000.0)
"""Maximum physical velocity in um/s. Frames exceeding this are unphysical (5 mm/s for 21 somites)."""

# Coherence gating for noisy/short vessels
COHERENCE_GATE_THRESHOLD: float = _get('quality', 'coherence_gate_threshold', 0.3)
"""Per-pixel coherence floor for velocity estimation. Pixels below this are
masked before computing v_hat(t). Set to 0 to disable gating."""

COLUMN_COHERENCE_MIN: float = _get('quality', 'column_coherence_min', 0.3)
"""Minimum fraction of frames a column must have good coherence to be included.
Used for non-contiguous column masking."""

BAND_COHERENCE_MIN: float = _get('quality', 'band_coherence_min', 0.5)
"""Minimum mean coherence for an entire radial band to be included in the
profile fit. Bands below this are rejected (except centerline, always kept)."""

ARC_QUALITY_THRESHOLD: float = _get('quality', 'arc_quality_threshold', 0.25)
"""Quality threshold for find_good_arc_region column selection."""

MIN_ARC_LENGTH_PX: int = _get('quality', 'min_arc_length_px', 15)
"""Minimum arc length in pixels for velocity analysis."""

# Poiseuille envelope filter for radial bands
ENVELOPE_INNER_FRACTION: float = _get('quality', 'envelope_inner_fraction', 0.5)
"""Fraction of R_seg used to select 'inner' bands for Poiseuille envelope fit.
Inner bands are more reliable; the envelope extrapolated outward catches
outer bands with anomalously high |v| (e.g., junction contamination)."""

ENVELOPE_SIGMA_TOLERANCE: float = _get('quality', 'envelope_sigma_tolerance', 1.0)
"""Number of inner-band median-σ above the Poiseuille envelope to tolerate.
Bands where |v̄| > |v_envelope(r)| + k×median(σ_inner) are rejected.
Uses median σ of inner bands only — the reliable noise floor estimate."""

# =============================================================================
# Visualization
# =============================================================================
VELOCITY_CMAP: str = "magma"
"""Colormap for velocity visualization."""

VELOCITY_VMAX_UM_S: float = 1000.0
"""Maximum velocity for colormap scaling (um/s)."""

SNR_DB_MIN: float = -10.0
"""Minimum SNR for alpha scaling (dB)."""

SNR_DB_MAX: float = 20.0
"""Maximum SNR for alpha scaling (dB)."""
