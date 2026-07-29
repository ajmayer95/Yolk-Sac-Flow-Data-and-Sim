"""
Synthetic kymograph generator for benchmarking velocity estimation.

Generates (T, M) kymographs with known ground-truth velocity by directly
rasterizing diagonal streaks.  Each streak enters one edge of the arc and
traverses the full visible region, just like real blood cells.

Noise model (in order of application):
  1. Streak signal (diagonal Gaussians, possibly out-of-focus)
  2. Stationary bright spots (stuck beads / debris)
  3. Static horizontal bands (vessel wall echoes)
  4. Spatially varying background (autofluorescence gradient)
  5. DC background (uniform autofluorescence / camera offset)
  6. Temporal intensity flicker (lamp arc wander / LED driver noise)
  7. Poisson shot noise  →  readout noise (or plain Gaussian)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .flow import doc_weight


# =============================================================================
# Hardware-constrained optics helpers
# =============================================================================

def compute_doc_z_c(
    NA: float = 0.3,
    wavelength_um: float = 0.605,
    bead_diameter_um: float = 0.5,
    n_medium: float = 1.0,
    magnification: float = 10.0,
    epsilon: float = 0.01,
) -> float:
    """Compute DOC half-depth z_c from Olsen & Adrian (2000).

    δz_corr = 2√{ C(ε) · [f#² d_p² + 5.95(M+1)²λ²f#⁴/M²] }

    where C(ε) = (1−√ε)/√ε and f# = n₀/(2·NA).
    Returns z_c = δz_corr / 2.

    Parameters
    ----------
    NA : float
        Numerical aperture of the objective.
    wavelength_um : float
        Emission wavelength in micrometers.
    bead_diameter_um : float
        Tracer particle diameter in micrometers.
    n_medium : float
        Refractive index of immersion medium (1.0 for air, 1.33 for water).
    magnification : float
        Objective magnification.
    epsilon : float
        Correlation threshold (standard: 0.01).

    Returns
    -------
    z_c_um : float
        DOC half-depth in micrometers.
    """
    f_num = n_medium / (2.0 * NA)
    C_eps = (1.0 - np.sqrt(epsilon)) / np.sqrt(epsilon)
    # Geometric term (particle image)
    term_geom = f_num**2 * bead_diameter_um**2
    # Diffraction term
    term_diff = (5.95 * ((magnification + 1) / magnification)**2
                 * wavelength_um**2 * f_num**4)
    delta_z_corr = 2.0 * np.sqrt(C_eps * (term_geom + term_diff))
    return delta_z_corr / 2.0


def compute_psf_sigma_px(
    NA: float = 0.3,
    wavelength_um: float = 0.605,
    px_size_um: float = 1.7,
) -> float:
    """Compute PSF Gaussian σ in pixels (object-space).

    Uses σ_PSF ≈ 0.42 λ / NA (Born & Wolf Gaussian approximation to
    the Airy disk main lobe).  This is a theoretical lower bound;
    practical values may be 1.5–2× larger due to aberrations, vibration,
    and finite pixel sampling.

    For sub-diffraction beads (d_p < Airy disk), the streak width is
    PSF-limited — the bead diameter does not matter.

    Parameters
    ----------
    NA : float
        Numerical aperture.
    wavelength_um : float
        Emission wavelength in micrometers.
    px_size_um : float
        Pixel size in µm (object-space equivalent).

    Returns
    -------
    sigma_px : float
        PSF σ in pixels.
    """
    sigma_um = 0.42 * wavelength_um / NA
    return sigma_um / px_size_um


def generate_synthetic_kymograph(
    # ---- Geometry ----
    T: int = 1000,
    M: int = 50,
    # ---- Velocity waveform ----
    v_mean: float = 2.0,
    v_amplitude: float = 0.0,
    v_harmonic: float = 0.0,
    phi_1: float = 0.0,
    phi_2: float = 0.0,
    f0_hz: float = 1.5,
    frame_dt_s: float = 0.004,
    v_t_override: Optional[np.ndarray] = None,
    # ---- Particle / streak properties ----
    particle_density: float = 0.3,
    streak_width: float = 0.6,
    intensity_mean: float = 1.0,
    intensity_spread: float = 0.3,
    v_spread: float = 0.0,
    # ---- Out-of-focus particle population ----
    oof_fraction: float = 0.0,
    oof_width_factor: float = 4.0,
    oof_intensity_factor: float = 0.5,
    # ---- Static artifacts ----
    static_bands: int = 0,
    static_band_strength: float = 0.3,
    n_bright_spots: int = 0,
    bright_spot_intensity: float = 2.0,
    bright_spot_width: float = 0.5,
    # ---- Background ----
    background: float = 0.0,
    bg_variation: float = 0.0,
    # ---- Noise ----
    noise_sigma: float = 0.2,
    poisson_noise: bool = False,
    gain: float = 1.0,
    flicker_sigma: float = 0.0,
    # ---- Spatial velocity variation ----
    radius_ratio: float = 1.0,
    taper_shape: str = 'linear',
    # ---- Other ----
    contrast_mode: str = 'bright',
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate a synthetic kymograph with known velocity ground truth.

    Parameters
    ----------
    T : int
        Number of frames (time axis).
    M : int
        Number of spatial columns (arc positions).
    v_mean : float
        Mean streak velocity in px/frame.  Positive = rightward.
    v_amplitude : float
        Fundamental pulsatile amplitude (px/frame).
        v(t) = v_mean + v_amplitude·sin(2πf₀t + phi_1)
               + v_harmonic·sin(4πf₀t + phi_2)
    v_harmonic : float
        Second-harmonic amplitude for cardiac waveform asymmetry.
    phi_1, phi_2 : float
        Phase offsets (radians) for the two harmonics.
    f0_hz : float
        Heart rate frequency (Hz).
    frame_dt_s : float
        Frame interval in seconds (default 0.004 = 250 fps).
    v_t_override : (T,) array, optional
        Use this directly as v(t), ignoring harmonic parameters.
    particle_density : float
        Expected new streaks entering per frame.
    streak_width : float
        Gaussian σ of in-focus streaks (pixels).
    intensity_mean : float
        Mean streak peak intensity.
    intensity_spread : float
        Lognormal σ for inter-particle intensity variation.
    v_spread : float
        Velocity spread (σ, px/frame) across particles, mimicking the
        Poiseuille velocity profile across the sampling band.  Particles
        at different radial positions have different speeds and will
        cross each other when v_spread is large.
    oof_fraction : float
        Fraction of particles that are out-of-focus.  Out-of-focus beads
        appear wider (oof_width_factor × streak_width) and dimmer
        (oof_intensity_factor × intensity).  Models beads above/below the
        focal plane — very common in practice.
    oof_width_factor : float
        Width multiplier for out-of-focus particles (default 3×).
    oof_intensity_factor : float
        Intensity multiplier for out-of-focus particles (default 0.3×).
    static_bands : int
        Number of static horizontal bands (wide Gaussian smears spanning
        the full time axis).  Models vessel wall autofluorescence echoes
        or out-of-focus fluorescent structures.
    static_band_strength : float
        Peak intensity of static bands.
    n_bright_spots : int
        Number of stationary point-source artifacts (stuck beads,
        fluorescent debris, autofluorescent patches).  Unlike static_bands
        (wide Gaussian), these are narrow (~0.5 px) and appear as thin
        vertical lines in the kymograph.
    bright_spot_intensity : float
        Mean peak intensity of bright spots (relative to intensity_mean).
    bright_spot_width : float
        Gaussian σ of bright spots in pixels (default 0.5 = near point source).
    background : float
        Uniform DC background level from autofluorescence, out-of-focus
        beads, and camera dark current.  In Poisson mode, non-zero background
        contributes shot noise in the "empty" regions — this is the main
        reason background degrades SNR even when streaks are bright.
        Typical value: 0.1–0.3 × intensity_mean.
    bg_variation : float
        Amplitude of spatially varying background (sinusoidal).  Models
        illumination falloff, autofluorescence gradient, or vignetting
        across the arc.  A value of 0.2 adds ±0.2 spatial modulation.
    noise_sigma : float
        Gaussian noise σ.  In Gaussian mode (poisson_noise=False): the
        only noise source.  In Poisson mode: readout/electronic noise
        added after the Poisson draw.  sCMOS readout noise ≈ 0.02–0.05.
    poisson_noise : bool
        Apply Poisson (photon shot noise) scaled by `gain`.  This is the
        dominant noise source in fluorescence microscopy.  Bright streaks
        get more noise than background (signal-dependent variance).
    gain : float
        Photon counts per intensity unit.  Controls the magnitude of
        Poisson shot noise: variance = signal / gain.  Higher gain =
        better detector / longer exposure = lower relative noise.
        Typical range: 5 (noisy) to 100 (excellent detector).
    flicker_sigma : float
        Temporal intensity flicker: each frame's signal is multiplied by
        (1 + N(0, flicker_sigma)).  Models lamp arc wander, LED driver
        noise, or heartbeat-modulated vessel distension changing
        fluorescence intensity.  Typical value: 0.02–0.10.
    radius_ratio : float
        Ratio R(x=M-1) / R(x=0).  Values < 1 mean the tube narrows
        (particles speed up) toward x=M; values > 1 mean it widens.
        Default 1.0 = uniform radius, no spatial variation.  By
        continuity, velocity scales as (R_ref / R(x))².
    taper_shape : str
        'linear' — R(x) interpolates linearly from R(0) to R(M-1).
        'smooth' — R(x) follows a smooth (cosine) taper.
    contrast_mode : str
        'bright' (default) — fluorescent particles on dark background.
        'dark' — absorbing particles on bright background (brightfield).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    kymo : (T, M) float64 array
    ground_truth : dict with keys 'v_t', 'streaks', 'params'
    """
    if contrast_mode not in ('bright', 'dark'):
        raise ValueError(f"contrast_mode must be 'bright' or 'dark', got {contrast_mode!r}")

    rng = np.random.default_rng(seed)

    # ---- Velocity time series ----
    if v_t_override is not None:
        v_t = np.asarray(v_t_override, dtype=np.float64)
        if len(v_t) != T:
            raise ValueError(f"v_t_override length {len(v_t)} != T={T}")
        v_entry = float(v_mean)
    else:
        t_arr = np.arange(T, dtype=np.float64)
        phase_t = 2.0 * np.pi * f0_hz * t_arr * frame_dt_s
        v_t = (v_mean
               + v_amplitude * np.sin(phase_t + phi_1)
               + v_harmonic  * np.sin(2.0 * phase_t + phi_2))
        v_entry = float(v_mean)

    # ---- Spatial velocity scaling from radius taper ----
    # R(x) varies from 1.0 at x=0 to radius_ratio at x=M-1.
    # By continuity: v(x) ∝ 1/R(x)², so v_scale(x) = (1/R(x))².
    # Normalized so median(v_scale_x) = 1 — the taper redistributes
    # velocity spatially but preserves the median level that GST recovers.
    if radius_ratio != 1.0 and M > 1:
        frac = np.linspace(0.0, 1.0, M)
        if taper_shape == 'smooth':
            frac = 0.5 * (1.0 - np.cos(np.pi * frac))  # cosine ease
        R_x = 1.0 + (radius_ratio - 1.0) * frac  # R(0)=1, R(M-1)=radius_ratio
        v_scale_x = 1.0 / R_x**2  # continuity: v ∝ 1/A ∝ 1/R²
        v_scale_x /= np.median(v_scale_x)  # preserve median velocity
        _has_taper = True
    else:
        v_scale_x = np.ones(M, dtype=np.float64)
        _has_taper = False

    # ---- Initialize kymograph ----
    kymo = np.zeros((T, M), dtype=np.float64)
    x_arr = np.arange(M, dtype=np.float64)

    # ---- Static horizontal bands (wide Gaussian, vessel wall echoes) ----
    if static_bands > 0:
        band_x = rng.uniform(0.1 * M, 0.9 * M, size=static_bands)
        band_w = rng.uniform(1.0, 3.0, size=static_bands)
        for bx, bw in zip(band_x, band_w):
            kymo += static_band_strength * np.exp(-0.5 * ((x_arr - bx) / bw) ** 2)[None, :]

    # ---- Stationary bright spots (stuck beads / debris) ----
    if n_bright_spots > 0:
        spot_x = rng.uniform(0.05 * M, 0.95 * M, size=n_bright_spots)
        spot_I = rng.lognormal(
            np.log(bright_spot_intensity) - 0.5 * 0.3**2, 0.3, size=n_bright_spots
        )
        for sx, si in zip(spot_x, spot_I):
            kymo += si * np.exp(-0.5 * ((x_arr - sx) / bright_spot_width) ** 2)[None, :]

    # ---- Spawn streaks ----
    v_abs = max(abs(v_entry), 0.5)
    crossing_frames = int(np.ceil(M / v_abs)) + 50
    n_streaks = rng.poisson(particle_density * (T + crossing_frames))
    t_starts = rng.uniform(-crossing_frames, T, size=n_streaks)

    v_offsets = rng.normal(0.0, v_spread, size=n_streaks) if v_spread > 0 else np.zeros(n_streaks)

    if intensity_spread > 0:
        intensities = rng.lognormal(
            np.log(intensity_mean) - 0.5 * intensity_spread**2,
            intensity_spread, size=n_streaks,
        )
    else:
        intensities = np.full(n_streaks, intensity_mean)

    # Out-of-focus flag per particle
    is_oof = rng.random(n_streaks) < oof_fraction
    widths = np.where(is_oof, streak_width * oof_width_factor, streak_width)
    intensities = intensities * np.where(is_oof, oof_intensity_factor, 1.0)

    entry_jitter = rng.uniform(-1.0, 1.0, size=n_streaks)
    x_starts = (-2.0 * streak_width + entry_jitter if v_entry >= 0
                else M + 2.0 * streak_width + entry_jitter)

    streak_sign = -1.0 if contrast_mode == 'dark' else 1.0
    streaks = []

    # ---- Rasterize each streak ----
    for i in range(n_streaks):
        t0_floor = int(np.floor(t_starts[i]))
        t_lo = max(0, t0_floor)
        dv = v_offsets[i]
        sw = widths[i]
        I_peak = intensities[i]

        x_center = x_starts[i]
        for t_pre in range(t0_floor, t_lo):
            v_base = (v_t[t_pre] if 0 <= t_pre < T else v_entry) + dv
            if _has_taper:
                xi = int(np.clip(np.round(x_center), 0, M - 1))
                v_base *= v_scale_x[xi]
            x_center += v_base

        if t_lo >= T:
            continue

        n_frames = T - t_lo
        if _has_taper:
            # Integrate step-by-step: velocity depends on current position
            x_traj = np.empty(n_frames)
            x_traj[0] = x_center
            for k in range(1, n_frames):
                xi = int(np.clip(np.round(x_traj[k - 1]), 0, M - 1))
                x_traj[k] = x_traj[k - 1] + (v_t[t_lo + k] + dv) * v_scale_x[xi]
        else:
            v_steps = np.zeros(n_frames)
            if n_frames > 1:
                v_steps[1:] = v_t[t_lo + 1:T] + dv
            x_traj = x_center + np.cumsum(v_steps)
        margin = 3.0 * sw
        visible = (x_traj >= -margin) & (x_traj <= M - 1 + margin)
        if not np.any(visible):
            continue

        vis = np.where(visible)[0]
        i0, i1 = int(vis[0]), int(vis[-1])
        t_vis = np.arange(t_lo + i0, t_lo + i1 + 1)
        x_vis = x_traj[i0: i1 + 1]

        dx = x_arr[None, :] - x_vis[:, None]
        kymo[t_vis, :] += streak_sign * I_peak * np.exp(-0.5 * (dx / sw) ** 2)
        streaks.append({'t_start': t_starts[i], 'x_start': x_starts[i],
                        'v_offset': dv, 'intensity': I_peak, 'oof': bool(is_oof[i])})

    # ---- Spatially varying background ----
    if bg_variation > 0:
        n_waves = rng.integers(1, 4)
        for _ in range(n_waves):
            freq = rng.uniform(0.5, 2.5) / M
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 1.0) * bg_variation
            kymo += amp * np.sin(2 * np.pi * freq * x_arr + phase)[None, :]

    # ---- Temporal flicker (multiplicative) ----
    if flicker_sigma > 0:
        flicker = 1.0 + rng.normal(0.0, flicker_sigma, size=T)
        kymo *= flicker[:, None]

    # ---- DC background + clip ----
    kymo += background
    kymo = np.clip(kymo, 0.0, None)

    # ---- Noise ----
    if poisson_noise:
        kymo = rng.poisson(np.maximum(kymo * gain, 0.0)).astype(np.float64) / gain
        if noise_sigma > 0:
            kymo += rng.normal(0.0, noise_sigma, size=kymo.shape)
    else:
        if noise_sigma > 0:
            kymo += rng.normal(0.0, noise_sigma, size=kymo.shape)

    # ---- 2D ground truth velocity field: v(t, x) = v_t(t) * v_scale(x) ----
    v_t_x = v_t[:, None] * v_scale_x[None, :]  # (T, M)

    ground_truth = {
        'v_t': v_t,
        'v_t_x': v_t_x,
        'v_scale_x': v_scale_x,
        'streaks': streaks,
        'params': {
            'T': T, 'M': M, 'v_mean': v_mean, 'v_amplitude': v_amplitude,
            'v_harmonic': v_harmonic, 'phi_1': phi_1, 'phi_2': phi_2,
            'f0_hz': f0_hz, 'frame_dt_s': frame_dt_s,
            'particle_density': particle_density, 'streak_width': streak_width,
            'intensity_mean': intensity_mean, 'intensity_spread': intensity_spread,
            'v_spread': v_spread, 'oof_fraction': oof_fraction,
            'n_bright_spots': n_bright_spots, 'static_bands': static_bands,
            'background': background, 'bg_variation': bg_variation,
            'noise_sigma': noise_sigma, 'poisson_noise': poisson_noise,
            'gain': gain, 'flicker_sigma': flicker_sigma,
            'radius_ratio': radius_ratio, 'taper_shape': taper_shape,
            'contrast_mode': contrast_mode, 'seed': seed,
        },
    }
    return kymo, ground_truth


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-band synthetic vessel with 3D particle physics
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_vessel(
    # ---- Geometry ----
    T: int = 800,
    M: int = 35,
    vessel_radius_px: float = 12.0,
    band_offsets: Optional[np.ndarray] = None,
    band_radius_px: float = 1.0,
    # ---- Velocity waveform (centerline v_max) ----
    v_mean: float = 2.0,
    v_amplitude: float = 1.5,
    v_harmonic: float = 0.5,
    phi_1: float = 0.0,
    phi_2: float = 0.0,
    f0_hz: float = 2.5,
    frame_dt_s: float = 0.004,
    # ---- Optics (hardware constraints) ----
    NA: float = 0.3,
    wavelength_um: float = 0.605,
    bead_diameter_um: float = 0.5,
    n_medium: float = 1.0,
    magnification: float = 10.0,
    px_size_um: float = 1.7,
    # ---- Particle / streak properties ----
    particle_density: float = 1.0,
    streak_width: Optional[float] = None,
    intensity_mean: float = 1.0,
    intensity_spread: float = 0.3,
    # ---- Depth-of-field model ----
    dof_z_c_px: Optional[float] = None,
    z_0_um: float = 0.0,
    doc_model: str = 'squared_lorentzian',
    doc_p: float = 2.0,
    single_depth_z: Optional[float] = None,
    oof_width_factor: float = 4.0,
    oof_intensity_factor: float = 0.5,
    # ---- Static artifacts ----
    static_bands: int = 0,
    static_band_strength: float = 0.3,
    n_bright_spots: int = 0,
    bright_spot_intensity: float = 2.0,
    bright_spot_width: float = 0.5,
    # ---- Background & noise ----
    background: float = 0.3,
    bg_variation: float = 0.0,
    noise_sigma: float = 0.05,
    poisson_noise: bool = True,
    gain: float = 15.0,
    flicker_sigma: float = 0.05,
    # ---- Other ----
    seed: Optional[int] = None,
) -> Tuple[Dict[float, np.ndarray], Dict]:
    """Generate kymographs at multiple radial offsets with 3D particle physics.

    Particles are spawned uniformly in the vessel cross-section (r, z).
    Each particle's velocity follows Poiseuille flow:
        v(r, z, t) = v_max(t) * max(0, 1 - (r² + z²) / R²)

    A particle at signed radial position r_particle appears in band i
    (centered at offset r_i) if |r_particle - r_i| <= band_radius_px.
    Its PSF width and intensity degrade continuously with depth, using
    the Olsen & Adrian DOC weighting W(z) = 1/(1 + ((z-z₀)/z_c)²)².

    The model is fully constrained by hardware parameters (NA, wavelength,
    bead diameter, pixel size) — streak width and DOC half-depth are
    derived from the optics unless explicitly overridden.

    Parameters
    ----------
    T, M : int
        Number of frames and spatial columns per kymograph.
    vessel_radius_px : float
        Vessel radius R in pixels.
    band_offsets : (N,) array, optional
        Signed offsets from centerline in pixels.  Default: 0, ±2, ±4, ...
        out to ±(R - 1).
    band_radius_px : float
        Half-width of each sampling band (a particle at offset r contributes
        to a band centered at r_i if |r - r_i| <= band_radius_px).
    v_mean, v_amplitude, v_harmonic, phi_1, phi_2, f0_hz, frame_dt_s :
        Centerline (v_max) pulsatile waveform parameters — same meaning as
        in ``generate_synthetic_kymograph``.
    NA : float
        Numerical aperture of the objective (default 0.3 for 10x).
    wavelength_um : float
        Emission wavelength in µm (default 0.605, FluoSpheres F-8812).
    bead_diameter_um : float
        Tracer particle diameter in µm (default 0.5).
    n_medium : float
        Refractive index of immersion medium (1.0 for air objective).
    magnification : float
        Objective magnification (default 10).
    px_size_um : float
        Pixel size in µm (default 1.7 for 10x objective).
    particle_density : float
        Expected new particles entering the visible arc per frame.
    streak_width : float or None
        In-focus Gaussian σ of a streak (pixels).  If None (default),
        computed from the PSF: σ = 0.42λ/(NA·px_size).  For sub-diffraction
        beads (d_p < Airy disk), the streak width is PSF-limited.
    intensity_mean, intensity_spread : float
        Peak intensity distribution (lognormal).
    dof_z_c_px : float or None
        DOC half-depth z_c in pixels.  If None (default), computed from
        the Olsen & Adrian (2000) formula using the optics parameters.
        Set explicitly to override the hardware-derived value.
    z_0_um : float
        Focal plane offset from vessel midplane in µm.  z₀ = 0 means
        the focal plane is at the vessel center.  Positive z₀ shifts the
        DOC weighting upward: W(z) → W(z − z₀/px_size).  This is a
        source of systematic uncertainty in real experiments — the
        experimentalist may have focused on the tissue surface rather
        than the vessel midplane.
    oof_width_factor : float
        Maximum PSF width multiplier for fully out-of-focus particles.
    oof_intensity_factor : float
        Minimum intensity multiplier for fully out-of-focus particles.
    static_bands, static_band_strength, n_bright_spots, bright_spot_intensity,
    bright_spot_width, background, bg_variation, noise_sigma, poisson_noise,
    gain, flicker_sigma :
        Noise model — same as ``generate_synthetic_kymograph``.
    seed : int, optional
        Random seed.

    Returns
    -------
    kymos : dict  {offset: (T, M) array}
        Kymograph per radial offset.
    ground_truth : dict
        'v_max_t': (T,) centerline velocity waveform (px/frame)
        'v_mean_band': {offset: (T,) array} — band-averaged velocity per offset
        'Q_t': (T,) ground truth flow rate = π R² v_max(t) / 2
        'offsets': sorted list of offsets
        'vessel_radius_px': R
        'params': dict of all input parameters
    """
    rng = np.random.default_rng(seed)
    R = vessel_radius_px

    # ---- Hardware-derived defaults ----
    if streak_width is None:
        streak_width = compute_psf_sigma_px(NA, wavelength_um, px_size_um)
    if dof_z_c_px is None:
        z_c_um = compute_doc_z_c(NA, wavelength_um, bead_diameter_um,
                                 n_medium, magnification)
        dof_z_c_px = z_c_um / px_size_um
    z_0_px = z_0_um / px_size_um  # focal plane offset in pixels

    # ---- Default band offsets: 0, ±2, ±4, ... ----
    if band_offsets is None:
        step = 2.0
        pos = np.arange(0, R - 0.5, step)
        band_offsets = np.sort(np.concatenate([-pos[1:][::-1], pos]))
    band_offsets = np.asarray(band_offsets, dtype=np.float64)

    # ---- Centerline velocity waveform v_max(t) ----
    t_arr = np.arange(T, dtype=np.float64)
    phase_t = 2.0 * np.pi * f0_hz * t_arr * frame_dt_s
    v_max_t = (v_mean
               + v_amplitude * np.sin(phase_t + phi_1)
               + v_harmonic  * np.sin(2.0 * phase_t + phi_2))
    v_entry = float(v_mean)

    # ---- Analytical ground truth ----
    # Band-averaged Poiseuille velocity at offset r_i, weighted by DOC:
    #   <v>(r_i, t) = v_max(t) * ∫ W(z-z₀) * (1 - (r²+z²)/R²) dz / ∫ W(z-z₀) dz
    # Integration is over z ∈ [-h, h] where h = sqrt(R²-r²)
    n_quad = 64
    z_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)
    z_c = max(dof_z_c_px, 0.1)

    v_mean_band = {}
    for r_i in band_offsets:
        if abs(r_i) >= R:
            v_mean_band[float(r_i)] = np.zeros(T)
        elif single_depth_z is not None:
            # Single-depth mode: no integration, just Poiseuille at (r_i, z_fixed)
            v_factor = max(0.0, 1.0 - (r_i**2 + single_depth_z**2) / R**2)
            v_mean_band[float(r_i)] = v_max_t * v_factor
        else:
            h = np.sqrt(R**2 - r_i**2)
            z = h * z_nodes
            w = h * w_nodes
            # DOC weight with focal plane offset
            W = doc_weight(z, z_c, z_0_px, model=doc_model, p=doc_p)
            # Poiseuille velocity factor at (r_i, z)
            rho2 = r_i**2 + z**2
            v_factor = np.maximum(0.0, 1.0 - rho2 / R**2)
            # DOC-weighted average
            poiseuille_factor = np.sum(W * v_factor * w) / np.sum(W * w)
            v_mean_band[float(r_i)] = v_max_t * poiseuille_factor

    # Q(t) = π R² v_max(t) / 2  (exact Poiseuille integral)
    Q_t = np.pi * R**2 * v_max_t / 2.0

    # ---- Initialize kymographs ----
    x_arr = np.arange(M, dtype=np.float64)
    kymos_signal = {float(r_i): np.zeros((T, M), dtype=np.float64) for r_i in band_offsets}

    # ---- Spawn 3D particles ----
    # Particles are uniformly distributed in the circular cross-section.
    # We spawn in (r_signed, z) where r_signed is the radial offset from
    # centerline (can be negative) and z is depth.
    v_abs = max(abs(v_entry), 0.5)
    crossing_frames = int(np.ceil(M / v_abs)) + 50
    n_particles = rng.poisson(particle_density * (T + crossing_frames))
    t_starts = rng.uniform(-crossing_frames, T, size=n_particles)

    # Sample (r, z) positions for each particle
    n_need = n_particles
    if single_depth_z is not None:
        # Single-depth mode: all particles at fixed z, r uniform in [-r_max, r_max]
        r_max = np.sqrt(max(R**2 - single_depth_z**2, 0.0))
        r_particles = rng.uniform(-r_max, r_max, size=n_need)
        z_particles = np.full(n_need, single_depth_z)
    else:
        # Full cross-section: uniform in circle via rejection sampling
        r_particles = np.empty(n_need)
        z_particles = np.empty(n_need)
        filled = 0
        while filled < n_need:
            batch = max(n_need - filled, 100)
            r_cand = rng.uniform(-R, R, size=batch)
            z_cand = rng.uniform(-R, R, size=batch)
            inside = (r_cand**2 + z_cand**2) < R**2
            n_accept = inside.sum()
            n_take = min(n_accept, n_need - filled)
            r_particles[filled:filled + n_take] = r_cand[inside][:n_take]
            z_particles[filled:filled + n_take] = z_cand[inside][:n_take]
            filled += n_take

    # Particle intensity (lognormal variation)
    if intensity_spread > 0:
        intensities = rng.lognormal(
            np.log(intensity_mean) - 0.5 * intensity_spread**2,
            intensity_spread, size=n_particles,
        )
    else:
        intensities = np.full(n_particles, intensity_mean)

    entry_jitter = rng.uniform(-1.0, 1.0, size=n_particles)
    x_starts = -2.0 * streak_width + entry_jitter  # assume positive v_mean

    # ---- Optics-based defocus (NOT DOC weighting) ----
    # Defocus physics: u = (z - z₀)/z_c, d_e = sqrt(1 + u²)
    #   Width grows:     σ(z) = σ₀ · sqrt(1 + u²)
    #   Intensity falls: A(z) = A₀ / (1 + u²)  (energy conservation)
    # The DOC model W(z) only affects ground truth velocity averaging, not rendering.
    u2 = ((z_particles - z_0_px) / z_c) ** 2
    width_scale = np.clip(np.sqrt(1.0 + u2), 1.0, oof_width_factor)
    widths = streak_width * width_scale
    intensity_scale = np.clip(1.0 / (1.0 + u2), oof_intensity_factor, 1.0)
    intensities = intensities * intensity_scale

    # ---- Poiseuille velocity per particle ----
    # v_particle(t) = v_max(t) * max(0, 1 - (r² + z²) / R²)
    rho2 = r_particles**2 + z_particles**2
    poiseuille_scale = np.maximum(0.0, 1.0 - rho2 / R**2)  # (n_particles,)

    # ---- Determine which band(s) each particle appears in ----
    # particle at r_particles[i] is visible in band j if
    # |r_particles[i] - band_offsets[j]| <= band_radius_px
    # Precompute per-particle band membership
    particle_bands = []  # list of lists
    for i in range(n_particles):
        bands = []
        for j, r_j in enumerate(band_offsets):
            if abs(r_particles[i] - r_j) <= band_radius_px:
                bands.append(j)
        particle_bands.append(bands)

    # ---- Rasterize each particle as a streak into its band kymographs ----
    streaks_info = []
    for i in range(n_particles):
        if not particle_bands[i]:
            continue  # outside all bands

        t0_floor = int(np.floor(t_starts[i]))
        t_lo = max(0, t0_floor)
        sw = widths[i]
        I_peak = intensities[i]
        pscale = poiseuille_scale[i]

        # Integrate trajectory: v(t) = v_max(t) * pscale
        x_center = x_starts[i]
        for t_pre in range(t0_floor, t_lo):
            v_t_pre = (v_max_t[t_pre] if 0 <= t_pre < T else v_entry)
            x_center += v_t_pre * pscale

        if t_lo >= T:
            continue

        n_frames = T - t_lo
        v_steps = np.zeros(n_frames)
        if n_frames > 1:
            v_steps[1:] = v_max_t[t_lo + 1:T] * pscale
        x_traj = x_center + np.cumsum(v_steps)

        margin = 3.0 * sw
        visible = (x_traj >= -margin) & (x_traj <= M - 1 + margin)
        if not np.any(visible):
            continue

        vis = np.where(visible)[0]
        i0, i1 = int(vis[0]), int(vis[-1])
        t_vis = np.arange(t_lo + i0, t_lo + i1 + 1)
        x_vis = x_traj[i0:i1 + 1]

        # Gaussian streak profile (same for all bands this particle appears in)
        dx = x_arr[None, :] - x_vis[:, None]
        streak_img = I_peak * np.exp(-0.5 * (dx / sw) ** 2)

        for j in particle_bands[i]:
            r_j = band_offsets[j]
            kymos_signal[float(r_j)][t_vis, :] += streak_img

        streaks_info.append({
            't_start': t_starts[i], 'r': float(r_particles[i]),
            'z': float(z_particles[i]), 'poiseuille_scale': float(pscale),
            'bands': [float(band_offsets[j]) for j in particle_bands[i]],
            'intensity': float(I_peak), 'width': float(sw),
        })

    # ---- Apply noise model to each band kymograph ----
    # Shared flicker (same lamp/LED fluctuation hits all bands)
    if flicker_sigma > 0:
        flicker = 1.0 + rng.normal(0.0, flicker_sigma, size=T)
    else:
        flicker = None

    kymos = {}
    for r_i in band_offsets:
        r_key = float(r_i)
        kymo = kymos_signal[r_key].copy()

        # Static bands & bright spots (per-band, seeded differently)
        if static_bands > 0:
            band_x = rng.uniform(0.1 * M, 0.9 * M, size=static_bands)
            band_w = rng.uniform(1.0, 3.0, size=static_bands)
            for bx, bw in zip(band_x, band_w):
                kymo += static_band_strength * np.exp(-0.5 * ((x_arr - bx) / bw) ** 2)[None, :]

        if n_bright_spots > 0:
            spot_x = rng.uniform(0.05 * M, 0.95 * M, size=n_bright_spots)
            spot_I = rng.lognormal(
                np.log(bright_spot_intensity) - 0.5 * 0.3**2, 0.3, size=n_bright_spots
            )
            for sx, si in zip(spot_x, spot_I):
                kymo += si * np.exp(-0.5 * ((x_arr - sx) / bright_spot_width) ** 2)[None, :]

        # Spatially varying background
        if bg_variation > 0:
            n_waves = rng.integers(1, 4)
            for _ in range(n_waves):
                freq = rng.uniform(0.5, 2.5) / M
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.5, 1.0) * bg_variation
                kymo += amp * np.sin(2 * np.pi * freq * x_arr + phase)[None, :]

        # Flicker (shared)
        if flicker is not None:
            kymo *= flicker[:, None]

        # DC background
        kymo += background
        kymo = np.clip(kymo, 0.0, None)

        # Noise
        if poisson_noise:
            kymo = rng.poisson(np.maximum(kymo * gain, 0.0)).astype(np.float64) / gain
            if noise_sigma > 0:
                kymo += rng.normal(0.0, noise_sigma, size=kymo.shape)
        else:
            if noise_sigma > 0:
                kymo += rng.normal(0.0, noise_sigma, size=kymo.shape)

        kymos[r_key] = kymo

    ground_truth = {
        'v_max_t': v_max_t,
        'v_mean_band': v_mean_band,
        'Q_t': Q_t,
        'offsets': sorted(kymos.keys()),
        'vessel_radius_px': R,
        'streaks': streaks_info,
        'params': {
            'T': T, 'M': M, 'vessel_radius_px': R,
            'band_offsets': band_offsets.tolist(),
            'band_radius_px': band_radius_px,
            'v_mean': v_mean, 'v_amplitude': v_amplitude,
            'v_harmonic': v_harmonic, 'phi_1': phi_1, 'phi_2': phi_2,
            'f0_hz': f0_hz, 'frame_dt_s': frame_dt_s,
            'NA': NA, 'wavelength_um': wavelength_um,
            'bead_diameter_um': bead_diameter_um, 'n_medium': n_medium,
            'magnification': magnification, 'px_size_um': px_size_um,
            'particle_density': particle_density, 'streak_width': streak_width,
            'intensity_mean': intensity_mean, 'intensity_spread': intensity_spread,
            'dof_z_c_px': dof_z_c_px, 'z_0_um': z_0_um,
            'doc_model': doc_model, 'doc_p': doc_p,
            'single_depth_z': single_depth_z,
            'oof_width_factor': oof_width_factor,
            'oof_intensity_factor': oof_intensity_factor,
            'background': background, 'noise_sigma': noise_sigma,
            'poisson_noise': poisson_noise, 'gain': gain,
            'flicker_sigma': flicker_sigma, 'seed': seed,
        },
    }
    return kymos, ground_truth


# =============================================================================
# 2D frame renderer — produces (T, H, W) video from particle physics
# =============================================================================

def render_2d_frames(ground_truth: Dict) -> np.ndarray:
    """Render synthetic 2D video frames from particle data.

    Each frame is (H, W) where H = 2*R (radial across vessel) and W = M
    (arc along vessel).  Particles appear as 2D Gaussian blobs at their
    (r, x(t)) positions with defocus-dependent width and intensity.

    Parameters
    ----------
    ground_truth : dict
        Output from ``generate_synthetic_vessel()``.  Must contain
        'streaks', 'v_max_t', 'vessel_radius_px', and 'params'.

    Returns
    -------
    frames : (T, H, W) float64
        Video stack with the same noise model as the kymographs.
    """
    params = ground_truth['params']
    R = ground_truth['vessel_radius_px']
    v_max_t = ground_truth['v_max_t']
    streaks = ground_truth['streaks']
    T = params['T']
    M = params['M']
    v_mean_param = params['v_mean']
    streak_width_base = params['streak_width']
    background = params['background']
    noise_sigma = params['noise_sigma']
    poisson_noise = params['poisson_noise']
    gain = params['gain']
    flicker_sigma = params['flicker_sigma']
    seed = params.get('seed', None)

    rng = np.random.default_rng(seed if seed is not None else 42)

    # Frame dimensions: rows = radial (-R to R), cols = arc (0 to M-1)
    H = int(2 * np.ceil(R))
    W = M
    r_center = H / 2.0  # row index corresponding to r=0

    r_row = np.arange(H, dtype=np.float64) - r_center  # radial position per row
    x_col = np.arange(W, dtype=np.float64)              # arc position per column

    # Signal accumulator
    frames = np.zeros((T, H, W), dtype=np.float64)

    v_entry = float(v_mean_param)

    for s in streaks:
        t_start = s['t_start']
        r_p = s['r']          # radial position (px, signed)
        pscale = s['poiseuille_scale']
        I_peak = s['intensity']
        sw = s['width']       # already defocus-scaled

        t0_floor = int(np.floor(t_start))
        t_lo = max(0, t0_floor)
        if t_lo >= T:
            continue

        # Reconstruct trajectory (same logic as kymograph renderer)
        x_center = -2.0 * streak_width_base  # starting x
        for t_pre in range(t0_floor, t_lo):
            v_t_pre = v_max_t[t_pre] if 0 <= t_pre < T else v_entry
            x_center += v_t_pre * pscale

        n_frames = T - t_lo
        v_steps = np.zeros(n_frames)
        if n_frames > 1:
            v_steps[1:] = v_max_t[t_lo + 1:T] * pscale
        x_traj = x_center + np.cumsum(v_steps)

        margin = 3.0 * sw
        visible = (x_traj >= -margin) & (x_traj <= W - 1 + margin)
        if not np.any(visible):
            continue

        vis = np.where(visible)[0]
        i0, i1 = int(vis[0]), int(vis[-1])

        # Row profile: Gaussian in radial direction (constant over time)
        dr = r_row - r_p
        row_profile = np.exp(-0.5 * (dr / sw) ** 2)  # (H,)

        # Rasterize each visible frame
        for k in range(i0, i1 + 1):
            t_idx = t_lo + k
            x_pos = x_traj[k]
            dx = x_col - x_pos
            col_profile = np.exp(-0.5 * (dx / sw) ** 2)  # (W,)
            frames[t_idx] += I_peak * row_profile[:, None] * col_profile[None, :]

    # ---- Vessel mask: zero outside the circle ----
    r_grid = r_row[:, None]  # (H, 1)
    outside = r_grid ** 2 > R ** 2
    frames[:, outside.squeeze()] = 0.0

    # ---- Noise model (same as kymograph) ----
    # Flicker
    if flicker_sigma > 0:
        flicker = 1.0 + rng.normal(0.0, flicker_sigma, size=T)
        frames *= flicker[:, None, None]

    # Background
    frames += background
    np.clip(frames, 0.0, None, out=frames)

    # Poisson + readout noise
    if poisson_noise:
        frames = rng.poisson(np.maximum(frames * gain, 0.0)).astype(
            np.float64) / gain
        if noise_sigma > 0:
            frames += rng.normal(0.0, noise_sigma, size=frames.shape)
    else:
        if noise_sigma > 0:
            frames += rng.normal(0.0, noise_sigma, size=frames.shape)

    return frames


# Centerline geometry + 2D video generator moved to synthetic_video.py


# =============================================================================
# GST depth response measurement
# =============================================================================

def measure_gst_depth_response(
    vessel_radius_px: float = 12.0,
    n_depths: int = 31,
    *,
    v_mean: float = 2.0,
    f0_hz: float = 2.5,
    frame_dt_s: float = 0.004,
    T: int = 800,
    M: int = 35,
    NA: float = 0.3,
    wavelength_um: float = 0.605,
    bead_diameter_um: float = 0.5,
    n_medium: float = 1.0,
    magnification: float = 10.0,
    px_size_um: float = 1.7,
    particle_density: float = 2.0,
    noise_sigma: float = 0.05,
    gain: float = 15.0,
    doc_model: str = 'squared_lorentzian',
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Measure the effective GST depth response W_gst(z).

    Renders particles at a single depth z, runs GST velocity estimation,
    and compares measured velocity to the true Poiseuille velocity at that
    depth.  Sweeps z from -2z_c to +2z_c.

    The ratio  W_gst(z) = v_measured(z) / v_true(z)  gives the effective
    weight that the GST assigns to particles at depth z.  This can be
    compared to the theoretical DOC weight W_doc(z).

    Parameters
    ----------
    vessel_radius_px : float
        Vessel radius in pixels.
    n_depths : int
        Number of depth points to sample (default 31).
    v_mean, f0_hz, frame_dt_s, T, M : float/int
        Kymograph generation parameters.
    NA, wavelength_um, bead_diameter_um, n_medium, magnification, px_size_um
        Optics parameters (passed to generate_synthetic_vessel).
    particle_density : float
        Higher density for better statistics per depth.
    noise_sigma, gain : float
        Noise parameters.
    doc_model : str
        DOC model for theoretical comparison.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        z_um : (n_depths,) depth positions in µm
        z_norm : (n_depths,) z / z_c (normalized)
        v_true : (n_depths,) true Poiseuille velocity at (r=0, z)
        v_measured : (n_depths,) GST-measured velocity
        W_gst : (n_depths,) effective GST weight = v_measured / v_true
        W_doc : (n_depths,) theoretical DOC weight
        z_c_um : float, DOC half-depth
        vessel_radius_um : float
    """
    from .kymograph import compute_gst_velocity
    from .config import GST_WINDOWS

    z_c_um = compute_doc_z_c(
        NA=NA, wavelength_um=wavelength_um, bead_diameter_um=bead_diameter_um,
        n_medium=n_medium, magnification=magnification,
    )
    R_um = vessel_radius_px * px_size_um

    # Depth range: -2 z_c to +2 z_c (in µm), clipped to vessel radius
    z_max_um = min(2.0 * z_c_um, R_um * 0.95)
    z_positions_um = np.linspace(-z_max_um, z_max_um, n_depths)

    v_true = np.zeros(n_depths)
    v_measured = np.zeros(n_depths)

    if verbose:
        print(f"GST depth response: R={R_um:.1f} µm, z_c={z_c_um:.1f} µm, "
              f"{n_depths} depths in [{-z_max_um:.1f}, {z_max_um:.1f}] µm")

    for i, z_um in enumerate(z_positions_um):
        z_px = z_um / px_size_um

        # True Poiseuille velocity at (r=0, z)
        # v(r=0, z) = v_max × (1 - z²/R²)
        r_norm_sq = (z_um / R_um) ** 2
        if r_norm_sq >= 1.0:
            v_true[i] = 0.0
            v_measured[i] = 0.0
            continue
        v_true[i] = v_mean * (1.0 - r_norm_sq)

        # Generate single-depth kymograph (centerline band only, offset=0)
        kymos, gt = generate_synthetic_vessel(
            T=T, M=M,
            vessel_radius_px=vessel_radius_px,
            band_offsets=[0.0],  # Centerline only
            v_mean=v_mean, v_amplitude=0.0, v_harmonic=0.0,
            f0_hz=f0_hz, frame_dt_s=frame_dt_s,
            NA=NA, wavelength_um=wavelength_um,
            bead_diameter_um=bead_diameter_um,
            n_medium=n_medium, magnification=magnification,
            px_size_um=px_size_um,
            particle_density=particle_density,
            noise_sigma=noise_sigma, gain=gain,
            poisson_noise=True,
            single_depth_z=z_px,
            seed=seed + i,
        )

        # Run GST on the centerline kymograph
        kymo = kymos[0.0]
        try:
            gst_result = compute_gst_velocity(
                kymo, GST_WINDOWS,
                coherence_threshold=0.1,  # Low threshold — we want any signal
            )
            v_gst = gst_result.get('velocity', np.nan)
            if isinstance(v_gst, np.ndarray):
                v_gst = float(np.nanmedian(v_gst))
            v_measured[i] = abs(v_gst) if np.isfinite(v_gst) else 0.0
        except Exception:
            v_measured[i] = 0.0

        if verbose and (i % 5 == 0 or i == n_depths - 1):
            print(f"  z={z_um:+6.1f} µm ({z_um / z_c_um:+5.2f} z_c): "
                  f"v_true={v_true[i]:.3f}, v_meas={v_measured[i]:.3f}")

    # Compute effective weights
    with np.errstate(divide='ignore', invalid='ignore'):
        W_gst = np.where(v_true > 1e-6, v_measured / v_true, 0.0)

    # Theoretical DOC weight for comparison
    z_c_px = z_c_um / px_size_um
    W_doc = np.array([
        doc_weight(z / px_size_um, z_c_px, 0.0, doc_model)
        for z in z_positions_um
    ])
    # Normalize W_doc to peak=1
    if np.max(W_doc) > 0:
        W_doc /= np.max(W_doc)
    # Normalize W_gst to peak=1
    if np.max(W_gst) > 0:
        W_gst /= np.max(W_gst)

    if verbose:
        print(f"\nGST depth response summary:")
        print(f"  Peak W_gst location: z = {z_positions_um[np.argmax(W_gst)]:.1f} µm")
        print(f"  W_gst FWHM: ~{_estimate_fwhm(z_positions_um, W_gst):.1f} µm")
        print(f"  W_doc FWHM: ~{_estimate_fwhm(z_positions_um, W_doc):.1f} µm")

    return {
        'z_um': z_positions_um,
        'z_norm': z_positions_um / z_c_um,
        'v_true': v_true,
        'v_measured': v_measured,
        'W_gst': W_gst,
        'W_doc': W_doc,
        'z_c_um': z_c_um,
        'vessel_radius_um': R_um,
    }


def _estimate_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate FWHM of a peaked function by linear interpolation."""
    half_max = np.max(y) / 2.0
    above = y >= half_max
    if not np.any(above):
        return 0.0
    indices = np.where(above)[0]
    return float(x[indices[-1]] - x[indices[0]])
