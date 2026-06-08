"""Per-edge analysis (slim build).

Read-only-viewer runtime closure only: calibration constants, the
graph helpers used by `get_chain_coords`, the harmonic fit, and the
transmission-line solver consumed by the BC simulation tab.
"""
from .config import (
    PX_SIZE_UM, FRAME_DT_S, FPS,
    FMIN_HZ, FMAX_HZ, N_HARMONICS,
    GST_WINDOWS, COHERENCE_THRESHOLD,
    VELOCITY_CMAP, VELOCITY_VMAX_UM_S, SNR_DB_MIN, SNR_DB_MAX,
)
from .harmonic import estimate_f0_in_band, fit_harmonics
from .flow import get_chain_coords, trace_vessel_chain
from .transmission_line import (
    TransmissionLineResult,
    solve_transmission_line,
    plot_transmission_line_result,
    compute_eta_from_harmonics,
    compute_eta_from_qt,
)
