"""Optional classical tile registration refinement."""

from __future__ import annotations


def phase_correlation_shift(fixed, moving) -> tuple[float, float] | None:
    """Return a rough y/x shift using skimage phase cross-correlation if available."""
    try:
        from skimage.registration import phase_cross_correlation
        shift, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=10)
        return float(shift[0]), float(shift[1])
    except Exception:
        return None


def refine_transform_pair(fixed, moving, method: str = "none"):
    if method == "none":
        return None
    if method == "phase":
        return phase_correlation_shift(fixed, moving)
    if method in {"ecc", "orb"}:
        try:
            __import__("cv2")
        except ImportError as exc:
            raise ImportError(f"OpenCV is required for {method} registration") from exc
        return None
    raise ValueError(f"Unknown registration method: {method}")
