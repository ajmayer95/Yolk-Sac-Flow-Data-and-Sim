"""Metrics for segmentation and stitched mosaic quality."""

from __future__ import annotations


def binary_metrics(pred, target, eps: float = 1e-8) -> dict[str, float]:
    np = __import__("numpy")
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "pixel_accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
    }


def ncc(a, b, eps: float = 1e-8) -> float:
    np = __import__("numpy")
    aa = np.asarray(a, dtype="float32")
    bb = np.asarray(b, dtype="float32")
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float((aa * bb).mean() / (aa.std() * bb.std() + eps))


def normalize01(x):
    np = __import__("numpy")
    arr = np.asarray(x, dtype="float32")
    lo, hi = np.percentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def image_metrics(pred, ref) -> dict[str, float | None]:
    np = __import__("numpy")
    p = normalize01(pred)
    r = normalize01(ref)
    h = min(p.shape[0], r.shape[0])
    w = min(p.shape[1], r.shape[1])
    p = p[:h, :w]
    r = r[:h, :w]
    out = {
        "normalized_cross_correlation": ncc(p, r),
        "mean_absolute_error": float(np.abs(p - r).mean()),
        "overlap_coverage_fraction": float(((p > 0) & (r > 0)).sum() / max(1, (r > 0).sum())),
    }
    try:
        from skimage.metrics import structural_similarity
        out["ssim"] = float(structural_similarity(p, r, data_range=1.0))
    except Exception:
        out["ssim"] = None
    return out
