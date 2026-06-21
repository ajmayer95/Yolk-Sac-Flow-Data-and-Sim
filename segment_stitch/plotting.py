"""Matplotlib QC plots for segmentation and stitching."""

from __future__ import annotations

from pathlib import Path


def save_segmentation_qc(path: str | Path, image, pseudo=None, prob=None, binary=None) -> None:
    np = __import__("numpy")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image[0] if getattr(image, "ndim", 0) == 3 else image
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    panels = [("projection", base), ("pseudo", pseudo), ("prob", prob), ("binary", binary)]
    for ax, (title, arr) in zip(axes[:4], panels):
        ax.set_title(title)
        ax.imshow(np.zeros_like(base) if arr is None else arr, cmap="gray")
        ax.axis("off")
    axes[4].set_title("overlay")
    axes[4].imshow(base, cmap="gray")
    if binary is not None:
        axes[4].imshow(np.ma.masked_where(binary <= 0, binary), cmap="autumn", alpha=0.45)
    axes[4].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_overlay(path: str | Path, image, mask, title: str = "overlay") -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.imshow(image, cmap="gray")
    ax.imshow(__import__("numpy").ma.masked_where(mask <= 0, mask), cmap="autumn", alpha=0.4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
