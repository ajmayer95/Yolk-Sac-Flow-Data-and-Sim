"""Rendering convenience wrappers for stitched QC outputs."""

from __future__ import annotations

from pathlib import Path

from .baseline import stitch_dataset


def render_manual_outputs(data_root: str | Path, output_dir: str | Path) -> Path:
    return stitch_dataset(data_root, output_dir)
