"""Minimal install spec for the PerTileFlow read-only viewer.

The wider PerTileFlow project includes mask segmentation, vectorization,
batch analysis, MCMC inference, and an editing viewer — none of which
ship in this slim package.  Only the read-only viewer
(`pertile.viewer.mosaic_readonly_app`) and its runtime dependencies are
present.  Pure-Python; no C extensions, no Cython, no build tools.
"""
from setuptools import setup, find_packages

setup(
    name="pertile",
    version="0.1.0",
    description="PerTileFlow read-only viewer — pure-Python slim build.",
    packages=find_packages(),
    include_package_data=True,
    package_data={"pertile": ["../configs/default.json"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tifffile",
        "networkx",
        "scikit-image",
        "opencv-python",
        "qtpy",
        "napari[pyqt5]",
    ],
)
