# Somites21 — embryonic yolk-sac flow demo dataset

Self-contained viewer bundle for a stage-21 (HH-21) embryonic chick
yolk-sac vasculature, acquired at 250 fps under 10× magnification with
fluorescent microspheres.  Per-edge flow waveforms `Q(t)` recovered
from PIV / Farneback optical flow are stored in the analyzed graph.
Stitched per-tile videos are included for visual inspection.  The
PerTileFlow viewer code rides along inside this zip — no second
download required.

> **Companion bundle**: stage-27 embryos live at the same Releases page
> as `Somites27_demo.zip`.

## Contents

```
Somites21_demo/
├── README.md              ← this file
├── LAUNCH.txt             launch instructions + CLI flags
├── CODE_CONTEXT.md        codebase orientation for humans + AI agents
├── PerTileFlow/           bundled viewer code (~680 KB, pip-installable)
└── emb1/
    ├── config.json        relative-path config consumed by the viewer
    ├── analyzed/
    │   ├── mosaic_graph_analyzed.gpickle   ~210 MB
    │   ├── stitched_linear.tif             ~45 MB
    │   └── tile_positions_manual.json
    └── videos/                              ~1.1 GB, 26 of 53 tiles
```

Total unzipped size: ~1.4 GB.  See `LAUNCH.txt` for the exact tile list
and the downsampling caveats.

## Requirements

- **Python 3.9 or newer.**  Works on macOS, Linux, and Windows.
- Either conda/mamba or Python's built-in `venv`.
- No C compiler, no Cython, no system libraries to install separately.

## Setup A — conda (recommended)

Works identically on macOS, Linux, and Windows.

```bash
unzip Somites21_demo.zip
cd Somites21_demo

conda create -n yolk-sac python=3.11 -y
conda activate yolk-sac
pip install ./PerTileFlow

python -m pertile.viewer.mosaic_readonly_app --config emb1/config.json
```

## Setup B — pip + venv (no conda)

```bash
unzip Somites21_demo.zip
cd Somites21_demo

python -m venv .venv

# Activate (pick one for your OS / shell):
source .venv/bin/activate                # macOS / Linux
.\.venv\Scripts\Activate.ps1             # Windows PowerShell
.\.venv\Scripts\activate.bat             # Windows cmd.exe

python -m pip install --upgrade pip
python -m pip install ./PerTileFlow

python -m pertile.viewer.mosaic_readonly_app --config emb1/config.json
```

The single `pip install ./PerTileFlow` pulls in napari, PyQt5, numpy,
scipy, matplotlib, tifffile, networkx, scikit-image, opencv-python, and
qtpy automatically.  No separate `pip install napari[pyqt5]` needed, no
quoting gymnastics required.

`LAUNCH.txt` has the full CLI flag reference and troubleshooting notes.

## What the viewer shows

- napari window with the stitched mosaic and the vessel network as
  colored segments overlaid.
- Per-edge colormap driven by a 4-selector model
  (Source × Quantity × Property × Harmonic).  Properties include
  Magnitude, Phase, Resolution (per-harmonic Z), Total SNR, Pulsatility
  Index, Frequency, geometry, and the categorical harmonic class.
- Tile filter to restrict measured-Q fields to a single tile's
  measurements instead of best-of-edge across the network.
- Click-to-inspect for each edge: metadata, Q(t) plot, harmonic
  decomposition, per-harmonic SNR / Rayleigh-tier panel.
- BC simulation tab (transmission-line solve with measured or custom
  arterial / venous waveforms).
- Per-tile inference tab (boundary-pressure + distensibility, LM + FGLS).

## Limitations of this bundle

Demonstration bundle — partial tile coverage (top 50% by signal
quality) and 2× spatially downsampled videos, capped at 600 frames
each.  The optical-flow re-analysis button is hidden in this build
because the absolute-Q scaling becomes fragile under spatial
downsampling.  The analyzed graph's per-edge measurements remain at
full original quality; visual inspection, harmonic analysis, network
simulation, and per-tile inference all work unaffected.

## Citation

If this dataset contributes to a publication, please cite the
PerTileFlow methodology paper (in preparation).

## License

Data: CC-BY-4.0 unless noted otherwise.
PerTileFlow viewer code (in `PerTileFlow/`): see that subdirectory.
