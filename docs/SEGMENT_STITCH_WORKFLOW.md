# Segment Stitch Workflow

This package provides an in-house supervised Python/PyTorch workflow for tiled
Somites microscopy videos. It trains on `Somites27_demo`, tests on
`Somites21_demo`, and uses a whole-mosaic label/target image plus manual tile
placement metadata to generate tile-level target masks.

## Expected Data

Each dataset should look like:

```text
SomitesXX_demo/
  emb1/
    videos/*.tif
    analyzed/
      tile_positions_manual.json
      stitched_linear.tif
      mosaic_segmentation_labels.tif
    config.json
```

The mosaic label filename can also be passed explicitly with `--train-labels`
and `--test-labels`. If omitted, the code tries common label filenames in
`emb1/analyzed/`, then falls back to the configured full mosaic
`stitched_linear.tif`.

The workflow uses noncontiguous tile IDs parsed from raw video filenames and
intersects them with IDs present in `tile_positions_manual.json`.

## Modules

`segment_stitch.io` resolves dataset paths, reads TIFFs, JSON metadata, manual
tile positions, and usable tile IDs.

`segment_stitch.projections` creates mean, max, standard deviation, and median
temporal projections from each raw tile video. The default model input is a
3-channel `[mean, max, std]` tensor with per-channel 1st/99th percentile
normalization.

`segment_stitch.extract_masks` converts the whole-mosaic label/target image
into tile-local ground-truth masks. For each raw tile, it maps tile-local pixels
into the mosaic coordinate frame using the manual affine/scale/translation
metadata, samples the mosaic image, binarizes it, and saves a mask aligned to
the raw video tile.

`segment_stitch.models.unet` implements a lightweight GroupNorm U-Net:
encoder channels `32, 64, 128, 256`, bottleneck `512`, mirrored decoder, and a
single-channel logits output.

`segment_stitch.train_unet` trains with `BCEWithLogitsLoss + DiceLoss`, CUDA
when available, mixed precision on CUDA, and deterministic seed handling where
practical.

`segment_stitch.predict_masks` writes probability masks, thresholded masks, and
per-tile QC overlays.

`segment_stitch.stitch.baseline` renders projection and predicted-mask mosaics
using manual tile positions and average/max blending.

`segment_stitch.evaluate_segmentation` compares Somites21 predictions to the
extracted tile-level ground-truth masks and writes per-tile CSV plus summary
JSON.

`segment_stitch.evaluate_stitching` compares the stitched projection to
`stitched_linear.tif` and compares the reconstructed stitched mask mosaic to
the original whole-mosaic labels.

`segment_stitch.stitch.pairwise_dataset`, `pairwise_model`, and `pose_graph`
remain scaffolds for future learned stitching. Learned pairwise/GNN stitching
is not implemented in the primary workflow.

## Commands

Generate projections:

```bash
python -m segment_stitch.projections \
  --data-root /mnt/home/sswee/ceph/Somites27_demo \
  --output-dir segment_stitch/work
```

Extract supervised tile masks from the default full mosaic:

```bash
python -m segment_stitch.extract_masks \
  --data-root /mnt/home/sswee/ceph/Somites27_demo \
  --output-dir segment_stitch/work
```

Train:

```bash
python -m segment_stitch.train_unet \
  --train-root /mnt/home/sswee/ceph/Somites27_demo \
  --output-dir segment_stitch/work \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --num-workers 4 \
  --device cuda
```

Predict on Somites21:

```bash
python -m segment_stitch.predict_masks \
  --model segment_stitch/work/checkpoints/best_model.pt \
  --data-root /mnt/home/sswee/ceph/Somites21_demo \
  --output-dir segment_stitch/work \
  --device cuda
```

Stitch manually:

```bash
python -m segment_stitch.stitch.baseline \
  --data-root /mnt/home/sswee/ceph/Somites21_demo \
  --output-dir segment_stitch/work
```

Evaluate:

```bash
python -m segment_stitch.evaluate_segmentation \
  --data-root /mnt/home/sswee/ceph/Somites21_demo \
  --output-dir segment_stitch/work

python -m segment_stitch.evaluate_stitching \
  --data-root /mnt/home/sswee/ceph/Somites21_demo \
  --output-dir segment_stitch/work
```

Full workflow:

```bash
python -m segment_stitch.run_workflow \
  --train-root /mnt/home/sswee/ceph/Somites27_demo \
  --test-root /mnt/home/sswee/ceph/Somites21_demo \
  --output-dir segment_stitch/work \
  --device cuda \
  --epochs 50 \
  --batch-size 8
```

Or use the launcher:

```bash
segment_stitch/scripts/run_somites27_train_somites21_test.sh
```

For Slurm:

```bash
sbatch segment_stitch/scripts/run_somites27_train_somites21_test.sbatch
```

Reverse direction, using Somites21 for training and Somites27 for testing:

```bash
segment_stitch/scripts/run_somites21_train_somites27_test.sh
sbatch segment_stitch/scripts/run_somites21_train_somites27_test.sbatch
```

Generate or regenerate the final metrics/visual report from an existing run:

```bash
python segment_stitch/scripts/generate_qc_report.py \
  --output-dir segment_stitch/work \
  --report-dir segment_stitch/outputs \
  --dataset Somites21 \
  --reference-mosaic /mnt/home/sswee/ceph/Somites21_demo/emb1/analyzed/stitched_linear.tif \
  --overwrite-metrics
```

## Work Artifacts

```text
segment_stitch/work/
  projections/Somites27/
  projections/Somites21/
  extracted_masks/Somites27/
  extracted_masks/Somites21/
  checkpoints/best_model.pt
  checkpoints/last_model.pt
  predictions/Somites21/
  stitched/Somites21/
  metrics/
```

`segment_stitch/work/` is rerunnable working state: projections, extracted
tile masks, checkpoints, predictions, stitched TIFFs, and raw metric files.

## Final Outputs

```text
segment_stitch/outputs/
  Somites21/
    metrics/
      training_summary.json
      training_metrics.csv
      segmentation_metrics_summary.json
      segmentation_metrics_per_tile.csv
      stitching_metrics_summary.json
    visuals/
      training_loss.png
      segmentation_metrics_per_tile.png
      mask_coverage_by_tile.png
      segmentation_overlay_montage.png
      stitched_qc_panel.png
    QC_REPORT.md
```

`segment_stitch/outputs/` is the folder meant for inspection: metrics and
visuals comparing tile predictions and whole-mosaic stitching.

Metrics include tile-level Dice, IoU, precision, recall, F1, pixel accuracy,
optional skeleton metrics, stitched-image normalized cross-correlation,
optional SSIM, MAE, overlap coverage, stitched mosaic mask Dice/IoU/F1, and
manual transform error placeholders.

## Known Limitations

This is not publication-grade yet. There are only two embryos, many frames but
few independent mosaics, the whole-mosaic label quality controls the tile-level
ground truth quality, intensity and scale can vary across tiles, tile IDs are
missing/noncontiguous, manual JSON files can contain more tile entries than raw
videos, and learned stitching should not be trusted until validated against
manual transforms.

## Future Extensions

Future work should add pairwise Siamese tile compatibility, tile-level graph
neural networks for global placement, differentiable spatial-transformer
stitching, active learning for manual mask correction, and graph-aware vessel
continuity losses.
