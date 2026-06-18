# `default_mosaic_tile_profiles.py` Command Reference

Run from `Somites21_demo/PerTileFlow`:

```bash
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json
```

## Quick Command Matrix

View the mosaic in the read-only Napari app:

```bash
python -m pertile.viewer.mosaic_readonly_app \
  --config ../emb1/config.json
```

Run the simulation/recovery profile workflow:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json
```

Run ordinary least-squares simulation/recovery for comparison:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --ordinary-least-squares
```

Run measured-data frequentist inference:

```bash
python scripts/infer_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json
```

Run measured-data frequentist inference with ordinary least squares:

```bash
python scripts/infer_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --ordinary-least-squares
```

Run Section-9 Bayesian inference with H1 only:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json
```

Run Section-9 Bayesian inference with H1 + H2:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --use-second-harmonic
```

## What This Command Does

This command runs the first-pass whole-mosaic-to-tile profile-likelihood
experiment using settings that mirror the default simulation controls in:

```bash
python -m pertile.viewer.mosaic_readonly_app --config ../emb1/config.json
```

It:

1. Loads the analyzed mosaic graph from `../emb1/config.json`.
2. Runs one whole-mosaic transmission-line solve using the viewer-default
   boundary-condition setup.
3. Converts the simulated whole-mosaic edge flows into tile-local synthetic
   flow measurements.
4. Runs a tile-by-tile distensibility profile-likelihood scan.
5. Sums the tile profile chi2 curves to produce one global
   constant-distensibility profile.
6. Writes CSV outputs and an interactive HTML dashboard.

The HTML dashboard is self-contained: the profile data are embedded directly
inside the file. You can move or share `default_mosaic_tile_profiles.html`
by itself after it has been generated. The CSV files are still useful for
analysis, but the dashboard does not need them to render.

Default outputs:

```text
renders/meeting/default_mosaic_tile_profiles/
  mosaic_solve_result.pkl
  global_profile_constant_D.csv
  interior_profile_constant_D.csv
  periphery_profile_constant_D.csv
  tile_profiles.csv
  tile_profile_summary.csv
  default_mosaic_tile_profiles.html
```

The plain baseline command uses sigma-weighted least squares, matching the
original profile-likelihood behavior. If `--weighted-least-squares` is used,
the same objective is run but outputs are written with `_weighted` before the
file extension:

```text
renders/meeting/default_mosaic_tile_profiles/
  mosaic_solve_result_weighted.pkl
  global_profile_constant_D_weighted.csv
  interior_profile_constant_D_weighted.csv
  periphery_profile_constant_D_weighted.csv
  tile_profiles_weighted.csv
  tile_profile_summary_weighted.csv
  default_mosaic_tile_profiles_weighted.html
```

Open the dashboard with:

```bash
open renders/meeting/default_mosaic_tile_profiles/default_mosaic_tile_profiles.html
```

## Required Argument

### `--config ../emb1/config.json`

Path to the bundle config file.

The config tells the script where to find the analyzed graph:

```json
"mosaic_graph": "analyzed/mosaic_graph_analyzed.gpickle"
```

Relative paths inside the config are resolved relative to the config file's
directory, so `../emb1/config.json` resolves the graph to:

```text
../emb1/analyzed/mosaic_graph_analyzed.gpickle
```

## Common Optional Arguments

### `--tiles 22 26 38`

Run only selected tile IDs instead of every measured tile.

Useful for a fast smoke test:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --tiles 22 26 38
```

### `--all-tiles`

Run every tile with PIV measurements.

This is already the default when `--tiles` is omitted, so these are equivalent:

```bash
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json --all-tiles
```

### `--out-dir PATH`

Write outputs somewhere other than:

```text
renders/meeting/default_mosaic_tile_profiles
```

Example:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --out-dir renders/meeting/default_profiles_test
```

### `--reuse-mosaic-result PATH`

Skip the whole-mosaic solve and reuse an existing `mosaic_solve_result.pkl`.

This is useful when changing tile profile settings but keeping the same
whole-mosaic forward simulation:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --reuse-mosaic-result renders/meeting/default_mosaic_tile_profiles/mosaic_solve_result.pkl
```

## Whole-Mosaic Simulation Arguments

These control the synthetic whole-mosaic forward solve. For the first pass,
the defaults intentionally mirror the viewer's default simulation controls.

### `--D-mosaic 1e-3`

Distensibility used in the whole-mosaic forward simulation.

Default:

```text
1.0e-3 1/Pa
```

In the dashboard, this value is shown as the red vertical line. It is the
known input D used to generate synthetic mosaic flows, not a fitted tile value.

### `--target-flux 1.0`

Target total source/sink DC flux in nL/s for the viewer-default equal-split
boundary-condition setup.

Default:

```text
1.0 nL/s
```

### `--n-harmonics-mosaic 3`

Number of AC harmonics used in the whole-mosaic transmission-line solve.

Default:

```text
3
```

This produces DC + H1 + H2 + H3 in the whole-mosaic simulated flow field.

### `--f0-hz VALUE`

Fundamental heart frequency used by the solve.

Default comes from the existing analysis constants imported by the script.
Use this only if you intentionally want to override the default.

## Tile Profile-Likelihood Arguments

These control the per-tile D profile scans after synthetic flows have been
generated from the whole-mosaic solve.

### `--tile-harmonics 1 2`

AC harmonics included in each tile profile-likelihood scan.

Default:

```text
1 2
```

Meaning: use DC + H1 + H2 tile observations. H3 is left for later sensitivity
analysis.

### `--D-min 1e-6`

Minimum D value in the profile-likelihood scan.

Default:

```text
1e-6
```

### `--D-max 1e-1`

Maximum D value in the profile-likelihood scan.

Default:

```text
1e-1
```

### `--D-count 41`

Number of logarithmically spaced D values between `--D-min` and `--D-max`.

Default:

```text
41
```

Increase this for smoother curves. Decrease it for faster smoke tests.

## Noise Model Arguments

These set the measurement-noise model used by the default weighted objective
and by `--weighted-least-squares`.
The additive floors are in nL/s. Multiplicative terms are dimensionless.

By default, residuals are divided by their estimated uncertainty before
squaring:

```text
chi2(D) = sum_i ((observed_i - predicted_i) / sigma_i)^2
```

This is the likelihood-style objective where the `delta chi2 = 1` and
`delta chi2 = 3.84` reference levels have their usual interpretation.

### `--weighted-least-squares`

Use sigma-weighted least squares for the tile profile fits and write outputs
with `_weighted` in their filenames. This is the same objective as the
baseline command; the flag is useful when you want explicitly named weighted
outputs next to another run.

Example:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --weighted-least-squares
```

### `--ordinary-least-squares`

Use ordinary least squares with one constant sigma for every residual. The
constant sigma is the average of the weighted-noise sigma values for that
tile. This keeps the ordinary-LS relative weighting, while avoiding
meaningless near-zero chi2 values from unscaled SI-unit flow residuals.

This option does not use per-observation heteroscedastic weights. It only
rescales the ordinary-LS objective onto a comparable chi2 scale.

Example:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --ordinary-least-squares
```

### `--a-dc 0.061`

DC additive noise floor.

### `--a-h1 0.012`

H1 additive noise floor.

### `--a-h2 0.030`

H2 additive noise floor.

### `--b-dc 0.29`

DC multiplicative noise coefficient.

### `--b-h1 0.0`

H1 multiplicative noise coefficient.

### `--b-h2 0.0`

H2 multiplicative noise coefficient.

## Graph Path Override

### `--graph PATH`

Use a graph path directly instead of reading it from `--config`.

Example:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --graph ../emb1/analyzed/mosaic_graph_analyzed.gpickle
```

## Suggested Runs

Fast smoke test:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --tiles 22 26 38 \
  --D-count 15
```

Full first-pass run:

```bash
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json
```

Weighted least-squares run:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --weighted-least-squares
```

To verify the objective mode from fresh outputs, rerun the baseline and
explicit weighted modes:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json

python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --weighted-least-squares
```

Both should report/write `objective=weighted_least_squares`; the explicit
weighted run uses `_weighted` filenames. To run the constant-sigma ordinary
LS comparison, use:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --ordinary-least-squares
```

A fresh ordinary-LS run writes `objective=ordinary_least_squares` and
`weight_mode=constant_average_sigma` into the CSV rows. A weighted run writes
`objective=weighted_least_squares` and `weight_mode=sigma`, plus sigma
diagnostic columns in the tile summary CSV.

In the HTML payload, `defaults.objective` records the active objective.
`defaults.weighted_output_names` records whether the `_weighted` filename
flag was used.

Reuse the same mosaic solve but make smoother profiles:

```bash
python scripts/default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --reuse-mosaic-result renders/meeting/default_mosaic_tile_profiles/mosaic_solve_result.pkl \
  --D-count 81
```

## Measured-Data Inference Commands

`infer_default_mosaic_tile_profiles.py` does not run a whole-mosaic forward
simulation. It profiles D directly against measured tile flow observations.

Weighted least-squares inference, default:

```bash
python scripts/infer_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json
```

Ordinary least-squares inference:

```bash
python scripts/infer_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --ordinary-least-squares
```

Default outputs:

```text
renders/meeting/infer_default_mosaic_tile_profiles/
  measured_global_profile_constant_D.csv
  measured_interior_profile_constant_D.csv
  measured_periphery_profile_constant_D.csv
  measured_tile_profiles.csv
  measured_tile_profile_summary.csv
  infer_default_mosaic_tile_profiles.html
```

Ordinary LS outputs use `_ordinary` before the file extension.

## Bayesian Inference Commands

`infer_bayes_default_mosaic_tile_profiles.py` implements the Section-9
Bayesian tile inference. It uses measured AC harmonic flow observations and
analytically marginalizes tile-boundary forcing. By default it uses H1 only.

Bayesian H1-only run:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json
```

Bayesian H1 + H2 run:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --use-second-harmonic
```

H1-only outputs:

```text
renders/meeting/infer_bayes_default_mosaic_tile_profiles/
  bayes_global_posterior_constant_D.csv
  bayes_interior_posterior_constant_D.csv
  bayes_periphery_posterior_constant_D.csv
  bayes_tile_posteriors.csv
  bayes_tile_posterior_summary.csv
  infer_bayes_default_mosaic_tile_profiles.html
```

H1 + H2 outputs use `_h1h2` before the file extension:

```text
renders/meeting/infer_bayes_default_mosaic_tile_profiles/
  bayes_global_posterior_constant_D_h1h2.csv
  bayes_interior_posterior_constant_D_h1h2.csv
  bayes_periphery_posterior_constant_D_h1h2.csv
  bayes_tile_posteriors_h1h2.csv
  bayes_tile_posterior_summary_h1h2.csv
  infer_bayes_default_mosaic_tile_profiles_h1h2.html
```

Useful Bayesian sensitivity arguments:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --use-second-harmonic \
  --boundary-sigma-pa 7 \
  --h1-weight-source h1_z \
  --h2-weight-source h2_z
```

## Dashboard Notes

The red vertical line is the D value used to generate the whole-mosaic
synthetic flow field. It is an input to the forward simulation.

The best-D reference lines are determined by summing profile chi2 values at
each scanned D and taking the D where that summed chi2 is smallest:

```text
all-tiles best D   = argmin_D sum_all_tiles chi2_tile(D)
interior best D    = argmin_D sum_interior_tiles chi2_tile(D)
periphery best D   = argmin_D sum_periphery_tiles chi2_tile(D)
```

These are grid-search profile minima over the D values in the scan, not a
continuous optimizer between grid points.

Individual tile D values in `tile_profile_summary.csv` are fitted tile-by-tile
from each tile's own profile likelihood. They are not fitted globally.

The selected-tile overlay intentionally shows only the selected tile profiles.
It does not draw the mosaic simulation D or grouped best-D reference lines,
because those are global/group summaries rather than selected-tile fits.

Tiles are labeled as `periphery` or `interior` in the dashboard summary table.
The all-tile overlay includes a filter for:

```text
All tiles
Interior
Periphery
```

Periphery tiles are:

```text
1, 2, 3, 4, 8, 10, 11, 16, 17, 18, 24, 25,
31, 32, 37, 38, 43, 44, 48, 49, 50, 51, 52, 53
```

Every measured tile not listed above is labeled `interior`.
