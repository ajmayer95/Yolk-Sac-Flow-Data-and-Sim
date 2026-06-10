# Distensibility Analysis Methods and Current Results

Date: 2026-06-09

This note summarizes the methods attempted so far for the Somites21 and Somites27 yolk-sac mosaic datasets, with comments on what the current results suggest about distensibility. The emphasis is on tile-by-tile recovery/inference, profile likelihood/posterior shape, and phase-based diagnostics.

## Main Scripts

Somites21:

- `Somites21_demo/PerTileFlow/scripts/default_mosaic_tile_profiles.py`
- `Somites21_demo/PerTileFlow/scripts/infer_default_mosaic_tile_profiles.py`
- `Somites21_demo/PerTileFlow/scripts/infer_bayes_default_mosaic_tile_profiles.py`
- `Somites21_demo_light/PerTileFlow/scripts/infer_bayes_default_mosaic_tile_profiles.py`

Somites27:

- `Somites27_demo/PerTileFlow/scripts/default_mosaic_tile_profiles.py`
- `Somites27_demo/PerTileFlow/scripts/infer_default_mosaic_tile_profiles.py`
- `Somites27_demo/PerTileFlow/scripts/infer_bayes_default_mosaic_tile_profiles.py`

The newest phase-correlation dashboard features are currently in the Somites21 Bayesian script and the Somites21 light copy. They have not yet been ported into the Somites27 Bayesian script.

## Methods Attempted

### 1. Whole-Mosaic Simulation Followed by Tile-by-Tile Recovery

Script:

```bash
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json
```

Purpose:

- Solve the full mosaic network using a chosen distensibility value.
- Generate tile-level boundary conditions and synthetic tile flow measurements.
- Refit/recover distensibility on each tile.
- Ask whether a known global distensibility can be recovered tile-by-tile.

Variants:

```bash
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json --ordinary
python scripts/default_mosaic_tile_profiles.py --config ../emb1/config.json --weighted
```

Comments:

- This is a simulation/recovery workflow, not inference from measured data.
- It is useful as a sanity check for whether the tile graph, boundary construction, noise scaling, and profile-likelihood calculation behave as expected.
- Early WLS/OLS work exposed a noise-scale issue: ordinary least squares initially produced artificially tiny reduced chi-square values. OLS was adjusted to use a reasonable shared noise scale rather than effectively zero residual scale.

Current output folders:

- `Somites21_demo/PerTileFlow/renders/meeting/default_mosaic_tile_profiles/`
- `Somites27_demo/PerTileFlow/renders/meeting/default_mosaic_tile_profiles/`

Current simulation summary:

| Dataset | Tiles | OLS median D_hat | WLS median D_hat | Comment |
|---|---:|---:|---:|---|
| Somites21 | 53 | `1.0e-3` | `1.0e-3` | Recovery centers near the simulated value, with some tile variability. |
| Somites27 | 84 | `1.0e-3` | `1.0e-3` | Recovery also centers near the simulated value, but tile variability can be wider. |

Interpretation:

- The recovery workflow can recover the imposed global value in the median.
- Tile-to-tile variation still appears, which is expected because each tile has different local topology, constraints, harmonics, and boundary information.
- This supports using the framework, but it does not prove that measured-data inference is reliable.

### 2. Frequentist Measured-Data Inference

Script:

```bash
python scripts/infer_default_mosaic_tile_profiles.py --config ../emb1/config.json
```

Purpose:

- Use measured graph flow information rather than synthetic measurements generated from a chosen D.
- Build a profile likelihood over D for each tile.
- Compare OLS and WLS variants.

Current output folders:

- `Somites21_demo/PerTileFlow/renders/meeting/infer_default_mosaic_tile_profiles/`
- `Somites27_demo/PerTileFlow/renders/meeting/infer_default_mosaic_tile_profiles/`

Current measured-data summary:

| Dataset | Tiles | OLS median D_hat | OLS chi2_red median | WLS median D_hat | WLS chi2_red median |
|---|---:|---:|---:|---:|---:|
| Somites21 | 53 | `2.37e-4` | `0.74` | not used as main current result | not used as main current result |
| Somites27 | 84 | `1.78e-3` | `2.64` | `1.78e-3` | `215.8` |

Interpretation:

- Somites21 measured-data OLS has reduced chi-square values closer to the expected order of magnitude.
- Somites27 measured-data OLS has a median reduced chi-square above 1, suggesting either model mismatch, under-estimated noise, more difficult boundary conditions, or stronger biological/measurement heterogeneity.
- Somites27 WLS produces very large reduced chi-square values in the current output, so those WLS chi-square values should be treated cautiously until the weighting/noise assumptions are revisited.

### 3. Bayesian Tile Inference With Marginalized Boundary Forcing

Script:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py --config ../emb1/config.json
```

H1 + H2 variant:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --use-second-harmonic
```

Purpose:

- Infer D from measured graph data using a Bayesian posterior over D.
- Use a log-normal prior on D.
- Marginalize boundary forcing terms.
- Compare H1-only versus H1+H2 information.

Current output folders:

- `Somites21_demo/PerTileFlow/renders/meeting/infer_bayes_default_mosaic_tile_profiles/`
- `Somites27_demo/PerTileFlow/renders/meeting/infer_bayes_default_mosaic_tile_profiles/`

Important caution:

- In the Bayesian script, noise rescaling makes `chi2_h1_red_at_mode` close to 1 by construction. Therefore, reduced chi-square in these Bayesian summaries is not an independent goodness-of-fit diagnostic. Posterior width, posterior curvature, mode boundary behavior, and H1 versus H1+H2 shifts are more informative.

Current Bayesian summary:

| Dataset | Harmonics | Tiles | Median D_hat | D_hat range | Median width | Boundary modes |
|---|---|---:|---:|---:|---:|---:|
| Somites21 | H1 | 53 | `3.16e-4` | `1.26e-5` to `5.01e-2` | `1.10` decades, LR width | 0 |
| Somites21 | H1+H2 | 53 | `2.00e-4` | `1.00e-5` to `6.31e-2` | `0.73` decades, 95% credible width | 1 |
| Somites27 | H1 | 84 | `7.94e-4` | `5.01e-5` to `1.00e-1` | `1.00` decades, LR width | 5 |
| Somites27 | H1+H2 | 84 | `3.98e-4` | `3.16e-5` to `1.00e-1` | `0.80` decades, LR width | 1 |

Number of tiles with relatively narrow posteriors:

| Dataset | Harmonics | Width <= 0.5 decades | Width <= 1.0 decades | Width <= 1.5 decades |
|---|---|---:|---:|---:|
| Somites21 | H1 | 9 / 53 | 24 / 53 | 38 / 53 |
| Somites21 | H1+H2 | 14 / 53 | 42 / 53 | 50 / 53 |
| Somites27 | H1 | 25 / 84 | 41 / 84 | 58 / 84 |
| Somites27 | H1+H2 | 27 / 84 | 54 / 84 | 68 / 84 |

Interior/periphery summary:

| Dataset | Harmonics | Interior median D_hat | Periphery median D_hat | Interior median width | Periphery median width |
|---|---|---:|---:|---:|---:|
| Somites21 | H1 | `5.01e-4` | `2.25e-4` | `1.20` | `1.05` |
| Somites21 | H1+H2 | `3.16e-4` | `1.58e-4` | `0.77` | `0.68` |
| Somites27 | H1 | `7.94e-4` | `6.31e-4` | `1.10` | `1.00` |
| Somites27 | H1+H2 | `3.98e-4` | `3.98e-4` | `0.90` | `0.70` |

Interpretation:

- Adding H2 generally narrows posterior widths and shifts median D_hat downward in both datasets.
- The H2 effect suggests that second-harmonic information is contributing useful timing/amplitude constraints, but it may also expose model mismatch if H2 behavior is not well captured by the current model.
- Periphery tiles often have slightly narrower posterior widths than interior tiles in the current summaries. This may reflect stronger boundary influence, simpler local constraints, or better measured harmonic content near larger/clearer vessels.
- Some Somites27 H1 modes hit the grid boundary, especially in H1-only inference. H1+H2 reduces this issue.

### 4. Location-Based Tile Grouping

Argument:

```bash
--cluster
```

Purpose:

- Split tiles into coarse anatomical/proximity groups.
- Somites21 periphery/interior labels are hard-coded from the specified tile list.
- Interior subgroups: proximal versus distal, with tile 14 and overlapping tile-boundary neighbors defining proximal.
- Periphery subgroups: venous end versus arterial end, using the line between the two source A nodes as a reference.

Comments:

- The clustering did not reveal a clean, obvious pattern in likelihood profiles.
- This does not mean location is irrelevant; it means the current coarse grouping may not align with the dominant sources of variability.
- More local features may matter more: vessel diameter, harmonic SNR, edge-to-edge phase jumps, boundary proximity, and graph topology.

### 5. H1 Versus H1+H2 Sensitivity

Argument:

```bash
--use-second-harmonic
```

Purpose:

- Test whether including H2 changes inferred D or posterior identifiability.

Current comment:

- H1+H2 tends to narrow profiles/posteriors.
- H1+H2 also tends to lower median D_hat relative to H1-only in both datasets.
- This is potentially informative, but it should be interpreted with care. If H2 is more sensitive to nonlinearities, waveform shape, phase measurement error, or model mismatch, it may constrain D while also revealing limitations of the current model.

### 6. Tile Phase and Harmonic Visualization

Somites21 command:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --tile-visualization phase
```

Somites21 H1+H2 command:

```bash
python scripts/infer_bayes_default_mosaic_tile_profiles.py \
  --config ../emb1/config.json \
  --use-second-harmonic \
  --tile-visualization phase
```

Purpose:

- Display the original tile crop from the stitched mosaic.
- Overlay vessel harmonic phase and amplitude.
- Show phase-over-time animation.
- Show H1/H2 amplitude and phase distributions.
- Allow selected-edge comparison of H1/H2 amplitudes, phases, and SNR.

Comments:

- The phase map is static vessel timing: which vessels peak earlier or later in the fitted harmonic cycle.
- Phase over time animates the fitted harmonic signal through one cycle.
- Large phase variation in large-diameter vessels may indicate that diameter-dependent compliance, waveform propagation, or measurement dominance from large vessels should be considered.

### 7. Contour and Smooth Phase Gradient Experiments

Methods attempted:

- Direct phase contour maps at mode, mode +/- pi/6, and mode +/- pi/3.
- Robust phase-plane fitting.
- Phase-gradient arrows from the fitted phase plane.

Comments:

- Raw contour overlays were visually noisy and hard to interpret.
- Finer contours risk tracking measurement noise rather than meaningful propagation structure.
- The robust phase-plane view is cleaner, but it is still a visualization aid rather than direct evidence of constant distensibility.
- The main lesson was that phase often appears piecewise or edge-to-edge heterogeneous rather than smoothly varying within each vessel segment.

### 8. Within-Edge Phase Gradient

Method:

- Estimate node phase from incident edge phases.
- For each edge, compute wrapped node-to-node phase difference divided by vessel length.

Comments:

- The within-edge phase gradient was often small.
- This suggests phase does not necessarily change much along individual segmented edges.
- The more interesting behavior appears to be phase changes between adjacent edges or across junctions.

### 9. Edge-to-Edge Phase Change

Method:

- For each edge, compare its phase to neighboring edges sharing a node.
- Summarize median, p90, and maximum wrapped phase jump.
- Normalize by length when possible to get a rad/mm-style diagnostic.

Interpretation:

- Low edge-to-edge phase change means neighboring vessel segments pulse with similar timing.
- High edge-to-edge phase change means adjacent segments have different fitted timing.
- High jumps may reflect propagation delay, attenuation, vessel-type transitions, boundary effects, local topology, poor SNR, or model mismatch.
- This is more relevant to current observations than within-edge gradient, because phase differences appear stronger across edges than along edges.

Somites21 H1+H2 phase-identifiability output:

- `Somites21_demo/PerTileFlow/renders/meeting/infer_bayes_default_mosaic_tile_profiles/bayes_tile_phase_identifiability_metrics_h1h2.csv`

Current Somites21 H1 phase-change summary from that file:

| Metric | Min | Median | Max |
|---|---:|---:|---:|
| H1 edge jump p90 | `1.25` rad | `2.25` rad | `2.80` rad |
| H1 edge jump p90 normalized | `28.8` rad/mm | `43.4` rad/mm | `60.0` rad/mm |

### 10. Phase Change Versus Distensibility Identifiability

Current Somites21 dashboard panel:

- `Edge-to-Edge Phase Change vs Distensibility Inference`

Available controls:

- Tile group: all, interior, periphery.
- X-axis: H1/H2 edge-to-edge phase-change metrics.
- Y-axis: posterior curvature, credible interval width, likelihood-ratio width, or `D_hat`.
- `D_hat` is displayed as `log10(D_hat)`.
- Points are labeled by tile number.
- Points are colored by interior/periphery.
- A least-squares line, R-squared, and approximate p-value are shown.
- Error bars are shown when `D_hat` is plotted, using the posterior 95% interval.

Current interpretation:

- This panel is exploratory. It asks whether phase heterogeneity is associated with posterior sharpness, uncertainty, or inferred D.
- A positive relationship with posterior curvature would suggest that stronger edge-to-edge timing structure carries more information about D.
- A positive relationship with credible interval width would instead suggest that phase heterogeneity is mostly model mismatch or noise.
- A relationship with `log10(D_hat)` would suggest the fitted distensibility estimate itself depends on phase propagation structure.

Somites27 status:

- The same phase-correlation panel has not yet been ported into the Somites27 Bayesian script.
- Somites27 has Bayesian H1 and H1+H2 posterior outputs, but not the latest tile phase-identifiability metrics in the main output folder.

## Distensibility Comments

### Identifiability

The current evidence suggests that distensibility is only moderately identifiable on a tile-by-tile basis. Some tiles have narrow posteriors, but many still have widths near or above one order of magnitude. Adding H2 improves identifiability in both datasets, but does not make all tiles sharply identifiable.

### Reliability

A sharp posterior/profile is not automatically reliable. It can mean the model has an obvious optimum, but if the model is misspecified, the optimum can be precise and biased. The ideal case is:

- a sharp posterior/profile,
- no boundary-mode artifact,
- reasonable residual behavior under a noise model that was not artificially tuned to force chi-square near 1,
- consistency across H1 and H1+H2,
- and biological plausibility relative to vessel geometry and phase propagation.

### Diameter Dependence

The visual observation that larger vessels show stronger phase changes may support testing diameter-dependent distensibility. It is not by itself proof that D varies with diameter, because larger vessels also tend to dominate flow amplitude, SNR, boundary effects, and network connectivity. A natural next model would be something like:

```text
D_e = D0 * (r_e / r0)^alpha
```

or a grouped model:

```text
D_large != D_small
```

This could be compared against constant-D using posterior predictive checks or information criteria.

### Spatial Variation

Interior/periphery differences are present but not decisive. Periphery tiles currently show slightly narrower posterior widths in several runs, but this may reflect measurement and boundary-condition strength rather than a true biological difference in D.

### Phase Heterogeneity

Phase heterogeneity may be one of the most useful qualitative diagnostics. Smooth phase variation would be more consistent with simple propagation through a locally coherent compliant network. Sharp edge-to-edge jumps suggest local transitions, attenuation, topology changes, or model mismatch. These jumps are now measurable and can be compared directly against posterior curvature, credible interval width, and `log10(D_hat)`.

## Recommended Next Steps

1. Port the newest Somites21 phase-correlation dashboard features to Somites27.
2. Re-run Somites21 and Somites27 Bayesian inference with `--use-second-harmonic --tile-visualization phase`.
3. Use the correlation panel to check whether edge-to-edge phase change predicts:
   - posterior curvature,
   - credible interval width,
   - likelihood-ratio width,
   - or `log10(D_hat)`.
4. Add vessel-diameter summaries per tile and correlate diameter/diameter variance with the same inference metrics.
5. Test a diameter-dependent D model against the current constant-D model.
6. Treat WLS results cautiously until the noise/weight model is reviewed against measured harmonic SNR and residual structure.

