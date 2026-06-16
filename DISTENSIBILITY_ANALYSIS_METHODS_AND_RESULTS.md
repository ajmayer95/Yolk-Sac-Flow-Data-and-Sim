# Distensibility Analysis Methods and Current Results

Date: 2026-06-10

This note summarizes the methods attempted so far for the Somites21 and Somites27 yolk-sac mosaic datasets, with comments on what the current results suggest about distensibility. The emphasis is on tile-by-tile recovery/inference, profile likelihood/posterior shape, and phase-based diagnostics.

## Main Scripts

Somites21:

- `Somites21_demo/PerTileFlow/scripts/default_mosaic_tile_profiles.py`
- `Somites21_demo/PerTileFlow/scripts/infer_default_mosaic_tile_profiles.py`
- `Somites21_demo/PerTileFlow/scripts/infer_bayes_default_mosaic_tile_profiles.py`
- `Somites21_demo/PerTileFlow/scripts/self_consistency.py`
- `Somites21_demo/PerTileFlow/scripts/global_inverse_shared_D.py`
- `Somites21_demo_light/PerTileFlow/scripts/infer_bayes_default_mosaic_tile_profiles.py`

Somites27:

- `Somites27_demo/PerTileFlow/scripts/default_mosaic_tile_profiles.py`
- `Somites27_demo/PerTileFlow/scripts/infer_default_mosaic_tile_profiles.py`
- `Somites27_demo/PerTileFlow/scripts/infer_bayes_default_mosaic_tile_profiles.py`

The newest phase-correlation dashboard features are currently in the Somites21 Bayesian script and the Somites21 light copy. They have not yet been ported into the Somites27 Bayesian script.

Additional GNN and Bayesian analysis folders:

- `Somites21_demo/PerTileFlow/gnn_edge/train_gnn_edge.py`
- `Somites21_demo/PerTileFlow/gnn_edge_heldout/train_gnn_edge.py`
- `Somites21_demo/PerTileFlow/gnn_edge_within/train_gnn_edge.py`
- `Somites21_demo/PerTileFlow/gnn_edge_v2/train_gnn_edge.py`
- `Somites21_demo/PerTileFlow/bayesian/bayesian_tile_distensibility.py`
- `Somites21_demo/PerTileFlow/bayesian/bayesian_output_dashboard.py`
- `Somites21_demo/PerTileFlow/bayesian/infer_bayes_default_mosaic_tile_profiles.py`
- `Somites27_demo/PerTileFlow/gnn_edge/train_gnn_edge.py`

These runs divide into two related but different families. The GNN scripts are physics-embedded flow models used to infer pressure/flow scaffolds and diagnose residual structure. The Bayesian scripts are posterior-based distensibility inference workflows. Detailed Bayesian methods are summarized in Section 3, and GNN methods/results are summarized in Section 4.

Current GNN output folders used in the analysis:

- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260614_222704/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260614_223539/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260614_231036/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260616_092543/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_within_20260615_093931/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_within_20260615_104348/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_heldout_20260615_094419/`
- `Somites27_demo/PerTileFlow/renders/gnn_edge_dc_20260615_125237/`

Current Bayesian output folders used in the analysis:

- `Somites21_demo/PerTileFlow/bayesian/outputs/`
- `Somites21_demo/PerTileFlow/renders/meeting/infer_bayes_default_mosaic_tile_profiles/`
- `Somites27_demo/PerTileFlow/renders/meeting/infer_bayes_default_mosaic_tile_profiles/`

High-level current interpretation:

- The Bayesian outputs remain the primary posterior-based distensibility summaries.
- The GNN pressure fields are best treated as scaffolds, priors, and diagnostic views rather than direct D estimates.
- The GNN outputs are still scientifically useful because they expose pressure outliers, edge-level residual structure, harmonic error maps, and differences between whole-mosaic and tile-local pressure explanations.

New Somites21 outputs generated on 2026-06-10:

- `Somites21_demo/PerTileFlow/renders/meeting/self_consistency/self_consistency_summary.json`
- `Somites21_demo/PerTileFlow/renders/meeting/self_consistency/self_consistency_METHOD.md`
- `Somites21_demo/PerTileFlow/renders/meeting/self_consistency/self_consistency_shared_D_profile.csv`
- `Somites21_demo/PerTileFlow/renders/meeting/self_consistency/self_consistency_overlap_pairs.csv`
- `Somites21_demo/PerTileFlow/renders/meeting/self_consistency/self_consistency_report.html`
- `Somites21_demo/PerTileFlow/renders/meeting/global_inverse_shared_D/global_inverse_summary.json`
- `Somites21_demo/PerTileFlow/renders/meeting/global_inverse_shared_D/global_inverse_profile.csv`
- `Somites21_demo/PerTileFlow/renders/meeting/global_inverse_shared_D/global_inverse_residuals_at_best.csv`
- `Somites21_demo/PerTileFlow/renders/meeting/global_inverse_shared_D/global_inverse_shared_D.html`

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
- `Somites21_demo/PerTileFlow/bayesian/outputs/`

Additional Bayesian folder workflow:

```bash
python bayesian/bayesian_tile_distensibility.py --config ../emb1/config.json
python bayesian/bayesian_output_dashboard.py --out-dir bayesian/outputs/<run_dir>
```

Folder-specific method:

- `bayesian/infer_bayes_default_mosaic_tile_profiles.py` is the dashboard-style Bayesian workflow used for measured tile data. It evaluates the marginal posterior over `D` on a log grid and writes global/tile posterior CSV files plus a standalone HTML dashboard.
- `bayesian/bayesian_tile_distensibility.py` is a deterministic Bayesian grid-integration implementation. For each tile and each `D0` grid point, it builds a harmonic transfer operator from unknown boundary pressure phasors to observed edge-flow phasors.
- Boundary pressure phasors are analytically marginalized under a zero-mean Gaussian prior. Nuisance scales for boundary forcing and observation noise are integrated on log grids using log-sum-exp.
- H1 is used by default; H2 can be included with `--harmonics 1 2`. The observation vector is real-stacked so complex harmonic flow phasors contribute real and imaginary residual components.
- `bayesian/bayesian_output_dashboard.py` reads `tile_D_posterior_curves.csv` and `tile_D_posterior_summary.csv` and writes an interactive HTML dashboard with likelihood profiles, selected-tile prior/posterior curves, and per-tile summary tables.

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

### 4. Physics-Embedded GNN Flow/Pressure Scaffolds

Whole-mosaic scripts:

```bash
python gnn_edge/train_gnn_edge.py --config ../emb1/config.json --sweep
python gnn_edge_v2/train_gnn_edge.py --config ../emb1/config.json --sweep
```

Tile-local scripts:

```bash
python gnn_edge_within/train_gnn_edge.py --mosaic-scaffold-dir <gnn_edge_sweep_or_run>
python gnn_edge_heldout/train_gnn_edge.py --mosaic-scaffold-dir <gnn_edge_sweep_or_run>
```

Somites27 port:

```bash
python gnn_edge/train_gnn_edge.py --config ../../emb1/config.json --sweep
```

Purpose:

- Fit a physics-embedded GNN that predicts an edge conductance correction `delta`.
- Convert Poiseuille conductance to learned conductance using `G_hat = G_pois * exp(delta)`.
- Solve a resistive network pressure problem and reconstruct DC flow as `Q_hat = G_hat * pressure_drop`.
- Use the learned pressure field as a scaffold/pressure prior for later tile-profile analyses.
- Diagnose where measured flows, vessel geometry, graph topology, and pressure consistency disagree.

Core model:

- Edge features include geometric/Poiseuille quantities such as radius, length, `r^4 / L`, baseline conductance, degree information, and, in v2, optional harmonic descriptors.
- A message-passing edge GNN predicts `delta` on each edge. The exponential parameterization keeps learned conductances positive.
- For whole-mosaic runs, the model couples the learned conductances to a pressure solve over the full graph and optimizes flow reconstruction on train edges while monitoring held-out validation edges.
- For scaffolded tile-local runs, the whole-mosaic pressure solution is loaded from a previous GNN run. Each tile then fits only low-dimensional pressure nuisance terms around that scaffold while the shared edge model learns local conductance corrections.
- Reported errors are in nL/s. Normalized RMSE divides RMSE by the RMS magnitude of the observed flow on the scored split.

Model variants:

- `gnn_edge`: whole-mosaic DC model. It trains on observed DC edge flows, solves one mosaic pressure field, and writes pressure/flow diagnostics.
- `gnn_edge_within`: tile-local scaffolded model. It loads a whole-mosaic GNN scaffold, then trains/evaluates with held-out edges within each tile.
- `gnn_edge_heldout`: tile-local scaffolded model with held-out tiles. It tests whether the shared correction model transfers to tiles not used for fitting.
- `gnn_edge_v2`: harmonic-aware extension. It supports `--flow-components dc`, `--flow-components dc-h1`, and `--flow-components dc-h1-h2`. H1/H2 auxiliary losses are SNR-weighted and controlled by `--lambda-h1` and `--lambda-h2`.
- Somites27 `gnn_edge`: port of the whole-mosaic DC workflow. It includes an observed-divergence fallback for boundary injections because Somites27 does not expose the same explicit source/sink metadata as Somites21.

Main artifacts:

- `metrics.csv`: aggregate train/validation RMSE, normalized RMSE, MAE, correlation, and R2.
- `edge_predictions*.csv`: observed and predicted flow by edge, learned correction `delta`, conductance multiplier `C`, and pressure-drop diagnostics.
- `node_pressures_physics_gnn.csv`: fitted node pressure field for the whole-mosaic GNN.
- `diagnostics.json`: summary of correction magnitudes, conductance multipliers, and split sizes.
- Spatial figures: pressure maps, pressure-drop maps, residual maps, predicted-versus-observed plots, conductance histograms, and, for v2, harmonic flow/error maps.

Current output folders:

- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260614_222704/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260614_223539/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260614_231036/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_dc_20260616_092543/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_within_20260615_093931/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_within_20260615_104348/`
- `Somites21_demo/PerTileFlow/renders/gnn_edge_heldout_20260615_094419/`
- `Somites27_demo/PerTileFlow/renders/gnn_edge_dc_20260615_125237/`

Whole-mosaic DC GNN summary:

| Dataset/run | Sweep size | Best run | Best val NRMSE | Best val RMSE | Median C | Max C | Comment |
|---|---:|---|---:|---:|---:|---:|---|
| Somites21 `gnn_edge_dc_20260614_231036` | 45 | `K3_hidden128_lambda0p0001_seed0` | `0.720` | `0.110` nL/s | `1.52` | `128` | Best older DC sweep; conductance scale remains near order 1 in the median but has high-C outliers. |
| Somites21 `gnn_edge_dc_20260616_092543` | 20 | `K3_hidden64_lambda0p001_seed0` | `0.801` | `0.123` nL/s | `0.0071` | `16.8` | Harmonic-aware/v2-era run; useful diagnostics, but median conductance scale is much lower and should be interpreted cautiously. |
| Somites27 `gnn_edge_dc_20260615_125237` | 4 | `K0_hidden128_lambda0p0001_seed0` | `0.138` | `0.039` nL/s | `1.27` | `102` | Small Somites27 sweep; validation flow error is substantially lower than current Somites21 sweeps. |

Tile-local GNN summary:

| Dataset/run | Split style | Val NRMSE | Val RMSE | Diagnostic notes |
|---|---|---:|---:|---|
| Somites21 `gnn_edge_within_20260615_093931` | held-out edges within tiles | `1.04` | `0.190` nL/s | Physics GNN improves strongly over Poiseuille baseline (`11.6` val NRMSE), but remains worse than the best whole-mosaic DC validation. |
| Somites21 `gnn_edge_within_20260615_104348` | scaffolded local pressure, held-out edges | `1.25` for `scaffold_local_pressure_gnn` | `0.207` nL/s | `scale_only` baseline (`0.657` val NRMSE) outperformed the local-pressure GNN in this run; local pressure flexibility can overfit or destabilize validation. |
| Somites21 `gnn_edge_heldout_20260615_094419` | held-out tiles | `28.0` | `6.42` nL/s | Held-out-tile generalization is poor in this run; this suggests tile-specific pressure/flow structure is not captured well by the current shared model. |

Harmonic-aware GNN diagnostics:

- `gnn_edge_v2` adds harmonic edge features derived from H1/H2 amplitudes, phases, and SNR.
- H1/H2 auxiliary losses are computed on complex harmonic flow residuals and weighted by the SNR-derived flow uncertainty.
- The current harmonic output includes `harmonic_predictions_physics_gnn.csv`, per-harmonic absolute error maps, and per-harmonic observed flow magnitude maps.
- The maps named `harmonic_H1_flow_magnitude_map.png` and `harmonic_H2_flow_magnitude_map.png` show observed harmonic flow magnitudes, not predicted magnitudes.
- The maps named `harmonic_H1_absolute_error_map.png` and `harmonic_H2_absolute_error_map.png` compare predicted and observed complex harmonic flows.
- The current v2 model does not solve a harmonic pressure field. Harmonic pressure maps and harmonic pressure-drop maps would require an additional harmonic pressure model or an explicit complex network solve.

Pressure-map outlier analysis:

- Some whole-mosaic Somites21 GNN runs produced very high or very low pressure outliers that visually dominated pressure spatial maps.
- Additional filtered maps were generated for selected runs by removing the single highest-pressure node, and separately by removing the ten highest and ten lowest pressure nodes.
- These filtered maps are visualization-only diagnostics. They should not be interpreted as refit models; they reveal lower-pressure spatial variation that is otherwise compressed by the color scale.

Interpretation:

- The whole-mosaic GNNs are useful pressure/flow scaffolds and produce interpretable residual, pressure, conductance, and topology diagnostics.
- The GNN outputs should not be treated as direct estimates of distensibility `D`; they fit conductance corrections and pressure fields, not compliance dynamics.
- GNN pressure fields are useful as priors for frequentist tile-profile scans because they partially remove the pressure nuisance-parameter ambiguity.
- Tile-local GNN results show that generalization is harder than edge-level interpolation: held-out-tile performance can be poor, and local pressure nuisance terms can dominate.
- Harmonic-aware GNN results are promising as diagnostics, but the harmonic component is currently an auxiliary flow-prediction head rather than a full harmonic pressure solver.

### 5. Location-Based Tile Grouping

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

### 6. H1 Versus H1+H2 Sensitivity

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

### 7. Tile Phase and Harmonic Visualization

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

### 8. Contour and Smooth Phase Gradient Experiments

Methods attempted:

- Direct phase contour maps at mode, mode +/- pi/6, and mode +/- pi/3.
- Robust phase-plane fitting.
- Phase-gradient arrows from the fitted phase plane.

Comments:

- Raw contour overlays were visually noisy and hard to interpret.
- Finer contours risk tracking measurement noise rather than meaningful propagation structure.
- The robust phase-plane view is cleaner, but it is still a visualization aid rather than direct evidence of constant distensibility.
- The main lesson was that phase often appears piecewise or edge-to-edge heterogeneous rather than smoothly varying within each vessel segment.

### 9. Within-Edge Phase Gradient

Method:

- Estimate node phase from incident edge phases.
- For each edge, compute wrapped node-to-node phase difference divided by vessel length.

Comments:

- The within-edge phase gradient was often small.
- This suggests phase does not necessarily change much along individual segmented edges.
- The more interesting behavior appears to be phase changes between adjacent edges or across junctions.

### 10. Edge-to-Edge Phase Change

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

### 11. Phase Change Versus Distensibility Identifiability

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

### 12. Shared-D Self-Consistency and Tile-Order Phase Correction

Script:

```bash
python scripts/self_consistency.py \
  --config ../emb1/config.json \
  --use-second-harmonic
```

Recommended no-correction baseline:

```bash
python scripts/self_consistency.py \
  --config ../emb1/config.json \
  --use-second-harmonic \
  --timing-mode none
```

Purpose:

- Combine the existing tile-level Bayesian marginal likelihoods into one shared-D profile.
- Plot the shared-D profile likelihood, prior, and posterior.
- Test whether a simple tile-order phase correction improves consistency between overlapping measurements of the same graph edge.
- Use tile acquisition order as prior information: lower tile numbers were measured earlier, and all Somites21 tiles were acquired over roughly five minutes.

Phase-correction modes:

- `--timing-mode none`: no phase correction; this is the cleanest baseline.
- `--timing-mode fit_beta`: fit a signed global linear phase drift across tile order using overlap phase mismatch.
- `--timing-mode fixed_acquisition_time`: impose a phase drift from the nominal acquisition duration and median frequency.

How the phase correction works:

```text
theta_t = beta * timing_coordinate_t
phi_corrected = phi_measured - h * theta_t
```

where `h` is the harmonic number. H2 receives twice the phase correction of H1.

Current Somites21 findings:

- The 2026-06-10 saved baseline used all 53 Somites21 tiles, H1+H2, tile-order timing, and `--timing-mode none`.
- It included 23,594 harmonic measurements and 19,838 overlapping tile-pair phase comparisons.
- With `--timing-mode none`, the shared-D profile likelihood is already narrow and the posterior is sharply peaked.
- The saved baseline has shared-D likelihood mode `3.16e-4`, posterior mode `3.16e-4`, posterior median `2.78e-4`, and approximate 95% posterior interval `2.09e-4` to `3.14e-4`.
- The raw overlap absolute phase mismatch in this no-correction run had mean `1.57` rad, median `1.56` rad, and p90 `2.83` rad.
- The prior is comparatively broad/flat over the plotted D range, so the sharp posterior is not simply prior-driven.
- Earlier/alternate phase-correction runs indicated that introducing tile-order phase corrections does not materially change the shared-D likelihood/posterior profiles.
- The fitted tile-order correction is not uniformly helpful for overlap phases. It can give a modest average improvement but helps some overlap pairs while worsening others.
- The raw-versus-corrected mismatch scatter can look nearly symmetric, and largest improvements/worsenings can approach `pi` because phase differences are wrapped.
- The fitted `beta` is multi-modal and search-range-dependent because the objective uses wrapped phases. Therefore, fitted beta should be treated as a diagnostic correction, not as a literal acquisition-time estimate.

Interpretation:

- These results are evidence that the current shared-D inference is not strongly dependent on the simple phase-adjustment model.
- A global linear tile-order phase correction is probably too crude to explain stitched phase mismatch.
- The phase-correction experiment is still useful as a stress test for acquisition-order artifacts, but it should not be part of the main distensibility result unless a clearer, physically interpretable correction model is developed.
- For current reporting, `--timing-mode none` is the preferred baseline, while `fit_beta` and `fixed_acquisition_time` are sensitivity analyses.

Important limitation:

- The self-consistency script still does not solve the full global pressure inverse problem. It multiplies tile-level marginal likelihoods and separately analyzes overlap phase consistency. It does not yet fit one global pressure/flow field to all measured tile observations simultaneously.

### 13. First-Pass Full-Mosaic Shared-D Inverse

Script:

```bash
python scripts/global_inverse_shared_D.py \
  --config ../emb1/config.json
```

H1 + H2 variant:

```bash
python scripts/global_inverse_shared_D.py \
  --config ../emb1/config.json \
  --use-second-harmonic
```

Purpose:

- Solve the whole mosaic once per D value.
- Enforce one mosaic-wide pressure/flow field for all graph edges.
- Score measured edge flows against the global predicted edge flows.
- Produce a global shared-D profile likelihood, prior, and posterior.

This differs from the tile inference scripts in an important way:

- Tile inference fits or marginalizes tile-local boundary conditions.
- `self_consistency.py` multiplies tile-level likelihoods.
- `global_inverse_shared_D.py` scores all measured observations against one whole-mosaic solve, so tile boundaries are no longer independently free.

Current first-pass assumptions:

- A/V boundary flow waveforms are fixed from the same viewer-default boundary setup used in the simulation scripts.
- D is the only fitted physical parameter.
- Per-tile measurements can be scored directly with `--observation-source per-tile`.
- Top-level graph edge measurements can be scored with `--observation-source top-level`.
- H1-only or H1+H2 can be used.

Current saved Somites21 run:

- The 2026-06-10 saved run used top-level observations, H1 only, oriented observations, a 41-point D grid from `1e-5` to `1e-1`, viewer-default fixed A/V flow boundary conditions, target flux `1.0` nL/s, and `f0 = 2.773` Hz.
- The likelihood and posterior modes were both `2.51e-3`.
- The posterior median was `2.51e-3`, with approximate 95% posterior interval `2.25e-3` to `2.80e-3`.
- The best-fit residual was still high: `chi2_min = 1.82e5`, `chi2_red_min = 11.8`, with 15,430 scalar observations represented by 10,287 scored rows.
- The previous very coarse smoke test landed at the high end of its tested range (`D=0.1`), but the current 41-point saved top-level H1 run has an interior optimum near `2.5e-3`.

Interpretation:

- This is the first actual global inverse scaffold: one D, one mosaic solve, one global residual.
- The interior optimum is encouraging as a numerical profile, but the large reduced chi-square means this should not yet be treated as a final biological estimate.
- The high residuals likely reflect the restrictive fixed viewer-default A/V boundary forcing, remaining observation/orientation/noise mismatch, or missing physics/heterogeneity.
- The next methodological step is to relax or infer global boundary forcing rather than keeping the viewer-default A/V flow waveforms fixed.
- A useful intermediate model would keep a single shared D but fit low-dimensional source/sink boundary amplitudes or complex harmonic scale factors jointly with D.

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

### Phase Correction and D Identifiability

The current self-consistency results suggest that simple tile-order phase correction is not a major driver of the shared-D estimate. The 2026-06-10 no-correction H1+H2 run is already narrow and sharply peaked, with shared-D posterior mode `3.16e-4`. Earlier phase-correction comparisons did not meaningfully alter the D profile. This supports the interpretation that, under the current model, D identifiability is coming mostly from the harmonic flow/pressure response encoded in the tile likelihoods rather than from the tile-order phase adjustment.

The full-mosaic shared-D inverse now has a concrete 41-point top-level H1 run with an interior optimum near `2.51e-3`, but its `chi2_red_min = 11.8` indicates substantial mismatch under fixed viewer-default A/V boundary forcing. This should be viewed as evidence that the global inverse scaffold is working computationally, not yet as a settled D estimate.

This does not rule out noise or stitching artifacts. Rather, it suggests that the current one-parameter phase correction is too blunt to explain them. If acquisition timing matters, it may require tile-specific offsets, local overlap constraints, or a full global inverse model rather than a single linear drift across tile order.

## Recommended Next Steps

1. Treat `self_consistency.py --timing-mode none --use-second-harmonic` as the current Somites21 shared-D baseline.
2. Use `fit_beta` and `fixed_acquisition_time` only as timing-artifact sensitivity checks unless a stronger physical phase-correction model is developed.
3. Continue building the full global inverse problem from `global_inverse_shared_D.py`:
   - one global pressure/flow field,
   - one shared D initially,
   - all tile harmonic observations included jointly,
   - A/V boundary conditions imposed at the mosaic level,
   - then fit or marginalize low-dimensional global A/V boundary forcing,
   - optional tile phase offsets as nuisance parameters only after the no-correction baseline is working.
4. Re-run `global_inverse_shared_D.py` with H1+H2 and with `--observation-source per-tile`, then compare the global optimum and residual structure against the saved top-level H1 run.
5. Add low-dimensional fitted global A/V forcing to the global inverse before interpreting its D mode biologically.
6. Port the newest Somites21 phase-correlation dashboard features to Somites27.
7. Re-run Somites21 and Somites27 Bayesian inference with `--use-second-harmonic --tile-visualization phase`.
8. Use the correlation panel to check whether edge-to-edge phase change predicts:
   - posterior curvature,
   - credible interval width,
   - likelihood-ratio width,
   - or `log10(D_hat)`.
9. Add vessel-diameter summaries per tile and correlate diameter/diameter variance with the same inference metrics.
10. Test a diameter-dependent D model against the current constant-D model.
11. Treat WLS results cautiously until the noise/weight model is reviewed against measured harmonic SNR and residual structure.
