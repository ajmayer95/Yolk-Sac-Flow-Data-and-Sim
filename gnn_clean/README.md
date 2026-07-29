# `gnn_clean`

Clean repository for DC and AC graph-based flow studies.

## DC Studies

### Step 0: Ideal Poiseuille Solver

Step 0 is the ideal steady-state pressure-and-flow model for the DC workflow.

- Main script: `scripts/python/poiseuille_only_baseline.py`

This script supports both:

- a direct gauge-fixed Poiseuille solve
- a reduced soft-constrained least-squares solve via `--dc-solve-mode reduced-soft-constrained-lstsq`

Example:

```bash
python scripts/python/poiseuille_only_baseline.py \
  datasets/emb1_mosaic_graph_analyzed.gpickle \
  --output-dir outputs/dc/00_ideal_models/poiseuille_only_baseline \
  --run-name default_partitioned
```

### Step 1: Boundary Parameter Calibration

Step 1 calibrates the DC boundary-constraint weight by reusing the Step 0 least-squares solver.

- `scripts/python/run_boundary_weight_sweep.py`
- `scripts/python/plot_boundary_weight_sweep.py`

Example:

```bash
python scripts/python/run_boundary_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/01_boundary_parameter_calibration
```

```bash
python scripts/python/plot_boundary_weight_sweep.py \
  --input-csv outputs/dc/01_boundary_parameter_calibration/boundary_weight_summary.csv \
  --output-dir outputs/dc/01_boundary_parameter_calibration
```

This step runs the Step 0 baseline multiple times. The main change between runs is:

- `--dc-solve-mode reduced-soft-constrained-lstsq`
- pressure constraint: `P_A1 = P_A2` and `P_V1 = P_V2`
- flow constraint: total inflow through arterial nodes equals total outflow through venous nodes
- `lambda_q = 1` and `lambda_k = 1` are held fixed
- `lambda_b` is swept over `1, 10, 100, 1000`

The combined summary is written to:

- `outputs/dc/01_boundary_parameter_calibration/boundary_weight_summary.csv`

### Step 2: Physics Weight Sweep

Step 2 reuses the same least-squares solver after fixing the Step 1 boundary setting.

- `scripts/python/run_physics_weight_sweep.py`
- `scripts/python/analyze_physics_weight_sweep.py`
- `scripts/python/plot_physics_weight_sweep.py`

Example:

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep \
  --aggregate-after
```

```bash
python scripts/python/analyze_physics_weight_sweep.py \
  --input-root outputs/dc/02_physics_weight_sweep
```

```bash
python scripts/python/plot_physics_weight_sweep.py \
  --input-root outputs/dc/02_physics_weight_sweep \
  --output-dir outputs/dc/02_physics_weight_sweep/figures
```

For the Poiseuille baseline runs in this sweep:

- `lambda_b` is fixed at `100`
- pressure constraint: `P_A1 = P_A2` and `P_V1 = P_V2`
- flow constraint: total inflow through arterial nodes equals total outflow through venous nodes
- `lambda_q` and `lambda_k` are swept over `0.1, 1, 10, 100`

For GNN runs in this sweep:

- `lambda_b = 100`
- `lambda_q`, `lambda_k`, and `lambda_delta` are swept over `0.1, 1, 10, 100`
- the launcher calls `scripts/python/gnn_flow.py`
- GNN settings used here: `K = 2`, `hidden_dim = 64`, correction bounds `[-0.5, 0.5]`
- pressure solver settings used here: `reduced-soft-constrained-lstsq`, `pressure_constraints = ["equal-a-equal-v"]`, `pressure_detach = False`

Model selection in Step 2 is handled by `scripts/python/analyze_physics_weight_sweep.py`.

- completed runs are aggregated into summary CSV files
- GNN runs are grouped into weighting regimes: flow-prioritized, balanced, conservation-prioritized, and correction-regularized
- Pareto ranks are computed using flow RMSE and Kirchhoff RMS
- representative models are then selected within each regime and written to `representative_configurations.csv`
- selected representatives are labeled `F1`, `F2`, `B1`, `B2`, `K1`, `K2`, `C1`, `C2`, ... for plotting

`scripts/python/plot_physics_weight_sweep.py` generates the main Step 2 summary figures, including:

- flow vs Kirchhoff Pareto plots
- flow RMSE vs correction RMS
- Kirchhoff RMS vs correction RMS
- supplementary plots showing trends against `lambda_q / lambda_k` and `lambda_delta`
- representative label tables for the plotted selected models

The workflow entry points for users who want to test different parameter grids are:

- `scripts/python/run_boundary_weight_sweep.py`
- `scripts/python/run_physics_weight_sweep.py`

The combined outputs are written under:

- `outputs/dc/02_physics_weight_sweep/`

### Step 3: Pressure Constraint Sensitivity

Step 3 reuses the selected Step 2 models and varies the pressure constraints.

- `scripts/python/run_pressure_constraint_sensitivity.py`
- `scripts/python/analyze_pressure_constraint_sensitivity.py`
- `scripts/python/plot_pressure_constraint_sensitivity.py`

Pressure-constraint settings used here:

- `gauge_only`: venous gauge only
- `equal_av`: `P_A1 = P_A2` and `P_V1 = P_V2`
- `equal_drop`: equal arterial-to-venous pressure drops
- `fixed_drop_10pa`: prescribed mean arterial-minus-venous drop of `10 Pa` with equal venous pressure

Example:

```bash
python scripts/python/run_pressure_constraint_sensitivity.py \
  --output-root outputs/dc/03_pressure_constraint_sensitivity \
  --aggregate-after
```

```bash
python scripts/python/analyze_pressure_constraint_sensitivity.py \
  --input-root outputs/dc/03_pressure_constraint_sensitivity
```

```bash
python scripts/python/plot_pressure_constraint_sensitivity.py \
  --input-root outputs/dc/03_pressure_constraint_sensitivity \
  --output-dir outputs/dc/03_pressure_constraint_sensitivity/figures
```

To run only the Poiseuille baseline cases:

```bash
python scripts/python/run_pressure_constraint_sensitivity.py \
  --mode poiseuille \
  --output-root outputs/dc/03_pressure_constraint_sensitivity \
  --aggregate-after
```

This step reuses:

- selected Step 2 GNN representatives from `outputs/dc/02_physics_weight_sweep/representative_configurations.csv`
- the corresponding Poiseuille `lambda_q`, `lambda_k` settings from Step 2

### Step 4: Message-Passing Sensitivity

Step 4 reuses the selected balanced Step 2 GNN model and varies the number of message-passing layers.

- `scripts/python/run_message_passing_depth_sweep.py`
- `scripts/python/compute_message_passing_field_similarity.py`

The tested depths are `K = 0, 1, 2, 3, 4`.

Example:

```bash
python scripts/python/run_message_passing_depth_sweep.py \
  --output-root outputs/dc/04_message_passing_sensitivity
```

```bash
python scripts/python/compute_message_passing_field_similarity.py \
  --input outputs/dc/04_message_passing_sensitivity/combined_gnn_message_passing_depth_summary.csv \
  --output outputs/dc/04_message_passing_sensitivity/message_passing_field_similarity.csv
```

### Step 5: Radius Corrections

Step 5 applies selected radius corrections and reruns the Poiseuille baseline and GNN comparisons.

- `scripts/python/run_radius_correction_experiment.py`
- `scripts/python/analyze_radius_correction_experiment.py`
- `scripts/python/plot_radius_correction_experiment.py`

The main radius-correction strategies used here are:

- `targeted_166`
- `low_snr_20pct`

The main comparison conditions are:

- `p_original`
- `p_corrected`
- `g_original`
- `g_fixed`
- `g_retrained`

Example:

```bash
python scripts/python/run_radius_correction_experiment.py \
  --output-root outputs/dc/05_radius_corrections \
  --aggregate-after \
  --plot-after
```

```bash
python scripts/python/analyze_radius_correction_experiment.py \
  --input-root outputs/dc/05_radius_corrections
```

```bash
python scripts/python/plot_radius_correction_experiment.py \
  --input-root outputs/dc/05_radius_corrections \
  --output-dir outputs/dc/05_radius_corrections/figures
```
