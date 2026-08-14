# Harmonized DC Release Bundle

This folder contains the lightweight DC results bundle for the harmonized dataset.

## Contents

- `00_ideal_models/`
  Poiseuille-only baseline summaries and figures.
- `01_boundary_parameter_calibration/`
  Boundary-weight calibration summaries and figures.
- `02_physics_weight_sweep/`
  Full sweep summary tables, representative-configuration table, and figures.
- `03_pressure_constraint_sensitivity/`
  Pressure-constraint summary tables, pairwise/correlation tables, and figures.
- `04_message_passing_sensitivity/`
  Message-passing depth summary files and K-sweep figures.

## Notes

- This is the lightweight bundle intended for GitHub release upload.
- It includes summary tables, metadata tables, and final figures.
- It does not include the full raw run directories for the DC sweeps.
- For Step 3 and Step 4, the packaged outputs correspond to the harmonized run that used:
  - `lambda_q = 10`
  - `lambda_k = 10`
  - `lambda_delta = 0.1`

## Key files

- Step 2 representative configuration table:
  `02_physics_weight_sweep/representative_configurations.csv`
- Step 3 GNN summary:
  `03_pressure_constraint_sensitivity/pressure_constraint_gnn_summary.csv`
- Step 4 summary:
  `04_message_passing_sensitivity/summary.csv`
