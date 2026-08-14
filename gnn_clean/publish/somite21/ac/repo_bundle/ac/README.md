# Somite21 AC Release Bundle

This folder contains the lightweight AC results bundle for the Somite21 dataset.

## Contents

- `01_boundary_parameter_calibration/`
  Harmonic-specific boundary-parameter calibration summaries and figures.
- `02_physics_weight_sweep/`
  Harmonic-specific physics-weight sweep summaries, representative tables, and figures.
- `03_distensibility_alpha_profiles/`
  Harmonic-specific distensibility alpha/D0 summary tables, representative tables, and figures.

## Notes

- This is the lightweight bundle intended for GitHub release upload.
- It includes summary tables, metadata tables, and final figures.
- It does not include the full raw sweep trees for the AC studies.
- The packaged outputs correspond to the Somite21 AC run set that used:
  - `lambda_q = 100`
  - `lambda_k = 0.1`
  - `lambda_delta = 0.1`
  - arterial/venous boundary mode suffix `_all_observed`
  - representative label `B1`

## Key files

- Step 1 summaries:
  `01_boundary_parameter_calibration/boundary_parameter_calibration_summary_H1.csv`
  `01_boundary_parameter_calibration/boundary_parameter_calibration_summary_H2.csv`
- Step 2 representatives:
  `02_physics_weight_sweep/H1/ac_physics_weight_representatives.csv`
  `02_physics_weight_sweep/H2/ac_physics_weight_representatives.csv`
- Step 3 representatives:
  `03_distensibility_alpha_profiles/H1/representative_configurations.csv`
  `03_distensibility_alpha_profiles/H2/representative_configurations.csv`
