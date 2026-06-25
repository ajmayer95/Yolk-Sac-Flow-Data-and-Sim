# Generated Outputs Policy

This directory is reserved for outputs derived from the committed code and
datasets. For the GitHub upload, only a very small curated subset is kept in
version control.

## Committed artifacts

The following solver-comparison summaries are intentionally preserved because
they are lightweight and easy to inspect in GitHub:

- `figures/solver_comparison/manifest.json`
- `figures/solver_comparison/configuration_summary.csv`
- `figures/solver_comparison/all_solver_results.csv`

## Omitted artifacts

These are considered generated and should usually be regenerated locally or
stored as release/archive assets instead of committed:

- `runs/`: per-run predictions, parameter surfaces, and manifests.
- Most of `figures/`: dashboards and large collections of per-run PNG files.
- Most of `metrics/`: detailed per-configuration summaries.
- `tables/`: regenerated exports unless a small publication-ready table is
  explicitly curated later.

## Suggested external storage

If you need to share heavier result bundles, prefer:

- GitHub Releases for browsable packaged outputs.
- Zenodo or another archival artifact store for persistent citations.
- Project documentation links back to those external artifacts.

## Regeneration

Recreate outputs from the committed sources:

- `python scripts/run_solver.py ...`
- `python scripts/run_linear_solver_gpu.py ...`
- `python scripts/run_bayesian_solver_gpu.py ...`
- `python scripts/make_classical_solver_comparison.py`
- `python scripts/make_gnn_108_dashboard.py`
