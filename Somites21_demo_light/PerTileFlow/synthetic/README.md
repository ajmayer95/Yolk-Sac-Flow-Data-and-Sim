# Synthetic Distensibility Repro Package

This `synthetic/` subtree is organized as a GitHub-friendly reproducibility
package. It keeps the code, configurations, lightweight synthetic datasets,
and a small curated set of summary artifacts in version control while omitting
bulk generated outputs that are straightforward to regenerate.

## What is included

- `configs/`: experiment and solver configuration files.
- `data/`: synthetic dataset documentation plus the compact `.npz` archives
  and `manifest.csv` needed for turnkey reproduction.
- `scripts/`: command-line and batch entry points used to generate data,
  run solvers, train models, and assemble comparison dashboards.
- `src/`: reusable implementation modules that the scripts call into.
- `outputs/`: curated summary artifacts only.

## What is intentionally omitted

The following are treated as generated artifacts and should generally stay out
of Git history:

- `outputs/runs/`: per-run predictions, parameter surfaces, manifests, and
  other bulk solver artifacts.
- Most of `outputs/figures/`: mass-generated PNG collections and dashboard
  outputs.
- Most of `outputs/metrics/`: per-configuration metrics that can be rebuilt.
- `scripts/logs/`: scheduler and batch logs.
- `__pycache__/`: Python cache directories.

## Curated results kept in Git

The committed results are intentionally small and publication-oriented:

- `outputs/figures/solver_comparison/manifest.json`
- `outputs/figures/solver_comparison/configuration_summary.csv`
- `outputs/figures/solver_comparison/all_solver_results.csv`

If you want browsable heavy artifacts such as large dashboards, keep them in
GitHub Releases, Zenodo, or another artifact store and link them from project
documentation instead of committing the whole generated tree.

## Regenerating omitted outputs

Run commands from this `synthetic/` directory unless noted otherwise.

Generate synthetic datasets:

```bash
python scripts/make_synthetic.py
```

Run the classical inverse solvers:

```bash
python scripts/run_solver.py DATASET --method METHOD [options]
```

Run the tiled GPU campaigns:

```bash
sbatch scripts/run_all_tile_solvers_gpu.sbatch
sbatch scripts/run_gnn_conditioned_tile_solvers_gpu.sbatch
```

Run the GNN training and aggregation workflows:

```bash
python scripts/run_gnn_grid.py
python scripts/make_classical_solver_comparison.py
python scripts/make_gnn_108_dashboard.py
```

See `data/README.md`, `scripts/README.md`, and `src/README.md` for the
dataset schema, entry points, and workflow details.
