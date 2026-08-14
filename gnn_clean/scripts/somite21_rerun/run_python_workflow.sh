#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GRAPH="datasets/somite21_mosaic_cut_pipeline_ready.gpickle"
DC_ROOT="outputs/somite21/dc"
AC_ROOT="outputs/somite21/ac"
PUBLISH_ROOT="publish/somite21"

run() {
  echo
  echo "[run] $*"
  "$@"
}

mkdir -p logs

run python scripts/python/poiseuille_only_baseline.py \
  "$GRAPH" \
  --output-dir "$DC_ROOT/00_ideal_models/poiseuille_only_baseline" \
  --run-name default_partitioned

run python scripts/python/plot_poiseuille_baseline.py \
  --input-dir "$DC_ROOT/00_ideal_models/poiseuille_only_baseline/default_partitioned" \
  --output-dir "$DC_ROOT/00_ideal_models/poiseuille_only_baseline/default_partitioned/figures"

run python scripts/python/run_boundary_weight_sweep.py \
  --graph "$GRAPH" \
  --output-root "$DC_ROOT/01_boundary_parameter_calibration"

run python scripts/python/plot_boundary_weight_sweep.py \
  --input-csv "$DC_ROOT/01_boundary_parameter_calibration/boundary_weight_summary.csv" \
  --input-root "$DC_ROOT/01_boundary_parameter_calibration" \
  --output-dir "$DC_ROOT/01_boundary_parameter_calibration/figures" \
  --lambda-b 100

run python scripts/python/run_physics_weight_sweep.py \
  --graph "$GRAPH" \
  --output-root "$DC_ROOT/02_physics_weight_sweep" \
  --aggregate-after

run python scripts/python/plot_physics_weight_sweep.py \
  --input-root "$DC_ROOT/02_physics_weight_sweep"

run python scripts/python/run_pressure_constraint_sensitivity.py \
  --graph "$GRAPH" \
  --output-root "$DC_ROOT/03_pressure_constraint_sensitivity" \
  --aggregate-after

run python scripts/python/plot_pressure_constraint_sensitivity.py \
  --input-root "$DC_ROOT/03_pressure_constraint_sensitivity"

run python scripts/python/run_message_passing_depth_sweep.py \
  --graph "$GRAPH" \
  --output-root "$DC_ROOT/04_message_passing_sensitivity"

run python scripts/python/harmonic_stage1_admittance_model_comparison.py \
  --graph-path "$GRAPH" \
  --dc-step2-root "$DC_ROOT/02_physics_weight_sweep" \
  --harmonic-number 1 \
  --output-dir "$AC_ROOT/00_ideal_models/harmonic_stage1_admittance_model_comparison/H1" \
  --overwrite

run python scripts/python/harmonic_stage1_admittance_model_comparison.py \
  --graph-path "$GRAPH" \
  --dc-step2-root "$DC_ROOT/02_physics_weight_sweep" \
  --harmonic-number 2 \
  --output-dir "$AC_ROOT/00_ideal_models/harmonic_stage1_admittance_model_comparison/H2" \
  --overwrite

run python scripts/python/run_ac_distensibility_sweep.py \
  --graph-path "$GRAPH" \
  --scratch-root "$AC_ROOT/00_ideal_models/distensibility_sweep/_raw_runs" \
  --output-root "$AC_ROOT/00_ideal_models/distensibility_sweep" \
  --plot-after

run python scripts/python/run_ac_boundary_parameter_calibration.py \
  --graph-path "$GRAPH" \
  --dc-step2-root "$DC_ROOT/02_physics_weight_sweep" \
  --output-root "$AC_ROOT/01_boundary_parameter_calibration" \
  --aggregate-after \
  --plot-after

run python scripts/python/run_ac_physics_weight_sweep.py \
  --graph-path "$GRAPH" \
  --dc-step2-root "$DC_ROOT/02_physics_weight_sweep" \
  --output-root "$AC_ROOT/02_physics_weight_sweep" \
  --aggregate-after \
  --plot-after

run python scripts/python/run_ac_distensibility_alpha_profiles.py \
  --graph-path "$GRAPH" \
  --step2-root "$AC_ROOT/02_physics_weight_sweep" \
  --output-root "$AC_ROOT/03_distensibility_alpha_profiles" \
  --aggregate-after \
  --plot-after

run python scripts/python/package_dc_results_for_release.py \
  --outputs-root "$DC_ROOT" \
  --output-root "$PUBLISH_ROOT"

run python scripts/python/package_ac_results_for_release.py \
  --outputs-root "$AC_ROOT" \
  --output-root "$PUBLISH_ROOT"
