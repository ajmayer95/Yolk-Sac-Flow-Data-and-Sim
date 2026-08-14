#!/usr/bin/env bash
#SBATCH --job-name=ac_step1_canonical_agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ac_step1_canonical_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_ac_boundary_parameter_calibration.py \
  --graph-path datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --dc-step2-root outputs_canonical/dc/02_physics_weight_sweep \
  --output-root outputs_canonical/ac/01_boundary_parameter_calibration \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --aggregate-only \
  --plot-after
