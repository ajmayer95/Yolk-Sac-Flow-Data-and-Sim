#!/usr/bin/env bash
#SBATCH --job-name=dc_step3_canonical_agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_step3_canonical_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_pressure_constraint_sensitivity.py \
  --representative-csv outputs_canonical/dc/02_physics_weight_sweep/representative_configurations.csv \
  --graph datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --output-root outputs_canonical/dc/03_pressure_constraint_sensitivity \
  --aggregate-only

conda run -n yolk-sac python scripts/python/plot_pressure_constraint_sensitivity.py \
  --input-root outputs_canonical/dc/03_pressure_constraint_sensitivity
