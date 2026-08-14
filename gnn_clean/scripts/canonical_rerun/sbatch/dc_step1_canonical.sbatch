#!/usr/bin/env bash
#SBATCH --job-name=dc_step1_canonical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dc_step1_canonical.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_boundary_weight_sweep.py \
  --graph datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --output-root outputs_canonical/dc/01_boundary_parameter_calibration

conda run -n yolk-sac python scripts/python/plot_boundary_weight_sweep.py \
  --input-csv outputs_canonical/dc/01_boundary_parameter_calibration/boundary_weight_summary.csv \
  --input-root outputs_canonical/dc/01_boundary_parameter_calibration \
  --output-dir outputs_canonical/dc/01_boundary_parameter_calibration/figures \
  --lambda-b 100
