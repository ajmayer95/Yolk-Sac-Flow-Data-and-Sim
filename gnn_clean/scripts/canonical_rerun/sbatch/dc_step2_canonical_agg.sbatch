#!/usr/bin/env bash
#SBATCH --job-name=dc_step2_canonical_agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_step2_canonical_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

LAMBDA_B=100.0
echo "[info] DC Step 2 canonical sweep aggregates runs with fixed lambda_B=${LAMBDA_B}."

conda run -n yolk-sac python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --output-root outputs_canonical/dc/02_physics_weight_sweep \
  --aggregate-only

conda run -n yolk-sac python scripts/python/plot_physics_weight_sweep.py \
  --input-root outputs_canonical/dc/02_physics_weight_sweep
