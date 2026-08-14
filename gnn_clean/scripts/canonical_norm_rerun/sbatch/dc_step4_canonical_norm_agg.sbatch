#!/usr/bin/env bash
#SBATCH --job-name=dc_step4_canonical_norm_agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_step4_canonical_norm_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_message_passing_depth_sweep.py \
  --representative-csv outputs_canonical_norm/dc/02_physics_weight_sweep/representative_configurations.csv \
  --step2-root outputs_canonical_norm/dc/02_physics_weight_sweep \
  --graph datasets/mosaic_graph_norm_canonical.gpickle \
  --output-root outputs_canonical_norm/dc/04_message_passing_sensitivity \
  --aggregate-only
