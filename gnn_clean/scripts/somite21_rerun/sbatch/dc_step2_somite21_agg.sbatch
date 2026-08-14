#!/usr/bin/env bash
#SBATCH --job-name=dc_step2_somite21_agg_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_step2_somite21_agg_cpu.%j.out

# CPU aggregation job: summarize and plot existing sweep outputs only.

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/02_physics_weight_sweep \
  --aggregate-only

conda run -n yolk-sac python scripts/python/plot_physics_weight_sweep.py \
  --input-root outputs/somite21/dc/02_physics_weight_sweep
