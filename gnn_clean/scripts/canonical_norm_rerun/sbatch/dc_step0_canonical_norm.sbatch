#!/usr/bin/env bash
#SBATCH --job-name=dc_step0_canonical_norm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dc_step0_canonical_norm.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/poiseuille_only_baseline.py \
  datasets/mosaic_graph_norm_canonical.gpickle \
  --output-dir outputs_canonical_norm/dc/00_ideal_models/poiseuille_only_baseline \
  --run-name default_partitioned

conda run -n yolk-sac python scripts/python/plot_poiseuille_baseline.py \
  --input-dir outputs_canonical_norm/dc/00_ideal_models/poiseuille_only_baseline/default_partitioned \
  --output-dir outputs_canonical_norm/dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/figures
