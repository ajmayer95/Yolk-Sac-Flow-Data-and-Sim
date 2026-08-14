#!/usr/bin/env bash
#SBATCH --job-name=package_somite21
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/package_somite21.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/package_dc_results_for_release.py \
  --outputs-root outputs/somite21/dc \
  --output-root publish/somite21

conda run -n yolk-sac python scripts/python/package_ac_results_for_release.py \
  --outputs-root outputs/somite21/ac \
  --output-root publish/somite21
