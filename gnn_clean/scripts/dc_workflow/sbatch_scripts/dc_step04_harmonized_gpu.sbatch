#!/usr/bin/env bash
#SBATCH --job-name=dc_step04_harmonized_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dc_step04_harmonized_gpu.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/dc_workflow/solver/04_message_passing_sensitivity.py \
  --graph datasets_upload/harmonized_scaled_dataset.gpickle \
  --representative-csv outputs_harmonized/dc/02_physics_weight_sweep/representative_configurations.csv \
  --step2-root outputs_harmonized/dc/02_physics_weight_sweep \
  --device cuda \
  --require-cuda \
  --output-root outputs_harmonized/dc/04_message_passing_sensitivity \
  "$@"
