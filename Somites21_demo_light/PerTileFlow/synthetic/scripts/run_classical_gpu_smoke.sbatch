#!/bin/bash
# Small GPU validation run for the new tile engines.
#
# Submit:
#   sbatch scripts/run_classical_gpu_smoke.sbatch

#SBATCH --job-name=dist_gpu_smoke
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/dist_gpu_smoke-%j.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/dist_gpu_smoke-%j.err

set -euo pipefail

SYN_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${SYN_DIR}"

DATASET="data/synthetic/pl_d1e-03_a1_n10_s42.npz"

python scripts/run_linear_solver_gpu.py "${DATASET}" \
  --alpha-mode solved --harmonics h1_h2 --tiles 22 \
  --num-D0 5 --num-alpha 5 --chunk-size 5

python scripts/run_bayesian_solver_gpu.py "${DATASET}" \
  --alpha-mode solved --harmonics h1_h2 --tiles 22 \
  --num-D0 5 --num-alpha 5 --chunk-size 5
