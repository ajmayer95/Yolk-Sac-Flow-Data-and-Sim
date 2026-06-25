#!/bin/bash
# GPU campaign for tile-specific linear and Bayesian solvers.
#
# One array task owns one dataset and runs:
#   2 methods: linear_tile_gpu, bayesian_tile_gpu
#   4 alpha configurations: prescribed 0/1/2 and solved
#   2 harmonic configurations: H1 and H1+H2
# for 16 runs per dataset.
#
# Submit:
#   sbatch scripts/run_all_tile_solvers_gpu.sbatch

#SBATCH --job-name=dist_tile_gpu
#SBATCH --partition=gpu
#SBATCH --array=0-35%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/dist_tile_gpu-%A_%a.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/dist_tile_gpu-%A_%a.err

set -euo pipefail

SYN_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
MANIFEST="${SYN_DIR}/data/synthetic/manifest.csv"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${SYN_DIR}"

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")"
DATASET_FILE="${ROW%%,*}"
DATASET="data/synthetic/${DATASET_FILE}"
DATASET_NAME="${DATASET_FILE%.npz}"

METHODS=(linear bayesian)
ALPHAS=(prescribed_0 prescribed_1 prescribed_2 solved)
HARMONICS=(h1 h1_h2)

for METHOD in "${METHODS[@]}"; do
  SCRIPT="scripts/run_${METHOD}_solver_gpu.py"
  METHOD_DIR="${METHOD}_tile_gpu"
  for ALPHA_CONFIG in "${ALPHAS[@]}"; do
    for HARMONIC in "${HARMONICS[@]}"; do
      if [[ "${ALPHA_CONFIG}" == prescribed_* ]]; then
        VALUE="${ALPHA_CONFIG#prescribed_}"
        ALPHA_ARGS=(--alpha-mode prescribed --alpha "${VALUE}")
        CONFIG_NAME="alpha_prescribed_${VALUE}__${HARMONIC}"
      else
        ALPHA_ARGS=(--alpha-mode solved)
        CONFIG_NAME="alpha_solved__${HARMONIC}"
      fi
      SUMMARY="outputs/metrics/${METHOD_DIR}/${DATASET_NAME}/${CONFIG_NAME}/summary_metrics.json"
      COMMAND=(
        python "${SCRIPT}" "${DATASET}"
        "${ALPHA_ARGS[@]}"
        --harmonics "${HARMONIC}"
        --chunk-size "${CHUNK_SIZE}"
      )
      echo "${METHOD_DIR} ${CONFIG_NAME}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
      elif [[ "${SKIP_EXISTING}" == "1" && -s "${SUMMARY}" ]]; then
        echo "completed summary exists; skipping"
      else
        "${COMMAND[@]}"
      fi
    done
  done
done
