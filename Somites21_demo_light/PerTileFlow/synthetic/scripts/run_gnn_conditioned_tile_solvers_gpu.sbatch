#!/bin/bash
# GPU tile solvers conditioned on two selected DC pressure profiles:
#   1. best physics-informed GNN
#   2. K=0 edge-local MLP
#
# Selection minimizes validation DC relative RMSE across K/harmonic modes.
# One array task owns one dataset and runs both pressure sources through:
#   linear + Bayesian
#   prescribed alpha 0/1/2 + solved alpha
#   H1 + H1/H2
# for 32 runs per dataset.
#
# Submit:
#   sbatch scripts/run_gnn_conditioned_tile_solvers_gpu.sbatch

#SBATCH --job-name=gnn_tile_gpu
#SBATCH --partition=gpu
#SBATCH --array=0-35%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/gnn_tile_gpu-%A_%a.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/gnn_tile_gpu-%A_%a.err

set -euo pipefail

SYN_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
MANIFEST="${SYN_DIR}/data/synthetic/manifest.csv"
SELECTOR="${SYN_DIR}/scripts/select_best_gnn_pressure_runs.py"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
PRESSURE_MODE="${PRESSURE_MODE:-scaled}"
PRESSURE_WEIGHT="${PRESSURE_WEIGHT:-1.0}"
PRESSURE_SIGMA_PA="${PRESSURE_SIGMA_PA:-0.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${SYN_DIR}"

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")"
DATASET_FILE="${ROW%%,*}"
DATASET_NAME="${DATASET_FILE%.npz}"
DATASET="data/synthetic/${DATASET_FILE}"
GNN_DIR="outputs/runs/gnn/${DATASET_NAME}"

BEST_RUNS="$(
  python "${SELECTOR}" "${GNN_DIR}" \
    --models physics_informed_gnn edge_local_mlp \
    --group-by model \
    --metric splits.validation.dc_relative_rmse
)"

METHODS=(linear bayesian)
ALPHAS=(prescribed_0 prescribed_1 prescribed_2 solved)
HARMONICS=(h1 h1_h2)

while IFS=$'\t' read -r PRESSURE_RUN SCORE MODEL SOURCE_HARMONICS; do
  [[ -z "${PRESSURE_RUN}" ]] && continue
  PRESSURE_NAME="$(basename "${PRESSURE_RUN}")"
  SAFE_NAME="$(echo "${PRESSURE_NAME}" | tr -c '[:alnum:]_-' '_')"
  SAFE_NAME="${SAFE_NAME%_}"
  echo "Pressure source: ${MODEL} ${PRESSURE_NAME} validation_dc_rmse=${SCORE}"
  for METHOD in "${METHODS[@]}"; do
    METHOD_DIR="${METHOD}_tile_gpu"
    SCRIPT="scripts/run_${METHOD}_solver_gpu.py"
    for ALPHA_CONFIG in "${ALPHAS[@]}"; do
      for HARMONIC in "${HARMONICS[@]}"; do
        if [[ "${ALPHA_CONFIG}" == prescribed_* ]]; then
          VALUE="${ALPHA_CONFIG#prescribed_}"
          ALPHA_ARGS=(--alpha-mode prescribed --alpha "${VALUE}")
          BASE_NAME="alpha_prescribed_${VALUE}__${HARMONIC}"
        else
          ALPHA_ARGS=(--alpha-mode solved)
          BASE_NAME="alpha_solved__${HARMONIC}"
        fi
        CONFIG_NAME="${BASE_NAME}__pressure_${PRESSURE_MODE}__${SAFE_NAME}"
        SUMMARY="outputs/metrics/${METHOD_DIR}/${DATASET_NAME}/${CONFIG_NAME}/summary_metrics.json"
        COMMAND=(
          python "${SCRIPT}" "${DATASET}"
          "${ALPHA_ARGS[@]}"
          --harmonics "${HARMONIC}"
          --chunk-size "${CHUNK_SIZE}"
          --pressure-field "${PRESSURE_RUN}"
          --pressure-mode "${PRESSURE_MODE}"
          --pressure-weight "${PRESSURE_WEIGHT}"
          --pressure-sigma-pa "${PRESSURE_SIGMA_PA}"
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
done <<< "${BEST_RUNS}"
