#!/usr/bin/env bash
# Run whole-mosaic classical solvers on GPU using two selected neural pressure
# priors.
#
# One array task owns one synthetic dataset. It selects:
#   1. the best physics-informed GNN pressure field
#   2. the best K=0 edge-local MLP pressure field
# using validation DC relative RMSE, then sequentially runs:
#   2 whole-mosaic GPU methods
#   4 alpha configurations: prescribed 0/1/2 and solved
#   2 harmonic configurations: H1 and H1+H2
#
# Current neural artifacts contain DC pressure only. The default "scaled"
# conditioning uses the DC spatial shape as a prior for each complex harmonic
# boundary pressure while still fitting harmonic amplitude and phase.
#
# Dry run:
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 \
#     bash scripts/run_all_gnn_conditioned_solvers.sbatch
#
# Submit:
#   sbatch scripts/run_all_gnn_conditioned_solvers.sbatch

#SBATCH --job-name=gnn_dist_mosaic_gpu
#SBATCH --partition=gpu
#SBATCH --array=0-35%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/gnn_dist_mosaic_gpu-%A_%a.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/gnn_dist_mosaic_gpu-%A_%a.err

set -euo pipefail

PROJECT_ROOT="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
MANIFEST="${PROJECT_ROOT}/data/synthetic/manifest.csv"
CONFIG="${PROJECT_ROOT}/configs/solver_base.yaml"
GNN_ROOT="${PROJECT_ROOT}/outputs/runs/gnn"
CONDA_ROOT="/mnt/home/sswee/miniforge3"
CONDA_ENV="yolk-sac"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
PRESSURE_MODE="${PRESSURE_MODE:-scaled}"
PRESSURE_WEIGHT="${PRESSURE_WEIGHT:-1.0}"
PRESSURE_SIGMA_PA="${PRESSURE_SIGMA_PA:-0.0}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
# Space-separated model families that produce finite pressure fields.
PRESSURE_MODELS="${PRESSURE_MODELS:-physics_informed_gnn edge_local_mlp}"
SELECTOR="${PROJECT_ROOT}/scripts/select_best_gnn_pressure_runs.py"

METHODS=(linear bayesian)
ALPHA_CONFIGURATIONS=(prescribed_0 prescribed_1 prescribed_2 solved)
HARMONIC_MODES=(h1 h1_h2)
EXPECTED_DATASETS=36

DATASET_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( DATASET_INDEX < 0 || DATASET_INDEX >= EXPECTED_DATASETS )); then
  echo "ERROR: dataset index ${DATASET_INDEX} is outside 0-$((EXPECTED_DATASETS - 1))." >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: dataset manifest not found: ${MANIFEST}" >&2
  exit 2
fi

N_DATASETS=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( N_DATASETS != EXPECTED_DATASETS )); then
  echo "ERROR: expected ${EXPECTED_DATASETS} datasets, found ${N_DATASETS}." >&2
  exit 2
fi

MANIFEST_ROW="$(sed -n "$((DATASET_INDEX + 2))p" "${MANIFEST}")"
IFS=',' read -r DATASET_FILE TRUE_D0 TRUE_ALPHA R0 NOISE SEED \
  N_NODES N_EDGES HEART_FREQUENCY <<< "${MANIFEST_ROW}"
DATASET_PATH="${PROJECT_ROOT}/data/synthetic/${DATASET_FILE}"
DATASET_NAME="${DATASET_FILE%.npz}"
GNN_DATASET_DIR="${GNN_ROOT}/${DATASET_NAME}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ ! -d "${GNN_DATASET_DIR}" ]]; then
  echo "ERROR: GNN output directory not found: ${GNN_DATASET_DIR}" >&2
  exit 2
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/scripts/logs"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

read -r -a PRESSURE_MODEL_ARRAY <<< "${PRESSURE_MODELS}"
BEST_RUN_TABLE="$(
  python "${SELECTOR}" "${GNN_DATASET_DIR}" \
    --models "${PRESSURE_MODEL_ARRAY[@]}" \
    --group-by model \
    --metric splits.validation.dc_relative_rmse
)"
PRESSURE_RUNS=()
PRESSURE_SCORES=()
PRESSURE_CONFIGURATIONS=()
while IFS=$'\t' read -r RUN_PATH SCORE MODEL HARMONIC_MODE; do
  [[ -z "${RUN_PATH}" ]] && continue
  PRESSURE_RUNS+=("${RUN_PATH}")
  PRESSURE_SCORES+=("${SCORE}")
  PRESSURE_CONFIGURATIONS+=("${MODEL}/${HARMONIC_MODE}")
done <<< "${BEST_RUN_TABLE}"

if (( ${#PRESSURE_RUNS[@]} == 0 )); then
  echo "ERROR: no pressure-producing GNN runs found under ${GNN_DATASET_DIR}" >&2
  exit 2
fi

CONFIGURATIONS_PER_PRESSURE_RUN=16
TOTAL_CONFIGURATIONS=$(( ${#PRESSURE_RUNS[@]} * CONFIGURATIONS_PER_PRESSURE_RUN ))
echo "============================================================"
echo "Slurm job:          ${SLURM_JOB_ID:-local}"
echo "Dataset task:       ${DATASET_INDEX}/$((EXPECTED_DATASETS - 1))"
echo "Host:               $(hostname)"
echo "Start time:         $(date --iso-8601=seconds)"
echo "Dataset:            ${DATASET_FILE}"
echo "Truth:              D0=${TRUE_D0}, alpha=${TRUE_ALPHA}, noise=${NOISE}"
echo "Pressure runs:      ${#PRESSURE_RUNS[@]}"
for INDEX in "${!PRESSURE_RUNS[@]}"; do
  echo "  selected:         ${PRESSURE_CONFIGURATIONS[$INDEX]} validation_dc_rmse=${PRESSURE_SCORES[$INDEX]} run=$(basename "${PRESSURE_RUNS[$INDEX]}")"
done
echo "Solver runs:        ${TOTAL_CONFIGURATIONS}"
echo "Pressure mode:      ${PRESSURE_MODE}"
echo "Pressure weight:    ${PRESSURE_WEIGHT}"
echo "Pressure sigma Pa:  ${PRESSURE_SIGMA_PA}"
echo "Chunk size:         ${CHUNK_SIZE}"
echo "============================================================"

COMPLETED=0
SKIPPED=0
ATTEMPTED=0

for PRESSURE_INDEX in "${!PRESSURE_RUNS[@]}"; do
  PRESSURE_RUN="${PRESSURE_RUNS[$PRESSURE_INDEX]}"
  PRESSURE_NAME="$(basename "${PRESSURE_RUN}")"
  SAFE_PRESSURE_NAME="$(echo "${PRESSURE_NAME}" | tr -c '[:alnum:]_-' '_')"
  SAFE_PRESSURE_NAME="${SAFE_PRESSURE_NAME%_}"
  echo
  echo "Pressure source: ${PRESSURE_RUN}"
  echo "Selected group:  ${PRESSURE_CONFIGURATIONS[$PRESSURE_INDEX]}"
  echo "Validation DC RMSE: ${PRESSURE_SCORES[$PRESSURE_INDEX]}"

  for METHOD in "${METHODS[@]}"; do
    SCRIPT="scripts/run_${METHOD}_solver_gpu.py"
    for ALPHA_CONFIGURATION in "${ALPHA_CONFIGURATIONS[@]}"; do
      for HARMONICS in "${HARMONIC_MODES[@]}"; do
        ATTEMPTED=$((ATTEMPTED + 1))
        if [[ "${ALPHA_CONFIGURATION}" == prescribed_* ]]; then
          PRESCRIBED_ALPHA="${ALPHA_CONFIGURATION#prescribed_}"
          ALPHA_ARGS=(--alpha-mode prescribed --alpha "${PRESCRIBED_ALPHA}")
          ALPHA_LABEL="${PRESCRIBED_ALPHA//-/m}"
          ALPHA_LABEL="${ALPHA_LABEL//./p}"
          BASE_CONFIGURATION="alpha_prescribed_${ALPHA_LABEL}__${HARMONICS}"
        else
          ALPHA_ARGS=(--alpha-mode solved)
          BASE_CONFIGURATION="alpha_solved__${HARMONICS}"
        fi

        CONFIGURATION_NAME="${BASE_CONFIGURATION}__pressure_${PRESSURE_MODE}__${SAFE_PRESSURE_NAME}"
        METHOD_DIR="${METHOD}_mosaic_gpu"
        SUMMARY_PATH="${PROJECT_ROOT}/outputs/metrics/${METHOD_DIR}/${DATASET_NAME}/${CONFIGURATION_NAME}/summary_metrics.json"
        COMMAND=(
          python "${SCRIPT}"
          "${DATASET_PATH}"
          --config "${CONFIG}"
          --spatial-mode whole_mosaic
          "${ALPHA_ARGS[@]}"
          --harmonics "${HARMONICS}"
          --chunk-size "${CHUNK_SIZE}"
          --pressure-field "${PRESSURE_RUN}"
          --pressure-mode "${PRESSURE_MODE}"
          --pressure-weight "${PRESSURE_WEIGHT}"
          --pressure-sigma-pa "${PRESSURE_SIGMA_PA}"
        )

        echo "[$ATTEMPTED/${TOTAL_CONFIGURATIONS}] ${METHOD_DIR} ${ALPHA_CONFIGURATION} ${HARMONICS} ${PRESSURE_NAME}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '  '
          printf '%q ' "${COMMAND[@]}"
          printf '\n'
          SKIPPED=$((SKIPPED + 1))
          continue
        fi
        if [[ "${SKIP_EXISTING}" == "1" && -s "${SUMMARY_PATH}" ]]; then
          echo "  completed summary exists; skipping"
          SKIPPED=$((SKIPPED + 1))
          continue
        fi

        "${COMMAND[@]}"
        if [[ ! -s "${SUMMARY_PATH}" ]]; then
          echo "ERROR: solver exited without creating ${SUMMARY_PATH}" >&2
          exit 1
        fi
        COMPLETED=$((COMPLETED + 1))
      done
    done
  done
done

echo
echo "============================================================"
echo "End time:           $(date --iso-8601=seconds)"
echo "Dataset:            ${DATASET_FILE}"
echo "Completed now:      ${COMPLETED}"
echo "Skipped/dry-run:    ${SKIPPED}"
echo "============================================================"
