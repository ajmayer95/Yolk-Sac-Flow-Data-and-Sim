#!/usr/bin/env bash
# Run every classical solver configuration over every generated dataset.
#
# One Slurm array task owns one dataset and runs all 32 configurations:
#   4 methods
#   4 alpha configurations: prescribed 0, prescribed 1, prescribed 2, solved
#   2 harmonic sets: H1, H1+H2
#
# Total campaign:
#   36 array jobs * 32 sequential configurations = 1,152 solver runs.
#
# This coarse job structure avoids cluster limits on array size and submitted
# jobs. Retries are safe because completed summary files are skipped by default.
#
# Review without running:
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 \
#     bash scripts/run_all_classical_solvers.sbatch
#
# Submit:
#   sbatch scripts/run_all_classical_solvers.sbatch

#SBATCH --job-name=dist_solvers
# No partition is specified; Slurm will use the cluster default.
# Add, for example, "#SBATCH --partition=cpu" if your site requires one.
#SBATCH --array=0-35%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/dist_solvers-%A_%a.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/dist_solvers-%A_%a.err

set -euo pipefail

PROJECT_ROOT="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
MANIFEST="${PROJECT_ROOT}/data/synthetic/manifest.csv"
CONFIG="${PROJECT_ROOT}/configs/solver_base.yaml"
CONDA_ROOT="/mnt/home/sswee/miniforge3"
CONDA_ENV="yolk-sac"

# Set SKIP_EXISTING=0 to deliberately replace completed configurations:
#   sbatch --export=ALL,SKIP_EXISTING=0 scripts/run_all_classical_solvers.sbatch
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

METHODS=(linear_tile linear_mosaic bayesian_tile bayesian_mosaic)
ALPHA_CONFIGURATIONS=(prescribed_0 prescribed_1 prescribed_2 solved)
HARMONIC_MODES=(h1 h1_h2)
EXPECTED_DATASETS=36
CONFIGURATIONS_PER_DATASET=32

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
  echo "Update --array and EXPECTED_DATASETS before submitting." >&2
  exit 2
fi

# manifest columns:
# file,D0_per_pa,alpha,R0_m,noise_level,seed,n_nodes,n_edges,heart_frequency_hz
MANIFEST_ROW="$(sed -n "$((DATASET_INDEX + 2))p" "${MANIFEST}")"
IFS=',' read -r DATASET_FILE TRUE_D0 TRUE_ALPHA R0 NOISE SEED \
  N_NODES N_EDGES HEART_FREQUENCY <<< "${MANIFEST_ROW}"

if [[ -z "${DATASET_FILE}" || -z "${TRUE_ALPHA}" ]]; then
  echo "ERROR: could not decode manifest row ${DATASET_INDEX}." >&2
  exit 2
fi

DATASET_PATH="${PROJECT_ROOT}/data/synthetic/${DATASET_FILE}"
DATASET_NAME="${DATASET_FILE%.npz}"
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/scripts/logs"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "============================================================"
echo "Slurm job:       ${SLURM_JOB_ID:-local}"
echo "Dataset task:    ${DATASET_INDEX}/$((EXPECTED_DATASETS - 1))"
echo "Host:            $(hostname)"
echo "Start time:      $(date --iso-8601=seconds)"
echo "Dataset:         ${DATASET_FILE}"
echo "Truth:           D0=${TRUE_D0}, alpha=${TRUE_ALPHA}, noise=${NOISE}"
echo "Configurations:  ${CONFIGURATIONS_PER_DATASET}"
echo "Skip existing:   ${SKIP_EXISTING}"
echo "============================================================"

COMPLETED=0
SKIPPED=0

for METHOD in "${METHODS[@]}"; do
  for ALPHA_CONFIGURATION in "${ALPHA_CONFIGURATIONS[@]}"; do
    for HARMONICS in "${HARMONIC_MODES[@]}"; do
      if [[ "${ALPHA_CONFIGURATION}" == prescribed_* ]]; then
        PRESCRIBED_ALPHA="${ALPHA_CONFIGURATION#prescribed_}"
        ALPHA_ARGS=(--alpha-mode prescribed --alpha "${PRESCRIBED_ALPHA}")
        ALPHA_CANONICAL="$(awk -v value="${PRESCRIBED_ALPHA}" 'BEGIN {printf "%g", value}')"
        ALPHA_LABEL="${ALPHA_CANONICAL//-/m}"
        ALPHA_LABEL="${ALPHA_LABEL//./p}"
        CONFIGURATION_NAME="alpha_prescribed_${ALPHA_LABEL}__${HARMONICS}"
        if awk -v prescribed="${PRESCRIBED_ALPHA}" -v truth="${TRUE_ALPHA}" \
          'BEGIN {exit !(prescribed == truth)}'; then
          ALPHA_MATCH="matched"
        else
          ALPHA_MATCH="mismatched"
        fi
      else
        ALPHA_ARGS=(--alpha-mode solved)
        CONFIGURATION_NAME="alpha_solved__${HARMONICS}"
        ALPHA_MATCH="solved"
      fi

      SUMMARY_PATH="${PROJECT_ROOT}/outputs/metrics/${METHOD}/${DATASET_NAME}/${CONFIGURATION_NAME}/summary_metrics.json"
      COMMAND=(
        python scripts/run_solver.py
        "${DATASET_PATH}"
        --config "${CONFIG}"
        --method "${METHOD}"
        "${ALPHA_ARGS[@]}"
        --harmonics "${HARMONICS}"
      )

      echo
      echo "------------------------------------------------------------"
      echo "Configuration:   $((COMPLETED + SKIPPED + 1))/${CONFIGURATIONS_PER_DATASET}"
      echo "Method:          ${METHOD}"
      echo "Alpha config:    ${ALPHA_CONFIGURATION} (${ALPHA_MATCH})"
      echo "Harmonics:       ${HARMONICS}"
      echo "Output:          ${SUMMARY_PATH}"
      printf 'Command:         '
      printf '%q ' "${COMMAND[@]}"
      printf '\n'

      if [[ "${DRY_RUN}" == "1" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      if [[ "${SKIP_EXISTING}" == "1" && -s "${SUMMARY_PATH}" ]]; then
        echo "Completed summary exists; skipping."
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

echo
echo "============================================================"
echo "End time:        $(date --iso-8601=seconds)"
echo "Dataset:         ${DATASET_FILE}"
echo "Completed now:   ${COMPLETED}"
echo "Skipped:         ${SKIPPED}"
echo "============================================================"
