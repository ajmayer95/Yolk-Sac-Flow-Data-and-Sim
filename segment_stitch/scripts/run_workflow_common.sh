#!/usr/bin/env bash

detect_labels() {
  local root="$1"
  local analyzed="${root}/emb1/analyzed"
  local names=(
    mosaic_segmentation.tif
    mosaic_segmentation_labels.tif
    stitched_segmentation.tif
    stitched_labels.tif
    segmentation_labels.tif
    labels.tif
    mask.tif
    mosaic_mask.tif
    stitched_linear.tif
  )
  local name
  for name in "${names[@]}"; do
    if [[ -f "${analyzed}/${name}" ]]; then
      printf '%s\n' "${analyzed}/${name}"
      return 0
    fi
  done
  return 1
}

is_placeholder_path() {
  [[ "$1" == /path/to/* || "$1" == *"<"* || "$1" == *">"* ]]
}

if [[ -n "${TRAIN_LABELS}" ]] && is_placeholder_path "${TRAIN_LABELS}"; then
  echo "Ignoring placeholder TRAIN_LABELS=${TRAIN_LABELS}" >&2
  TRAIN_LABELS=""
fi
if [[ -n "${TEST_LABELS}" ]] && is_placeholder_path "${TEST_LABELS}"; then
  echo "Ignoring placeholder TEST_LABELS=${TEST_LABELS}" >&2
  TEST_LABELS=""
fi
if [[ -z "${TRAIN_LABELS}" ]]; then
  TRAIN_LABELS="$(detect_labels "${TRAIN_ROOT}" || true)"
fi
if [[ -z "${TEST_LABELS}" ]]; then
  TEST_LABELS="$(detect_labels "${TEST_ROOT}" || true)"
fi
if [[ -z "${TRAIN_LABELS}" || -z "${TEST_LABELS}" ]]; then
  echo "ERROR: supervised workflow requires whole-mosaic label TIFFs." >&2
  echo "Set TRAIN_LABELS and TEST_LABELS, or place common label filenames in emb1/analyzed/." >&2
  echo "TRAIN_LABELS=${TRAIN_LABELS:-missing}" >&2
  echo "TEST_LABELS=${TEST_LABELS:-missing}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_LABELS}" || ! -f "${TEST_LABELS}" ]]; then
  echo "ERROR: mosaic label path does not exist." >&2
  echo "TRAIN_LABELS=${TRAIN_LABELS}" >&2
  echo "TEST_LABELS=${TEST_LABELS}" >&2
  exit 2
fi
if [[ "$(basename "${TRAIN_LABELS}")" == "stitched_linear.tif" || "$(basename "${TEST_LABELS}")" == "stitched_linear.tif" ]]; then
  echo "WARNING: using stitched_linear.tif as the full-mosaic label source." >&2
fi

mkdir -p "${WORK_DIR}" "${REPORT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONHASHSEED="${SEED}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

echo "Project: ${PROJECT_DIR}"
echo "Train root: ${TRAIN_ROOT}"
echo "Test root: ${TEST_ROOT}"
echo "Train labels: ${TRAIN_LABELS}"
echo "Test labels: ${TEST_LABELS}"
echo "Work dir: ${WORK_DIR}"
echo "Report dir: ${REPORT_DIR}"
echo "Seed: ${SEED}"

python -m segment_stitch.run_workflow \
  --train-root "${TRAIN_ROOT}" \
  --test-root "${TEST_ROOT}" \
  --train-labels "${TRAIN_LABELS}" \
  --test-labels "${TEST_LABELS}" \
  --output-dir "${WORK_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  "${OVERWRITE_ARGS[@]}"

python "${PROJECT_DIR}/segment_stitch/scripts/generate_qc_report.py" \
  --output-dir "${WORK_DIR}" \
  --report-dir "${REPORT_DIR}" \
  --dataset "$(basename "${TEST_ROOT}" _demo)" \
  --reference-mosaic "${TEST_LABELS}" \
  --overwrite-metrics
