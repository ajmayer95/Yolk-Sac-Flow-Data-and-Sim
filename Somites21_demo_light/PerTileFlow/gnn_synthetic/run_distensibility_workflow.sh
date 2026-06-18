#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-$PROJECT_ROOT/../emb1/config.json}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-mps}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR}"
PRIMARY_RUN="${PRIMARY_RUN:-masked_edge_validation_15pct}"

POSITIONAL_D_VALUES=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --d-values)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "ERROR: --d-values requires a quoted list, e.g. --d-values \"1e-3 3.16e-4\"" >&2
        exit 2
      fi
      # shellcheck disable=SC2206
      POSITIONAL_D_VALUES+=($1)
      ;;
    --device)
      shift
      DEVICE="${1:?ERROR: --device requires auto, cpu, or mps}"
      ;;
    --epochs)
      shift
      EPOCHS="${1:?ERROR: --epochs requires a value}"
      ;;
    --seed)
      shift
      SEED="${1:?ERROR: --seed requires a value}"
      ;;
    --python-bin)
      shift
      PYTHON_BIN="${1:?ERROR: --python-bin requires a value}"
      ;;
    --config)
      shift
      CONFIG="${1:?ERROR: --config requires a path}"
      ;;
    --results-root)
      shift
      RESULTS_ROOT="${1:?ERROR: --results-root requires a path}"
      ;;
    -h|--help)
      D_VALUES=()
      ;;
    --)
      shift
      POSITIONAL_D_VALUES+=("$@")
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      exit 2
      ;;
    *)
      POSITIONAL_D_VALUES+=("$1")
      ;;
  esac
  shift || true
done

if [[ "${#POSITIONAL_D_VALUES[@]}" -gt 0 ]]; then
  D_VALUES=("${POSITIONAL_D_VALUES[@]}")
elif [[ -n "${D_VALUES:-}" ]]; then
  # shellcheck disable=SC2206
  D_VALUES=(${D_VALUES})
else
  cat >&2 <<EOF
Usage: $(basename "$0") D_VALUE [D_VALUE ...]

Example:
  $(basename "$0") 1e-3 3.16e-4
  $(basename "$0") --d-values "1e-3 3.16e-4 1e-2"
  $(basename "$0") --device mps --epochs 100 1e-3

Useful environment overrides:
  PYTHON_BIN=python
  CONFIG=$CONFIG
  EPOCHS=$EPOCHS
  SEED=$SEED
  DEVICE=$DEVICE
  RESULTS_ROOT=$RESULTS_ROOT
  MOSAIC_EXTRA_ARGS="--tiles 22"
  TRAIN_EXTRA_ARGS="--no-tqdm"
  DASHBOARD_EXTRA_ARGS=""
  INFER_EXTRA_ARGS=""
EOF
  exit 2
fi

split_extra_args() {
  local var_name="$1"
  local value="${!var_name:-}"
  if [[ -n "$value" ]]; then
    # Intentional shell-style word splitting for simple extra CLI flags.
    # Use paths without spaces for these override strings.
    # shellcheck disable=SC2206
    EXTRA_ARGS=($value)
  else
    EXTRA_ARGS=()
  fi
}

d_label() {
  "$PYTHON_BIN" - "$1" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

raw = sys.argv[1].strip()
try:
    value = Decimal(raw)
except InvalidOperation as exc:
    raise SystemExit(f"Invalid distensibility value {raw!r}: {exc}")

prefix = "dm" if value < 0 else "d"
value = abs(value)
plain = format(value.normalize(), "f")
if "." in plain:
    plain = plain.rstrip("0").rstrip(".")
if plain == "":
    plain = "0"
print(prefix + plain.replace(".", "p"))
PY
}

run_stage() {
  local label="$1"
  local log_file="$2"
  shift 2
  echo
  echo "=== $label ==="
  echo "Log: $log_file"
  mkdir -p "$(dirname "$log_file")"
  "$@" 2>&1 | tee "$log_file"
}

echo "Distensibility workflow"
echo "Config: $CONFIG"
echo "Python: $PYTHON_BIN"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Results root: $RESULTS_ROOT"
echo "D values: ${D_VALUES[*]}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

for D in "${D_VALUES[@]}"; do
  LABEL="$(d_label "$D")"
  RESULTS_DIR="$RESULTS_ROOT/${LABEL}_results"
  LOG_DIR="$RESULTS_DIR/logs"
  SYN_GRAPH="$RESULTS_DIR/synthetic_mosaic_graph.gpickle"
  GNN_DIR="$RESULTS_DIR/gnn_edge_dc"
  VALIDATED_GNN_DIR="$GNN_DIR/$PRIMARY_RUN"
  INFER_DIR="$RESULTS_DIR/infer_validated_gnn_tile_profiles"

  mkdir -p "$RESULTS_DIR" "$LOG_DIR"

  echo
  echo "############################################################"
  echo "# D=$D -> $RESULTS_DIR"
  echo "############################################################"

  split_extra_args MOSAIC_EXTRA_ARGS
  MOSAIC_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  run_stage "simulate/profile mosaic D=$D" "$LOG_DIR/01_default_mosaic_tile_profiles.log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/default_mosaic_tile_profiles.py" \
      --config "$CONFIG" \
      --D-mosaic "$D" \
      --out-dir "$RESULTS_DIR" \
      ${MOSAIC_ARGS[@]+"${MOSAIC_ARGS[@]}"}

  split_extra_args TRAIN_EXTRA_ARGS
  TRAIN_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  run_stage "train GNN D=$D" "$LOG_DIR/02_train_gnn_edge.log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/train_gnn_edge.py" \
      --config "$CONFIG" \
      --graph "$SYN_GRAPH" \
      --synthetic-input "$SYN_GRAPH" \
      --out-dir "$GNN_DIR" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --device "$DEVICE" \
      ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}

  split_extra_args DASHBOARD_EXTRA_ARGS
  DASHBOARD_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  run_stage "build GNN dashboard D=$D" "$LOG_DIR/03_gnn_edge_dashboard.log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/gnn_edge_dashboard.py" \
      --results-dir "$GNN_DIR" \
      --primary-run "$PRIMARY_RUN" \
      ${DASHBOARD_ARGS[@]+"${DASHBOARD_ARGS[@]}"}

  split_extra_args INFER_EXTRA_ARGS
  INFER_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  run_stage "infer tile distensibility D=$D" "$LOG_DIR/04_infer_default_mosaic_tile_profiles.log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/infer_default_mosaic_tile_profiles.py" \
      --config "$CONFIG" \
      --graph "$SYN_GRAPH" \
      --pressure-prior-dir "$VALIDATED_GNN_DIR" \
      --pressure-prior-mode scaled \
      --out-dir "$INFER_DIR" \
      ${INFER_ARGS[@]+"${INFER_ARGS[@]}"}

  echo
  echo "Completed D=$D"
  echo "Results: $RESULTS_DIR"
done
