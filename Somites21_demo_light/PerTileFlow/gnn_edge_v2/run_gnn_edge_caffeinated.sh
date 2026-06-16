#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/renders/gnn_edge_dc_${STAMP}}"
CONFIG="${CONFIG:-../emb1/config.json}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP="${SWEEP:-1}"
K_VALUES=(${K_VALUES:-0 1 2 3 4})
HIDDEN_DIM_VALUES=(${HIDDEN_DIM_VALUES:-64 128})
LAMBDA_DELTA_VALUES=(${LAMBDA_DELTA_VALUES:-1e-4 1e-3})
SEEDS=(${SEEDS:-$SEED})
FLOW_COMPONENTS="${FLOW_COMPONENTS:-dc-h1-h2}"
LAMBDA_H1="${LAMBDA_H1:-1.0}"
LAMBDA_H2="${LAMBDA_H2:-1.0}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

LOG="$OUT_DIR/train.log"
echo "GNN edge DC run under caffeinate"
echo "Output: $OUT_DIR"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Python: $PYTHON_BIN"
echo "Sweep: $SWEEP"
echo "K values: ${K_VALUES[*]}"
echo "Hidden dims: ${HIDDEN_DIM_VALUES[*]}"
echo "lambda_delta values: ${LAMBDA_DELTA_VALUES[*]}"
echo "Flow components: $FLOW_COMPONENTS"
echo "lambda_h1: $LAMBDA_H1"
echo "lambda_h2: $LAMBDA_H2"
echo "Seeds: ${SEEDS[*]}"
echo "Log: $LOG"

cmd=(caffeinate -dimsu "$PYTHON_BIN" -X faulthandler gnn_edge_v2/train_gnn_edge.py
  --config "$CONFIG"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --out-dir "$OUT_DIR"
  --flow-components "$FLOW_COMPONENTS"
  --lambda-h1 "$LAMBDA_H1"
  --lambda-h2 "$LAMBDA_H2"
  "$@")

if [[ "$SWEEP" == "1" ]]; then
  cmd+=(--sweep)
  cmd+=(--K-values "${K_VALUES[@]}")
  cmd+=(--hidden-dim-values "${HIDDEN_DIM_VALUES[@]}")
  cmd+=(--lambda-delta-values "${LAMBDA_DELTA_VALUES[@]}")
  cmd+=(--seeds "${SEEDS[@]}")
fi

"${cmd[@]}" 2>&1 | tee "$LOG"
