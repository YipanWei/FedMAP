#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAINER="VPT_Ma"
DATASET="fedisic"
SEED="1"
GPU="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trainer)
      TRAINER="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage:
  bash Scripts/run_experiment.sh --trainer VPT_Ma --dataset fedisic --seed 1 --gpu 0 [extra federated_main.py args]

Defaults:
  --trainer VPT_Ma
  --dataset fedisic
  --seed 1
  --gpu 0

Extra arguments are passed through to federated_main.py.
USAGE
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$SCRIPT_DIR/output" "$SCRIPT_DIR/logs"

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" federated_main.py \
  --trainer "$TRAINER" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --root "$SCRIPT_DIR/data" \
  --semantic_root "$SCRIPT_DIR" \
  "${EXTRA_ARGS[@]}"
