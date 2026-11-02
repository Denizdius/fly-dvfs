#!/bin/bash
# One TokenPowerBench case.
# Usage: run_workload.sh BATCH OUTPUT_TOKENS TAG MAX_MODEL_LEN
# TAG is the result folder name under trial_results/.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BATCH OUTPUT_TOKENS TAG MAX_MODEL_LEN" >&2
  exit 2
fi

BATCH="$1"
OUTPUT_TOKENS="$2"
TAG="$3"
MAX_MODEL_LEN="$4"
ROOT=/data/fly-dvfs

set +u
source "$ROOT/TokenPowerBench/.venv/bin/activate"
set -u

cd "$ROOT"
echo "[workload] tag=${TAG} batch=${BATCH} output_tokens=${OUTPUT_TOKENS} max_model_len=${MAX_MODEL_LEN} num_samples=1024"
exec tokenpowerbench \
  --model "$ROOT/TokenPowerBench/Qwen3.5-27B" \
  --dataset alpaca \
  --num-samples 1024 \
  --batch-sizes "$BATCH" \
  --output-tokens "$OUTPUT_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --seed 42 \
  --language-model-only \
  --monitor gpu_only \
  --output-dir "$ROOT/trial_results/$TAG"
