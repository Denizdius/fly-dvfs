#!/bin/bash
# Baselines first, then Fly DVFS with a 210s delay.
# Cases: batch 256 / 500 output tokens, then batch 256 / 2000 output tokens.
# Request count is 1024, same as the earlier batch-128 trial.
set -u
ROOT=/data/fly-dvfs
LOG="$ROOT/campaign_logs/b256_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/campaign_logs" "$ROOT/flydvfs-results"
exec > >(tee -a "$LOG") 2>&1

echo "log $LOG"
echo "campaign start $(date -Is)"
nvidia-smi --query-gpu=name,memory.used,clocks.sm,clocks.mem,power.draw --format=csv

set +u
source "$ROOT/TokenPowerBench/.venv/bin/activate"
set -u
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

RESULTS=()

reset_clocks() {
  nvidia-smi -rgc || true
}

run_baseline() {
  local batch="$1" out="$2" maxlen="$3"
  local tag="baseline_b${batch}_o${out}"
  echo "===== BASELINE ${tag} $(date -Is) ====="
  reset_clocks
  if "$ROOT/scripts/run_workload.sh" "$batch" "$out" "$tag" "$maxlen"; then
    RESULTS+=("OK ${tag}")
  else
    local rc=$?
    RESULTS+=("FAIL ${rc} ${tag}")
    echo "baseline ${tag} failed rc=${rc}"
  fi
}

run_dvfs() {
  local batch="$1" out="$2" maxlen="$3"
  local tag="dvfs_b${batch}_o${out}"
  echo "===== DVFS delay=210 ${tag} $(date -Is) ====="
  reset_clocks
  if python -m flydvfs \
      --delay 210 \
      --cap-window 1 \
      --output-dir "$ROOT/flydvfs-results/${tag}" \
      -- "$ROOT/scripts/run_workload.sh" "$batch" "$out" "$tag" "$maxlen"; then
    RESULTS+=("OK ${tag}")
  else
    local rc=$?
    RESULTS+=("FAIL ${rc} ${tag}")
    echo "dvfs ${tag} failed rc=${rc}"
  fi
  reset_clocks
}

run_baseline 256 500 2048
run_baseline 256 2000 4096
run_dvfs 256 500 2048
run_dvfs 256 2000 4096

echo "===== SUMMARY $(date -Is) ====="
printf '%s\n' "${RESULTS[@]}"
echo "campaign end $(date -Is)"
