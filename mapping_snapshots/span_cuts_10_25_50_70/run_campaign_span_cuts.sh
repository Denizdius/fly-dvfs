#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/fly-dvfs
source TokenPowerBench/.venv/bin/activate
export PYTHONPATH=/data/fly-dvfs

echo "Campaign started $(date -Is)"
echo "Plan: dummy alpaca, dummy longbench, dvfs run1, dvfs run2"
echo "Shapes: alpaca then longbench, batches 8 16 32 64 128 256, output_tokens=500, then batch 256 output_tokens=2000"
echo "Mapping: span_cuts_10_25_50_70"
echo "Fly DVFS delay=35 cap-window=1, cooldown=15s, GPU reset before every DVFS run"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "Previous baseline and two_level DVFS results are left in place"
echo "Results: trial_results/dvfs_span_cuts and flydvfs-results/span_cuts"

python3 - << 'PY'
from pathlib import Path
path = Path("/usr/local/lib/python3.12/dist-packages/transformers/dependency_versions_table.py")
text = path.read_text()
old = '"huggingface-hub": "huggingface-hub>=1.5.0,<2.0",'
new = '"huggingface-hub": "huggingface-hub>=1.5.0",'
if old in text:
    path.write_text(text.replace(old, new, 1))
    cache = path.parent / "__pycache__"
    for name in (
        "dependency_versions_table.cpython-312.pyc",
        "dependency_versions_check.cpython-312.pyc",
    ):
        file = cache / name
        if file.exists():
            file.unlink()
    print("Removed the huggingface-hub upper bound in this container")
else:
    print("huggingface-hub pin already allows 2.x")
PY

gpu_defaults() {
  echo "----------------------------------------------------------------"
  echo "Reset GPU to defaults before the next run"
  nvidia-smi -rgc || true
  nvidia-smi -rmc || true
  watts=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%d", $1 + 0.5}')
  nvidia-smi -pl "$watts"
  nvidia-smi -pm 1
  python - << 'PY'
from flydvfs.nvml import Nvml
with Nvml() as nvml:
    nvml.unlock_sm_clock(0)
print("SM clock unlocked")
PY
  nvidia-smi --query-gpu=persistence_mode,power.limit,power.default_limit,clocks.sm --format=csv
  echo "----------------------------------------------------------------"
}

cooldown() {
  echo "Cooldown 15 seconds $(date -Is)"
  sleep 15
}

run_bench() {
  local phase="$1"
  local run="$2"
  local name="$3"
  local script="$4"
  echo "############################################################"
  echo "START phase=$phase run=$run name=$name $(date -Is)"
  echo "script=TokenPowerBench/$script"
  echo "############################################################"
  OUTPUT_DIR="trial_results/$phase/$run/$name" bash "TokenPowerBench/$script"
  local code=$?
  echo "END phase=$phase run=$run name=$name exit=$code $(date -Is)"
  echo "$phase $run $name $code" >> /data/fly-dvfs/trial_results/campaign_status_span_cuts.txt
  cooldown
  return 0
}

run_dvfs() {
  local run="$1"
  local name="$2"
  local script="$3"
  echo "############################################################"
  echo "START phase=dvfs run=$run name=$name $(date -Is)"
  echo "script=TokenPowerBench/$script"
  echo "flydvfs delay=35 cap-window=1 mapping=span_cuts_10_25_50_70"
  echo "############################################################"
  gpu_defaults
  OUTPUT_DIR="trial_results/dvfs_span_cuts/$run/$name" \
    python -u -m flydvfs --delay 35 --cap-window 1 \
      --output-dir "flydvfs-results/span_cuts/$run/$name" -- \
      bash "TokenPowerBench/$script"
  local code=$?
  echo "END phase=dvfs run=$run name=$name exit=$code $(date -Is)"
  echo "dvfs $run $name $code" >> /data/fly-dvfs/trial_results/campaign_status_span_cuts.txt
  cooldown
  return 0
}

mkdir -p /data/fly-dvfs/trial_results
echo "Campaign status log: /data/fly-dvfs/trial_results/campaign_status_span_cuts.txt"
: > /data/fly-dvfs/trial_results/campaign_status_span_cuts.txt

jobs=(
  alpaca_b8_o500
  alpaca_b16_o500
  alpaca_b32_o500
  alpaca_b64_o500
  alpaca_b128_o500
  alpaca_b256_o500
  alpaca_b256_o2000
  longbench_b8_o500
  longbench_b16_o500
  longbench_b32_o500
  longbench_b64_o500
  longbench_b128_o500
  longbench_b256_o500
  longbench_b256_o2000
)

gpu_defaults

run_bench dummy span_cuts alpaca_b128_o500 alpaca_b128_o500.sh
run_bench dummy span_cuts longbench_b128_o500 longbench_b128_o500.sh

for run in run1 run2; do
  for name in "${jobs[@]}"; do
    run_dvfs "$run" "$name" "$name.sh"
  done
done

echo "############################################################"
echo "Campaign finished $(date -Is)"
echo "Exit codes:"
cat /data/fly-dvfs/trial_results/campaign_status_span_cuts.txt
echo "############################################################"
