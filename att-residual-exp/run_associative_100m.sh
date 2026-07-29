#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for seed in 1337 2027 3407; do
  run_name="full_100m_seed${seed}"
  run_dir="att-residual-exp/runs/associative_read_depth_kda/${run_name}"
  summary_path="${run_dir}/summary.json"
  checkpoint_path="${run_dir}/checkpoints/latest.pt"

  if [[ -f "$summary_path" ]]; then
    echo "skip complete run: ${run_name}"
    continue
  fi

  resume_args=()
  if [[ -f "$checkpoint_path" ]]; then
    echo "resume run: ${run_name}"
    resume_args=(--resume "$checkpoint_path")
  elif [[ -e "$run_dir" ]]; then
    echo "run directory exists without a resumable checkpoint: ${run_dir}" >&2
    exit 1
  else
    echo "start run: ${run_name}"
  fi

  python -u att-residual-exp/associative_read_depth_kda.py \
    --mode full \
    --dataset parameter_golf_sp1024 \
    --precision bf16 \
    --seed "$seed" \
    --target-train-tokens 100000000 \
    --checkpoint-interval 500 \
    --run-name "$run_name" \
    "${resume_args[@]}"
done

echo "all associative-reader 100M runs complete"
