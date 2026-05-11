#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/250010072/zlh/king/code_tailored_dataset
RL=$ROOT/train/rl
SFT=$ROOT/train/sft/models
LOG=$RL/log/pipeline_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$RL/log"

run_one() {
  local size=$1
  local base=$2
  local tag=$3
  local served=$4
  local model_path="$SFT/${base}-reasoning-sft"

  echo "waiting for $model_path" | tee -a "$LOG"
  while [[ ! -f "$model_path/config.json" ]]; do
    sleep 300
  done

  bash "$RL/stop_vllm.sh" || true
  local vllm_log
  vllm_log=$(bash "$RL/serve_vllm_4ports.sh" "$model_path" 35000 "$served" 0,1,2,3)
  echo "vllm logs: $vllm_log" | tee -a "$LOG"
  sleep 180

  bash "$RL/run_filter_all.sh" "$tag" "$served" 2>&1 | tee -a "$LOG"

  bash "$RL/stop_vllm.sh" || true
  bash "$RL/train_dapo.sh" "$size" 0,1,2,3 2>&1 | tee -a "$LOG"
}

run_one 1.5B Qwen2.5-Coder-1.5B-Instruct qwen25_coder_1_5b qwen25-coder-1.5b-sft
run_one 3B Qwen2.5-Coder-3B-Instruct qwen25_coder_3b qwen25-coder-3b-sft
