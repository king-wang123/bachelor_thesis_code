#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/250010072/zlh/king/code_tailored_dataset
LOG_DIR=$ROOT/train/sft/log
mkdir -p "$LOG_DIR"

cd "$ROOT"

bash train/sft/train.sh 1.5B 0,1,2,3 > "$LOG_DIR/qwen25_coder_1_5b_sft.log" 2>&1
bash train/sft/train.sh 3B 0,1,2,3 > "$LOG_DIR/qwen25_coder_3b_sft.log" 2>&1

conda activate qwen3-next
nohup env CUDA_VISIBLE_DEVICES="0,1,2,3" python -m vllm.entrypoints.openai.api_server --served-model-name default --model="/data/share/Qwen3-Coder-Next" --trust-remote-code --tensor-parallel-size=4 --port="35000" > /data/250010072/zlh/king/code_tailored_dataset/log/qwen0.log 2>&1 &