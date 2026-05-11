#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?model path required}
BASE_PORT=${2:-35000}
MODEL_NAME=${3:-reasoning-sft}
GPUS=${4:-0,1,2,3}

ROOT=/data/250010072/zlh/king/code_tailored_dataset
ENV=/data/250010072/zlh/conda_envs/verl
LOG_DIR=$ROOT/train/rl/log/vllm_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
export PATH="$ENV/bin:$PATH"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
for idx in "${!GPU_LIST[@]}"; do
  gpu=${GPU_LIST[$idx]}
  port=$((BASE_PORT + idx))
  echo "starting $MODEL_NAME on gpu=$gpu port=$port"
  CUDA_VISIBLE_DEVICES=$gpu \
  nohup "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "$port" \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12288 \
    > "$LOG_DIR/port_${port}.log" 2>&1 &
  echo $! > "$LOG_DIR/port_${port}.pid"
done

echo "$LOG_DIR"
