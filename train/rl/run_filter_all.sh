#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/250010072/zlh/king/code_tailored_dataset
RL=$ROOT/train/rl
ENV=/data/250010072/zlh/conda_envs/verl
DATA=${DATA:-$ROOT/data/kodcode_reasoning_sft.jsonl}
HOST=${HOST:-http://127.0.0.1}
BASE_PORT=${BASE_PORT:-35000}
WORKERS=${WORKERS:-128}
SAMPLE_TIMES=${SAMPLE_TIMES:-10}
MAX_ITEMS=${MAX_ITEMS:-0}

MODEL_TAG=${1:?model tag required, e.g. qwen25_coder_1_5b}
SERVED_NAME=${2:?served model name required}

export PATH="$ENV/bin:$PATH"

ENDPOINTS="$HOST:$BASE_PORT,$HOST:$((BASE_PORT+1)),$HOST:$((BASE_PORT+2)),$HOST:$((BASE_PORT+3))"

"$ENV/bin/python" "$RL/filter_dapo_data.py" \
  --input "$DATA" \
  --out_dir "$RL/data" \
  --model "$SERVED_NAME" \
  --model_tag "$MODEL_TAG" \
  --endpoints "$ENDPOINTS" \
  --sample_times "$SAMPLE_TIMES" \
  --workers "$WORKERS" \
  --max_items "$MAX_ITEMS"
