#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/250010072/zlh/king/code_tailored_dataset
TASK_DIR=$ROOT/multi_lang_task
TASK_FILE=$TASK_DIR/multi_task_data.jsonl
REASONING_FILE=$TASK_DIR/multi_task_reasoning.jsonl

cd "$ROOT"

python3 "$TASK_DIR/gen_task.py" \
  --data_file "$ROOT/data/kodcode.jsonl" \
  --save_file "$TASK_FILE" \
  --ratio 0.15 \
  --workers 32 \
  --ip 10.120.0.102 \
  --port 35000

python3 "$TASK_DIR/gen_reasoning.py" \
  --data_file "$TASK_FILE" \
  --save_file "$REASONING_FILE" \
  --workers 32 \
  --ip 10.120.0.102 \
  --port 35000
