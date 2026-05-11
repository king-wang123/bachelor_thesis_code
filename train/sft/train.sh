#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE=${1:-1.5B}
GPUS=${2:-0,1,2,3}
MAX_SAMPLES=${3:-}

ROOT=/data/250010072/zlh/king/code_tailored_dataset
LLAMA_FACTORY=/data/250010072/zlh/king/LLaMA-Factory
CONDA=/data/250010072/zlh/miniconda3
SFT_ENV=/data/250010072/zlh/conda_envs/sft
DATASET=kodcode_reasoning_sft
DATASET_DIR=$ROOT/data
MODEL_ROOT=/data/share
OUT_ROOT=$ROOT/train/sft/models

if [[ "$MODEL_SIZE" == "1.5B" ]]; then
  MODEL_NAME=Qwen2.5-Coder-1.5B-Instruct
elif [[ "$MODEL_SIZE" == "3B" ]]; then
  MODEL_NAME=Qwen2.5-Coder-3B-Instruct
else
  echo "MODEL_SIZE must be 1.5B or 3B"
  exit 1
fi

MODEL_PATH=$MODEL_ROOT/$MODEL_NAME
OUTPUT_DIR=$OUT_ROOT/${MODEL_NAME}-reasoning-sft

EXTRA_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
  OUTPUT_DIR=${OUTPUT_DIR}-smoke
fi

mkdir -p "$OUT_ROOT"
cd "$LLAMA_FACTORY"

source "$CONDA/etc/profile.d/conda.sh"
conda activate "$SFT_ENV"

export PATH="$SFT_ENV/bin:$PATH"
export PYTHONPATH="$LLAMA_FACTORY/src:${PYTHONPATH:-}"
export HF_HOME=/data/250010072/zlh/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$GPUS")
export MASTER_PORT=${MASTER_PORT:-29500}

"$SFT_ENV/bin/python" -m torch.distributed.run \
  --nnodes 1 \
  --node_rank 0 \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  "$LLAMA_FACTORY/src/llamafactory/launcher.py" \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --deepspeed "$LLAMA_FACTORY/examples/deepspeed/ds_z2_config.json" \
  --model_name_or_path "$MODEL_PATH" \
  --dataset "$DATASET" \
  --dataset_dir "$DATASET_DIR" \
  --template qwen \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 8192 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 2 \
  --plot_loss \
  --bf16 \
  --gradient_checkpointing \
  --save_total_limit 3 \
  "${EXTRA_ARGS[@]}"
