#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE=${1:-1.5B}
GPUS=${2:-0,1,2,3}

ROOT=/data/250010072/zlh/king/code_tailored_dataset
VERL=/data/250010072/zlh/king/verl
ENV=/data/250010072/zlh/conda_envs/verl
SFT_ROOT=$ROOT/train/sft/models
RL_ROOT=$ROOT/train/rl
DATA_ROOT=$RL_ROOT/data
OUT_ROOT=$RL_ROOT/models
LOG_ROOT=$RL_ROOT/log

if [[ "$MODEL_SIZE" == "1.5B" ]]; then
  BASE=Qwen2.5-Coder-1.5B-Instruct
  TAG=qwen25_coder_1_5b
elif [[ "$MODEL_SIZE" == "3B" ]]; then
  BASE=Qwen2.5-Coder-3B-Instruct
  TAG=qwen25_coder_3b
else
  echo "MODEL_SIZE must be 1.5B or 3B"
  exit 1
fi

MODEL_PATH=${MODEL_PATH:-$SFT_ROOT/${BASE}-reasoning-sft}
TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/${TAG}_parquet/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_ROOT/${TAG}_parquet/val.parquet}
OUT_DIR=${OUT_DIR:-$OUT_ROOT/${BASE}-reasoning-dapo}

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$VERL"

export CUDA_VISIBLE_DEVICES=$GPUS
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$VERL:$RL_ROOT:${PYTHONPATH:-}"
export HF_HOME=/data/250010072/zlh/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1

NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")
N_RESP=${N_RESP:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-}

EXTRA_ARGS=()
if [[ -n "$TOTAL_STEPS" ]]; then
  EXTRA_ARGS+=(trainer.total_training_steps="$TOTAL_STEPS")
fi

"$ENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=10 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.gen_batch_size="$GEN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESPONSE_LEN" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$N_RESP" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BSZ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BSZ" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.reward_manager.name=dapo \
  reward.num_workers=32 \
  reward.custom_reward_function.path="$RL_ROOT/reward_function.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=kodcode_reasoning_rl \
  trainer.experiment_name="${TAG}_dapo" \
  trainer.logger=console \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$OUT_DIR" \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.device=cuda \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_ROOT/${TAG}_dapo_$(date +%Y%m%d_%H%M%S).log"
