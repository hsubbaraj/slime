#!/bin/bash
set -ex

# Default paths - can be overridden by environment variables
SLIME_HOME=${SLIME_HOME:-"/root/slime"}
MODEL_PATH=${MODEL_PATH:-"/root/models/Qwen3-8B-Instruct"}
DATA_PATH=${DATA_PATH:-"/root/data/airline_sft_formatted.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"/root/checkpoints/Qwen3-8B-sft-airline"}
MEGATRON_PATH=${MEGATRON_PATH:-"/root/Megatron-LM"}

# Ensure output directory exists
mkdir -p "$OUTPUT_PATH"

# Source the Qwen3-8B model arguments
# Try to locate the script relative to this file or use SLIME_HOME
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_CONFIG_SCRIPT="$SLIME_HOME/scripts/models/qwen3-4B-Instruct-2507.sh"
source "$MODEL_CONFIG_SCRIPT"

# Increase sequence length for Tau-bench (long contexts)
MODEL_ARGS+=(
    --seq-length 8192
    --max-position-embeddings 8192
)

# Define Checkpoint Arguments
CKPT_ARGS=(
   --hf-checkpoint "$MODEL_PATH"
   --ref-load "${MODEL_PATH}_torch_dist"
   --save "$OUTPUT_PATH"
   --save-interval 1000
   --rotary-base 5000000
)

# Define SFT Arguments
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "$DATA_PATH"
   --input-key messages
   --rollout-shuffle
   --num-epoch ${NUM_EPOCH:-3}
   --rollout-batch-size ${BATCH_SIZE:-128}
   --global-batch-size ${BATCH_SIZE:-128}

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# Define Performance Arguments (tuned for 4B)
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# Define Optimizer Arguments
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

# WandB Arguments (set WANDB_API_KEY env var)
# WANDB_ARGS=(
#    --use-wandb
#    --wandb-project slime-tau
#    --wandb-group qwen3-4b-sft
# )

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Check if ray is already running
if ! ray status > /dev/null 2>&1; then
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $(nvidia-smi -L | wc -l) --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Runtime environment
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$SLIME_HOME:$MEGATRON_PATH\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_ALGO\": \"Ring\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\"
  }
}"

# Run directly with unbuffered output for better debugging
python3 -u "$SLIME_HOME/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $(nvidia-smi -L | wc -l) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $(nvidia-smi -L | wc -l) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
