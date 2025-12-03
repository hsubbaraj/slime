#!/bin/bash
set -ex

# Paths
SLIME_HOME=${SLIME_HOME:-"/root/slime"}
MODEL_ROOT=${MODEL_ROOT:-"/root/models"}
HF_MODEL_NAME=${HF_MODEL_NAME:-"Qwen3-4B-Instruct-2507"}
HF_PATH="$MODEL_ROOT/$HF_MODEL_NAME"
MEGATRON_PATH=${MEGATRON_PATH:-"/root/Megatron-LM"}
OUTPUT_PATH="${HF_PATH}_torch_dist"

# Source Model Config
MODEL_CONFIG_SCRIPT="$SLIME_HOME/scripts/models/qwen3-4B-Instruct-2507.sh"
source "$MODEL_CONFIG_SCRIPT"

# Export Environment
export PYTHONPATH="$SLIME_HOME:$MEGATRON_PATH"

echo "Starting conversion from $HF_PATH to $OUTPUT_PATH"

# Run Conversion
python3 "$SLIME_HOME/tools/convert_hf_to_torch_dist.py" \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "$HF_PATH" \
    --save "$OUTPUT_PATH"

echo "Conversion complete: $OUTPUT_PATH"
