#!/bin/bash
# Run one memory type's baseline on a chosen GPU, optionally on a dataset shard.
# Usage:  scripts/run_memory.sh <memory_type> <gpu_id> [shard_i] [shard_n] [run_dir]
# Examples:
#   scripts/run_memory.sh full_history 0              # GPU 0, full dataset
#   scripts/run_memory.sh rl_memory 0 0 2             # GPU 0, shard 0 of 2
#   scripts/run_memory.sh rl_memory 1 1 2 results/X   # GPU 1, shard 1 of 2, write into results/X

set -euo pipefail

MEMORY=${1:?usage: $0 <memory_type> <gpu_id> [shard_i] [shard_n] [run_dir] [batch_size]}
GPU=${2:?usage: $0 <memory_type> <gpu_id> [shard_i] [shard_n] [run_dir] [batch_size]}
SHARD_I=${3:-0}
SHARD_N=${4:-1}
RUN_DIR=${5:-}
BATCH_SIZE=${6:-16}

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/scratch/aroot/hf_cache}"
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

mkdir -p logs
LOG="logs/${MEMORY}_gpu${GPU}_shard${SHARD_I}of${SHARD_N}_$(date +%Y%m%d_%H%M%S).log"

echo "[run_memory] MEMORY=$MEMORY GPU=$GPU SHARD=${SHARD_I}/${SHARD_N} BATCH_SIZE=$BATCH_SIZE HF_HOME=$HF_HOME RUN_DIR=${RUN_DIR:-<auto>}" | tee "$LOG"

ARGS=(--memory "$MEMORY" --shard-i "$SHARD_I" --shard-n "$SHARD_N" --batch-size "$BATCH_SIZE" --skip-eval)
if [[ -n "$RUN_DIR" ]]; then
    ARGS+=(--run-dir "$RUN_DIR")
fi

python -u main.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
