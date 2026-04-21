#!/bin/bash
# Run one memory type's baseline on a chosen GPU.
# Usage: scripts/run_memory.sh <memory_type> <gpu_id>
# Example: scripts/run_memory.sh full_history 0

set -euo pipefail

MEMORY=${1:?usage: $0 <memory_type> <gpu_id>}
GPU=${2:?usage: $0 <memory_type> <gpu_id>}

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/scratch/aroot/hf_cache}"
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

mkdir -p logs
LOG="logs/${MEMORY}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"

echo "[run_memory] MEMORY=$MEMORY GPU=$GPU HF_HOME=$HF_HOME" | tee "$LOG"
python -u main.py --memory "$MEMORY" --skip-eval 2>&1 | tee -a "$LOG"
