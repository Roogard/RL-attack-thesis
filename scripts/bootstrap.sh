#!/usr/bin/env bash
# One-time setup for the Microway box (or any single-node Linux w/ CUDA).
# Idempotent — safe to re-run. Assumes you're already inside the cloned repo.
#
# Usage:  bash scripts/bootstrap.sh

set -euo pipefail

# ── Preconditions ─────────────────────────────────────────────────────────────
if [[ ! -f "main.py" || ! -d "memory" ]]; then
    echo "ERROR: run this from the repo root (where main.py lives)." >&2
    exit 1
fi

REPO_DIR="$(pwd)"
HF_CACHE="${HF_HOME:-/scratch/aroot/hf_cache}"

echo "==> Repo:      $REPO_DIR"
echo "==> HF cache:  $HF_CACHE"
echo

# ── venv ──────────────────────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "==> Creating venv at .venv/"
    python3 -m venv .venv
else
    echo "==> venv already exists, reusing"
fi

PIP=".venv/bin/pip"
PY=".venv/bin/python"

# ── pip installs ──────────────────────────────────────────────────────────────
echo "==> Upgrading pip"
"$PIP" install --quiet --upgrade pip

# Install torch from the CUDA 12.4 wheel index (forward-compatible with driver 580 / CUDA 13.0)
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "==> Installing torch (cu124 wheels)"
    "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "==> torch already installed ($("$PY" -c 'import torch; print(torch.__version__)'))"
fi

echo "==> Installing requirements.txt"
"$PIP" install --quiet -r requirements.txt

# ── HF cache env ──────────────────────────────────────────────────────────────
mkdir -p "$HF_CACHE"
if ! grep -q "HF_HOME=$HF_CACHE" ~/.bashrc 2>/dev/null; then
    echo "==> Adding HF_HOME to ~/.bashrc"
    echo "export HF_HOME=$HF_CACHE" >> ~/.bashrc
else
    echo "==> HF_HOME already set in ~/.bashrc"
fi
export HF_HOME="$HF_CACHE"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo
echo "==> Sanity check"
"$PY" - <<'PY'
import torch
print(f"  torch:        {torch.__version__}")
print(f"  cuda avail:   {torch.cuda.is_available()}")
print(f"  gpu count:    {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  gpu {i}:        {name} ({free/1e9:.1f}/{total/1e9:.1f} GB free)")
PY

# ── Model pre-download (optional — slow, but saves time during first run) ─────
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
    echo
    echo "==> Pre-downloading HF models to $HF_CACHE"
    echo "    (set SKIP_MODEL_DOWNLOAD=1 to skip)"
    "$PY" - <<'PY'
# Keep this list in sync with _ANSWER_MODEL_ID / _JUDGE_MODEL_ID in harness.py
# and _MODEL_ID in memory/rl_memory.py
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "driaforall/mem-agent"]:
    print(f"  fetching {m} ...", flush=True)
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m)
print("  done.")
PY
fi

# ── Data files reminder ───────────────────────────────────────────────────────
if [[ ! -f "LongMemEval/data/longmemeval_s_cleaned.json" || ! -f "LongMemEval/data/longmemeval_oracle.json" ]]; then
    echo
    echo "⚠  LongMemEval data files are NOT present yet (they're gitignored)."
    echo "   From your laptop, run:"
    echo "     scp LongMemEval/data/longmemeval_s_cleaned.json \\"
    echo "         LongMemEval/data/longmemeval_oracle.json \\"
    echo "         aroot@microway.cis.umassd.edu:$REPO_DIR/LongMemEval/data/"
fi

echo
echo "==> Bootstrap complete."
echo "    Next:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --memory full_history --limit 5 --skip-eval"
