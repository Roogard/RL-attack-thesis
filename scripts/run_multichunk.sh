#!/bin/bash
# Multi-chunk-per-hub Stage B sweep — see ATTACK.md option 3.
#
# Builds 4 poison files at varying (K, M), evals each on the test split with
# GPT-4o, prints recall-only splits. Sequential on 2 GPUs.
#
# Launch (no nohup needed):
#   tmux new -s mchunk -d "bash scripts/run_multichunk.sh 2>&1 | tee logs/multichunk_main.log"
# Or (no tmux):
#   setsid bash scripts/run_multichunk.sh > logs/multichunk_main.log 2>&1 < /dev/null &
#   disown
# Watch:  tail -f logs/multichunk_main.log
# Reattach (tmux): tmux attach -t mchunk
#
# GPU layout assumed: cuda:0 = vLLM answer engine, cuda:1 = local 7B judge.
# If you have 4 GPUs and want to halve wall-clock, split the eval loop into
# two parallel passes pinning CUDA_VISIBLE_DEVICES=0,1 and =2,3 (see
# scripts/run_stage_b_overnight.sh for the pattern).

set -e
set -o pipefail

mkdir -p logs results/stage_a results/stage_b

PY=.venv/bin/python
STAMP() { date +'%H:%M:%S'; }
banner() { echo "=== [$(STAMP)] $* ==="; }

# ── Step 1 ── hubs ──────────────────────────────────────────────────────
banner "save_hubs K=30"
CUDA_VISIBLE_DEVICES=0 $PY scripts/save_hubs.py \
    --K 30 --method kmeans --seed 0 \
    --out results/stage_a/hubs_K30.pkl

banner "save_hubs K=10"
CUDA_VISIBLE_DEVICES=0 $PY scripts/save_hubs.py \
    --K 10 --method kmeans --seed 0 \
    --out results/stage_a/hubs_K10.pkl

# ── Step 2 ── poison files (4 configs) ──────────────────────────────────
banner "poison file: K30 M=1 (sanity vs prior +14.0pp baseline)"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K30.pkl --rounds_per_hub 1 \
    --out  results/stage_b/retrieval_K30_M1.json

banner "poison file: K30 M=3"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K30.pkl --rounds_per_hub 3 \
    --out  results/stage_b/retrieval_K30_M3.json

banner "poison file: K30 M=5 (headline)"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K30.pkl --rounds_per_hub 5 \
    --out  results/stage_b/retrieval_K30_M5.json

banner "poison file: K10 M=3 (budget control vs K30 M=1)"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K10.pkl --rounds_per_hub 3 \
    --out  results/stage_b/retrieval_K10_M3.json

# ── Step 3 ── eval each config on test split with GPT-4o ────────────────
for TAG in K30_M1 K30_M3 K30_M5 K10_M3; do
    banner "eval: $TAG"
    CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
        --config configs/attack_rag.yaml --split test \
        --poison_file results/stage_b/retrieval_${TAG}.json \
        --output     results/stage_b/eval_test_retrieval_${TAG}.json \
        --use_gpt4o \
        2>&1 | tee logs/eval_${TAG}.log
done

# ── Step 4 ── recall-only splits ────────────────────────────────────────
banner "recall-only splits"
for TAG in K30_M1 K30_M3 K30_M5 K10_M3; do
    echo "--- $TAG ---"
    $PY scripts/split_hubs_eval_by_abs.py \
        results/stage_b/eval_test_retrieval_${TAG}.json --judge gpt4o \
        2>&1 | tee logs/split_${TAG}.log
done

banner "ALL DONE"
ls -la results/stage_b/
echo "Headline numbers: grep -E 'drop|hub_share|hub/top' logs/split_*.log"
