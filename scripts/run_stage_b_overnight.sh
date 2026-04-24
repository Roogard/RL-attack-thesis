#!/bin/bash
# Stage B overnight run — 4 GPUs, no tmux required.
#
# Launch: nohup bash scripts/run_stage_b_overnight.sh > logs/stage_b_main.log 2>&1 &
#         disown
# Check:  tail -f logs/stage_b_main.log
# Timing: ~1.5-2.5 hrs wall clock.
#
# GPU assignment:
#   Step 1 (save_hubs)     : GPU 0 — brief
#   Step 2 (retrieval+BoN) : GPU 0 + GPU 1 in parallel
#   Step 3 (grad)          : GPU 0 only (MiniLM is small)
#   Step 4 (evals, x3)     : two parallel on GPUs 0,1 and GPUs 2,3; then one on GPUs 0,1

set -e  # abort on any command failure
set -o pipefail

mkdir -p logs results/stage_b

PY=.venv/bin/python
STAMP() { date +'%H:%M:%S'; }
banner() { echo "=== [$(STAMP)] $* ==="; }

# ── Step 1 ── save hubs once ────────────────────────────────────────────
banner "save_hubs"
CUDA_VISIBLE_DEVICES=0 $PY scripts/save_hubs.py \
    --K 30 --method kmeans --seed 0 \
    --out results/stage_a/hubs_K30.pkl

# ── Step 2 ── retrieval (GPU 0) + BoN (GPU 1) in parallel ──────────────
banner "retrieval + BoN (parallel)"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K30.pkl \
    --out  results/stage_b/retrieval_K30.json \
    > logs/retrieval.log 2>&1 &
RPID=$!
CUDA_VISIBLE_DEVICES=1 $PY -m attacks.hubness.stage_b_bon \
    --hubs results/stage_a/hubs_K30.pkl \
    --N 64 \
    --out  results/stage_b/bon_K30_N64.json \
    > logs/bon.log 2>&1 &
BPID=$!
wait $RPID $BPID
banner "retrieval + BoN done"

# ── Step 3 ── gradient (HotFlip), seeded from BoN ──────────────────────
# Runtime is uncertain: 20 min to ~2 hr depending on MiniLM forward cost.
banner "grad HotFlip (seeded from BoN)"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_grad \
    --hubs results/stage_a/hubs_K30.pkl \
    --seed_from results/stage_b/bon_K30_N64.json \
    --n_iter 400 --top_k_cand 20 \
    --out results/stage_b/grad_K30_bon_seed.json \
    > logs/grad.log 2>&1
banner "grad done"

# ── Step 4a ── eval retrieval + eval BoN in parallel ───────────────────
# Each eval needs 2 GPUs: vLLM answer engine on cuda:0 (local), 7B judge on cuda:1.
# Under CUDA_VISIBLE_DEVICES=X,Y a process sees cuda:0=X, cuda:1=Y.
banner "eval retrieval (GPUs 0,1) + eval BoN (GPUs 2,3) in parallel"
CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
    --config configs/attack_rag.yaml --split test \
    --poison_file results/stage_b/retrieval_K30.json \
    --output results/stage_b/eval_test_retrieval.json --use_gpt4o \
    > logs/eval_retrieval.log 2>&1 &
ERPID=$!
CUDA_VISIBLE_DEVICES=2,3 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
    --config configs/attack_rag.yaml --split test \
    --poison_file results/stage_b/bon_K30_N64.json \
    --output results/stage_b/eval_test_bon.json --use_gpt4o \
    > logs/eval_bon.log 2>&1 &
EBPID=$!
wait $ERPID $EBPID
banner "eval retrieval + eval BoN done"

# ── Step 4b ── eval grad (all GPUs free now) ───────────────────────────
banner "eval grad (GPUs 0,1)"
CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
    --config configs/attack_rag.yaml --split test \
    --poison_file results/stage_b/grad_K30_bon_seed.json \
    --output results/stage_b/eval_test_grad.json --use_gpt4o \
    > logs/eval_grad.log 2>&1
banner "eval grad done"

# ── Step 5 ── post-hoc splits (CPU, fast) ──────────────────────────────
banner "post-hoc abstention splits"
for name in retrieval bon grad; do
    echo "--- $name ---"
    $PY scripts/split_hubs_eval_by_abs.py \
        results/stage_b/eval_test_${name}.json --judge gpt4o \
        > logs/split_${name}.log 2>&1
done
banner "ALL DONE"

echo "Outputs:"
ls -la results/stage_b/
echo "Per-method splits in: logs/split_{retrieval,bon,grad}.log"
