#!/bin/bash
# Multi-chunk + PI append. M=3 was the Option-B sweet spot (+20pp recall acc
# drop) but cnf_Δ ≈ 0 — the attack flips correct→incorrect, doesn't induce
# abstention. Mixing PI-template bait into each chunk's assistant_msg is the
# attempt to also drive abstention. Encoded vector stays anchored to the
# natural-text round (preserving hub-share@10), but indexed text now carries
# an override the reader sees.
#
# Variants:
#   K30_M3_PI : sweet-spot displacement + PI bait
#   K30_M5_PI : higher slot claim + PI bait (in case PI compounds at higher M)
#
# Launch:  nohup bash scripts/run_multichunk_pi.sh > logs/pi_main.log 2>&1 &
#          disown
# Watch:   tail -f logs/pi_main.log

set -e
set -o pipefail

mkdir -p logs results/stage_b

PY=.venv/bin/python
STAMP() { date +'%H:%M:%S'; }
banner() { echo "=== [$(STAMP)] $* ==="; }

# Sanity-check hubs_K30 exists; bail early if you forgot to run the prior sweep.
if [ ! -f results/stage_a/hubs_K30.pkl ]; then
    echo "missing results/stage_a/hubs_K30.pkl — run scripts/run_multichunk.sh first"
    exit 1
fi

# ── Step 1 ── poison files (PI append) ──────────────────────────────────
banner "poison file: K30 M=3 + PI append"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K30.pkl \
    --rounds_per_hub 3 --pi_mode append \
    --out  results/stage_b/retrieval_K30_M3_PI.json

banner "poison file: K30 M=5 + PI append"
CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
    --hubs results/stage_a/hubs_K30.pkl \
    --rounds_per_hub 5 --pi_mode append \
    --out  results/stage_b/retrieval_K30_M5_PI.json

# ── Step 2 ── eval each on test split with GPT-4o ───────────────────────
for TAG in K30_M3_PI K30_M5_PI; do
    banner "eval: $TAG"
    CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
        --config configs/attack_rag.yaml --split test \
        --poison_file results/stage_b/retrieval_${TAG}.json \
        --output     results/stage_b/eval_test_retrieval_${TAG}.json \
        --use_gpt4o \
        2>&1 | tee logs/eval_${TAG}.log
done

# ── Step 3 ── recall-only splits ────────────────────────────────────────
banner "recall-only splits"
for TAG in K30_M3_PI K30_M5_PI; do
    echo "--- $TAG ---"
    $PY scripts/split_hubs_eval_by_abs.py \
        results/stage_b/eval_test_retrieval_${TAG}.json --judge gpt4o \
        2>&1 | tee logs/split_${TAG}.log
done

banner "ALL DONE (multi-chunk + PI)"
echo
echo "Inspect:"
echo "  cat logs/split_K30_M3_PI.log"
echo "  cat logs/split_K30_M5_PI.log"
echo "  python3 -c \"import json; d=json.load(open('results/stage_b/retrieval_K30_M3_PI.json')); print('post-PI cos:', d.get('post_pi_mean_cos_to_hub'))\""
