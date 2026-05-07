#!/bin/bash
# Re-eval existing multi-chunk poison files under Option B packaging.
#
# Option B = each of the M selected rounds per hub is injected as its own
# 1-round session (distinct session_id per round) instead of bundling all
# M into one fake "coherent conversation" session. Same K*M chunks, same
# retrieval slot claim, but no multi-round-history-coherence artifact.
#
# A/B comparison: original outputs are eval_test_retrieval_<TAG>.json,
# Option B outputs go to eval_test_retrieval_<TAG>_optB.json. Both stay on
# disk so you can diff them.
#
# Reuses existing poison files in results/stage_b/retrieval_<TAG>.json — no
# regeneration needed (poison file format is unchanged; only injection at
# eval time changed).
#
# Launch (no nohup needed for tmux; school server uses nohup):
#   nohup bash scripts/run_multichunk_optb.sh > logs/optb_main.log 2>&1 &
#   disown
# Watch:  tail -f logs/optb_main.log

set -e
set -o pipefail

mkdir -p logs results/stage_b

PY=.venv/bin/python
STAMP() { date +'%H:%M:%S'; }
banner() { echo "=== [$(STAMP)] $* ==="; }

for TAG in K30_M1 K30_M3 K30_M5 K10_M3; do
    banner "eval (Option B): $TAG"
    CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
        --config configs/attack_rag.yaml --split test \
        --poison_file results/stage_b/retrieval_${TAG}.json \
        --output     results/stage_b/eval_test_retrieval_${TAG}_optB.json \
        --use_gpt4o \
        2>&1 | tee logs/eval_${TAG}_optB.log
done

banner "recall-only splits"
for TAG in K30_M1 K30_M3 K30_M5 K10_M3; do
    echo "--- $TAG (Option B) ---"
    $PY scripts/split_hubs_eval_by_abs.py \
        results/stage_b/eval_test_retrieval_${TAG}_optB.json --judge gpt4o \
        2>&1 | tee logs/split_${TAG}_optB.log
done

banner "ALL DONE (Option B)"
echo
echo "=== Compare Option A (M-round session) vs Option B (M single-round sessions) ==="
for TAG in K30_M1 K30_M3 K30_M5 K10_M3; do
    echo "--- $TAG ---"
    echo "  Option A:"
    grep -E "recall .*acc_Δ" logs/split_${TAG}.log 2>/dev/null | head -1 || echo "    (no prior log)"
    echo "  Option B:"
    grep -E "recall .*acc_Δ" logs/split_${TAG}_optB.log | head -1
done
