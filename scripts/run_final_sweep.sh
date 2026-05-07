#!/bin/bash
# Consolidated Stage B retrieval-hub sweep: K∈{10,30,100} × M∈{1,3,5} × PI∈{none,append}
# = 18 text-mode configs evaluated against ONE shared clean baseline per qid
# (noise-free cross-config comparison).
#
# Outputs:
#   results/stage_b/eval_test_final_sweep.{json,csv}    18 per_config entries
#   results/stage_b/POISON_EXAMPLES.md                  3 chunks × 18 configs
#   logs/final_sweep_main.log                            full run log
#
# Estimated wall clock on 2 GPUs (cuda:0 vLLM, cuda:1 judge):
#   - save_hubs K=100               : ~30 sec
#   - 18 poison files (sequential)  : ~30–40 min (CPU/GPU corpus encode each)
#   - 1 eval_hubs (18 configs)      : ~3–4 hr (54 qids × 19 contexts each + GPT-4o judge)
#   - split + render examples       : ~1 min
#
# Launch (school server, no tmux):
#   nohup bash scripts/run_final_sweep.sh > logs/final_sweep_main.log 2>&1 &
#   disown
#
# Watch:  tail -f logs/final_sweep_main.log
# Smoke before full run:  add `--limit 8` to the eval_hubs invocation below.

set -e
set -o pipefail

mkdir -p logs results/stage_a results/stage_b

PY=.venv/bin/python
STAMP() { date +'%H:%M:%S'; }
banner() { echo "=== [$(STAMP)] $* ==="; }

# ── Step 1 ── hubs (K=10 and K=30 should already exist from prior sweep) ──
for K in 10 30 100; do
    if [ ! -f "results/stage_a/hubs_K${K}.pkl" ]; then
        banner "save_hubs K=${K}"
        CUDA_VISIBLE_DEVICES=0 $PY scripts/save_hubs.py \
            --K ${K} --method kmeans --seed 0 \
            --out results/stage_a/hubs_K${K}.pkl
    else
        echo "=== [$(STAMP)] hubs_K${K}.pkl already exists, skipping save_hubs ==="
    fi
done

# ── Step 2 ── build the 18 poison files ─────────────────────────────────
# Each call re-encodes the corpus (~1.5–2 min). 18 × ~2 min ≈ 30–40 min total.
banner "building 18 poison files (K∈{10,30,100} × M∈{1,3,5} × PI∈{none,append})"
for K in 10 30 100; do
    for M in 1 3 5; do
        for PI in none append; do
            if [ "$PI" = "none" ]; then
                OUT="results/stage_b/retrieval_K${K}_M${M}.json"
                PI_FLAG=""
                TAG="K${K}_M${M}"
            else
                OUT="results/stage_b/retrieval_K${K}_M${M}_PI.json"
                PI_FLAG="--pi_mode append"
                TAG="K${K}_M${M}_PI"
            fi
            if [ -f "$OUT" ]; then
                echo "  $TAG: $OUT exists, skipping"
                continue
            fi
            echo "--- $TAG ---"
            CUDA_VISIBLE_DEVICES=0 $PY -m attacks.hubness.stage_b_retrieval \
                --hubs results/stage_a/hubs_K${K}.pkl \
                --rounds_per_hub ${M} ${PI_FLAG} \
                --out "$OUT" \
                2>&1 | tee logs/build_${TAG}.log
        done
    done
done

# ── Step 3 ── one consolidated eval_hubs run with all 18 poison files ───
# Single invocation = single clean baseline per qid → noise-free comparison.
banner "consolidated eval (18 configs, test split, GPT-4o)"
POISON_FILES=$(ls results/stage_b/retrieval_K{10,30,100}_M{1,3,5}{,_PI}.json 2>/dev/null | tr '\n' ' ')
echo "Poison files: $POISON_FILES"

CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 $PY -m attacks.eval_hubs \
    --config configs/attack_rag.yaml --split test \
    --poison_file $POISON_FILES \
    --output results/stage_b/eval_test_final_sweep.json \
    --use_gpt4o \
    2>&1 | tee logs/final_sweep_eval.log

# ── Step 4 ── recall-only split (writes per-config table to log) ────────
banner "recall-only split"
$PY scripts/split_hubs_eval_by_abs.py \
    results/stage_b/eval_test_final_sweep.json --judge gpt4o \
    2>&1 | tee logs/final_sweep_split.log

# ── Step 5 ── examples doc ──────────────────────────────────────────────
banner "render poison examples"
$PY scripts/render_poison_examples.py \
    --poison_files $POISON_FILES \
    --out results/stage_b/POISON_EXAMPLES.md \
    2>&1 | tee logs/final_sweep_examples.log

banner "ALL DONE (final sweep)"
echo
echo "Headline numbers:"
echo "  cat logs/final_sweep_split.log"
echo "Per-config mechanism summary:"
echo "  python3 -c \"
import json
d = json.load(open('results/stage_b/eval_test_final_sweep.json'))
for k, v in sorted(d['per_config'].items()):
    m, a = v['mechanism'], v['abstention']
    print(f'{k:40s}  hub/top={m[\\\"mean_hubs_in_topk\\\"]:5.2f}  hub_share={m[\\\"hub_share_topk\\\"]:5.1%}  abst_p={a[\\\"poisoned_rate\\\"]:5.1%}')\""
echo "Examples doc:"
echo "  less results/stage_b/POISON_EXAMPLES.md"
