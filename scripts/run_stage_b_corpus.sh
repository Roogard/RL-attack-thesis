#!/usr/bin/env bash
# Convenience driver for the Stage B corpus pipeline.
# Reads paths from environment, defaults to project conventions.
# See data/corpora/README.md for the per-step explanation.

set -euo pipefail

HUBS=${HUBS:-results/stage_a/hubs_K30.pkl}
CONFIG=${CONFIG:-configs/attack_rag.yaml}
OUT_DIR=${OUT_DIR:-results/stage_b_corpus}
SCORING_POOL=${SCORING_POOL:-${OUT_DIR}/scoring_pool.pkl}
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}
TOP_N=${TOP_N:-200}

CORPORA=(nl_web nl_memory nl_persona
         synth_generic synth_topic synth_persona
         adv_inject adv_confuse adv_optimized)

mkdir -p "${OUT_DIR}" data/corpora

step=${1:-all}

build_corpora() {
  python -m attacks.corpora.nl_web      --out data/corpora/nl_web.jsonl
  python -m attacks.corpora.nl_memory   --out data/corpora/nl_memory.jsonl
  python -m attacks.corpora.nl_persona  --out data/corpora/nl_persona.jsonl
  python -m attacks.corpora.synth_generic --out data/corpora/synth_generic.jsonl
  python -m attacks.corpora.synth_topic   --hubs "${HUBS}" --out data/corpora/synth_topic.jsonl
  python -m attacks.corpora.synth_persona --out data/corpora/synth_persona.jsonl
  python -m attacks.corpora.adv_inject    --out data/corpora/adv_inject.jsonl
  python -m attacks.corpora.adv_confuse   --out data/corpora/adv_confuse.jsonl
  python -m attacks.corpora.adv_optimized --out data/corpora/adv_optimized.jsonl
}

build_scoring_pool() {
  python -m attacks.hubness.reader_ppx \
      --config "${CONFIG}" --split val --n_questions 10 --seed 0 \
      --out "${SCORING_POOL}"
}

per_corpus_select() {
  for cid in "${CORPORA[@]}"; do
    python -m attacks.hubness.stage_b_corpus \
        --hubs "${HUBS}" \
        --corpus_id "${cid}" \
        --corpus_path "data/corpora/${cid}.jsonl" \
        --scoring_pool "${SCORING_POOL}" \
        --alpha "${ALPHA}" --beta "${BETA}" --top_n "${TOP_N}" \
        --out "${OUT_DIR}/poison_${cid}.json"
  done
}

union_select() {
  paths=()
  for cid in "${CORPORA[@]}"; do
    paths+=("data/corpora/${cid}.jsonl")
  done
  python -m attacks.hubness.stage_b_corpus \
      --hubs "${HUBS}" \
      --union \
      --corpus_paths "${paths[@]}" \
      --corpus_ids "${CORPORA[@]}" \
      --scoring_pool "${SCORING_POOL}" \
      --alpha "${ALPHA}" --beta "${BETA}" --top_n "${TOP_N}" \
      --out "${OUT_DIR}/poison_union.json"
}

eval_all() {
  for cid in "${CORPORA[@]}" union; do
    python -m attacks.eval_hubs \
        --config "${CONFIG}" --split test --hub_scope global \
        --poison_file "${OUT_DIR}/poison_${cid}.json" \
        --output "${OUT_DIR}/eval_${cid}.json" \
        --use_gpt4o
    python scripts/split_hubs_eval_by_abs.py \
        "${OUT_DIR}/eval_${cid}.json" --judge gpt4o \
        > "${OUT_DIR}/split_${cid}.log"
  done
}

case "${step}" in
  corpora)        build_corpora ;;
  scoring_pool)   build_scoring_pool ;;
  select)         per_corpus_select ;;
  union)          union_select ;;
  eval)           eval_all ;;
  all)
    build_corpora
    build_scoring_pool
    per_corpus_select
    union_select
    eval_all
    ;;
  *)
    echo "usage: $0 [corpora|scoring_pool|select|union|eval|all]" >&2
    exit 2
    ;;
esac
