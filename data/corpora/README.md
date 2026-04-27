# Stage B corpus pipeline — runbook

End-to-end recipe for the corpus-construction extension described in
[../../plans/i-m-working-on-a-cheerful-whistle.md](../../../.claude/plans/i-m-working-on-a-cheerful-whistle.md)
("Next: corpus construction + reader-aware fitness").

Output: ten Stage B eval result sets — 9 per-corpus + 1 union — each
shaped like the existing `results/stage_b/eval_test_retrieval.json`.

All commands assume the project root as cwd and the cluster `.venv/`
already populated with the existing dependencies (`vllm`,
`sentence-transformers`, `transformers`, `chromadb`, `pyyaml`,
`tqdm`). The corpora pipeline additionally needs `datasets` (HuggingFace),
which should already be present from the existing project.

---

## 0. Pre-flight

The corpora list is open-only. Two of the originally-planned datasets
(WildChat-1M, LMSYS-Chat-1M) are gated and were swapped for
UltraChat-200k + ShareGPT-Vicuna-Unfiltered. No HF login required.

If you ever want to add WildChat or LMSYS-Chat back in:
- `huggingface-cli login`
- Accept the dataset's terms on its HF page
- Add the loader inside `attacks/corpora/nl_web.py`

Disk: budget ~30 GB for raw JSONL + ~15 GB per encoded `.npy` (cached
under `results/stage_b_corpus/encodings/`).

GPU: corpus build needs vLLM (Qwen2.5-3B attacker) for the synthetic
and adv_optimized corpora. Encoding only needs MiniLM. RPR scoring
shares the answer engine with `eval_hubs.py`.

---

## 1. Build the 9 corpora

Natural-language corpora are streamed off HuggingFace Hub; no GPU.

```bash
python -m attacks.corpora.nl_web      --out data/corpora/nl_web.jsonl
python -m attacks.corpora.nl_memory   --out data/corpora/nl_memory.jsonl
python -m attacks.corpora.nl_persona  --out data/corpora/nl_persona.jsonl
```

Synthetic corpora need the attacker vLLM engine. Run sequentially —
each one saturates the engine.

```bash
CUDA_VISIBLE_DEVICES=0 python -m attacks.corpora.synth_generic \
    --out data/corpora/synth_generic.jsonl

CUDA_VISIBLE_DEVICES=0 python -m attacks.corpora.synth_topic \
    --hubs results/stage_a/hubs_K30.pkl \
    --out data/corpora/synth_topic.jsonl

CUDA_VISIBLE_DEVICES=0 python -m attacks.corpora.synth_persona \
    --out data/corpora/synth_persona.jsonl
```

Adversarial corpora — `adv_inject` and `adv_confuse` are template-driven
(no GPU). `adv_optimized` runs the attacker engine.

```bash
python -m attacks.corpora.adv_inject  --out data/corpora/adv_inject.jsonl
python -m attacks.corpora.adv_confuse --out data/corpora/adv_confuse.jsonl

CUDA_VISIBLE_DEVICES=0 python -m attacks.corpora.adv_optimized \
    --out data/corpora/adv_optimized.jsonl
```

Knobs each builder respects (full list: `python -m attacks.corpora.<id> --help`):
- `--samples_per_topic` / `--samples_per_hub` for synth corpora
- `--n_topics`, `--n_personas` for breadth
- `--max_tokens`, `--batch_size`, `--seed`

---

## 2. Build the scoring-pool fixture

One-time. Picks 10 val-split questions, captures clean retrieval
context, and precomputes baseline log P(refusal phrase | clean ctx, q)
for every (question × refusal). Cached as a pickle.

```bash
CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 .venv/bin/python -m attacks.hubness.reader_ppx \
    --config configs/attack_rag.yaml \
    --split val --n_questions 10 --seed 0 \
    --out results/stage_b_corpus/scoring_pool.pkl
```

(Single-GPU fallback: drop `JUDGE_DEVICE`. The local judge is not used
here, but `eval_hubs.py` will load it later, so leaving it pinned is
cheap.)

---

## 3. Per-corpus selection (9 runs)

For each corpus, run the two-axis selector. Encoded vectors are cached
under `results/stage_b_corpus/encodings/` so re-running with different
α/β doesn't re-encode.

```bash
for cid in nl_web nl_memory nl_persona \
           synth_generic synth_topic synth_persona \
           adv_inject adv_confuse adv_optimized; do
  CUDA_VISIBLE_DEVICES=0 python -m attacks.hubness.stage_b_corpus \
      --hubs results/stage_a/hubs_K30.pkl \
      --corpus_id $cid \
      --corpus_path data/corpora/$cid.jsonl \
      --scoring_pool results/stage_b_corpus/scoring_pool.pkl \
      --alpha 1.0 --beta 1.0 --top_n 200 \
      --out results/stage_b_corpus/poison_${cid}.json
done
```

For a cos-only baseline run on any corpus (apples-to-apples vs the
existing `+14.8 pp` retrieval number), drop `--scoring_pool` or pass
`--beta 0`.

---

## 4. Union selection

```bash
CUDA_VISIBLE_DEVICES=0 python -m attacks.hubness.stage_b_corpus \
    --hubs results/stage_a/hubs_K30.pkl \
    --union \
    --corpus_paths \
        data/corpora/nl_web.jsonl \
        data/corpora/nl_memory.jsonl \
        data/corpora/nl_persona.jsonl \
        data/corpora/synth_generic.jsonl \
        data/corpora/synth_topic.jsonl \
        data/corpora/synth_persona.jsonl \
        data/corpora/adv_inject.jsonl \
        data/corpora/adv_confuse.jsonl \
        data/corpora/adv_optimized.jsonl \
    --scoring_pool results/stage_b_corpus/scoring_pool.pkl \
    --alpha 1.0 --beta 1.0 --top_n 200 \
    --out results/stage_b_corpus/poison_union.json
```

The poison file's `provenance` field tells you which corpus contributed
each of the 30 winning chunks.

---

## 5. Per-corpus + union eval

Existing `eval_hubs.py` ingests every poison file unchanged.

```bash
for cid in nl_web nl_memory nl_persona \
           synth_generic synth_topic synth_persona \
           adv_inject adv_confuse adv_optimized union; do
  CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 .venv/bin/python -m attacks.eval_hubs \
      --config configs/attack_rag.yaml \
      --split test --hub_scope global \
      --poison_file results/stage_b_corpus/poison_${cid}.json \
      --output results/stage_b_corpus/eval_${cid}.json \
      --use_gpt4o
  python scripts/split_hubs_eval_by_abs.py \
      results/stage_b_corpus/eval_${cid}.json --judge gpt4o \
      > results/stage_b_corpus/split_${cid}.log
done
```

Cost: ~$0.15-0.20 of GPT-4o judging per run × 10 runs = ~$2.

---

## 6. Sanity checks (run before trusting the numbers)

1. **Refactor regression**: `python -m attacks.hubness.stage_b_retrieval`
   should still reproduce `+14.8 pp` ± 1 pp on the test split. The
   shared `iter_longmemeval_train_rounds` was swapped in but ordering /
   dedup behavior is preserved.
2. **RPR signal pilot**: run `stage_b_corpus.py` on `nl_memory` with
   `--top_n 50 --beta 0` (cos-only) and again with `--beta 1.0`. The
   two should pick *different* chunks for some hubs; if RPR picks
   identical chunks across the board, the scorer is dead and beta is
   wasted compute.
3. **poison_*.json shape**: every file should have exactly 30 sessions,
   each with `cos_to_hub`, `meta.source_corpus`, and (when scoring
   pool was used) `meta.rpr`.
4. **Union ≥ best single**: union acc-drop should be at least
   `max(per-corpus acc-drop) − 1 pp`. If it isn't, the union encoding
   cache is probably stale — clear `results/stage_b_corpus/encodings/`
   and re-run.
5. **Provenance histogram**: in the union eval, check `provenance` in
   `poison_union.json`. If one corpus contributes 30/30 winners, the
   rest are dead weight — flag in the writeup but it's still a finding.

---

## 7. What to expect cost-wise

| step | wall time (4× H200) | $ |
|---|---|---|
| Corpora 1-3 (natural, network) | 2-6 hr each, parallelizable | 0 |
| Corpora 4-6 (synthetic, vLLM) | 4-8 hr each, sequential | 0 |
| Corpora 7-9 (adversarial) | <1 hr (7,8); 3 hr (9) | 0 |
| Encoding (per corpus) | 5-30 min | 0 |
| Scoring pool fixture | 5 min | 0 |
| Per-corpus selection (RPR) | 5-15 min × 9 | 0 |
| Union selection | 30 min | 0 |
| Per-corpus + union eval (GPT-4o) | 30 min × 10 | ~$2 |
| **Total** | ~2-3 days | ~$2 |
