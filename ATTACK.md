# Phase 2 — Adversarial Hubness Attack on Memory-Augmented LLMs

This document is the engineering spec for the current primary attack track under [attacks/](attacks/). An earlier REINFORCE-based approach under [attack/](attack/) is **parked** — see the [Previous approach](#previous-approach-reinforce-parked) section at the bottom for context.

## Attack in one paragraph

The attacker injects K synthetic "hub" vectors directly into the victim's retrieval index. Hubs are chosen as k-means centroids of the training-split question embeddings, so they sit in the region of MiniLM space that user questions cluster in. For arbitrary unseen queries they win top-k slots from legitimate documents. When the answer model reads the resulting contaminated context, it sees enough empty / uncertain / instruction-like signals to either abstain ("I don't know") or hallucinate. The primary attack metric is **confident-answer rate drop** — the memory system's usefulness is measured by how often it still gives a direct answer, not only by whether that answer is correct.

## Threat model

- Attacker has **domain knowledge** (personal-assistant memory, multi-session conversations) and **memory read access**.
- Attacker does **not** see the victim's queries, only the existing memory contents.
- Attacker is **budget-constrained**: K injected hub chunks out of a ~250-chunk haystack.
- In Stage A (current), the attacker has privileged **write access at the vector level** (`index_raw_embeddings`). This is the upper-bound setting — Stage B will drop this privilege.

## Primary attack metric: confident-answer rate

Traditional benchmark accuracy conflates two things:
- **Wrong confident answers** — a user who trusts them gets false information.
- **Correct abstentions** — on `_abs` questions, abstention is marked correct.

An attack that makes the memory system abstain on everything pushes both buckets up. Net accuracy change is ambiguous (some questions got "more correct" via coerced abstention, some got "wrong" via abstention-on-recall-questions). But from the user's perspective the memory system is clearly broken: it's refusing to answer half the time.

**Confident-answer rate** = `1 − abstention_rate`. Drop in this metric is our headline number. Secondary metrics: recall-only accuracy drop (from `split_hubs_eval_by_abs.py`), hub share in top-k (mechanism confirmation), decoupling cosine (how far the Stage A ceiling is from realizable Stage B text-mode attacks).

## Package: [attacks/](attacks/)

| File | Purpose |
|---|---|
| [attacks/__init__.py](attacks/__init__.py) | `PoisonSession`, `WritePolicy`, `Attack` protocol. Two injection modes: text (realistic) or raw vectors (vector-mode upper bound). |
| [attacks/hubness/stage_a_hubs.py](attacks/hubness/stage_a_hubs.py) | `compute_hubs(Q, K, D, method)` — spherical k-means or facility-location on the unit sphere. Returns (hubs, diagnostics: mean_max_sim, displacement_rate, …). |
| [attacks/eval_hubs.py](attacks/eval_hubs.py) | End-to-end evaluator. Computes hubs from train split, injects per eval question, measures clean vs poisoned. Reports confident-answer rate + accuracy + hub-share + abstention delta + decoupling cosine. |

## Memory plumbing

- [memory/base.py](memory/base.py) — `MemoryStore.index_raw_embeddings(embeddings, metadatas, ids, documents)` — vector-mode injection hook. Raises NotImplementedError on memories without an addressable vector layer.
- [memory/rag.py](memory/rag.py) — RAGMemory override. Bypasses the encoder; writes provided vectors straight into Chroma.

## Scripts

| File | Purpose |
|---|---|
| [scripts/validate_stage_a_real.py](scripts/validate_stage_a_real.py) | Part 1 — hub quality on real LongMemEval queries. Measures displacement rate + hub-share@top-k. No GPU required. |
| [scripts/smoke_stage_a.py](scripts/smoke_stage_a.py) | Original synthetic-data smoke test. Mechanism validation only. |
| [scripts/split_hubs_eval_by_abs.py](scripts/split_hubs_eval_by_abs.py) | Post-hoc split of `eval_hubs.py` output by abstention vs recall questions + per-question-type breakdown. Primary column is confident-answer rate. |

## Configs

| File | Purpose |
|---|---|
| [configs/attack_rag.yaml](configs/attack_rag.yaml) | Shared config (`embed_model`, `top_k`, `answer_batch_size`, `data_path`, `split_path`). The REINFORCE-specific hyperparameters (group_size, lr, max_steps) are kept for historical compatibility but unused by the hubs pipeline. |
| [configs/splits/rag_attack.json](configs/splits/rag_attack.json) | 80/10/10 stratified split from `scripts/make_split.py`. 398 train / 48 val / 52 test. |

## Results so far

### Part 1 — Real-data hub quality (val, n=15)

| method | K | displacement_rate | hub_share@10 |
|---|---:|---:|---:|
| kmeans | 10 | 0.600 | 0.380 |
| kmeans | 20 | 0.667 | 0.467 |
| facility | 10 | 0.600 | 0.400 |

Hubs work at the retrieval layer. `kmeans` is cheap and competitive with `facility`; `kmeans` is the default going forward.

### Part 2 — End-to-end attack on val split (n=48, local 7B judge)

| config | acc drop (recall) | confident-answer drop | abst_p |
|---|---:|---:|---:|
| K10 placeholder | +7.0pp | +10.4pp | 22.9% |
| K10 prompt_injection | +11.6pp | +25.0pp | 37.5% |
| K30 prompt_injection | +9.3pp | +39.6pp | 52.1% |
| K100 contradictory | +11.6pp | +12.5pp | 25.0% |

### Part 2c — Test-split headline (n=54, GPT-4o judged) ⭐

These are the numbers to cite in the thesis writeup.

**Overall (n=54):**

| config | acc drop | confident-answer drop | hub/top | abst_p |
|---|---:|---:|---:|---:|
| K10_placeholder | +5.6pp | +3.7pp | 3.57 | 7.4% |
| K10_contradictory | +9.3pp | 0.0pp | 3.57 | 3.7% |
| K10_prompt_injection | +16.7pp | +22.2pp | 3.57 | 25.9% |
| K30_placeholder | +16.7pp | +7.4pp | 5.04 | 11.1% |
| K30_contradictory | +16.7pp | +7.4pp | 5.04 | 11.1% |
| **K30_prompt_injection** | **+20.4pp** | **+31.5pp** | **5.04** | **35.2%** |

**Recall-only (n=50, the real attacker metric):**

| config | acc drop | cnf drop | abst_p |
|---|---:|---:|---:|
| K30_placeholder | +18.0pp | +6.0pp | 10.0% |
| K30_contradictory | +18.0pp | +6.0pp | 10.0% |
| **K30_prompt_injection** | **+24.0pp** | **+26.0pp** | **30.0%** |

Clean baseline (recall): 66.0% accuracy, 96.0% confident-answer rate.
Attacked (K30_prompt_injection): 42.0% accuracy, 70.0% confident-answer rate.

**Per-task vulnerability (K30_prompt_injection, GPT-4o):**

| task type | n | acc drop |
|---|---:|---:|
| knowledge-update | 9 | +33.3pp |
| single-session-assistant | 3 | +33.3pp |
| single-session-user | 7 | +28.6pp |
| multi-session | 14 | +21.4pp |
| temporal-reasoning | 14 | +14.3pp |
| single-session-preference | 7 | 0.0pp |

Preference questions are resistant across all configs — the judge's rubric accepts degraded context. Knowledge-update is the softest target.

### Attribution: displacement vs weaponization

On recall-only (GPT-4o):
- K30 **placeholder** (no payload, just displacement): **+18.0pp**
- K30 **contradictory** (fake-correction text): **+18.0pp** (identical)
- K30 **prompt_injection** (abstention-bait text): **+24.0pp** (+6.0pp extra)

**~75% of the attack is pure geometric displacement; ~25% is adversarial payload weaponization.** The hubs winning top-k slots matters far more than what text they carry. This is the primary finding — it means the attack mechanism is fundamentally geometric, not content-based, and should transfer to Stage B (text-mode) as long as Stage B can get text encoded near hubs regardless of content.

Flag for verification: K30_placeholder and K30_contradictory produce identical numbers across every bucket and task type on test. Likely coincidence at n=50 (same 9 questions flip under K=30 displacement regardless of payload text), but worth a one-liner per-question cross-check before writeup.

### Decoupling caveat

Across Part 2/2c configs, `cos(encode(payload_text), hub_vector)` is in the 0.04–0.13 range. Vector-mode lets us pick payload text and hub vector independently; text-mode (Stage B) cannot. The Part 2 numbers are therefore an *inflated ceiling* — a realistic text-mode attack produces lower drops because the text's encoded vector won't land exactly at a hub centroid. Given the 75% displacement / 25% weaponization split, **a Stage B attack that matches Stage A's displacement effect would preserve most of the attack** (~14-18pp). The 6pp weaponization component is harder to transfer because it requires adversarial content under a naturalness constraint.

## Architecture and data flow

```
LongMemEval JSON
        │
        ▼
scripts/make_split.py ── configs/splits/rag_attack.json
        │
        ▼
attacks/hubness/stage_a_hubs.py
        H = compute_hubs(Q_train, K, method="kmeans")
        │
        ▼
┌─── attacks/eval_hubs.py (per eval question) ─────────────────────┐
│                                                                  │
│   build RAGMemory, index(haystack)                               │
│   clean_ctx    = mem.retrieve(q, date)                           │
│   for each (K, payload) config:                                  │
│       mem.index_raw_embeddings(H, hub_payload_text)              │
│       poisoned_ctx = mem.retrieve(q, date)                       │
│       mem.delete(hub_ids)                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
harness.ask_qwen_batch(contexts)  — vLLM-batched Qwen2.5-3B answers
harness.judge_answer_local_batch(...) — local 7B judge (iteration)
harness.judge_answer(...) — GPT-4o (final, optional)
        │
        ▼
results/stage_a/<name>.json + .csv   (confident-answer rate primary)
        │
        ▼
scripts/split_hubs_eval_by_abs.py   (recall vs abstention split)
```

## Run directions

### 0. Environment

On the cluster, inside `.venv/`:
```bash
pip install sentence-transformers<4 transformers<5 chromadb pyyaml tqdm
# vLLM, torch, peft are already installed from the REINFORCE bootstrap.
```

### 1. Part 1 — Stage A quality on real queries (no GPU)

```bash
python scripts/validate_stage_a_real.py \
    --n_val 15 --K_values 1 3 5 10 20 --methods kmeans facility \
    --out results/stage_a/real_data_validation.json
```

Pass criteria (currently pass): kmeans K=10 displacement ≥ 0.30 AND hub_share@10 ≥ 0.30.

### 2. Part 2 — End-to-end on val (with diagnostics)

```bash
CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 .venv/bin/python -m attacks.eval_hubs \
    --config configs/attack_rag.yaml \
    --split val \
    --hub_scope global \
    --K_values 10 30 100 --payloads placeholder contradictory prompt_injection \
    --output results/stage_a/eval_val_diag.json
```

Single-GPU fallback: `CUDA_VISIBLE_DEVICES=0 VLLM_GPU_MEM_UTIL=0.3 python -m attacks.eval_hubs ...`

Inspect recall-vs-abstention breakdown:
```bash
python scripts/split_hubs_eval_by_abs.py results/stage_a/eval_val_diag.json
```

### 3. Part 2c — Test-split headline number

```bash
CUDA_VISIBLE_DEVICES=0,1 JUDGE_DEVICE=cuda:1 .venv/bin/python -m attacks.eval_hubs \
    --config configs/attack_rag.yaml \
    --split test \
    --hub_scope global \
    --K_values 10 30 --payloads placeholder contradictory prompt_injection \
    --output results/stage_a/eval_test_conf.json \
    --use_gpt4o

python scripts/split_hubs_eval_by_abs.py results/stage_a/eval_test_conf.json --judge gpt4o
```

### 4. Part 4 — Stage B (scope + open questions, not yet built)

**Goal.** Produce realistic user/assistant conversation turns that, when indexed normally through the defender's encoder (`all-MiniLM-L6-v2`, no vector-mode privilege), land near a target hub vector. Then ship those turns through the regular memory pipeline and measure the resulting confident-answer drop. The "natural-looking + hub-like" pair is the whole problem.

**What we learned from Part 2 that changes Stage B scope:**

1. **75% of the attack is displacement, only 25% is payload content.** Stage B doesn't need adversarial content to recover most of the attack — it just needs text that encodes near hub centroids. This makes the realism constraint much less painful than I initially feared.
2. **Prompt-injection-style payloads add +6pp.** If Stage B can layer a stealthy abstention-bait signal into otherwise natural-looking turns, we pick that up. But it's a secondary objective.
3. **Decoupling cos is 0.04-0.13 today.** Stage B needs to raise it to some threshold — unclear what threshold (0.3? 0.5?) is enough to still win top-k.

**Design questions to resolve tomorrow:**

1. **Generation approach.**
   - **BoN (best-of-N sampling)** — prompt an attacker chat LLM for candidate turns, encode each, keep the one closest to the hub. Simple, no training. Scales with compute.
   - **Gradient-based** — discrete text optimization (GCG / HotFlip / PEZ) on token embeddings to minimize distance to hub. More precise but slower and less fluent.
   - **Hybrid** — BoN to find a rough candidate, gradient to refine. Best of both worlds.
   - **RL** — treat candidate generation as a policy, optimize against hub-distance + fluency + abstention-proxy. Most expensive; probably overkill given displacement does most of the work.

2. **Fitness function.** For candidate text `t`, target hub `h`:
   - `cos(encode(t), h)` — retrieval fitness (primary)
   - fluency score (e.g., attacker LLM log-prob) — naturalness constraint
   - abstention-signal proxy — optional. Could be "text mentions uncertainty / denial / no-memory-here." Measure separately.

3. **Budget.** Stage A used K=30 hubs. If each hub needs its own generated text, that's 30 text chunks to produce per attack. If we can reuse one text per multiple hubs (or one text per cluster of related hubs), budget drops.

4. **Target-hub selection.** Stage A used all K k-means centroids. Stage B might need to be picky — generate text for only the 5-10 easiest-to-hit hubs (ones whose region has dense natural-text neighbors in the encoder's training distribution).

5. **Evaluation.** Plug Stage B output into the same `attacks/eval_hubs.py` pipeline via the text-mode injection path (not `index_raw_embeddings` — actual `mem.index` with generated sessions). Same metrics (confident-answer rate drop, recall-only acc drop). Same test split.

**Minimal first cut (to discuss tomorrow):**

A small script `attacks/hubness/stage_b_bon.py` that:
1. Loads the K=30 hubs from Stage A.
2. For each hub, runs BoN=64 candidate generations from an attacker chat LLM (Qwen2.5-3B, already loaded via `_vllm_engines.get_attacker_engine`), prompting for plausible user/assistant turns about generic memory topics (work, health, relationships, etc.).
3. Encodes each candidate, keeps the one with highest `cos(encode(turn), hub)`.
4. Emits the 30 selected turns as a list of `PoisonSession`s.
5. Runs `eval_hubs.py` in text-mode (new flag `--stage b_sessions <file.json>`) on the test split.

Estimated effort: ~200 lines new code + ~30 line patch to `eval_hubs.py` to accept pre-built `PoisonSession` inputs. Half a day.

Open before coding: which attacker LLM, what prompt, what BoN budget, does the existing `_vllm_engines.get_attacker_engine` already fit the bill or do we need a fresh generation engine.

## Status

- Part 1 — Stage A hub quality — **done**, passing.
- Part 2 — End-to-end vector-mode eval with diagnostics (val split) — **done**.
- Part 2c — Test-split + GPT-4o headline — **done**. K30_prompt_injection: +20.4pp accuracy drop, +31.5pp confident-answer drop overall; +24.0pp accuracy drop on recall-only.
- Part 3 — Docs rewrite for hubs-primary framing — **done** (this document + [direction.md](direction.md)).
- Part 4 — Stage B text realization — **next**. Scope sketched above. User wants to focus on this next session; minimal first cut is ~200 lines code + eval_hubs.py patch. Stage B design questions to discuss before implementation.

## Tonight's session summary (2026-04-23)

Kicked off the pivot from REINFORCE to hubness. Sequence:
1. Built `scripts/validate_stage_a_real.py` — confirmed hubs displace real docs on LongMemEval queries (60% displacement at K=10 kmeans).
2. Built `attacks/eval_hubs.py` — end-to-end attack evaluator. Confirmed mechanism: hubs reach top-10 on ~90-100% of queries.
3. Val-split diagnostic runs uncovered the abstention-bucket confound. Reframed primary metric to confident-answer rate.
4. Added diagnostics: hub-count-in-top-k, abstention-markers detection, decoupling cos.
5. Added three payload modes: `placeholder`, `contradictory`, `prompt_injection`.
6. Test split + GPT-4o: locked in headline numbers, found 75% of the attack is pure displacement.
7. Rewrote ATTACK.md and direction.md for the new framing.

**Key decisions made tonight:**
- REINFORCE (under [attack/](attack/)) is parked permanently. Hubs is the primary track.
- Confident-answer rate is the primary attack metric, not overall accuracy.
- The displacement effect is the main contribution — the text carried by hubs matters less than expected.
- Stage B is next but unscoped in code.

**Thesis path still TBD** — I sketched three options in [direction.md](direction.md#phased-execution-revised). User was too tired to decide tonight; picking this up next session.

## Previous approach (REINFORCE, parked)

The [attack/](attack/) package contains an earlier GRPO-trained text-generation attacker (Qwen2.5-3B + LoRA rank 16, 5 composite reward components, curriculum schedule, 2000-step training loop). Infrastructure reached a working state (chunked logprob backward, vLLM LoRA hot-swap, multi-GPU judge split), but full training was not converged at the time of the pivot, and the contribution direction shifted to geometric hubs with a different primary metric. The REINFORCE code is preserved for ablation comparisons if the thesis needs them, but is not the active track. See commits `01798e8`..`7d671aa` for the infrastructure work and `1be8518` for the pivot.
