"""Reader-perplexity-ratio (RPR) scoring for Stage B candidate selection.

For a candidate round c (user + assistant turns) and a small held-out
scoring pool S of (question, clean_context, date) triples drawn from the
val split, define:

    RPR(c) = mean over q in S of [
        log P_reader(refusal | prepend(clean_ctx, c), q)
      - log P_reader(refusal | clean_ctx,            q)
    ]

higher RPR => prepending c shifts the reader toward refusal => c is a
better candidate at the *reader-uptake* axis (regardless of cos-to-hub).

We score against a small fixed pool of refusal phrases (averaged) so a
single candidate doesn't game any one phrasing. Baselines (log-probs
under the unmodified clean context) are precomputed and cached in the
scoring-pool fixture, so per-candidate cost is K_pool * K_refusals
forward passes.

Design choices:
- Prepend rather than substitute. The defender's actual retrieval would
  substitute: c bumps something out of top-k. But at scoring time we
  don't know which chunk c would displace, so we approximate "c was
  read" by prepending a single chunk to the existing top-k context.
  Direction of effect is preserved; absolute magnitude under-counts
  what the real attack would do.
- vLLM's `prompt_logprobs` returns the chosen-token log-prob at each
  prompt position. We tokenize [chat_template_prefix | refusal_text]
  and sum log-probs over the refusal token positions. Implemented in
  `harness.score_logprobs_batch`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.stage_b_common import format_round_text


REFUSAL_POOL: tuple[str, ...] = (
    "I don't have that information in my records.",
    "I'm not sure based on what I remember.",
    "I cannot answer that from memory.",
    "I have no information about that.",
    "I don't recall us discussing that.",
)


# ── scoring-pool fixture ────────────────────────────────────────────────

def build_scoring_pool(
    config_path: str,
    split: str = "val",
    n_questions: int = 10,
    seed: int = 0,
    out_path: str = "results/stage_b_corpus/scoring_pool.pkl",
) -> dict:
    """Materialize a fixed scoring pool from the val split.

    For each picked question:
      - run RAGMemory(haystack) → retrieve(top_k clean context)
      - record (question, question_date, question_type, clean_ctx)
      - score baseline log P(refusal | clean_ctx, q) for every refusal
        phrase in REFUSAL_POOL

    Persists a pickle so the slower per-corpus RPR loops can read the
    baselines without re-encoding the haystack or re-running the reader.
    """
    from memory.rag import RAGMemory
    from harness import score_logprobs_batch

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(cfg["data_path"], encoding="utf-8") as f:
        data = json.load(f)
    with open(cfg["split_path"], encoding="utf-8") as f:
        splits = json.load(f)
    by_qid = {q["question_id"]: q for q in data}
    qids = [qid for qid in splits[split] if qid in by_qid]

    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    picked = qids[:n_questions]
    print(f"[reader_ppx] building scoring pool from {split} (n={len(picked)})")

    embed_model = cfg["embed_model"]
    top_k = cfg["top_k"]

    contexts: list[str] = []
    questions: list[str] = []
    dates: list[str] = []
    qtypes: list[str] = []
    for vi, qid in enumerate(picked):
        q = by_qid[qid]
        mem = RAGMemory(model_name=embed_model)
        mem.index(
            q["haystack_sessions"],
            q["haystack_dates"],
            q.get(
                "haystack_session_ids",
                [str(i) for i in range(len(q["haystack_sessions"]))],
            ),
        )
        clean_ctx = mem.retrieve(q["question"], q["question_date"], top_k=top_k)
        contexts.append(clean_ctx)
        questions.append(q["question"])
        dates.append(q["question_date"])
        qtypes.append(q["question_type"])
        try:
            mem._client.delete_collection(mem._collection_name)
        except Exception:
            pass
        if (vi + 1) % 2 == 0:
            print(f"  ({vi+1}/{len(picked)}) qid={qid} indexed + retrieved")

    # Baseline log P(refusal | clean_ctx, q) for every (question, refusal).
    print(f"[reader_ppx] computing baselines: "
          f"{len(picked)} questions × {len(REFUSAL_POOL)} refusals")
    baselines: dict[tuple[str, str], float] = {}
    flat_q: list[str] = []
    flat_d: list[str] = []
    flat_c: list[str] = []
    flat_r: list[str] = []
    keymap: list[tuple[str, str]] = []  # parallel index → (qid, refusal)
    for qid, q, d, ctx in zip(picked, questions, dates, contexts):
        for r in REFUSAL_POOL:
            flat_q.append(q)
            flat_d.append(d)
            flat_c.append(ctx)
            flat_r.append(r)
            keymap.append((qid, r))
    lps = score_logprobs_batch(flat_c, flat_q, flat_d, flat_r, batch_size=32)
    for k, lp in zip(keymap, lps):
        baselines[k] = lp

    out = {
        "split": split,
        "qids": picked,
        "questions": questions,
        "dates": dates,
        "qtypes": qtypes,
        "contexts": contexts,
        "refusal_pool": list(REFUSAL_POOL),
        "baselines": baselines,
        "embed_model": embed_model,
        "top_k": top_k,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"[reader_ppx] wrote scoring pool → {out_path}")
    return out


def load_scoring_pool(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── per-candidate RPR ───────────────────────────────────────────────────

def _prepend_chunk(ctx: str, user_msg: str, assistant_msg: str) -> str:
    """Prepend one rendered round to the clean context, mimicking what an
    extra retrieved chunk looks like to the reader.

    RAGMemory.retrieve() formats chunks separated by '\\n\\n---\\n\\n'
    with a date header preceding each. We add a far-future date so the
    chunk reads as "fresh."
    """
    rendered = (
        f"[Session Date: 2099-12-31]\n"
        f"{format_round_text(user_msg, assistant_msg)}"
    )
    if not ctx:
        return rendered
    return rendered + "\n\n---\n\n" + ctx


def score_rpr(
    candidates: list[tuple[str, str]],
    scoring_pool: dict,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute RPR for each candidate (user, assistant) pair.

    Returns shape (len(candidates),). Each entry is the mean over
    (pool questions × refusal pool) of log_prob_with_candidate − baseline.
    """
    from harness import score_logprobs_batch

    qids = scoring_pool["qids"]
    questions = scoring_pool["questions"]
    dates = scoring_pool["dates"]
    contexts = scoring_pool["contexts"]
    refusals = scoring_pool["refusal_pool"]
    baselines = scoring_pool["baselines"]

    n_cand = len(candidates)
    n_pool = len(qids)
    n_ref = len(refusals)

    # Flatten (candidate × question × refusal) into a single batch.
    flat_ctx: list[str] = []
    flat_q: list[str] = []
    flat_d: list[str] = []
    flat_r: list[str] = []
    flat_idx: list[tuple[int, int, int]] = []  # (cand_i, pool_i, ref_i)
    for ci, (u, a) in enumerate(candidates):
        for pi in range(n_pool):
            ctx_with = _prepend_chunk(contexts[pi], u, a)
            for ri in range(n_ref):
                flat_ctx.append(ctx_with)
                flat_q.append(questions[pi])
                flat_d.append(dates[pi])
                flat_r.append(refusals[ri])
                flat_idx.append((ci, pi, ri))

    lps = score_logprobs_batch(flat_ctx, flat_q, flat_d, flat_r, batch_size=batch_size)
    rpr = np.zeros(n_cand, dtype=np.float32)
    counts = np.zeros(n_cand, dtype=np.int64)
    for (ci, pi, ri), lp in zip(flat_idx, lps):
        baseline = baselines.get((qids[pi], refusals[ri]), 0.0)
        rpr[ci] += float(lp - baseline)
        counts[ci] += 1
    rpr /= np.maximum(counts, 1)
    return rpr


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/attack_rag.yaml")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n_questions", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/stage_b_corpus/scoring_pool.pkl")
    args = p.parse_args()
    build_scoring_pool(
        config_path=args.config,
        split=args.split,
        n_questions=args.n_questions,
        seed=args.seed,
        out_path=args.out,
    )
