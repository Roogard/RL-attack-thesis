"""Stage B — Retrieval from natural-text corpus.

For each hub vector h, find the round (user/assistant pair) from the
training-split haystack rounds whose encoding is closest to h. These are
bona fide human-authored chat turns — naturalness is guaranteed by
construction, at the cost of being bounded by whatever happens to live
in the corpus.

Naturalness caveat: uses LongMemEval train-split haystack rounds as the
corpus. Some sessions may appear in both train and eval haystacks (as
shared distractors), so there's a small non-zero chance the selected
poison text already exists in the target's haystack. Document this in
the writeup; does not invalidate the attack (re-indexing an existing
round still adds a chunk) but is a caveat to flag.

Usage:
    python -m attacks.hubness.stage_b_retrieval \\
        --hubs results/stage_a/hubs_K30.pkl \\
        --out  results/stage_b/retrieval_K30.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.stage_b_common import (
    MultiRoundPoisonSessionRecord,
    encode_many,
    format_round_text,
    iter_longmemeval_train_rounds,
    l2_normalize,
    load_hubs,
    normalize_text,
    write_poison_file,
)


def _collect_corpus_rounds(
    data_path: str, train_qids: list[str]
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Walk every train-split question's haystack, pull out 2-turn rounds
    as (user_msg, assistant_msg) pairs. Returns pairs + metadata for each.

    Wraps the shared `iter_longmemeval_train_rounds` and applies the same
    in-memory normalized-text dedup the original implementation used.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    metas: list[dict] = []
    for u, a, meta in iter_longmemeval_train_rounds(data_path, train_qids):
        key = normalize_text(u) + "\n|\n" + normalize_text(a)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((u, a))
        # Back-compat: original meta fields the rest of this script reads.
        metas.append({
            "source_qid": meta["source_qid"],
            "source_session_id": meta["source_session_id"],
            "source_date": meta["source_date"],
            "round_index": meta["round_index"],
        })
    return pairs, metas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hubs", required=True, help="Path to hubs.pkl from save_hubs.py.")
    p.add_argument("--data", default="LongMemEval/data/longmemeval_s_cleaned.json")
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--model", default=None,
                   help="Encoder override; defaults to hubs['model'].")
    p.add_argument("--rounds_per_hub", type=int, default=1,
                   help="M: number of corpus rounds picked per hub (top-M by "
                        "cos-to-hub). M=1 reproduces the original single-pick "
                        "behavior; M>1 builds one M-round session per hub so "
                        "each hub's region claims M slots in top-k retrieval.")
    args = p.parse_args()
    if args.rounds_per_hub < 1:
        raise SystemExit("--rounds_per_hub must be >= 1")

    hubs_blob = load_hubs(args.hubs)
    H = hubs_blob["hubs"]
    train_qids = hubs_blob["train_qids"]
    model_name = args.model or hubs_blob["model"]
    print(f"[stage_b/retrieval] K={len(H)}  encoder={model_name}")

    print(f"[stage_b/retrieval] collecting train-split corpus rounds ...")
    pairs, metas = _collect_corpus_rounds(args.data, train_qids)
    print(f"  {len(pairs)} unique rounds across {len(train_qids)} train-split haystacks")

    from sentence_transformers import SentenceTransformer
    print(f"[stage_b/retrieval] loading encoder ...")
    encoder = SentenceTransformer(model_name)

    # Encode rounds in the EXACT format RAGMemory will reindex them in.
    print(f"[stage_b/retrieval] encoding {len(pairs)} rounds (batch={args.batch_size}) ...")
    t0 = time.time()
    texts = [format_round_text(u, a) for (u, a) in pairs]
    vecs = encode_many(encoder, texts, batch_size=args.batch_size)
    print(f"  encoded in {time.time()-t0:.1f}s  (shape={vecs.shape})")

    # For each hub, take top-M by cosine over the corpus.
    M = args.rounds_per_hub
    if M > len(pairs):
        raise SystemExit(
            f"--rounds_per_hub={M} exceeds corpus size {len(pairs)}"
        )
    print(f"[stage_b/retrieval] picking top-{M} per hub "
          f"({len(H)} hubs × {M} = {len(H) * M} chunks total) ...")
    sims = vecs @ H.T  # (n_corpus, K)
    records: list[MultiRoundPoisonSessionRecord] = []
    all_corpus_idxs: list[int] = []  # for cross-hub-dedup diagnostic
    for k in range(len(H)):
        col = sims[:, k]
        # argpartition gets the M largest indices (unsorted); then sort that
        # slice descending by sim to get a stable top-M ordering.
        if M < len(col):
            cand = np.argpartition(-col, M)[:M]
        else:
            cand = np.arange(len(col))
        cand_sorted = cand[np.argsort(-col[cand])]
        rounds = []
        for j in cand_sorted:
            j_int = int(j)
            u, a = pairs[j_int]
            rounds.append({
                "user_msg": u,
                "assistant_msg": a,
                "cos_to_hub": float(col[j_int]),
                "source": metas[j_int],
            })
            all_corpus_idxs.append(j_int)
        records.append(MultiRoundPoisonSessionRecord(
            hub_idx=k,
            rounds=rounds,
            meta={"M": M},
        ))

    cos_values = np.array(
        [r["cos_to_hub"] for rec in records for r in rec.rounds]
    )
    distinct_rounds = len(set(all_corpus_idxs))
    print(f"[stage_b/retrieval] cos to hub:  mean={cos_values.mean():.3f}  "
          f"min={cos_values.min():.3f}  max={cos_values.max():.3f}")
    print(f"[stage_b/retrieval] distinct corpus rounds across all selections: "
          f"{distinct_rounds} / {len(H) * M} "
          f"({100.0 * distinct_rounds / max(len(H) * M, 1):.1f}% unique)")

    write_poison_file(
        args.out, method="retrieval", hubs_source=args.hubs,
        sessions=records,
        extra={
            "corpus_size": len(pairs),
            "mean_cos_to_hub": float(cos_values.mean()),
            "M": M,
            "distinct_rounds_selected": distinct_rounds,
            "total_chunks_injected": len(H) * M,
        },
    )
    print(f"[stage_b/retrieval] wrote {args.out}")


if __name__ == "__main__":
    main()
