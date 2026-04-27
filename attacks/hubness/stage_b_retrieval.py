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
    PoisonSessionRecord,
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
    args = p.parse_args()

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

    # For each hub, argmax cosine over the corpus.
    print(f"[stage_b/retrieval] picking top-1 per hub ...")
    sims = vecs @ H.T  # (n_corpus, K)
    records: list[PoisonSessionRecord] = []
    for k in range(len(H)):
        best_idx = int(sims[:, k].argmax())
        best_cos = float(sims[best_idx, k])
        u, a = pairs[best_idx]
        records.append(PoisonSessionRecord(
            hub_idx=k,
            user_msg=u,
            assistant_msg=a,
            cos_to_hub=best_cos,
            meta={"retrieval": metas[best_idx]},
        ))

    cos_values = np.array([r.cos_to_hub for r in records])
    print(f"[stage_b/retrieval] cos to hub:  mean={cos_values.mean():.3f}  "
          f"min={cos_values.min():.3f}  max={cos_values.max():.3f}")

    write_poison_file(
        args.out, method="retrieval", hubs_source=args.hubs,
        sessions=records,
        extra={"corpus_size": len(pairs), "mean_cos_to_hub": float(cos_values.mean())},
    )
    print(f"[stage_b/retrieval] wrote {args.out}")


if __name__ == "__main__":
    main()
