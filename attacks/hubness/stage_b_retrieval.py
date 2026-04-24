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
    l2_normalize,
    load_hubs,
    write_poison_file,
)


def _collect_corpus_rounds(
    data_path: str, train_qids: list[str]
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Walk every train-split question's haystack, pull out 2-turn rounds
    as (user_msg, assistant_msg) pairs. Returns pairs + metadata for each.
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    by_qid = {q["question_id"]: q for q in data}
    seen_texts: set[str] = set()
    pairs: list[tuple[str, str]] = []
    metas: list[dict] = []
    for qid in train_qids:
        q = by_qid.get(qid)
        if not q:
            continue
        sessions = q["haystack_sessions"]
        dates = q["haystack_dates"]
        sids = q.get(
            "haystack_session_ids",
            [str(i) for i in range(len(sessions))],
        )
        for session, date, sid in zip(sessions, dates, sids):
            i = 0
            round_idx = 0
            while i < len(session):
                if (
                    i + 1 < len(session)
                    and session[i]["role"] == "user"
                    and session[i + 1]["role"] == "assistant"
                ):
                    u = session[i]["content"].strip()
                    a = session[i + 1]["content"].strip()
                    key = f"{u}\n|\n{a}"
                    if key not in seen_texts:
                        seen_texts.add(key)
                        pairs.append((u, a))
                        metas.append({
                            "source_qid": qid,
                            "source_session_id": sid,
                            "source_date": date,
                            "round_index": round_idx,
                        })
                    i += 2
                    round_idx += 1
                else:
                    i += 1
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
