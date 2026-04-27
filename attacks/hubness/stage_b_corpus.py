"""Stage B — corpus-based selection with two-axis (cos + RPR) fitness.

Pipeline per corpus:
  1. Load hubs (K × d, unit-normalized).
  2. Stream the corpus JSONL, encode every (user, assistant) round through
     the defender's MiniLM encoder in chunks. Cache encoded vectors to .npy.
  3. For each hub h:
       a. Take top-N candidates by cos(encode(round), h).      [Axis 1]
       b. Score those N with reader_ppx.score_rpr().          [Axis 2]
       c. Min-max normalize cos and RPR within the top-N pool.
       d. Pick argmax over (alpha * cos_norm + beta * rpr_norm).
  4. Emit a poison_file.json that `attacks/eval_hubs.py --poison_file`
     can ingest unchanged.

Union mode: same flow, but the candidate pool is concatenated across
multiple corpora. Each emitted poison-session record records its
source_corpus in `meta.source_corpus` so post-eval analysis can build
a per-hub provenance histogram.

Usage:
  # Per-corpus selection
  python -m attacks.hubness.stage_b_corpus \\
      --hubs results/stage_a/hubs_K30.pkl \\
      --corpus_id nl_web \\
      --corpus_path data/corpora/nl_web.jsonl \\
      --scoring_pool results/stage_b_corpus/scoring_pool.pkl \\
      --out results/stage_b_corpus/poison_nl_web.json

  # Union selection (multiple --corpus_path entries)
  python -m attacks.hubness.stage_b_corpus \\
      --hubs results/stage_a/hubs_K30.pkl \\
      --union \\
      --corpus_paths data/corpora/nl_web.jsonl data/corpora/nl_memory.jsonl ... \\
      --scoring_pool results/stage_b_corpus/scoring_pool.pkl \\
      --out results/stage_b_corpus/poison_union.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.reader_ppx import load_scoring_pool, score_rpr
from attacks.hubness.stage_b_common import (
    PoisonSessionRecord,
    encode_many,
    format_round_text,
    iter_corpus_jsonl,
    load_hubs,
    write_poison_file,
)


def _encode_corpus_streaming(
    corpus_paths: list[str],
    corpus_ids: list[str],
    encoder,
    batch_size: int = 128,
    cache_path: str | None = None,
) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    """Encode every round across one or more corpus JSONL files.

    Returns:
      vecs: (N_total, d) float32, L2-normalized
      pairs: list of (user, assistant) parallel to vecs
      sources: list of corpus_id parallel to vecs (for provenance)
    """
    if cache_path and Path(cache_path).exists():
        print(f"[stage_b_corpus] loading cached encoding {cache_path}")
        blob = np.load(cache_path, allow_pickle=True).item()
        return blob["vecs"], blob["pairs"], blob["sources"]

    pairs: list[tuple[str, str]] = []
    sources: list[str] = []
    chunks: list[np.ndarray] = []
    buf_texts: list[str] = []
    buf_pairs: list[tuple[str, str]] = []
    buf_src: list[str] = []
    n_total = 0
    t0 = time.time()
    BUF_FLUSH = 8192

    def flush():
        nonlocal buf_texts, buf_pairs, buf_src
        if not buf_texts:
            return
        v = encode_many(encoder, buf_texts, batch_size=batch_size)
        chunks.append(v)
        pairs.extend(buf_pairs)
        sources.extend(buf_src)
        buf_texts = []
        buf_pairs = []
        buf_src = []

    for path, cid in zip(corpus_paths, corpus_ids):
        if not Path(path).exists():
            print(f"[stage_b_corpus] WARNING: {path} missing — skipping {cid}")
            continue
        for row in iter_corpus_jsonl(path):
            u, a = row["user"], row["assistant"]
            buf_texts.append(format_round_text(u, a))
            buf_pairs.append((u, a))
            buf_src.append(cid)
            n_total += 1
            if len(buf_texts) >= BUF_FLUSH:
                flush()
                if n_total % (BUF_FLUSH * 4) == 0:
                    print(f"  encoded {n_total} rounds  ({time.time()-t0:.1f}s)")
    flush()

    if not chunks:
        raise RuntimeError(f"no rounds encoded from {corpus_paths}")
    vecs = np.concatenate(chunks, axis=0).astype(np.float32)
    print(f"[stage_b_corpus] encoded {n_total} rounds in {time.time()-t0:.1f}s "
          f"(shape={vecs.shape})")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, {"vecs": vecs, "pairs": pairs, "sources": sources},
                allow_pickle=True)
        print(f"[stage_b_corpus] cached → {cache_path}")
    return vecs, pairs, sources


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def select_per_hub(
    H: np.ndarray,
    vecs: np.ndarray,
    pairs: list[tuple[str, str]],
    sources: list[str],
    scoring_pool: dict | None,
    alpha: float,
    beta: float,
    top_n: int,
    rpr_batch: int,
) -> list[PoisonSessionRecord]:
    """Two-axis selection. Returns one PoisonSessionRecord per hub.

    If scoring_pool is None or beta == 0, falls back to cos-only
    (top-1 by cos within top_n), preserving compatibility with the
    plain stage_b_retrieval baseline.
    """
    K, d = H.shape
    sims = vecs @ H.T  # (N, K)
    records: list[PoisonSessionRecord] = []
    for hub_idx in range(K):
        col = sims[:, hub_idx]
        # Top-N candidates by cosine
        n_eff = min(top_n, len(col))
        top_idx = np.argpartition(-col, n_eff - 1)[:n_eff]
        top_idx = top_idx[np.argsort(-col[top_idx])]
        cos_top = col[top_idx]

        if scoring_pool is None or beta == 0.0:
            best_local = 0
            rpr_top = None
        else:
            cands = [pairs[i] for i in top_idx]
            t0 = time.time()
            rpr_top = score_rpr(cands, scoring_pool, batch_size=rpr_batch)
            cos_n = _minmax(cos_top.astype(np.float32))
            rpr_n = _minmax(rpr_top.astype(np.float32))
            score = alpha * cos_n + beta * rpr_n
            best_local = int(np.argmax(score))
            print(
                f"  hub {hub_idx:>2d}: cos[best]={cos_top[best_local]:.3f} "
                f"rpr[best]={rpr_top[best_local]:+.3f} "
                f"({time.time()-t0:.1f}s for {n_eff} cands)"
            )
        best_global = int(top_idx[best_local])
        u, a = pairs[best_global]
        records.append(PoisonSessionRecord(
            hub_idx=hub_idx,
            user_msg=u,
            assistant_msg=a,
            cos_to_hub=float(cos_top[best_local]),
            meta={
                "source_corpus": sources[best_global],
                "rpr": (float(rpr_top[best_local]) if rpr_top is not None else None),
                "top_n_pool_size": int(n_eff),
                "fitness_alpha": alpha,
                "fitness_beta": beta,
                "cos_top_best": float(cos_top[best_local]),
                "cos_top_min": float(cos_top.min()),
                "cos_top_max": float(cos_top.max()),
            },
        ))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hubs", required=True)
    p.add_argument("--scoring_pool", default=None,
                   help="Path to a scoring_pool.pkl from reader_ppx.build_scoring_pool. "
                        "If omitted, falls back to cos-only top-1 selection.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="weight on cos_to_hub (normalized within per-hub top-N)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="weight on RPR (normalized within per-hub top-N)")
    p.add_argument("--top_n", type=int, default=200,
                   help="per-hub candidate pool size to RPR-rerank")
    p.add_argument("--rpr_batch", type=int, default=32)
    p.add_argument("--encode_batch", type=int, default=128)
    p.add_argument("--cache_dir", default="results/stage_b_corpus/encodings",
                   help="If set, cache encoded vectors per (corpus list, encoder) here.")
    p.add_argument("--out", required=True, help="Output poison_file JSON.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--corpus_id", default=None,
                      help="Single-corpus selection. Use with --corpus_path.")
    mode.add_argument("--union", action="store_true",
                      help="Union selection across multiple corpora. Use with --corpus_paths.")

    p.add_argument("--corpus_path", default=None,
                   help="Path to a corpus JSONL when --corpus_id is set.")
    p.add_argument("--corpus_paths", nargs="+", default=None,
                   help="Multiple corpus JSONL paths when --union is set.")
    p.add_argument("--corpus_ids", nargs="+", default=None,
                   help="Optional explicit corpus IDs aligned with --corpus_paths "
                        "(default: derive from filename stems).")

    args = p.parse_args()

    if args.corpus_id is not None and args.corpus_path is None:
        raise SystemExit("--corpus_id requires --corpus_path")
    if args.union and not args.corpus_paths:
        raise SystemExit("--union requires --corpus_paths")

    if args.union:
        corpus_paths = args.corpus_paths
        corpus_ids = args.corpus_ids or [Path(p).stem for p in corpus_paths]
        method_label = "corpus_union"
        cache_key = "union_" + "_".join(sorted(corpus_ids))
    else:
        corpus_paths = [args.corpus_path]
        corpus_ids = [args.corpus_id]
        method_label = f"corpus_{args.corpus_id}"
        cache_key = args.corpus_id

    hubs_blob = load_hubs(args.hubs)
    H = hubs_blob["hubs"]
    encoder_name = hubs_blob["model"]
    print(f"[stage_b_corpus] K={len(H)}  encoder={encoder_name}  "
          f"alpha={args.alpha} beta={args.beta} top_n={args.top_n}")

    # Encode corpus (with caching)
    cache_path = (
        str(Path(args.cache_dir) / f"{cache_key}.npy") if args.cache_dir else None
    )
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(encoder_name)
    vecs, pairs, sources = _encode_corpus_streaming(
        corpus_paths, corpus_ids, encoder,
        batch_size=args.encode_batch, cache_path=cache_path,
    )

    scoring_pool = None
    if args.scoring_pool:
        scoring_pool = load_scoring_pool(args.scoring_pool)
        print(f"[stage_b_corpus] scoring pool loaded "
              f"({len(scoring_pool['qids'])} questions × "
              f"{len(scoring_pool['refusal_pool'])} refusals)")
    else:
        print("[stage_b_corpus] no scoring pool → cos-only (beta=0 effectively)")

    records = select_per_hub(
        H, vecs, pairs, sources,
        scoring_pool=scoring_pool,
        alpha=args.alpha,
        beta=args.beta,
        top_n=args.top_n,
        rpr_batch=args.rpr_batch,
    )

    cos_values = np.array([r.cos_to_hub for r in records])
    print(f"[stage_b_corpus] cos_to_hub: mean={cos_values.mean():.3f}  "
          f"min={cos_values.min():.3f}  max={cos_values.max():.3f}")
    if scoring_pool is not None:
        rpr_values = np.array([r.meta["rpr"] for r in records if r.meta.get("rpr") is not None])
        if len(rpr_values):
            print(f"[stage_b_corpus] rpr:        mean={rpr_values.mean():+.3f}  "
                  f"min={rpr_values.min():+.3f}  max={rpr_values.max():+.3f}")

    # Provenance histogram (always printed; especially useful in union mode).
    from collections import Counter
    prov = Counter(r.meta["source_corpus"] for r in records)
    print(f"[stage_b_corpus] provenance: " + ", ".join(
        f"{k}={v}" for k, v in prov.most_common()
    ))

    write_poison_file(
        args.out,
        method=method_label,
        hubs_source=args.hubs,
        sessions=records,
        extra={
            "corpora": corpus_ids,
            "alpha": args.alpha,
            "beta": args.beta,
            "top_n": args.top_n,
            "scoring_pool": args.scoring_pool,
            "mean_cos_to_hub": float(cos_values.mean()),
            "provenance": dict(prov),
        },
    )
    print(f"[stage_b_corpus] wrote {args.out}")


if __name__ == "__main__":
    main()
