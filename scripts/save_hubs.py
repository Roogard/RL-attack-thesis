"""Compute Stage A hubs once and save to disk for Stage B scripts to load.

Avoids recomputing hubs (and re-encoding training queries) across every
Stage B run. Writes a pickle with:
    hubs        : (K, d) float32, L2-normalized
    K           : int
    method      : str
    seed        : int
    train_qids  : list[str]
    train_texts : list[str]  (the question texts that were encoded)
    train_vecs  : (n_train, d) float32, L2-normalized — reusable for hub
                  topic derivation in BoN

Usage:
    python scripts/save_hubs.py --K 30 --method kmeans --seed 0 \\
        --out results/stage_a/hubs_K30.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.hubness.stage_a_hubs import compute_hubs


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="LongMemEval/data/longmemeval_s_cleaned.json")
    p.add_argument("--split_path", default="configs/splits/rag_attack.json")
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--method", default="kmeans", choices=["kmeans", "facility"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--out", default="results/stage_a/hubs_K30.pkl")
    args = p.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    with open(args.split_path, encoding="utf-8") as f:
        splits = json.load(f)
    by_qid = {q["question_id"]: q for q in data}
    train_qids = [qid for qid in splits["train"] if qid in by_qid]
    train_texts = [by_qid[qid]["question"] for qid in train_qids]

    print(f"[save_hubs] encoding {len(train_texts)} train-split questions with {args.model}...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model)
    Q_train = _l2_normalize(
        np.asarray(enc.encode(train_texts, normalize_embeddings=True), dtype=np.float32)
    )

    print(f"[save_hubs] computing K={args.K} hubs via {args.method} (seed={args.seed})...")
    H, diag = compute_hubs(Q_train, args.K, D=None, method=args.method, seed=args.seed)
    H = _l2_normalize(H.astype(np.float32))
    print(f"  mean_max_sim={diag.mean_max_sim:.3f}  objective={diag.objective:.3f}")

    out = {
        "hubs": H,
        "K": args.K,
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "train_qids": train_qids,
        "train_texts": train_texts,
        "train_vecs": Q_train,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"[save_hubs] wrote {out_path}  (K={args.K}, d={H.shape[1]}, train n={len(train_qids)})")


if __name__ == "__main__":
    main()
