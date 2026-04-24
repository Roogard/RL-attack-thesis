"""Stage A real-data validation on LongMemEval_S.

Measures whether hubs computed from training-split question embeddings
actually displace legitimate top-1 retrievals on held-out val questions,
and whether they grab top-k slots after injection into a RAGMemory built
from each val question's real haystack.

Unlike scripts/smoke_stage_a.py (synthetic, mechanism check), this runs
against real encoder outputs + real haystacks, and its pass criteria are
the ones the Part 1 plan commits to.

Output: results/stage_a/real_data_validation.json (+ a human-readable
summary table printed to stdout).

Usage (CPU-fast; MiniLM encoding dominates):
    python scripts/validate_stage_a_real.py \\
        [--n_val 15] [--top_k 10] [--seed 0] \\
        [--K_values 1 3 5 10 20] [--methods kmeans facility]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from attacks.hubness.stage_a_hubs import compute_hubs
from memory.rag import RAGMemory


DATASET_PATH = Path("LongMemEval/data/longmemeval_s_cleaned.json")
SPLIT_PATH = Path("configs/splits/rag_attack.json")
OUT_PATH = Path("results/stage_a/real_data_validation.json")


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _load_data():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)
    by_qid = {q["question_id"]: q for q in data}
    train_qids = [qid for qid in split["train"] if qid in by_qid]
    val_qids = [qid for qid in split["val"] if qid in by_qid]
    return by_qid, train_qids, val_qids


def _index_and_get_embeddings(question: dict, model_name: str) -> tuple[RAGMemory, np.ndarray, list[str]]:
    """Build RAGMemory for one question's haystack; return (mem, D_q, doc_ids).

    D_q is L2-normalized so cos sim = matmul.
    """
    mem = RAGMemory(model_name=model_name)
    mem.index(
        question["haystack_sessions"],
        question["haystack_dates"],
        question.get(
            "haystack_session_ids",
            [str(i) for i in range(len(question["haystack_sessions"]))],
        ),
    )
    got = mem._collection.get(include=["embeddings"])
    D_q = np.asarray(got["embeddings"], dtype=np.float32)
    D_q = _l2_normalize(D_q)
    return mem, D_q, list(got["ids"])


def _config_hub_share(
    mem: RAGMemory,
    H: np.ndarray,
    queries: list[str],
    top_k: int,
    hub_tag: str,
    question_dates: list[str],
) -> dict:
    """Inject H into mem, run retrieve for each query, count hub chunks,
    then delete the injected hubs. Returns per-config metrics.
    """
    K = len(H)
    hub_ids = [f"{hub_tag}_{i}" for i in range(K)]
    hub_docs = [f"[{hub_tag}_{i}_INJECTED]" for i in range(K)]
    # Dates chosen to lex-sort AFTER any real haystack date; RAG.retrieve only
    # uses date for post-hoc chronological ordering, not top-k selection.
    hub_metas = [
        {"date": "2099-12-31", "session_id": hub_tag, "round_index": i}
        for i in range(K)
    ]
    mem.index_raw_embeddings(H, hub_metas, hub_ids, documents=hub_docs)

    hub_counts = []
    for q, qdate in zip(queries, question_dates):
        ctx = mem.retrieve(q, qdate, top_k=top_k)
        chunks = ctx.split("\n\n---\n\n") if ctx else []
        n = sum(1 for c in chunks if f"{hub_tag}_" in c and "INJECTED" in c)
        hub_counts.append(n)

    # Remove injected hubs so subsequent configs see a clean store.
    mem._collection.delete(ids=hub_ids)

    mean_hub_count = float(np.mean(hub_counts))
    return {
        "mean_hub_count": mean_hub_count,
        "hub_share_top_k": mean_hub_count / top_k,
        "any_hub_rate": float(np.mean([c > 0 for c in hub_counts])),
        "per_query_counts": hub_counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_val", type=int, default=15,
                   help="Number of val questions to sample (each requires indexing ~50 sessions).")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K_values", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    p.add_argument("--methods", nargs="+", default=["kmeans", "facility"],
                   choices=["kmeans", "facility"])
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    by_qid, train_qids, val_qids = _load_data()
    print(f"[validate] loaded {len(by_qid)} questions; train={len(train_qids)} val={len(val_qids)}")

    val_sample = list(rng.choice(val_qids, size=min(args.n_val, len(val_qids)), replace=False))
    print(f"[validate] sampled {len(val_sample)} val questions: {val_sample[:5]}{'...' if len(val_sample) > 5 else ''}")

    print(f"[validate] loading encoder {args.model}...")
    encoder = SentenceTransformer(args.model)

    train_texts = [by_qid[qid]["question"] for qid in train_qids]
    val_texts = [by_qid[qid]["question"] for qid in val_sample]
    val_dates = [by_qid[qid]["question_date"] for qid in val_sample]

    print(f"[validate] encoding {len(train_texts)} train + {len(val_texts)} val questions...")
    t0 = time.time()
    Q_train = _l2_normalize(np.asarray(encoder.encode(train_texts, normalize_embeddings=True), dtype=np.float32))
    Q_val = _l2_normalize(np.asarray(encoder.encode(val_texts, normalize_embeddings=True), dtype=np.float32))
    print(f"  encoded in {time.time()-t0:.1f}s; Q_train={Q_train.shape}, Q_val={Q_val.shape}")

    # Precompute hubs per (method, K). Only depends on Q_train.
    print(f"[validate] computing hubs for methods={args.methods} K={args.K_values} on Q_train...")
    hubs_by_config: dict[tuple[str, int], tuple[np.ndarray, dict]] = {}
    for method in args.methods:
        for K in args.K_values:
            if K > len(Q_train):
                print(f"  skipping method={method} K={K} (exceeds Q_train size {len(Q_train)})")
                continue
            t0 = time.time()
            H, diag = compute_hubs(Q_train, K, D=None, method=method, seed=args.seed)
            dt = time.time() - t0
            hubs_by_config[(method, K)] = (
                _l2_normalize(H.astype(np.float32)),
                {
                    "train_mean_max_sim": diag.mean_max_sim,
                    "train_objective": diag.objective,
                    "compute_seconds": dt,
                },
            )
            print(f"  {method} K={K}: train mean_max_sim={diag.mean_max_sim:.3f} ({dt:.2f}s)")

    # Per-val-question metrics.
    per_question = []
    for vi, qid in enumerate(val_sample):
        q = by_qid[qid]
        q_vec = Q_val[vi : vi + 1]  # (1, d)
        print(f"[validate] ({vi+1}/{len(val_sample)}) indexing qid={qid} ({len(q['haystack_sessions'])} sessions)...")
        t0 = time.time()
        mem, D_q, _ = _index_and_get_embeddings(q, args.model)
        idx_time = time.time() - t0
        print(f"  indexed {D_q.shape[0]} chunks in {idx_time:.1f}s")

        # Legitimate top-1 cos for this query
        top_d_sim = float((q_vec @ D_q.T).max())

        q_entry = {
            "question_id": qid,
            "n_chunks": int(D_q.shape[0]),
            "index_seconds": idx_time,
            "legit_top_d_sim": top_d_sim,
            "configs": {},
        }

        for (method, K), (H, _) in hubs_by_config.items():
            # Displacement: does max cos(hub, q) beat max cos(doc, q)?
            hub_max_sim = float((q_vec @ H.T).max())
            displaced = hub_max_sim > top_d_sim

            # Hub share via real retrieval (top_k=args.top_k)
            share = _config_hub_share(
                mem, H, [q["question"]], args.top_k,
                hub_tag=f"HUB_{method}_K{K}", question_dates=[q["question_date"]],
            )

            q_entry["configs"][f"{method}_K{K}"] = {
                "hub_max_sim": hub_max_sim,
                "displaced": displaced,
                "hub_share_top_k": share["hub_share_top_k"],
                "any_hub_in_top_k": share["any_hub_rate"] == 1.0,
            }

        per_question.append(q_entry)
        # Drop mem to free Chroma state.
        try:
            mem._client.delete_collection(mem._collection_name)
        except Exception:
            pass

    # Aggregate per (method, K).
    summary = {}
    for (method, K), (_, hub_info) in hubs_by_config.items():
        key = f"{method}_K{K}"
        displacements = [qe["configs"][key]["displaced"] for qe in per_question]
        hub_shares = [qe["configs"][key]["hub_share_top_k"] for qe in per_question]
        any_hits = [qe["configs"][key]["any_hub_in_top_k"] for qe in per_question]
        summary[key] = {
            "method": method,
            "K": K,
            "train_mean_max_sim": hub_info["train_mean_max_sim"],
            "train_objective": hub_info["train_objective"],
            "val_displacement_rate": float(np.mean(displacements)),
            "val_mean_hub_share_top_k": float(np.mean(hub_shares)),
            "val_any_hub_in_top_k_rate": float(np.mean(any_hits)),
            "n_val": len(per_question),
        }

    # Pass criteria from plan: kmeans K=10 displacement >= 0.30 AND hub_share >= 0.30.
    key_gate = "kmeans_K10"
    gate = summary.get(key_gate)
    gate_pass = (
        gate is not None
        and gate["val_displacement_rate"] >= 0.30
        and gate["val_mean_hub_share_top_k"] >= 0.30
    )

    report = {
        "args": vars(args),
        "val_qids_sampled": list(val_sample),
        "per_question": per_question,
        "summary": summary,
        "gate": {
            "key": key_gate,
            "passed": bool(gate_pass),
            "values": gate,
            "thresholds": {"val_displacement_rate": 0.30, "val_mean_hub_share_top_k": 0.30},
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("STAGE A REAL-DATA VALIDATION — SUMMARY")
    print("=" * 72)
    print(f"{'config':<20} {'train_max':>10} {'val_disp':>10} {'hub_share':>10} {'any_hit':>10}")
    print("-" * 72)
    for key in sorted(summary.keys(), key=lambda k: (summary[k]["method"], summary[k]["K"])):
        s = summary[key]
        print(f"{key:<20} "
              f"{s['train_mean_max_sim']:>10.3f} "
              f"{s['val_displacement_rate']:>10.3f} "
              f"{s['val_mean_hub_share_top_k']:>10.3f} "
              f"{s['val_any_hub_in_top_k_rate']:>10.3f}")
    print("-" * 72)
    print(f"GATE ({key_gate}): displacement>=0.30 AND hub_share>=0.30  -> "
          f"{'PASS' if gate_pass else 'FAIL'}")
    print(f"Report: {out_path}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
