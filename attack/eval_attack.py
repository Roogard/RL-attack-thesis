"""Held-out evaluation of a trained attacker.

Loads a LoRA checkpoint, runs clean-vs-poisoned on the test split, judges
with GPT-4o (final numbers) AND the local 7B judge (for comparison).
Reports per-task and overall accuracy drop.

Usage:
    python -m attack.eval_attack \\
        --config configs/attack_rag.yaml \\
        --adapter results/rl_attacker/rag_v1/adapter_final \\
        --split test \\
        --n_poison 3 \\
        --output results/rl_attacker/rag_v1/eval_test_n3.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import defaultdict

import numpy as np
import torch

from attack.caches import CleanCache
from attack.environment import RolloutEnvironment
from attack.policy import AttackerPolicy
from attack.probes import DOMAIN_PROBES
from attack.rollout import run_rollout
from attack.train import load_config, load_dataset
from harness import judge_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter directory (or 'none' to eval untrained).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default=None,
                        help="Override cfg['data_path']. Use to eval a trained-on-_M "
                             "attacker against _S (or any other dataset file).")
    parser.add_argument("--split_path", default=None,
                        help="Override cfg['split_path']. Pair with --data to point at "
                             "an eval-only split file.")
    parser.add_argument("--cache_path", default=None,
                        help="Override the per-split cache path. Defaults to derivation "
                             "from cfg['cache_path'].")
    parser.add_argument("--n_poison", type=int, default=None,
                        help="Override n_poison from config (for budget sweeps).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--use_gpt4o", action="store_true",
                        help="Also run the GPT-4o judge on poisoned+clean answers.")
    parser.add_argument("--memory_read_access", action="store_true", default=None)
    parser.add_argument("--no_memory_read_access", dest="memory_read_access",
                        action="store_false")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_poison = args.n_poison if args.n_poison is not None else cfg["n_poison"]

    data_path = args.data or cfg["data_path"]
    split_path = args.split_path or cfg["split_path"]
    qs = load_dataset(data_path, split_path, args.split)
    print(f"[eval] {len(qs)} {args.split} questions from {data_path}")

    # Build (or load) cache for the eval split
    if args.cache_path:
        cache_path = args.cache_path
    else:
        base = os.path.splitext(os.path.basename(data_path))[0]  # e.g. longmemeval_s_cleaned
        cache_path = os.path.join(os.path.dirname(cfg["cache_path"]),
                                  f"{base}_{args.split}_clean_cache.pkl")
    if os.path.exists(cache_path):
        cache = CleanCache.load(cache_path)
    else:
        cache = CleanCache.build(qs, embed_model_name=cfg["embed_model"],
                                 top_k=cfg["top_k"],
                                 answer_batch_size=cfg["answer_batch_size"])
        cache.save(cache_path)

    adapter_path = None if args.adapter.lower() == "none" else args.adapter
    policy = AttackerPolicy(
        model_id=cfg["attacker_model_id"],
        lora_adapter_path=adapter_path,
        lora_rank=cfg["lora_rank"],
    )

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(cfg["embed_model"])

    if args.memory_read_access is None:
        mra = cfg.get("memory_read_access", True)
    else:
        mra = args.memory_read_access
    probes = DOMAIN_PROBES if mra else []

    # ── run rollouts on each test question ───────────────────────────────
    per_task = defaultdict(lambda: {"n": 0, "clean": 0, "poisoned": 0})
    per_task_gpt = defaultdict(lambda: {"n": 0, "clean": 0, "poisoned": 0})
    rows = []

    q_by_id = {q["question_id"]: q for q in qs}
    for qid in cache.qids():
        q = q_by_id[qid]
        entry = cache[qid]
        with torch.no_grad():
            r = run_rollout(
                q, entry, policy, embedder,
                n_poison=n_poison,
                domain=cfg["domain"],
                architecture_name=cfg["architecture_name"],
                probes=probes,
                top_k=cfg["top_k"],
                diversity_buffer=None,
            )

        t = q["question_type"]
        per_task[t]["n"] += 1
        per_task[t]["clean"] += int(entry.clean_correct)
        per_task[t]["poisoned"] += int(r.poisoned_correct)

        row = {
            "qid": qid,
            "question_type": t,
            "clean_pred": entry.clean_pred,
            "clean_correct": entry.clean_correct,
            "poisoned_pred": r.poisoned_pred,
            "poisoned_correct_local": r.poisoned_correct,
            "r_outcome": r.component_scores.r_outcome,
            "r_retrieval": r.component_scores.r_retrieval,
            "r_answer_div": r.component_scores.r_answer_div,
            "r_stealth": r.component_scores.r_stealth,
        }

        if args.use_gpt4o:
            row["clean_correct_gpt4o"] = judge_answer(
                q["question"], q["answer"], entry.clean_pred,
                q["question_type"], qid
            )
            row["poisoned_correct_gpt4o"] = judge_answer(
                q["question"], q["answer"], r.poisoned_pred,
                q["question_type"], qid
            )
            per_task_gpt[t]["n"] += 1
            per_task_gpt[t]["clean"] += int(row["clean_correct_gpt4o"])
            per_task_gpt[t]["poisoned"] += int(row["poisoned_correct_gpt4o"])

        rows.append(row)

    # ── aggregate ─────────────────────────────────────────────────────────
    def summarize(d):
        overall_n = sum(v["n"] for v in d.values())
        overall_clean = sum(v["clean"] for v in d.values())
        overall_poisoned = sum(v["poisoned"] for v in d.values())
        return {
            "per_task": {
                k: {
                    "n": v["n"],
                    "clean_acc": v["clean"] / max(v["n"], 1),
                    "poisoned_acc": v["poisoned"] / max(v["n"], 1),
                    "drop": (v["clean"] - v["poisoned"]) / max(v["n"], 1),
                }
                for k, v in d.items()
            },
            "overall": {
                "n": overall_n,
                "clean_acc": overall_clean / max(overall_n, 1),
                "poisoned_acc": overall_poisoned / max(overall_n, 1),
                "drop": (overall_clean - overall_poisoned) / max(overall_n, 1),
            },
        }

    summary = {
        "config": args.config,
        "adapter": adapter_path,
        "split": args.split,
        "n_poison": n_poison,
        "memory_read_access": mra,
        "local_judge": summarize(per_task),
    }
    if args.use_gpt4o:
        summary["gpt4o_judge"] = summarize(per_task_gpt)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with io.open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    # ── print ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"HELD-OUT EVAL ({args.split}, N={n_poison}, MRA={mra})")
    print("=" * 60)
    loc = summary["local_judge"]["overall"]
    print(f"LOCAL JUDGE: clean={loc['clean_acc']:.3f}  "
          f"poisoned={loc['poisoned_acc']:.3f}  drop={loc['drop']:.3f}")
    if args.use_gpt4o:
        gpt = summary["gpt4o_judge"]["overall"]
        print(f"GPT-4o:      clean={gpt['clean_acc']:.3f}  "
              f"poisoned={gpt['poisoned_acc']:.3f}  drop={gpt['drop']:.3f}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
