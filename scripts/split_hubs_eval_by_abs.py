"""Post-hoc split of an eval_hubs.py output by abstention vs recall questions.

LongMemEval questions come in two flavors for the accuracy metric:
  - recall: "what did I say about X" — abstention = wrong answer
  - abstention: "did we ever discuss Y" where Y wasn't discussed —
                abstention = correct answer (marked by '_abs' qid suffix)

An attack that makes the reader abstain more often will:
  - drop accuracy on recall questions (abstaining on real-recall = wrong)
  - raise accuracy on abstention questions (correctly refusing)

The overall accuracy mixes these two effects and can show near-zero drop
even when the attack is clearly working. This script breaks out each
config's drop on recall-only, abstention-only, and per-question-type.

Usage:
    python scripts/split_hubs_eval_by_abs.py results/stage_a/eval_val_diag.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def _is_abs(qid: str, question_type: str) -> bool:
    # harness.py treats '_abs' in question_id as the abstention marker.
    return "_abs" in qid or question_type == "abstention"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_json", help="Path to an eval_hubs.py --output JSON.")
    p.add_argument("--judge", default="local", choices=["local", "gpt4o"])
    args = p.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    n_eval = data.get("n_eval")
    top_k = data.get("top_k")

    # Rows are duplicated per config (one row per (qid, config)). Group by
    # config, then split each group by abstention/recall.
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)

    clean_key = f"clean_correct_{args.judge}"
    pois_key = f"poisoned_correct_{args.judge}"

    def bucket_stats(rows: list[dict], pred: callable) -> dict:
        sel = [r for r in rows if pred(r)]
        n = len(sel)
        clean = sum(1 for r in sel if r[clean_key])
        pois = sum(1 for r in sel if r[pois_key])
        # abstention behavior metrics
        a_clean = sum(1 for r in sel if r.get("clean_abstained"))
        a_pois = sum(1 for r in sel if r.get("poisoned_abstained"))
        return {
            "n": n,
            "clean_acc": (clean / n) if n else 0.0,
            "pois_acc": (pois / n) if n else 0.0,
            "drop": ((clean - pois) / n) if n else 0.0,
            "abst_c_rate": (a_clean / n) if n else 0.0,
            "abst_p_rate": (a_pois / n) if n else 0.0,
            "abst_delta": ((a_pois - a_clean) / n) if n else 0.0,
        }

    # ── recall vs abstention split ───────────────────────────────────────
    # Primary attack metric: confident-answer rate = 1 - abstention_rate.
    # Under the attacker's framing, abstention IS a successful attack —
    # a memory system that refuses to answer is a broken memory system.
    # The traditional accuracy drop is shown alongside as secondary.
    print()
    print("=" * 128)
    print(f"POST-HOC SPLIT — {args.eval_json}   (judge={args.judge}, n_eval={n_eval}, top_k={top_k})")
    print("=" * 128)
    header = (
        f"{'config':<22} {'bucket':<10} {'n':>3}   "
        f"{'cnf_c':>6} {'cnf_p':>6} {'cnf_Δ':>6}   "
        f"{'acc_c':>6} {'acc_p':>6} {'acc_Δ':>6}   "
        f"{'abst_c%':>7} {'abst_p%':>7}"
    )
    print(header)
    print("-" * len(header))
    for cfg in sorted(by_cfg):
        cfg_rows = by_cfg[cfg]
        recall = bucket_stats(cfg_rows, lambda r: not _is_abs(r["qid"], r["question_type"]))
        abst = bucket_stats(cfg_rows, lambda r: _is_abs(r["qid"], r["question_type"]))
        overall = bucket_stats(cfg_rows, lambda r: True)
        for bucket_name, s in [("recall", recall), ("abstention", abst), ("overall", overall)]:
            if s["n"] == 0:
                continue
            conf_c = 1.0 - s["abst_c_rate"]
            conf_p = 1.0 - s["abst_p_rate"]
            conf_d = conf_c - conf_p
            print(
                f"{cfg:<22} {bucket_name:<10} {s['n']:>3}   "
                f"{conf_c:>6.3f} {conf_p:>6.3f} {conf_d:>+6.3f}   "
                f"{s['clean_acc']:>6.3f} {s['pois_acc']:>6.3f} {s['drop']:>+6.3f}   "
                f"{100*s['abst_c_rate']:>6.1f}% {100*s['abst_p_rate']:>6.1f}%"
            )
        print()

    # ── per-question-type breakdown ──────────────────────────────────────
    print()
    print("=" * 110)
    print("PER-QUESTION-TYPE BREAKDOWN")
    print("=" * 110)
    # Group by (cfg, question_type) — show drops
    qts = sorted({r["question_type"] for r in rows})
    header = f"{'config':<22} " + " ".join(f"{qt[:14]:>14}" for qt in qts)
    print(header)
    print(f"{'':<22} " + " ".join(f"{'drop (n)':>14}" for _ in qts))
    print("-" * len(header))
    for cfg in sorted(by_cfg):
        cfg_rows = by_cfg[cfg]
        parts = [f"{cfg:<22}"]
        for qt in qts:
            sel = [r for r in cfg_rows if r["question_type"] == qt]
            n = len(sel)
            if n == 0:
                parts.append(f"{'—':>14}")
                continue
            clean = sum(1 for r in sel if r[clean_key])
            pois = sum(1 for r in sel if r[pois_key])
            drop = (clean - pois) / n
            parts.append(f"{drop:>+7.3f} (n={n:>2})")
        print(" ".join(parts))

    # ── abstention-only submetric: judge-credit-for-attack ────────────────
    # For abstention questions, the attack "helps" the metric (model
    # correctly refuses more often). For recall questions, it hurts.
    # The attacker's real figure of merit is the recall-only drop.
    print()
    print("=" * 110)
    print("TLDR")
    print("=" * 110)
    print("For each config, recall-only drop is the number the attacker actually wants to maximize.")
    print("Large positive recall-drop + zero overall-drop = attack works but is hidden by the benchmark's")
    print("abstention-mixing effect. Report recall-only drop as the main attack metric going forward.")


if __name__ == "__main__":
    main()
