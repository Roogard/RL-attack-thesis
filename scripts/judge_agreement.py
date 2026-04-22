"""
Judge-judge agreement: local Qwen2.5-7B vs GPT-4o.

The RL training loop calls `judge_answer_local` thousands of times per run
(once per rollout for R_outcome). If the local 7B and the published GPT-4o
judge disagree substantially, the attacker optimizes a proxy and the real
headline number (GPT-4o) won't move.

This script replays the existing rag/full_history/rl_memory prediction files,
runs the local judge on every row, and compares against the GPT-4o label
already stored as `autoeval_label`. Reports Cohen's kappa, raw agreement,
and per-task-type breakdown.

Plan target: Cohen's kappa >= 0.75. Below that, retune prompts or upgrade
the local judge to Qwen2.5-14B-Instruct.

Usage:
    python scripts/judge_agreement.py \\
        --data LongMemEval/data/longmemeval_s_cleaned.json \\
        --eval_files results/benchmark/rag.eval.jsonl \\
                     results/benchmark/full_history.eval.jsonl \\
        --n 200 \\
        --output results/judge_agreement/agreement.json
"""

import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import judge_answer_local


def load_dataset(path):
    with io.open(path, encoding="utf-8") as f:
        return {q["question_id"]: q for q in json.load(f)}


def load_eval_rows(path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "autoeval_label" in rec:
                rows.append(rec)
    return rows


def cohen_kappa(y1, y2):
    """Compute Cohen's kappa for two binary labelings."""
    assert len(y1) == len(y2) and len(y1) > 0
    n = len(y1)
    a = b = c = d = 0  # TT, TF, FT, FF
    for x, y in zip(y1, y2):
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif not x and y:
            c += 1
        else:
            d += 1
    p_obs = (a + d) / n
    p_yes1 = (a + b) / n
    p_yes2 = (a + c) / n
    p_exp = p_yes1 * p_yes2 + (1 - p_yes1) * (1 - p_yes2)
    if abs(1 - p_exp) < 1e-12:
        return 1.0 if p_obs > 1 - 1e-12 else 0.0
    return (p_obs - p_exp) / (1 - p_exp)


def run_agreement(dataset, rows, seed=0):
    rng = random.Random(seed)
    rng.shuffle(rows)
    gpt_labels, local_labels, q_types = [], [], []
    for i, rec in enumerate(rows):
        qid = rec["question_id"]
        q = dataset.get(qid)
        if q is None:
            continue
        hypothesis = rec.get("hypothesis", "")
        question = q["question"]
        correct = q["answer"]
        qtype = q["question_type"]
        try:
            local_label = judge_answer_local(
                question, correct, hypothesis, qtype, qid
            )
        except Exception as e:
            print(f"  [warn] local judge failed on {qid}: {e}")
            continue
        gpt_labels.append(bool(rec["autoeval_label"]))
        local_labels.append(bool(local_label))
        q_types.append(qtype)
        if (i + 1) % 25 == 0:
            k = cohen_kappa(gpt_labels, local_labels)
            agree = sum(a == b for a, b in zip(gpt_labels, local_labels)) / len(gpt_labels)
            print(f"  [{i+1:4d}] kappa={k:.3f}  agreement={agree:.3f}")
    return gpt_labels, local_labels, q_types


def breakdown_by_task(gpt_labels, local_labels, q_types):
    by_task = defaultdict(lambda: {"gpt": [], "local": []})
    for g, l, t in zip(gpt_labels, local_labels, q_types):
        by_task[t]["gpt"].append(g)
        by_task[t]["local"].append(l)
    out = {}
    for t, lbl in by_task.items():
        n = len(lbl["gpt"])
        if n < 5:
            out[t] = {"n": n, "kappa": None, "agreement": None}
            continue
        k = cohen_kappa(lbl["gpt"], lbl["local"])
        agree = sum(a == b for a, b in zip(lbl["gpt"], lbl["local"])) / n
        out[t] = {"n": n, "kappa": round(k, 3), "agreement": round(agree, 3)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument(
        "--eval_files", nargs="+",
        default=[
            "results/benchmark/rag.eval.jsonl",
            "results/benchmark/full_history.eval.jsonl",
        ],
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Target sample size (combined across eval files).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/judge_agreement/agreement.json")
    args = parser.parse_args()

    dataset = load_dataset(args.data)

    all_rows = []
    for ef in args.eval_files:
        rows = load_eval_rows(ef)
        for r in rows:
            r["_source"] = os.path.basename(ef)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {ef}")

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    all_rows = all_rows[: args.n]
    print(f"Sampled {len(all_rows)} rows for judging.\n")

    gpt_labels, local_labels, q_types = run_agreement(dataset, all_rows, seed=args.seed)

    n = len(gpt_labels)
    kappa = cohen_kappa(gpt_labels, local_labels)
    agree = sum(a == b for a, b in zip(gpt_labels, local_labels)) / max(n, 1)
    by_task = breakdown_by_task(gpt_labels, local_labels, q_types)

    # Confusion: GPT-yes/Local-yes etc.
    tt = sum(g and l for g, l in zip(gpt_labels, local_labels))
    tf = sum(g and not l for g, l in zip(gpt_labels, local_labels))
    ft = sum(not g and l for g, l in zip(gpt_labels, local_labels))
    ff = sum(not g and not l for g, l in zip(gpt_labels, local_labels))

    summary = {
        "n": n,
        "kappa": round(kappa, 3),
        "agreement": round(agree, 3),
        "confusion": {"gpt_yes_local_yes": tt, "gpt_yes_local_no": tf,
                      "gpt_no_local_yes": ft, "gpt_no_local_no": ff},
        "per_task": by_task,
        "target": 0.75,
        "pass": kappa >= 0.75,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with io.open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"JUDGE-JUDGE AGREEMENT (n={n})")
    print("=" * 60)
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  Raw agreement: {agree:.3f}")
    print(f"  Confusion    : tt={tt} tf={tf} ft={ft} ff={ff}")
    print(f"  Target kappa : >= 0.75")
    print(f"  Status       : {'PASS' if kappa >= 0.75 else 'FAIL — retune prompts or upgrade local judge'}")
    print(f"\n  Per task type:")
    for t, v in by_task.items():
        print(f"    {t:30s} n={v['n']:3d}  kappa={v['kappa']}  agree={v['agreement']}")
    print(f"\n  Written to {args.output}")


if __name__ == "__main__":
    main()
