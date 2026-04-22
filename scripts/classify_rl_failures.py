"""Stage 1 of the rl_memory failure investigation.

Classify every prediction in results/benchmark/rl_memory.eval.jsonl into one of:
  - correct
  - empty                  (hypothesis is empty string)
  - claims_no_info         (says "I don't have info / not in conversation history")
  - claims_mix_up          (says "mix-up / misunderstanding / question doesn't match")
  - confident_wrong_answer (asserts a specific fact that the judge marked wrong)

Cross-tabs the buckets against question_type, abstention flag, and whether
rag/full_history got the same question right (so we can isolate questions
where the answer was retrievable in principle but rl_memory missed it).

Writes results/benchmark/rl_memory.failure_modes.json + console summary.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict


RESULTS_DIR = "results/benchmark"
ORACLE_PATH = "LongMemEval/data/longmemeval_oracle.json"

NO_INFO_PATTERNS = [
    r"don'?t have (any )?(information|access|details|data|record)",
    r"no (information|mention|record|data) (about|of|on)",
    r"(isn'?t|wasn'?t|haven'?t|doesn'?t|didn'?t) (mention|specif|provide|include|share|tell)",
    r"not (mention|specif|provide|include|share)",
    r"no (specific )?(record|details?) (of|about|on)",
    r"there'?s? no (information|mention|record)",
    r"there is no (information|mention|record)",
    r"the conversation history (does not|doesn'?t) (mention|contain|include)",
]
MIX_UP_PATTERNS = [
    r"mix-?up",
    r"misunderstanding",
    r"there might be a mix",
    r"question doesn'?t match",
    r"doesn'?t match the conversation",
    r"unrelated to (the|your) (topic|question|conversation)",
    r"shift in (the )?topic",
    r"there (was|might be) a (shift|misunderstanding|mix)",
]


def classify(hyp: str, autoeval_label: bool) -> str:
    if autoeval_label:
        return "correct"
    if not hyp.strip():
        return "empty"
    h = hyp.lower()
    if any(re.search(p, h) for p in MIX_UP_PATTERNS):
        return "claims_mix_up"
    if any(re.search(p, h) for p in NO_INFO_PATTERNS):
        return "claims_no_info"
    return "confident_wrong_answer"


def load_jsonl(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[obj["question_id"]] = obj
    return out


def main() -> None:
    rl = load_jsonl(os.path.join(RESULTS_DIR, "rl_memory.eval.jsonl"))
    rag = load_jsonl(os.path.join(RESULTS_DIR, "rag.eval.jsonl"))
    fh = load_jsonl(os.path.join(RESULTS_DIR, "full_history.eval.jsonl"))

    with open(ORACLE_PATH, encoding="utf-8") as f:
        oracle = {q["question_id"]: q for q in json.load(f)}

    # Classify each rl_memory prediction
    bucket_of: dict[str, str] = {}
    for qid, entry in rl.items():
        bucket_of[qid] = classify(entry["hypothesis"], entry.get("autoeval_label", False))

    overall_buckets = Counter(bucket_of.values())

    # Bucket × question_type
    by_qtype: dict[str, Counter] = defaultdict(Counter)
    for qid, b in bucket_of.items():
        qt = oracle.get(qid, {}).get("question_type", "unknown")
        by_qtype[qt][b] += 1

    # Abstention vs not
    abs_counts = Counter()
    nonabs_counts = Counter()
    for qid, b in bucket_of.items():
        (abs_counts if "_abs" in qid else nonabs_counts)[b] += 1

    # Subset: rag-right & rl-wrong (where the answer was retrievable in principle)
    rag_right_rl_wrong = [
        qid for qid in rl
        if rag.get(qid, {}).get("autoeval_label") and not rl[qid].get("autoeval_label")
    ]
    rrrw_buckets = Counter(bucket_of[q] for q in rag_right_rl_wrong)

    # Subset: fh-right & rl-wrong
    fh_right_rl_wrong = [
        qid for qid in rl
        if fh.get(qid, {}).get("autoeval_label") and not rl[qid].get("autoeval_label")
    ]
    fhrrw_buckets = Counter(bucket_of[q] for q in fh_right_rl_wrong)

    # Triple intersection: questions BOTH rag and full_history got right but rl missed.
    # These are the highest-signal cases for stage 2 — we know two independent
    # pipelines found the answer, so retrievability isn't the issue.
    both_right_rl_wrong = [
        qid for qid in rag_right_rl_wrong
        if fh.get(qid, {}).get("autoeval_label")
    ]

    # Pick stage-2 candidate qids: 5 questions across diverse question types
    # where both rag and fh got it right but rl didn't.
    candidates_by_type: dict[str, list[str]] = defaultdict(list)
    for qid in both_right_rl_wrong:
        qt = oracle.get(qid, {}).get("question_type", "unknown")
        candidates_by_type[qt].append(qid)

    stage2_picks = []
    for qt in sorted(candidates_by_type):
        if candidates_by_type[qt]:
            stage2_picks.append({
                "question_id": candidates_by_type[qt][0],
                "question_type": qt,
                "rl_memory_hypothesis": rl[candidates_by_type[qt][0]]["hypothesis"][:200],
                "rag_hypothesis": rag[candidates_by_type[qt][0]]["hypothesis"][:200],
                "ground_truth": str(oracle[candidates_by_type[qt][0]]["answer"])[:200],
                "n_haystack_sessions": len(oracle[candidates_by_type[qt][0]].get("haystack_sessions", [])),
                "n_answer_sessions": len(oracle[candidates_by_type[qt][0]].get("answer_session_ids", [])),
            })

    summary = {
        "overall_buckets": dict(overall_buckets),
        "by_question_type": {qt: dict(c) for qt, c in by_qtype.items()},
        "abstention_buckets": dict(abs_counts),
        "non_abstention_buckets": dict(nonabs_counts),
        "rag_right_rl_wrong": {
            "count": len(rag_right_rl_wrong),
            "buckets": dict(rrrw_buckets),
        },
        "full_history_right_rl_wrong": {
            "count": len(fh_right_rl_wrong),
            "buckets": dict(fhrrw_buckets),
        },
        "both_baselines_right_rl_wrong": {
            "count": len(both_right_rl_wrong),
            "by_question_type": {
                qt: len(qids) for qt, qids in candidates_by_type.items()
            },
        },
        "stage2_candidates": stage2_picks,
    }

    out_path = os.path.join(RESULTS_DIR, "rl_memory.failure_modes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print(f"Wrote {out_path}\n")
    print("OVERALL BUCKETS (n=500)")
    for b, c in sorted(overall_buckets.items(), key=lambda x: -x[1]):
        print(f"  {b:<24} {c:>4}  ({100*c/500:5.1f}%)")
    print()
    print("BY QUESTION TYPE")
    for qt in sorted(by_qtype):
        total = sum(by_qtype[qt].values())
        print(f"  {qt}  (n={total})")
        for b, c in sorted(by_qtype[qt].items(), key=lambda x: -x[1]):
            print(f"    {b:<24} {c:>4}  ({100*c/total:5.1f}%)")
    print()
    print(f"ABSTENTION (n={sum(abs_counts.values())})")
    for b, c in sorted(abs_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:<24} {c:>4}")
    print(f"\nNON-ABSTENTION (n={sum(nonabs_counts.values())})")
    for b, c in sorted(nonabs_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:<24} {c:>4}")
    print()
    print(f"RAG-RIGHT & RL-WRONG: {len(rag_right_rl_wrong)} cases — what bucket is rl in?")
    for b, c in sorted(rrrw_buckets.items(), key=lambda x: -x[1]):
        print(f"  {b:<24} {c:>4}")
    print()
    print(f"BOTH-BASELINES-RIGHT & RL-WRONG: {len(both_right_rl_wrong)} cases (highest-signal stage-2 targets)")
    for qt, qids in sorted(candidates_by_type.items()):
        print(f"  {qt:<28} {len(qids)} candidates")
    print()
    print(f"Stage-2 candidate qids (one per question type): {[p['question_id'] for p in stage2_picks]}")


if __name__ == "__main__":
    main()
