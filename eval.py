import json
import os

import numpy as np

import harness

ORACLE_PATH = "LongMemEval/data/longmemeval_oracle.json"


def run_eval(results_dir):
    """Evaluate all prediction files in results_dir and produce a comparison.

    For each {name}.jsonl, judges answers via GPT-4o, writes {name}.eval.jsonl,
    and builds a comparison.json with per-task and overall accuracy.
    """
    # Load oracle for ground truth
    with open(ORACLE_PATH, encoding="utf-8") as f:
        oracle = json.load(f)
    qid_to_ref = {entry["question_id"]: entry for entry in oracle}

    # Find all prediction files
    pred_files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith(".jsonl") and not f.endswith(".eval.jsonl")
    )

    if not pred_files:
        print("No prediction files found in", results_dir)
        return

    comparison = {}

    for pred_file in pred_files:
        memory_name = pred_file.replace(".jsonl", "")
        pred_path = os.path.join(results_dir, pred_file)
        eval_path = os.path.join(results_dir, f"{memory_name}.eval.jsonl")

        print(f"\nEvaluating: {memory_name}")
        predictions = [json.loads(line) for line in open(pred_path, encoding="utf-8")]

        # Judge each answer
        qtype_correct = {}
        eval_results = []

        for entry in predictions:
            qid = entry["question_id"]
            if qid not in qid_to_ref:
                print(f"  Warning: {qid} not in oracle, skipping")
                continue

            ref = qid_to_ref[qid]
            label = harness.judge_answer(
                question=ref["question"],
                answer=ref["answer"],
                hypothesis=entry["hypothesis"],
                question_type=ref["question_type"],
                question_id=qid,
            )

            entry["autoeval_label"] = label
            eval_results.append(entry)

            qtype = ref["question_type"]
            if qtype not in qtype_correct:
                qtype_correct[qtype] = []
            qtype_correct[qtype].append(1 if label else 0)

        # Write eval results
        with open(eval_path, "w", encoding="utf-8") as f:
            for entry in eval_results:
                f.write(json.dumps(entry) + "\n")

        # Compute metrics
        all_scores = [e["autoeval_label"] for e in eval_results]
        overall_acc = np.mean(all_scores).item() if all_scores else 0.0
        task_accs = {
            task: round(np.mean(scores).item(), 4)
            for task, scores in sorted(qtype_correct.items())
        }
        task_avg = round(np.mean(list(task_accs.values())).item(), 4) if task_accs else 0.0

        comparison[memory_name] = {
            "overall": round(overall_acc, 4),
            "task_average": task_avg,
            "per_task": task_accs,
            "num_questions": len(eval_results),
        }

        # Print results
        print(f"  Overall: {overall_acc:.4f} ({len(eval_results)} questions)")
        print(f"  Task average: {task_avg:.4f}")
        for task, acc in task_accs.items():
            count = len(qtype_correct[task])
            print(f"    {task}: {acc} ({count})")

    # Write comparison
    comparison_path = os.path.join(results_dir, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")

    # Print comparison table
    if len(comparison) > 1:
        print(f"\n{'='*60}")
        print("Comparison Table")
        print(f"{'='*60}")
        all_tasks = sorted(set(
            task for metrics in comparison.values() for task in metrics["per_task"]
        ))
        header = f"{'Memory':<20} {'Overall':<10} " + " ".join(f"{t:<25}" for t in all_tasks)
        print(header)
        print("-" * len(header))
        for name, metrics in comparison.items():
            row = f"{name:<20} {metrics['overall']:<10.4f} "
            row += " ".join(f"{metrics['per_task'].get(t, 'N/A'):<25}" for t in all_tasks)
            print(row)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Judge prediction JSONLs in a results dir with GPT-4o.")
    p.add_argument("--results-dir", required=True, help="Directory containing <memory>.jsonl prediction files")
    args = p.parse_args()
    run_eval(args.results_dir)
