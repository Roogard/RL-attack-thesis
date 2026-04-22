import argparse
import json
import os
from datetime import datetime

import harness
from memory import MEMORY_REGISTRY
from eval import run_eval

DATASET_PATH = "LongMemEval/data/longmemeval_s_cleaned.json"

MEMORY_TYPES = ["full_history", "rag", "rl_memory"]


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval across memory architectures")
    parser.add_argument("--memory", type=str, default=None, help="Run only this memory type")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of questions (for testing)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip GPT-4o evaluation step")
    parser.add_argument("--shard-i", type=int, default=0, help="0-indexed shard of the dataset")
    parser.add_argument("--shard-n", type=int, default=1, help="Total number of shards")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Reuse an existing results dir (lets sibling shards write into the same dir)")
    args = parser.parse_args()

    if not (0 <= args.shard_i < args.shard_n):
        parser.error(f"--shard-i must satisfy 0 <= i < n, got i={args.shard_i} n={args.shard_n}")

    # Load dataset once
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {DATASET_PATH}")

    # Stride-shard the dataset (distributes question types evenly across shards)
    if args.shard_n > 1:
        data = data[args.shard_i :: args.shard_n]
        print(f"Shard {args.shard_i}/{args.shard_n}: {len(data)} questions")

    # Create (or reuse) results directory
    run_dir = args.run_dir or os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results directory: {run_dir}")

    # Determine which memory types to run
    types_to_run = [args.memory] if args.memory else MEMORY_TYPES
    for name in types_to_run:
        if name not in MEMORY_REGISTRY:
            print(f"Unknown memory type: {name}")
            print(f"Available: {list(MEMORY_REGISTRY.keys())}")
            return

    # Run each memory type
    for name in types_to_run:
        print(f"\n{'='*60}")
        print(f"Memory type: {name}")
        print(f"{'='*60}")
        store = MEMORY_REGISTRY[name]()
        if args.shard_n > 1:
            output_name = f"{name}.shard{args.shard_i}of{args.shard_n}.jsonl"
        else:
            output_name = f"{name}.jsonl"
        output_path = os.path.join(run_dir, output_name)
        harness.run_questions(data, store, output_path, limit=args.limit)

    # Evaluate (skip when running a single shard — merge first, then judge)
    if args.shard_n > 1 and not args.skip_eval:
        print("\n[main] Sharded run; skipping eval. Merge shards then run eval separately.")
        return

    if not args.skip_eval:
        print(f"\n{'='*60}")
        print("Evaluation")
        print(f"{'='*60}")
        run_eval(run_dir)


if __name__ == "__main__":
    main()
