import argparse
import json
import os
from datetime import datetime

import harness
from memory import MEMORY_REGISTRY
from eval import run_eval

DATASET_PATH = "LongMemEval/data/longmemeval_s_cleaned.json"

MEMORY_TYPES = ["full_history", "rag"]


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval across memory architectures")
    parser.add_argument("--memory", type=str, default=None, help="Run only this memory type")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of questions (for testing)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip GPT-4o evaluation step")
    args = parser.parse_args()

    # Load dataset once
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {DATASET_PATH}")

    # Create timestamped results directory
    run_dir = os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
        output_path = os.path.join(run_dir, f"{name}.jsonl")
        harness.run_questions(data, store, output_path, limit=args.limit)

    # Evaluate
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print("Evaluation")
        print(f"{'='*60}")
        run_eval(run_dir)


if __name__ == "__main__":
    main()
