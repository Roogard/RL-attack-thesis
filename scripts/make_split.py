"""Create the train/val/test split file used by attack/train.py.

Default: 400 train / 50 val / 50 test, stratified by question_type, seeded.
Produces configs/splits/rag_attack.json with keys {train, val, test} ->
list of question_ids.
"""

import argparse
import io
import json
import os
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--output", default="configs/splits/rag_attack.json")
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with io.open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    buckets = defaultdict(list)
    for q in data:
        buckets[q["question_type"]].append(q["question_id"])

    rng = random.Random(args.seed)
    train, val, test = [], [], []
    for qtype, qids in buckets.items():
        rng.shuffle(qids)
        n = len(qids)
        n_train = int(args.train_frac * n)
        n_val = int(args.val_frac * n)
        train.extend(qids[:n_train])
        val.extend(qids[n_train:n_train + n_val])
        test.extend(qids[n_train + n_val:])

    all_qids = train + val + test

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with io.open(args.output, "w", encoding="utf-8") as f:
        json.dump({"train": train, "val": val, "test": test, "all": all_qids}, f, indent=2)

    print(f"train={len(train)} val={len(val)} test={len(test)} all={len(all_qids)}  -> {args.output}")


if __name__ == "__main__":
    main()
