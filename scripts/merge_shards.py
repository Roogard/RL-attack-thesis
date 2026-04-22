"""Merge sharded prediction files into one JSONL per memory type.

For each <memory>.shard{i}of{n}.jsonl in the given results dir, concatenate
all shards into <memory>.jsonl. Existing unsharded files are left alone.

Usage:  python scripts/merge_shards.py results/<timestamp>
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from glob import glob


SHARD_RE = re.compile(r"^(.*)\.shard(\d+)of(\d+)\.jsonl$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"Not a directory: {args.results_dir}")

    # Group shard files by memory name
    shards: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for path in sorted(glob(os.path.join(args.results_dir, "*.shard*of*.jsonl"))):
        name = os.path.basename(path)
        m = SHARD_RE.match(name)
        if not m:
            continue
        memory_name, i, n = m.group(1), int(m.group(2)), int(m.group(3))
        shards[memory_name].append((i, n, path))

    if not shards:
        print(f"No shard files found in {args.results_dir}")
        return

    for memory_name, parts in shards.items():
        ns = {n for _, n, _ in parts}
        if len(ns) != 1:
            raise SystemExit(f"{memory_name}: inconsistent shard totals {ns}")
        n = ns.pop()
        present = {i for i, _, _ in parts}
        missing = set(range(n)) - present
        if missing:
            print(f"WARNING: {memory_name} is missing shards {sorted(missing)} (continuing)")

        parts.sort(key=lambda x: x[0])
        out_path = os.path.join(args.results_dir, f"{memory_name}.jsonl")
        total = 0
        with open(out_path, "w", encoding="utf-8") as out_f:
            for _, _, src in parts:
                with open(src, encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line if line.endswith("\n") else line + "\n")
                            total += 1
        print(f"  {memory_name}: {len(parts)} shards -> {out_path} ({total} rows)")


if __name__ == "__main__":
    main()
