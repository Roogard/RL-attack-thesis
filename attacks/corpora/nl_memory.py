"""nl_memory — memory-shaped conversation corpus.

LoCoMo / PerLTQA / DuLeMon are not reliably open on HF (gated, restructured,
or unavailable). Per the plan, we substitute with what's open and
distribution-matched to "long-term memory chat":

- LongMemEval train-split rounds (already in the repo)
- Soda (allenai/soda) — synthetic, multi-turn, persona-grounded dialogue

Reuses `iter_longmemeval_train_rounds` so the LongMemEval extraction is
identical to what stage_b_retrieval.py does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.stage_b_common import (
    iter_longmemeval_train_rounds,
    write_corpus_jsonl,
)


def _iter_longmemeval(data_path: str, split_path: str) -> Iterator[tuple[str, str, dict]]:
    with open(split_path, encoding="utf-8") as f:
        splits = json.load(f)
    train_qids = splits["train"]
    yield from iter_longmemeval_train_rounds(data_path, train_qids)


def _iter_soda(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    ds = load_dataset("allenai/soda", split="train", streaming=True)
    n = 0
    for ex in ds:
        speakers = ex.get("speakers") or []
        dialogue = ex.get("dialogue") or []
        # Soda alternates between two speakers; we pair consecutive
        # (speaker_A, speaker_B) turns and treat A as "user", B as "assistant".
        for i in range(len(dialogue) - 1):
            u = dialogue[i]
            a = dialogue[i + 1]
            if not u or not a:
                continue
            yield u, a, {
                "source": "soda",
                "head": ex.get("head"),
                "narrative": ex.get("narrative", "")[:200],
                "speaker_user": speakers[i] if i < len(speakers) else None,
                "speaker_asst": speakers[i + 1] if i + 1 < len(speakers) else None,
            }
            n += 1
            if n >= max_rounds:
                return


def build(
    out_path: str,
    data_path: str = "LongMemEval/data/longmemeval_s_cleaned.json",
    split_path: str = "configs/splits/rag_attack.json",
    soda_max: int = 200_000,
    **_,
) -> dict:
    def stream():
        for r in _iter_longmemeval(data_path, split_path):
            yield r
        for r in _iter_soda(soda_max):
            yield r

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["sources"] = ["longmemeval_train", "soda"]
    summary["caps"] = {"soda": soda_max}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/nl_memory.jsonl")
    p.add_argument("--data", default="LongMemEval/data/longmemeval_s_cleaned.json")
    p.add_argument("--split", default="configs/splits/rag_attack.json")
    p.add_argument("--soda_max", type=int, default=200_000)
    args = p.parse_args()
    s = build(args.out, data_path=args.data, split_path=args.split, soda_max=args.soda_max)
    print(s)
