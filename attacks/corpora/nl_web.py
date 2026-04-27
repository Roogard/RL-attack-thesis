"""nl_web — open web-chat corpora.

WildChat-1M and LMSYS-Chat-1M are gated on HuggingFace. Per the plan
("if anything is gated, just get a different dataset"), we substitute:

- HuggingFaceH4/ultrachat_200k (open; ~200K multi-turn instruction chats)
- anon8231489123/ShareGPT_Vicuna_unfiltered (open mirror of ShareGPT)

Both yield (user, assistant) rounds via streaming. The point is broad
stylistic coverage of the embedding space — not faithfulness to any
particular chat distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.stage_b_common import write_corpus_jsonl


def _iter_ultrachat(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )
    n = 0
    for ex in ds:
        msgs = ex.get("messages") or []
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                yield msgs[i]["content"], msgs[i + 1]["content"], {
                    "source": "ultrachat_200k",
                    "prompt_id": ex.get("prompt_id"),
                }
                n += 1
                if n >= max_rounds:
                    return


def _iter_sharegpt(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    # Open mirror of ShareGPT in HF datasets (no gating).
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        streaming=True,
    )
    n = 0
    for ex in ds:
        convs = ex.get("conversations") or []
        for i in range(len(convs) - 1):
            a, b = convs[i], convs[i + 1]
            if a.get("from") in ("human", "user") and b.get("from") in ("gpt", "assistant"):
                yield a.get("value", ""), b.get("value", ""), {
                    "source": "sharegpt_vicuna_unfiltered",
                    "id": ex.get("id"),
                }
                n += 1
                if n >= max_rounds:
                    return


def build(
    out_path: str,
    ultrachat_max: int = 800_000,
    sharegpt_max: int = 700_000,
    **_,
) -> dict:
    def stream():
        for r in _iter_ultrachat(ultrachat_max):
            yield r
        for r in _iter_sharegpt(sharegpt_max):
            yield r

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["sources"] = ["ultrachat_200k", "sharegpt_vicuna_unfiltered"]
    summary["caps"] = {"ultrachat": ultrachat_max, "sharegpt": sharegpt_max}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/nl_web.jsonl")
    p.add_argument("--ultrachat_max", type=int, default=800_000)
    p.add_argument("--sharegpt_max", type=int, default=700_000)
    args = p.parse_args()
    s = build(args.out, ultrachat_max=args.ultrachat_max, sharegpt_max=args.sharegpt_max)
    print(s)
