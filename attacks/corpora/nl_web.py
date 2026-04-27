"""nl_web — open web-chat corpora.

WildChat-1M and LMSYS-Chat-1M are gated on HuggingFace. Per the plan
("if anything is gated, just get a different dataset"), we use:

- HuggingFaceH4/ultrachat_200k (open; ~200K multi-turn instruction chats)
- anon8231489123/ShareGPT_Vicuna_unfiltered (open mirror; non-standard
  HF layout — disabled by default because the auto-inferred Parquet
  loader can't parse it. Set --sharegpt_max > 0 to attempt anyway; if
  it fails the build still yields whatever the other sources produced.)

Each source is wrapped so one bad loader doesn't kill the whole corpus.
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


def _safe(name: str, gen):
    """Wrap a source iterator so its failure logs and yields nothing,
    instead of killing the whole corpus build.
    """
    try:
        for r in gen:
            yield r
    except Exception as e:
        print(f"[nl_web] WARNING: source {name!r} failed: {type(e).__name__}: {e}",
              flush=True)


def build(
    out_path: str,
    ultrachat_max: int = 800_000,
    sharegpt_max: int = 0,
    **_,
) -> dict:
    sources_used: list[str] = []

    def stream():
        if ultrachat_max > 0:
            sources_used.append("ultrachat_200k")
            for r in _safe("ultrachat_200k", _iter_ultrachat(ultrachat_max)):
                yield r
        if sharegpt_max > 0:
            sources_used.append("sharegpt_vicuna_unfiltered")
            for r in _safe("sharegpt_vicuna_unfiltered", _iter_sharegpt(sharegpt_max)):
                yield r

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["sources"] = sources_used
    summary["caps"] = {"ultrachat": ultrachat_max, "sharegpt": sharegpt_max}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/nl_web.jsonl")
    p.add_argument("--ultrachat_max", type=int, default=800_000)
    p.add_argument("--sharegpt_max", type=int, default=0,
                   help="ShareGPT mirror has non-standard HF layout; off by default")
    args = p.parse_args()
    s = build(args.out, ultrachat_max=args.ultrachat_max, sharegpt_max=args.sharegpt_max)
    print(s)
