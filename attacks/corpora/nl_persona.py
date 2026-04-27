"""nl_persona — persona / chitchat / assistant-style human chat.

All open on HF (no gating):

- bavard/personachat_truecased
- daily_dialog
- OpenAssistant/oasst1
- Anthropic/hh-rlhf
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.stage_b_common import write_corpus_jsonl


def _iter_personachat(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    ds = load_dataset(
        "bavard/personachat_truecased",
        split="train",
        streaming=True,
    )
    n = 0
    for ex in ds:
        history = ex.get("history") or []
        candidates = ex.get("candidates") or []
        if not history or not candidates:
            continue
        # The user-side history's last item is the latest user utterance;
        # the gold candidate is the assistant's reply.
        u = history[-1]
        a = candidates[-1] if candidates else ""
        if not u or not a:
            continue
        yield u, a, {"source": "personachat_truecased", "conv_id": ex.get("conv_id")}
        n += 1
        if n >= max_rounds:
            return


def _iter_dailydialog(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    # The dataset script is unmaintained on some HF mirrors; fall back to
    # the community-mirrored "li2017dailydialog/daily_dialog" if the
    # canonical name fails.
    try:
        ds = load_dataset("daily_dialog", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        ds = load_dataset(
            "li2017dailydialog/daily_dialog",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    n = 0
    for ex in ds:
        dialog = ex.get("dialog") or []
        for i in range(len(dialog) - 1):
            u, a = dialog[i].strip(), dialog[i + 1].strip()
            if not u or not a:
                continue
            yield u, a, {"source": "daily_dialog"}
            n += 1
            if n >= max_rounds:
                return


def _iter_oasst1(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    # Build a parent->children map across the streamed messages, then yield
    # (parent prompter -> child assistant) pairs.
    children: dict[str | None, list[dict]] = defaultdict(list)
    by_id: dict[str, dict] = {}
    n = 0
    for ex in ds:
        by_id[ex["message_id"]] = ex
        children[ex.get("parent_id")].append(ex)
        if ex.get("role") == "assistant":
            parent = by_id.get(ex.get("parent_id"))
            if parent and parent.get("role") == "prompter":
                u = parent.get("text", "").strip()
                a = ex.get("text", "").strip()
                if u and a:
                    yield u, a, {
                        "source": "oasst1",
                        "tree_id": ex.get("message_tree_id"),
                        "lang": ex.get("lang"),
                    }
                    n += 1
                    if n >= max_rounds:
                        return


def _iter_hh_rlhf(max_rounds: int) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    n = 0
    for ex in ds:
        # HH-RLHF rows have "chosen" and "rejected" multi-turn strings that
        # interleave "Human: ... Assistant: ..." text. Parse only the chosen
        # branch's first H/A pair.
        text = ex.get("chosen", "") or ""
        if "Human:" not in text or "Assistant:" not in text:
            continue
        try:
            after_h = text.split("Human:", 1)[1]
            human, after_a = after_h.split("Assistant:", 1)
            asst = after_a.split("Human:", 1)[0]
        except ValueError:
            continue
        u, a = human.strip(), asst.strip()
        if not u or not a:
            continue
        yield u, a, {"source": "hh_rlhf"}
        n += 1
        if n >= max_rounds:
            return


def build(
    out_path: str,
    personachat_max: int = 60_000,
    dailydialog_max: int = 80_000,
    oasst_max: int = 80_000,
    hhrlhf_max: int = 80_000,
    **_,
) -> dict:
    def stream():
        for r in _iter_personachat(personachat_max):
            yield r
        for r in _iter_dailydialog(dailydialog_max):
            yield r
        for r in _iter_oasst1(oasst_max):
            yield r
        for r in _iter_hh_rlhf(hhrlhf_max):
            yield r

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["sources"] = ["personachat_truecased", "daily_dialog", "oasst1", "hh_rlhf"]
    summary["caps"] = {
        "personachat": personachat_max,
        "dailydialog": dailydialog_max,
        "oasst": oasst_max,
        "hhrlhf": hhrlhf_max,
    }
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/nl_persona.jsonl")
    p.add_argument("--personachat_max", type=int, default=60_000)
    p.add_argument("--dailydialog_max", type=int, default=80_000)
    p.add_argument("--oasst_max", type=int, default=80_000)
    p.add_argument("--hhrlhf_max", type=int, default=80_000)
    args = p.parse_args()
    s = build(
        args.out,
        personachat_max=args.personachat_max,
        dailydialog_max=args.dailydialog_max,
        oasst_max=args.oasst_max,
        hhrlhf_max=args.hhrlhf_max,
    )
    print(s)
