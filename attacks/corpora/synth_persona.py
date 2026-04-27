"""synth_persona — persona-conditioned LLM generation.

Pulls personas from `proj-persona/PersonaHub` (open) and crosses each with
a small topic seed list. Each (persona, topic) pair contributes one
generated (user, assistant) pair, biasing output to a wider stylistic
distribution than synth_generic alone.

Default budget: 5000 personas × 30 topics = 150_000 candidates pre-dedup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.corpora._synth_common import yield_pairs_from_themes
from attacks.hubness.stage_b_common import write_corpus_jsonl


_TOPIC_SEEDS: tuple[str, ...] = (
    "what I did last weekend", "a recent purchase",
    "a project I'm working on", "an injury or health update",
    "a friend I haven't seen in a while", "what I'm cooking tonight",
    "a movie or show I just watched", "a goal for next month",
    "an annoying thing about my commute", "a memory from childhood",
    "a song stuck in my head", "a recent argument",
    "where I want to live next", "a new habit I'm trying to form",
    "an upcoming trip", "a coworker who irritates me",
    "a hobby I picked up recently", "a book I started",
    "a pet I'm thinking of getting", "a piece of advice my mom gave me",
    "a problem with my apartment", "a class I'm taking",
    "a bill I forgot to pay", "a recipe that flopped",
    "what I'd order at my favorite restaurant",
    "an old friend reaching out", "an idea for a side business",
    "a gift I'm planning", "a relative's wedding",
    "what I'm wearing tomorrow",
)


def _load_personas(max_personas: int) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("proj-persona/PersonaHub", "persona", split="train", streaming=True)
    n = 0
    for ex in ds:
        p = (ex.get("persona") or "").strip()
        if not p:
            continue
        yield p
        n += 1
        if n >= max_personas:
            return


def build(
    out_path: str,
    n_personas: int = 5000,
    topics_per_persona: int | None = None,
    batch_size: int = 32,
    max_tokens: int = 160,
    seed: int = 0,
    **_,
) -> dict:
    if topics_per_persona is None:
        topics_per_persona = len(_TOPIC_SEEDS)
    topic_pool = list(_TOPIC_SEEDS)[:topics_per_persona]

    personas: list[str | None] = []
    themes: list[str] = []
    for p in _load_personas(n_personas):
        for t in topic_pool:
            personas.append(p)
            themes.append(t)

    def stream():
        yield from yield_pairs_from_themes(
            themes, personas,
            n_samples_per=1,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
            source_label="synth_persona",
        )

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["n_personas"] = n_personas
    summary["topics_per_persona"] = topics_per_persona
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/synth_persona.jsonl")
    p.add_argument("--n_personas", type=int, default=5000)
    p.add_argument("--topics_per_persona", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    s = build(
        args.out,
        n_personas=args.n_personas,
        topics_per_persona=args.topics_per_persona,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(s)
