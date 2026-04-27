"""Candidate-text corpora for Stage B retrieval-style attacks.

Each corpus is one of three categories:
- natural-language (real human chat from public datasets)
- synthetic (LLM-generated chat conditioned on topics or personas)
- adversarial (templated attack content phrased naturalistically)

Every corpus exposes the same on-disk format (JSONL of
{"user", "assistant", "meta", "hash"}) produced by
`attacks.hubness.stage_b_common.write_corpus_jsonl`. This lets
`attacks.hubness.stage_b_corpus` ingest any corpus the same way: encode →
per-hub top-N by cos → RPR-rerank → emit a poison_file.json that
`attacks.eval_hubs --poison_file` can run.

Each loader module exposes one function `build(out_path: str, **kwargs) -> dict`
that streams its rounds to disk and returns a summary dict (counts, source
metadata). Loaders that need network access fail loud if the underlying HF
dataset isn't reachable; they don't silently produce an empty corpus.

Default registry — IDs match plan and CLI flags:

    nl_web         attacks.corpora.nl_web
    nl_memory      attacks.corpora.nl_memory
    nl_persona     attacks.corpora.nl_persona
    synth_generic  attacks.corpora.synth_generic
    synth_topic    attacks.corpora.synth_topic
    synth_persona  attacks.corpora.synth_persona
    adv_inject     attacks.corpora.adv_inject
    adv_confuse    attacks.corpora.adv_confuse
    adv_optimized  attacks.corpora.adv_optimized
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

CORPUS_IDS: tuple[str, ...] = (
    "nl_web",
    "nl_memory",
    "nl_persona",
    "synth_generic",
    "synth_topic",
    "synth_persona",
    "adv_inject",
    "adv_confuse",
    "adv_optimized",
)


def get_builder(corpus_id: str) -> Callable[..., dict]:
    """Return the `build(out_path, **kwargs) -> dict` function for a corpus."""
    if corpus_id not in CORPUS_IDS:
        raise KeyError(f"unknown corpus id: {corpus_id!r}; known: {CORPUS_IDS}")
    mod = importlib.import_module(f"attacks.corpora.{corpus_id}")
    if not hasattr(mod, "build"):
        raise AttributeError(f"attacks.corpora.{corpus_id} has no build() function")
    return mod.build


def default_out_path(corpus_id: str, root: str = "data/corpora") -> str:
    return str(Path(root) / f"{corpus_id}.jsonl")
