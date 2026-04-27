"""synth_topic — hub-conditioned LLM generation.

For each of the K hubs, identify the 5 nearest training-question texts
(already in hubs.pkl) and use them as topic anchors. Then ask the
attacker LLM to write candidate (user, assistant) pairs *on those topics*.
This is BoN done correctly: separate offline mass generation from
per-hub selection, so the LLM's compute amortizes across all hubs.

Default budget per hub: 2000 candidates → 30 hubs × 2000 = 60_000.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.corpora._synth_common import yield_pairs_from_themes
from attacks.hubness.stage_b_common import load_hubs, write_corpus_jsonl


def _theme_for_hub(train_texts: list[str], top_idx: np.ndarray, k_anchors: int) -> str:
    """Compose a single theme string out of the k_anchors nearest train queries."""
    anchors = [train_texts[i] for i in top_idx[:k_anchors]]
    anchors_block = "; ".join(a.strip().rstrip("?.").strip() for a in anchors)
    return (
        f"a personal-life topic similar to these user questions: {anchors_block}"
    )


def build(
    out_path: str,
    hubs_path: str = "results/stage_a/hubs_K30.pkl",
    samples_per_hub: int = 2000,
    k_anchors: int = 5,
    batch_size: int = 16,
    max_tokens: int = 160,
    seed: int = 0,
    **_,
) -> dict:
    hubs_blob = load_hubs(hubs_path)
    H = hubs_blob["hubs"]
    train_texts = hubs_blob["train_texts"]
    Q_train = hubs_blob["train_vecs"]
    K = len(H)

    sims_hq = Q_train @ H.T  # (n_train, K)
    themes: list[str] = []
    personas: list[str | None] = []
    for hub_idx in range(K):
        top_idx = np.argsort(-sims_hq[:, hub_idx])[:k_anchors]
        themes.append(_theme_for_hub(train_texts, top_idx, k_anchors))
        personas.append(None)

    def stream():
        for u, a, meta in yield_pairs_from_themes(
            themes, personas,
            n_samples_per=samples_per_hub,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
            source_label="synth_topic",
        ):
            # Tag each generation with its hub so analysis can recover
            # which hub the LLM was conditioned on (downstream code never
            # *requires* this — the corpus is searched by cosine — but it
            # helps post-hoc analysis).
            theme = meta["theme"]
            hub_idx = themes.index(theme) if theme in themes else -1
            meta["target_hub_idx"] = hub_idx
            yield u, a, meta

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["K"] = K
    summary["samples_per_hub"] = samples_per_hub
    summary["k_anchors"] = k_anchors
    summary["hubs_source"] = hubs_path
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/synth_topic.jsonl")
    p.add_argument("--hubs", default="results/stage_a/hubs_K30.pkl")
    p.add_argument("--samples_per_hub", type=int, default=2000)
    p.add_argument("--k_anchors", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    s = build(
        args.out,
        hubs_path=args.hubs,
        samples_per_hub=args.samples_per_hub,
        k_anchors=args.k_anchors,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(s)
