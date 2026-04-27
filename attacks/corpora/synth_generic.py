"""synth_generic — LLM-generated chat across a wide topic spread.

Topics are taken from a static seed list spanning the kinds of subjects a
personal-assistant memory typically holds (work, health, relationships,
preferences, hobbies, travel, food, learning, finance, daily routines,
etc.). For each topic, sample N candidate (user, assistant) pairs from
the attacker LLM with high-temperature decoding for variety.

Default budget:
    300 topics × 500 samples / topic = 150_000 candidates pre-dedup.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.corpora._synth_common import yield_pairs_from_themes
from attacks.hubness.stage_b_common import write_corpus_jsonl


# Wide topic seeds — fits an embedding-space coverage strategy. These are
# *categories*; the attacker LLM elaborates concrete details under each.
_TOPIC_SEEDS: tuple[str, ...] = (
    # work / career
    "a recent meeting at work", "a work deadline that slipped", "a job I'm considering",
    "my morning standup", "a frustrating coworker", "a project I'm proud of",
    "a promotion I'm chasing", "remote work setup", "a contract I just signed",
    "feedback from my manager", "a client that's hard to please",
    "switching teams", "a late-night work session", "a side project",
    # health
    "a doctor's appointment", "a sleep schedule that broke", "a workout plan",
    "a recurring headache", "trying to cut sugar", "physiotherapy progress",
    "a dentist visit", "running my first 5K", "my therapist", "blood test results",
    # relationships
    "a fight with my partner", "planning a wedding", "introducing my parents",
    "a long-distance friendship", "a date that went well", "an awkward hangout",
    "moving in together", "an old friend reconnecting",
    # finance
    "saving for a house", "a credit card bill", "rent going up",
    "a tax refund", "trying to budget groceries", "investing in index funds",
    "selling my car", "a money argument with my sibling",
    # travel
    "a trip to Tokyo", "a delayed flight", "a road trip across Utah",
    "lost luggage", "a hostel that surprised me", "trying local food abroad",
    "a hike I survived", "planning a honeymoon", "a visa application",
    # food + cooking
    "trying to bake bread", "a restaurant I won't return to",
    "my new espresso machine", "a recipe my grandmother left",
    "going vegetarian for a month", "a wine tasting",
    # hobbies + leisure
    "learning piano", "a video game I can't put down", "running a D&D campaign",
    "fixing up my bike", "starting a garden", "knitting a sweater",
    "a podcast I started", "watching a slow movie",
    # learning + skill
    "an online course I finished", "studying for the GRE",
    "learning Spanish", "a coding bug that took two days", "reading a tough book",
    # tech + tools
    "switching to Linux", "a new productivity app", "configuring my keyboard",
    "fighting with my printer", "buying a new laptop",
    # daily life
    "morning coffee ritual", "a packed weekend", "a quiet Sunday",
    "decluttering my apartment", "a bad night's sleep", "errands I keep putting off",
    # preferences + opinions
    "my favorite kind of music", "the best pizza topping debate",
    "why I like cold weather", "a movie everyone hated but I loved",
    "an unpopular opinion about coffee", "what kind of books I avoid",
    # past events
    "what happened in third grade", "an embarrassing high school moment",
    "the year I lived abroad", "the first time I lived alone",
    "a job I quit on day three", "a concert in 2018",
    # family
    "my mom's recipe for soup", "my dad's woodworking",
    "a sister who never calls back", "my niece's first words",
    "a family reunion I dreaded", "a cousin getting married",
    # plans + intentions
    "what I want to do this summer", "where I want to live in five years",
    "a business idea I keep coming back to", "starting a YouTube channel",
    "applying to grad school",
    # memory-typical "did we talk about" hooks
    "what I told you last week about my back",
    "the name of that restaurant I mentioned",
    "the password manager I was going to switch to",
    "the gift I planned for my mom's birthday",
    "the apartment lease question",
    "the medication my doctor mentioned",
    "the coworker I was annoyed with",
    "the pet sitter I was looking up",
)


def _expand_topics(seeds: tuple[str, ...], n_target: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    base = list(seeds)
    out: list[str] = []
    while len(out) < n_target:
        rng.shuffle(base)
        out.extend(base)
    return out[:n_target]


def build(
    out_path: str,
    n_topics: int = 300,
    samples_per_topic: int = 500,
    batch_size: int = 32,
    max_tokens: int = 160,
    seed: int = 0,
    **_,
) -> dict:
    themes = _expand_topics(_TOPIC_SEEDS, n_topics, seed=seed)
    personas: list[str | None] = [None] * len(themes)

    def stream():
        yield from yield_pairs_from_themes(
            themes, personas,
            n_samples_per=samples_per_topic,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
            source_label="synth_generic",
        )

    summary = write_corpus_jsonl(out_path, stream(), dedup=True)
    summary["n_topics"] = n_topics
    summary["samples_per_topic"] = samples_per_topic
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/corpora/synth_generic.jsonl")
    p.add_argument("--n_topics", type=int, default=300)
    p.add_argument("--samples_per_topic", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    s = build(
        args.out,
        n_topics=args.n_topics,
        samples_per_topic=args.samples_per_topic,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(s)
