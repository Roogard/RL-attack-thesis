"""Shared helpers for synthetic-corpus generators (synth_*.py).

Wraps `memory._vllm_engines.get_attacker_engine()` and the standard
"User: ... / Assistant: ..." parsing that stage_b_bon.py uses.
"""
from __future__ import annotations

from typing import Iterator

ATTACKER_SYSTEM = (
    "You generate short realistic user/assistant conversation turns for a "
    "personal-assistant memory dataset. Output exactly one user message and "
    "one assistant response, no preamble, no quotation marks, no labels."
)


def attacker_prompt(theme: str, persona: str | None = None) -> str:
    """Build a single attacker user-prompt from a theme + optional persona."""
    persona_block = ""
    if persona:
        persona_block = f"User persona (tone, background, age, style): {persona}\n\n"
    return (
        "Produce one short, natural user/assistant exchange about the topic "
        "below. The user should say something plausible about their life, "
        "preferences, or past events; the assistant should give a short, "
        "plausible response. Do not ask clarifying questions — just make a "
        "statement.\n\n"
        f"{persona_block}"
        f"Topic: {theme}\n\n"
        "Format exactly as:\n"
        "User: <one-sentence user message>\n"
        "Assistant: <one-sentence assistant response>"
    )


def parse_user_assistant(text: str) -> tuple[str, str] | None:
    """Parse 'User: ...\\nAssistant: ...' out of a model completion."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    user, asst = None, None
    for ln in lines:
        low = ln.lower()
        if user is None and low.startswith("user:"):
            user = ln.split(":", 1)[1].strip()
        elif user is not None and asst is None and low.startswith("assistant:"):
            asst = ln.split(":", 1)[1].strip()
            break
    if not user or not asst:
        return None
    return user, asst


def sample_batch(
    llm,
    tokenizer,
    prompts_messages: list[list[dict]],
    n_samples: int = 1,
    max_tokens: int = 180,
    temperature: float = 0.95,
    top_p: float = 0.95,
    seed: int = 0,
) -> list[list[str]]:
    """One vLLM call producing n_samples completions per prompt.

    Returns list[list[str]] of shape (B, n_samples) of raw completion text.
    """
    from vllm import SamplingParams
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompts_messages
    ]
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_samples,
        seed=seed,
    )
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [[o.text for o in out.outputs] for out in outs]


def yield_pairs_from_themes(
    themes: list[str],
    personas: list[str | None],
    n_samples_per: int,
    batch_size: int = 32,
    max_tokens: int = 180,
    seed: int = 0,
    source_label: str = "synth",
) -> Iterator[tuple[str, str, dict]]:
    """For each (theme, persona), generate n_samples completions and yield
    every parseable (u, a) pair with metadata.

    Loads the attacker vLLM engine on first call.
    """
    from memory._vllm_engines import get_attacker_engine
    llm, tokenizer = get_attacker_engine()
    assert len(themes) == len(personas)
    for start in range(0, len(themes), batch_size):
        end = min(start + batch_size, len(themes))
        msgs = [
            [
                {"role": "system", "content": ATTACKER_SYSTEM},
                {"role": "user", "content": attacker_prompt(themes[i], personas[i])},
            ]
            for i in range(start, end)
        ]
        completions = sample_batch(
            llm, tokenizer, msgs,
            n_samples=n_samples_per,
            max_tokens=max_tokens,
            seed=seed + start,
        )
        for j, cands in enumerate(completions):
            theme = themes[start + j]
            persona = personas[start + j]
            for k, c in enumerate(cands):
                pr = parse_user_assistant(c)
                if pr is None:
                    continue
                u, a = pr
                yield u, a, {
                    "source": source_label,
                    "theme": theme,
                    "persona": persona,
                    "sample_idx": k,
                }
