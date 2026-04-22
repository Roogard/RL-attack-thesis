"""One end-to-end rollout: N autoregressive sessions + per-session counterfactual
turn rewards + final outcome reward.

Pseudocode is in the plan file. This module is the bridge between the policy
(one call generates one session) and the reward module (component scores).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from attack.caches import CacheEntry
from attack.policy import AttackerPolicy, GenerationResult, summarize_session
from attack.probes import read_memory, DOMAIN_PROBES
from attack.reward import (
    ComponentScores,
    r_answer_divergence,
    r_diversity,
    r_outcome,
    r_retrieval,
    r_stealth,
)
from harness import ask_qwen_batch, judge_answer_local
from memory.rag import RAGMemory


def _session_text(session: list[dict]) -> str:
    """Flatten a session into the same shape RAGMemory indexes it as."""
    parts = []
    for t in session:
        role = t["role"].capitalize()
        parts.append(f"{role}: {t['content'].strip()}")
    return "\n".join(parts)


@dataclass
class SessionRecord:
    index: int
    session: list[dict]
    raw_text: str
    session_embed: np.ndarray
    ppl: float
    token_logprobs_sum: float
    token_count: int
    turn_reward_raw: float                  # retrieval-shift contribution only
    retrieval_sim_delta: float              # for logging


@dataclass
class RolloutResult:
    qid: str
    sessions: list[SessionRecord]
    poisoned_ctx: str
    poisoned_pred: str
    poisoned_correct: bool
    poisoned_pred_embed: np.ndarray
    component_scores: ComponentScores       # flattened to one per rollout
    per_session_token_logprobs: list[float] # for GRPO at turn granularity


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def run_rollout(
    q: dict,
    cache_entry: CacheEntry,
    policy: AttackerPolicy,
    embedder,
    *,
    n_poison: int,
    domain: str,
    architecture_name: str,
    probes: Optional[list[str]] = None,
    top_k: int = 10,
    diversity_buffer: Optional[list[np.ndarray]] = None,
    seen_haystack_session_ids: Optional[list[str]] = None,
    max_new_tokens: int = 768,
) -> RolloutResult:
    """Single-rollout convenience wrapper around `run_rollout_group(..., G=1)`."""
    return run_rollout_group(
        q, cache_entry, policy, embedder,
        group_size=1,
        n_poison=n_poison,
        domain=domain,
        architecture_name=architecture_name,
        probes=probes,
        top_k=top_k,
        diversity_buffer=diversity_buffer,
        seen_haystack_session_ids=seen_haystack_session_ids,
        max_new_tokens=max_new_tokens,
    )[0]


def run_rollout_group(
    q: dict,
    cache_entry: CacheEntry,
    policy: AttackerPolicy,
    embedder,
    *,
    group_size: int,
    n_poison: int,
    domain: str,
    architecture_name: str,
    probes: Optional[list[str]] = None,
    top_k: int = 10,
    diversity_buffer: Optional[list[np.ndarray]] = None,
    seen_haystack_session_ids: Optional[list[str]] = None,
    max_new_tokens: int = 768,
) -> list[RolloutResult]:
    """Run G GRPO-group rollouts on the same question with batched generates.

    Per session t, all G generate calls are issued as ONE batched HF
    generate, all G perplexity forwards as one batched forward, and at
    the end the G final answers + judges are batched as well. Memory state
    is per-rollout (each rollout owns its own RAGMemory).
    """
    probes = probes if probes is not None else list(DOMAIN_PROBES)
    diversity_buffer = diversity_buffer if diversity_buffer is not None else []

    haystack_sids = q.get(
        "haystack_session_ids",
        [f"s{i}" for i in range(len(q["haystack_sessions"]))],
    )

    mems: list[RAGMemory] = [RAGMemory() for _ in range(group_size)]
    for mem in mems:
        mem.index(q["haystack_sessions"], q["haystack_dates"], haystack_sids)

    baseline_sim = _cosine(cache_entry.clean_top1_embed, cache_entry.q_embed)
    prev_retrieval_scores = [baseline_sim] * group_size
    prior_summaries_per: list[list[str]] = [[] for _ in range(group_size)]
    session_records_per: list[list[SessionRecord]] = [[] for _ in range(group_size)]

    for t in range(n_poison):
        # 1) per-rollout memory read (cheap, leave serial)
        readouts = [
            read_memory(mems[g], q["question_date"], top_k_per_probe=3, probes=probes)
            for g in range(group_size)
        ]
        prompts_data = [
            dict(
                domain=domain,
                architecture_name=architecture_name,
                memory_readout=readouts[g],
                prior_summaries=prior_summaries_per[g],
                budget_total=n_poison,
                step_index=t,
            )
            for g in range(group_size)
        ]

        # 2) BATCHED generate
        gens = policy.generate_session_batch(prompts_data, max_new_tokens=max_new_tokens)

        # 3) per-rollout: index + collect text
        sessions: list[list[dict]] = []
        session_texts: list[str] = []
        for g, gen in enumerate(gens):
            session = gen.session if gen.session else [
                {"role": "user", "content": "(empty)"},
                {"role": "assistant", "content": "(empty)"},
            ]
            sessions.append(session)
            mems[g].index([session], [q["question_date"]], [f"poison_{t}"])
            session_texts.append(_session_text(session))

        # 4) BATCHED embedding + perplexity
        sess_embeds_np = np.asarray(
            embedder.encode(session_texts, normalize_embeddings=True),
            dtype=np.float32,
        )  # (G, d)
        ppls = policy.perplexity_batch(session_texts)

        # 5) per-rollout turn reward + record
        for g in range(group_size):
            sess_embed = sess_embeds_np[g]
            curr_poison_embeds = [s.session_embed for s in session_records_per[g]] + [sess_embed]
            curr_top = max(
                [_cosine(e, cache_entry.q_embed) for e in curr_poison_embeds]
                + [baseline_sim]
            )
            turn_reward = curr_top - prev_retrieval_scores[g]
            prev_retrieval_scores[g] = curr_top

            gen = gens[g]
            if gen.token_logprobs is not None:
                token_sum = float(gen.token_logprobs.sum().item())
                token_count = int(gen.token_logprobs.shape[0])
            else:
                token_sum = 0.0
                token_count = 0

            session_records_per[g].append(SessionRecord(
                index=t,
                session=sessions[g],
                raw_text=gen.raw_text,
                session_embed=sess_embed,
                ppl=float(ppls[g]),
                token_logprobs_sum=token_sum,
                token_count=token_count,
                turn_reward_raw=float(turn_reward),
                retrieval_sim_delta=float(turn_reward),
            ))
            prior_summaries_per[g].append(summarize_session(sessions[g]))

    # 6) BATCHED final retrieve, answer, judge across G
    poisoned_ctxs = [
        mems[g].retrieve(q["question"], q["question_date"], top_k=top_k)
        for g in range(group_size)
    ]
    poisoned_preds = ask_qwen_batch(
        poisoned_ctxs,
        [q["question"]] * group_size,
        [q["question_date"]] * group_size,
    )
    poisoned_corrects = [
        judge_answer_local(
            q["question"], q["answer"], poisoned_preds[g],
            q["question_type"], q["question_id"]
        )
        for g in range(group_size)
    ]
    poisoned_pred_embeds = np.asarray(
        embedder.encode(poisoned_preds, normalize_embeddings=True),
        dtype=np.float32,
    )

    # 7) build RolloutResults
    results: list[RolloutResult] = []
    for g in range(group_size):
        records = session_records_per[g]
        session_embeds = [s.session_embed for s in records]
        ppe = poisoned_pred_embeds[g]
        scores = ComponentScores(
            r_outcome=r_outcome(cache_entry.clean_correct, poisoned_corrects[g]),
            r_retrieval=r_retrieval(session_embeds, cache_entry.q_embed,
                                    cache_entry.clean_top1_embed),
            r_answer_div=r_answer_divergence(cache_entry.clean_pred_embed, ppe),
            r_stealth=float(np.mean([r_stealth(s.ppl) for s in records])) if records else 0.0,
            r_diversity=float(np.mean([
                r_diversity(s.session_embed, diversity_buffer) for s in records
            ])) if (diversity_buffer and records) else 0.0,
        )
        mems[g].clear()
        results.append(RolloutResult(
            qid=q["question_id"],
            sessions=records,
            poisoned_ctx=poisoned_ctxs[g],
            poisoned_pred=poisoned_preds[g],
            poisoned_correct=bool(poisoned_corrects[g]),
            poisoned_pred_embed=ppe,
            component_scores=scores,
            per_session_token_logprobs=[s.token_logprobs_sum for s in records],
        ))

    del mems
    return results
