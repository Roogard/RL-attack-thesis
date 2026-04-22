"""Training environment that produces GRPO-style grouped rollouts.

Design note on frameworks: the plan originally named Will Brown's
`verifiers` library for multi-turn agentic RL. Verifiers' public API has
shifted across releases and tying the training loop to a specific version
has proven brittle. This module instead exposes a framework-agnostic
`RolloutBatch` dataclass and a `RolloutEnvironment` that any GRPO trainer
(custom loop, `trl.GRPOTrainer`, or a verifiers adapter) can drive.

Minimal adapter surface:
    env = RolloutEnvironment(...)
    batch = env.sample_group(qid, group_size=G)
    # batch contains G RolloutResult objects + final composite rewards ready
    # to become GRPO advantages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from attack.caches import CacheEntry, CleanCache
from attack.policy import AttackerPolicy
from attack.reward import (
    ComponentScores,
    compose_group,
    curriculum_weights_for_step,
)
from attack.rollout import RolloutResult, run_rollout_group


@dataclass
class RolloutBatch:
    qid: str
    rollouts: list[RolloutResult]
    final_rewards: list[float]          # length G, composite + group-normalized
    step: int                           # training step used for curriculum


class RolloutEnvironment:
    def __init__(
        self,
        cache: CleanCache,
        policy: AttackerPolicy,
        embedder,
        *,
        n_poison: int,
        domain: str,
        architecture_name: str,
        probes: Optional[list[str]] = None,
        top_k: int = 10,
        diversity_buffer_size: int = 512,
        max_new_tokens: int = 768,
    ):
        self.cache = cache
        self.policy = policy
        self.embedder = embedder
        self.n_poison = n_poison
        self.domain = domain
        self.architecture_name = architecture_name
        self.probes = probes
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self._diversity_buffer: list[np.ndarray] = []
        self._diversity_buffer_size = diversity_buffer_size

    def _update_diversity_buffer(self, rollouts: Sequence[RolloutResult]):
        # Add successful-rollout session embeds to the diversity memory
        for r in rollouts:
            if r.poisoned_correct is False and (
                r.component_scores.r_outcome > 0 or r.component_scores.r_retrieval > 0
            ):
                for s in r.sessions:
                    self._diversity_buffer.append(s.session_embed)
        # Truncate
        if len(self._diversity_buffer) > self._diversity_buffer_size:
            self._diversity_buffer = self._diversity_buffer[-self._diversity_buffer_size :]

    def sample_group(
        self,
        qid: str,
        question: dict,
        group_size: int,
        step: int = 0,
    ) -> RolloutBatch:
        """Run `group_size` rollouts on the same question.

        GRPO uses group-relative advantages, so each group must share a
        single (question, haystack) pair. The only stochasticity comes from
        the attacker policy's sampling.
        """
        cache_entry: CacheEntry = self.cache[qid]

        rollouts: list[RolloutResult] = run_rollout_group(
            question,
            cache_entry,
            self.policy,
            self.embedder,
            group_size=group_size,
            n_poison=self.n_poison,
            domain=self.domain,
            architecture_name=self.architecture_name,
            probes=self.probes,
            top_k=self.top_k,
            diversity_buffer=self._diversity_buffer,
            seen_haystack_session_ids=question.get("haystack_session_ids"),
            max_new_tokens=self.max_new_tokens,
        )

        weights = curriculum_weights_for_step(step)
        component_list: list[ComponentScores] = [r.component_scores for r in rollouts]
        final_rewards = compose_group(component_list, weights)

        self._update_diversity_buffer(rollouts)

        return RolloutBatch(
            qid=qid, rollouts=rollouts, final_rewards=final_rewards, step=step
        )
