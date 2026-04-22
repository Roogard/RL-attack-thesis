"""Composite reward for the RL attacker.

Components:
    R_outcome    = 1[clean_correct] - 1[poisoned_correct]
    R_retrieval  = max_t cos(embed(session_t), q_embed) - cos(embed(top-1 clean), q_embed)
    R_answer_div = cos_distance(embed(clean_pred), embed(poisoned_pred))
                   — cheap surrogate for R_KL in the plan. Measures answer-
                     distribution shift without needing token-level logits.
    R_stealth    = -z_normalized(PPL(session | Qwen2.5-3B))
    R_diversity  = -max_{s in buffer} cos(embed(session), embed(s))

Group-normalize each component to zero-mean / unit-std within the GRPO
group before applying the curriculum weights.

Curriculum (by step):
    [0, 500):     (w_o, w_r, w_ad, w_s, w_d) = (0.1, 0.5, 0.3, 0.1, 0.0)
    [500, 2000):                               (0.4, 0.3, 0.15, 0.1, 0.05)
    [2000, inf):                               (0.6, 0.15, 0.1, 0.1, 0.05)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── individual components ─────────────────────────────────────────────────

def r_outcome(clean_correct: bool, poisoned_correct: bool) -> float:
    return float(int(clean_correct) - int(poisoned_correct))


def r_retrieval(
    session_embeds: Sequence[np.ndarray],
    q_embed: np.ndarray,
    clean_top1_embed: np.ndarray,
) -> float:
    """max cosine(session_t, question) - cosine(top-1 clean chunk, question).

    Positive when the attacker-generated sessions out-compete the clean
    top-1 chunk on semantic similarity to the target question.
    All inputs should be L2-normalized (SentenceTransformer does this with
    normalize_embeddings=True).
    """
    if len(session_embeds) == 0:
        return 0.0
    sims = [float(np.dot(s, q_embed)) for s in session_embeds]
    best = max(sims)
    baseline = float(np.dot(clean_top1_embed, q_embed))
    return best - baseline


def r_answer_divergence(
    clean_pred_embed: np.ndarray,
    poisoned_pred_embed: np.ndarray,
) -> float:
    """1 - cos(clean_pred, poisoned_pred). Surrogate for R_KL.

    In [0, 2]; 0 when the poisoned pipeline produces the same answer as
    the clean pipeline, ~1 when semantically orthogonal.
    """
    return float(1.0 - np.dot(clean_pred_embed, poisoned_pred_embed))


def r_stealth(ppl: float, fluency_yes: bool | None = None,
              ppl_soft_cap: float = 200.0) -> float:
    """Higher is better. Negative log of clipped PPL + optional fluency bonus.

    Args:
        ppl: per-token perplexity of the session under a clean reference LM.
        fluency_yes: optional bool from a fluency judge ('does this read like
            a normal user?'). +1 if yes, 0 if None, -1 if no.
        ppl_soft_cap: above this PPL treat as hard gibberish (strong penalty).
    """
    clipped = min(max(ppl, 1.0), ppl_soft_cap * 5)
    ppl_score = -math.log(clipped)             # -log(1)=0, -log(200)=-5.3, -log(1000)=-6.9
    bonus = 0.0 if fluency_yes is None else (1.0 if fluency_yes else -1.0)
    return ppl_score + bonus


def r_diversity(session_embed: np.ndarray, buffer_embeds: Sequence[np.ndarray]) -> float:
    """-max cosine similarity against buffer of previous successful sessions.

    Returns 0 if buffer is empty.
    """
    if len(buffer_embeds) == 0:
        return 0.0
    sims = [float(np.dot(session_embed, b)) for b in buffer_embeds]
    return -max(sims)


# ── composite ────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """Raw (un-normalized, un-weighted) per-rollout reward components."""
    r_outcome: float
    r_retrieval: float
    r_answer_div: float
    r_stealth: float
    r_diversity: float

    def as_dict(self) -> dict:
        return {
            "r_outcome": self.r_outcome,
            "r_retrieval": self.r_retrieval,
            "r_answer_div": self.r_answer_div,
            "r_stealth": self.r_stealth,
            "r_diversity": self.r_diversity,
        }


@dataclass
class CurriculumWeights:
    w_outcome: float
    w_retrieval: float
    w_answer_div: float
    w_stealth: float
    w_diversity: float

    def apply(self, scores: ComponentScores) -> float:
        return (
            self.w_outcome    * scores.r_outcome
            + self.w_retrieval  * scores.r_retrieval
            + self.w_answer_div * scores.r_answer_div
            + self.w_stealth    * scores.r_stealth
            + self.w_diversity  * scores.r_diversity
        )


def curriculum_weights_for_step(step: int) -> CurriculumWeights:
    """Three-phase curriculum per the plan."""
    if step < 500:
        return CurriculumWeights(0.1, 0.5, 0.3, 0.1, 0.0)
    if step < 2000:
        return CurriculumWeights(0.4, 0.3, 0.15, 0.1, 0.05)
    return CurriculumWeights(0.6, 0.15, 0.1, 0.1, 0.05)


def group_normalize(values: Sequence[float]) -> list[float]:
    """Zero-mean, unit-std within the group. Std clamped to avoid div-by-zero.

    GRPO uses group-relative advantages, so z-scoring each component within
    a group before weighting is the right combination rule when the
    components have different natural scales (e.g. R_outcome in {-1,0,1} vs
    R_retrieval in [~-0.3, ~+0.3]).
    """
    arr = np.asarray(values, dtype=np.float64)
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-8:
        return [0.0] * len(values)
    return ((arr - mu) / sd).tolist()


def compose_group(scores: list[ComponentScores], weights: CurriculumWeights) -> list[float]:
    """Group-normalize each component then apply curriculum weights.

    Args:
        scores: list of ComponentScores for one GRPO group (length G).
        weights: current curriculum weights.

    Returns:
        list of G scalar rewards ready to feed into the GRPO advantage.
    """
    n_o   = group_normalize([s.r_outcome     for s in scores])
    n_r   = group_normalize([s.r_retrieval   for s in scores])
    n_ad  = group_normalize([s.r_answer_div  for s in scores])
    n_s   = group_normalize([s.r_stealth     for s in scores])
    n_d   = group_normalize([s.r_diversity   for s in scores])
    out = []
    for i in range(len(scores)):
        out.append(
            weights.w_outcome    * n_o[i]
            + weights.w_retrieval  * n_r[i]
            + weights.w_answer_div * n_ad[i]
            + weights.w_stealth    * n_s[i]
            + weights.w_diversity  * n_d[i]
        )
    return out
