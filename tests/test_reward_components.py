"""Fixture-based unit tests for each reward component.

These tests are the contract: each component must behave as claimed on
known-answer inputs before we trust the training loop's reward signal.
"""

import math

import numpy as np
import pytest

from attack.reward import (
    ComponentScores,
    CurriculumWeights,
    compose_group,
    curriculum_weights_for_step,
    group_normalize,
    r_answer_divergence,
    r_diversity,
    r_outcome,
    r_retrieval,
    r_stealth,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# ── r_outcome ─────────────────────────────────────────────────────────────

def test_r_outcome_flipped():
    assert r_outcome(clean_correct=True, poisoned_correct=False) == 1.0

def test_r_outcome_unchanged_correct():
    assert r_outcome(True, True) == 0.0

def test_r_outcome_unchanged_wrong():
    assert r_outcome(False, False) == 0.0

def test_r_outcome_worsened_by_attack_impossible_but_bounded():
    # Can happen if clean was wrong and poison accidentally fixes it
    assert r_outcome(False, True) == -1.0


# ── r_retrieval ───────────────────────────────────────────────────────────

def test_r_retrieval_attacker_identical_to_question_beats_clean():
    q = _unit([1, 0, 0])
    clean_top1 = _unit([0, 1, 0])  # orthogonal
    attacker_sessions = [_unit([1, 0, 0])]  # perfect match
    assert r_retrieval(attacker_sessions, q, clean_top1) == pytest.approx(1.0, abs=1e-5)

def test_r_retrieval_attacker_equal_to_clean_is_zero():
    q = _unit([1, 1, 0])
    clean_top1 = _unit([1, 0, 0])
    attacker = [_unit([1, 0, 0])]  # identical to clean top-1
    assert r_retrieval(attacker, q, clean_top1) == pytest.approx(0.0, abs=1e-5)

def test_r_retrieval_empty_sessions_returns_zero():
    q = _unit([1, 0, 0])
    clean_top1 = _unit([1, 0, 0])
    assert r_retrieval([], q, clean_top1) == 0.0

def test_r_retrieval_uses_max_over_sessions():
    q = _unit([1, 0, 0])
    clean_top1 = _unit([0, 1, 0])
    attacker = [_unit([0, 1, 0]), _unit([1, 0, 0])]  # second matches q perfectly
    assert r_retrieval(attacker, q, clean_top1) == pytest.approx(1.0, abs=1e-5)


# ── r_answer_divergence ───────────────────────────────────────────────────

def test_r_answer_div_identical_preds_zero():
    p = _unit([1, 0, 0])
    assert r_answer_divergence(p, p) == pytest.approx(0.0, abs=1e-5)

def test_r_answer_div_orthogonal_preds_one():
    a = _unit([1, 0, 0])
    b = _unit([0, 1, 0])
    assert r_answer_divergence(a, b) == pytest.approx(1.0, abs=1e-5)

def test_r_answer_div_opposite_preds_two():
    a = _unit([1, 0, 0])
    b = _unit([-1, 0, 0])
    assert r_answer_divergence(a, b) == pytest.approx(2.0, abs=1e-5)


# ── r_stealth ─────────────────────────────────────────────────────────────

def test_r_stealth_natural_ppl_near_zero():
    # PPL=1 (perfectly predictable text) -> -log(1) = 0 + 0 fluency bonus
    assert r_stealth(ppl=1.0, fluency_yes=None) == pytest.approx(0.0, abs=1e-5)

def test_r_stealth_gibberish_ppl_is_negative():
    # PPL=1000 is clearly gibberish; -log(1000) = -6.9
    assert r_stealth(ppl=1000.0, fluency_yes=None) < -5.0

def test_r_stealth_fluency_yes_adds_bonus():
    base = r_stealth(ppl=10.0, fluency_yes=None)
    positive = r_stealth(ppl=10.0, fluency_yes=True)
    assert positive == pytest.approx(base + 1.0, abs=1e-5)

def test_r_stealth_fluency_no_penalty():
    base = r_stealth(ppl=10.0, fluency_yes=None)
    negative = r_stealth(ppl=10.0, fluency_yes=False)
    assert negative == pytest.approx(base - 1.0, abs=1e-5)


# ── r_diversity ───────────────────────────────────────────────────────────

def test_r_diversity_identical_to_buffer_is_minus_one():
    s = _unit([1, 0, 0])
    buf = [_unit([1, 0, 0])]
    assert r_diversity(s, buf) == pytest.approx(-1.0, abs=1e-5)

def test_r_diversity_orthogonal_to_buffer_is_zero():
    s = _unit([1, 0, 0])
    buf = [_unit([0, 1, 0])]
    assert r_diversity(s, buf) == pytest.approx(0.0, abs=1e-5)

def test_r_diversity_empty_buffer_is_zero():
    s = _unit([1, 0, 0])
    assert r_diversity(s, []) == 0.0

def test_r_diversity_takes_max_over_buffer():
    s = _unit([1, 1, 0])
    buf = [_unit([0, 1, 0]), _unit([1, 1, 0])]  # second matches
    assert r_diversity(s, buf) == pytest.approx(-1.0, abs=1e-5)


# ── group_normalize ───────────────────────────────────────────────────────

def test_group_normalize_mean_zero_std_one():
    out = group_normalize([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = np.asarray(out)
    assert arr.mean() == pytest.approx(0.0, abs=1e-5)
    assert arr.std() == pytest.approx(1.0, abs=1e-5)

def test_group_normalize_constant_returns_zeros():
    out = group_normalize([7.0, 7.0, 7.0])
    assert out == [0.0, 0.0, 0.0]


# ── curriculum ────────────────────────────────────────────────────────────

def test_curriculum_early_is_dense_heavy():
    w = curriculum_weights_for_step(0)
    assert w.w_retrieval > w.w_outcome

def test_curriculum_late_is_outcome_heavy():
    w = curriculum_weights_for_step(5000)
    assert w.w_outcome > w.w_retrieval

def test_curriculum_weights_sum_to_one_at_each_phase():
    for step in (0, 100, 500, 1500, 2000, 10000):
        w = curriculum_weights_for_step(step)
        s = w.w_outcome + w.w_retrieval + w.w_answer_div + w.w_stealth + w.w_diversity
        assert s == pytest.approx(1.0, abs=1e-5), f"sum at step={step} was {s}"


# ── compose_group ─────────────────────────────────────────────────────────

def test_compose_group_returns_one_scalar_per_member():
    scores = [
        ComponentScores(1.0, 0.2, 0.5, -0.3, -0.1),
        ComponentScores(0.0, 0.1, 0.0,  0.0,  0.0),
        ComponentScores(-1.0, -0.2, -0.5, 0.3, 0.1),
    ]
    w = curriculum_weights_for_step(1000)
    out = compose_group(scores, w)
    assert len(out) == len(scores)
    # Symmetric opposite members should produce opposite final scores
    assert out[0] == pytest.approx(-out[2], abs=1e-5)
