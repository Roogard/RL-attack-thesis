"""RL-trained adversarial memory-poisoning attacker (Phase 2).

See plans file and direction.md for the research framing. The package lays
out:

    probes.py        fixed domain-probe set for attacker memory read-access
    caches.py        per-epoch cache of clean artifacts (avoid recomputation)
    reward.py        composite reward components + curriculum schedule
    policy.py        Qwen2.5-3B + LoRA attacker wrapped for generation
    rollout.py       one end-to-end rollout (N autoregressive sessions + rewards)
    environment.py   verifiers.Environment subclass wiring memory + harness
    train.py         GRPO entrypoint
    eval_attack.py   held-out eval (clean vs poisoned, GPT-4o judge)
"""
