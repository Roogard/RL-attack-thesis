"""GRPO training entrypoint for the RL memory-poisoning attacker.

Design: minimal custom GRPO loop over HF-transformers + peft. Each step:
  1. Pick one question (cycling through clean_correct train set).
  2. Sample G rollouts from the RolloutEnvironment.
  3. Compute final group-normalized composite rewards.
  4. Compute per-rollout advantage = R_i - mean(R).
  5. Backprop policy gradient on sum of per-session token logprobs weighted by advantage.
  6. Apply KL penalty vs. reference (frozen base) every K steps.

Why custom and not trl/verifiers: at the time of writing, trl.GRPOTrainer
assumes single-prompt-single-completion scalar rewards and doesn't naturally
support the multi-turn observe-act-observe rollouts used here, while
verifiers' multi-turn API has been moving. Writing the loop directly (~150
lines) keeps us unblocked and debuggable.

Usage:
    python -m attack.train --config configs/attack_rag.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from attack.caches import CleanCache
from attack.environment import RolloutEnvironment
from attack.policy import AttackerPolicy
from attack.probes import DOMAIN_PROBES


def load_config(path: str) -> dict:
    with io.open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(data_path: str, split_path: str, split: str) -> list[dict]:
    with io.open(data_path, encoding="utf-8") as f:
        all_qs = json.load(f)
    lookup = {q["question_id"]: q for q in all_qs}
    with io.open(split_path, encoding="utf-8") as f:
        split_info = json.load(f)
    if split == "all" and "all" not in split_info:
        qids = split_info["train"] + split_info["val"] + split_info["test"]
    else:
        qids = split_info[split]
    out = []
    for qid in qids:
        if qid in lookup:
            q = lookup[qid]
            q.setdefault("haystack_session_ids",
                         [f"s{i}" for i in range(len(q["haystack_sessions"]))])
            out.append(q)
    return out


def _chunked_policy_gradient_step(
    policy: AttackerPolicy,
    rollouts: list,
    advantages: np.ndarray,
    chunk_size: int,
) -> float:
    """Chunked forward+backward for the REINFORCE policy gradient.

    The monolithic version (single forward of all group_size * n_poison
    sessions, then one loss.backward) peaks activation memory well past
    what fits on a single GPU when two vLLM engines + a 7B judge + the
    attacker model all share it. This version processes up to
    `chunk_size` sequences per forward, calls backward per chunk, and
    lets gradients accumulate in the LoRA parameters. Mathematically
    identical to the monolithic version thanks to the linearity of
    the loss:  loss = -sum_seq(adv[rollout(seq)] * logp(seq)) / G.

    Returns the scalar loss value (for logging). Callers should call
    optim.step() after this returns; grad accumulation leaves the
    aggregate gradient ready to apply.
    """
    flat_texts: list[str] = []
    rollout_idx: list[int] = []
    for ri, r in enumerate(rollouts):
        for sess in r.sessions:
            if not sess.session or len(sess.session) == 0:
                continue
            text = sess.raw_text or ""
            if not text.strip():
                continue
            flat_texts.append(text)
            rollout_idx.append(ri)

    G = len(rollouts)
    if not flat_texts:
        return 0.0

    if policy.tokenizer.pad_token_id is None:
        policy.tokenizer.pad_token = policy.tokenizer.eos_token
    prev_side = policy.tokenizer.padding_side
    policy.tokenizer.padding_side = "right"
    try:
        enc = policy.tokenizer(
            flat_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        )
    finally:
        policy.tokenizer.padding_side = prev_side

    adv_tensor = torch.tensor(advantages, device=policy.device, dtype=torch.float32)
    rollout_idx_tensor = torch.tensor(
        rollout_idx, device=policy.device, dtype=torch.long
    )

    N = len(flat_texts)
    total_loss = 0.0
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        ids = enc.input_ids[start:end].to(policy.device)
        mask = enc.attention_mask[start:end].to(policy.device)

        out = policy.model(ids, attention_mask=mask)
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        shift_mask = mask[:, 1:].float()

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        label_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        label_lp = label_lp * shift_mask
        seq_logp = label_lp.sum(dim=1)  # (chunk,)

        chunk_adv = adv_tensor[rollout_idx_tensor[start:end]]
        chunk_loss = -(chunk_adv * seq_logp).sum() / G
        chunk_loss.backward()
        total_loss += float(chunk_loss.detach().cpu().item())

    return total_loss


def _save_checkpoint(out_dir: Path, step: int, policy: AttackerPolicy,
                     optim: torch.optim.Optimizer, rng: random.Random):
    ckpt_dir = out_dir / f"adapter_step_{step}"
    policy.save_adapter(str(ckpt_dir))
    state = {
        "step": step,
        "optim": optim.state_dict(),
        "py_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")
    return ckpt_dir


def _load_checkpoint(path: str, optim: torch.optim.Optimizer, rng: random.Random) -> int:
    state_path = Path(path) / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"No trainer_state.pt at {state_path} — was this checkpoint saved by "
            "the new resumable trainer? Older adapter-only checkpoints can't resume."
        )
    state = torch.load(state_path, map_location="cpu")
    optim.load_state_dict(state["optim"])
    rng.setstate(state["py_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    return int(state["step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit_one", action="store_true",
                        help="Debug: overfit a single question (sanity check).")
    parser.add_argument("--resume_from", default=None,
                        help="Path to an adapter_step_N directory saved by a prior "
                             "run. Restores LoRA weights, optimizer state, RNGs, "
                             "and step counter — training continues from step N+1.")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="Graceful stop: save a resumable checkpoint and exit "
                             "once wall time exceeds this many hours. Use slightly "
                             "below your true budget (e.g. 7.5 for an 8h window) so "
                             "the final checkpoint save has time to finish.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Pre-load BOTH vLLM engines (answer + attacker) BEFORE any HF model
    # touches CUDA in this process. Once the parent has CUDA state,
    # vLLM's forked EngineCore subprocess fails to init. The attacker
    # engine is the rollout-generation path (LoRA hot-swapped by
    # sync_lora_to_vllm); the answer engine is the victim answer path.
    from memory import _vllm_engines
    _vllm_engines.get_answer_engine()
    _vllm_engines.get_attacker_engine(max_lora_rank=cfg["lora_rank"])

    # ── data & cache ─────────────────────────────────────────────────────
    print("[train] loading train split ...")
    train_qs = load_dataset(cfg["data_path"], cfg["split_path"], "train")
    print(f"[train] {len(train_qs)} train questions")

    cache_path = cfg["cache_path"]
    if os.path.exists(cache_path):
        print(f"[train] loading cache from {cache_path}")
        cache = CleanCache.load(cache_path)
    else:
        print(f"[train] building cache -> {cache_path}")
        cache = CleanCache.build(
            train_qs,
            embed_model_name=cfg["embed_model"],
            top_k=cfg["top_k"],
            answer_batch_size=cfg["answer_batch_size"],
        )
        cache.save(cache_path)
    cache = cache.filter_clean_correct()
    print(f"[train] cache has {len(cache)} clean_correct entries")

    # ── policy ───────────────────────────────────────────────────────────
    print("[train] initializing attacker policy ...")
    # Scope the vLLM LoRA cache dir to this run so parallel trainings
    # on different configs don't overwrite each other's adapter-on-disk.
    vllm_lora_cache_dir = os.path.join(cfg["output_dir"], "vllm_lora_cache")
    policy = AttackerPolicy(
        model_id=cfg["attacker_model_id"],
        lora_adapter_path=args.resume_from,    # None on cold start
        lora_rank=cfg["lora_rank"],
        vllm_lora_cache_dir=vllm_lora_cache_dir,
    )

    # ── embedder (shared with RAG) ───────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(cfg["embed_model"])

    # ── environment ──────────────────────────────────────────────────────
    probes = DOMAIN_PROBES if cfg.get("memory_read_access", True) else []
    env = RolloutEnvironment(
        cache=cache,
        policy=policy,
        embedder=embedder,
        n_poison=cfg["n_poison"],
        domain=cfg["domain"],
        architecture_name=cfg["architecture_name"],
        probes=probes,
        top_k=cfg["top_k"],
        diversity_buffer_size=cfg.get("diversity_buffer_size", 512),
        max_new_tokens=cfg.get("max_new_tokens", 768),
    )

    # ── optimizer ────────────────────────────────────────────────────────
    optim = torch.optim.AdamW(
        (p for p in policy.model.parameters() if p.requires_grad),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    # ── resume? ─────────────────────────────────────────────────────────
    rng = random.Random(cfg.get("seed", 0))
    start_step = 0
    if args.resume_from:
        start_step = _load_checkpoint(args.resume_from, optim, rng) + 1
        print(f"[train] resumed from {args.resume_from}; starting at step {start_step}")

    # Push the initial (or resumed) LoRA weights to vLLM so the first
    # rollout samples from the correct distribution.
    policy.sync_lora_to_vllm()

    # ── training loop ────────────────────────────────────────────────────
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    train_qids = cache.qids()
    if args.overfit_one:
        train_qids = train_qids[:1]
        print(f"[train] OVERFIT MODE — single qid={train_qids[0]}")

    q_lookup = {q["question_id"]: q for q in train_qs}

    max_steps = cfg["max_steps"]
    group_size = cfg["group_size"]
    log_every = cfg.get("log_every", 10)
    ckpt_every = cfg.get("ckpt_every", 500)

    log_mode = "a" if args.resume_from else "w"
    deadline_seconds = args.max_hours * 3600 if args.max_hours else None
    start_wall = time.time()
    stopped_early_at: Optional[int] = None
    with io.open(log_path, log_mode, encoding="utf-8") as logf:
        for step in tqdm(range(start_step, max_steps), desc="train",
                         initial=start_step, total=max_steps):
            if deadline_seconds is not None and (time.time() - start_wall) > deadline_seconds:
                tqdm.write(
                    f"[train] --max_hours={args.max_hours} reached "
                    f"(elapsed {(time.time() - start_wall)/3600:.2f}h). "
                    f"Saving resumable checkpoint at step {step} and stopping."
                )
                ckpt = _save_checkpoint(out_dir, step, policy, optim, rng)
                tqdm.write(f"  [ckpt] saved {ckpt}")
                stopped_early_at = step
                break
            qid = rng.choice(train_qids)
            question = q_lookup[qid]

            # Gradient checkpointing is needed to fit the backward pass in
            # memory, but it forces use_cache=False during HF generate(),
            # which kills KV caching and makes attacker session generation
            # ~5-10x slower. Disable during rollouts (inference only),
            # re-enable for the logprob recompute + backward.
            policy.model.gradient_checkpointing_disable()
            policy.model.eval()
            with torch.no_grad():
                batch = env.sample_group(qid, question, group_size=group_size, step=step)

            rewards = np.asarray(batch.final_rewards, dtype=np.float32)
            baseline = rewards.mean()
            advantages = rewards - baseline

            policy.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            policy.model.train()
            optim.zero_grad(set_to_none=True)
            # Chunked forward+backward keeps peak activation memory to
            # chunk_size sequences through the 3B attacker. Override the
            # default via LOGPROB_CHUNK_SIZE if you need to tune.
            chunk_size = int(os.environ.get("LOGPROB_CHUNK_SIZE", "8"))
            loss_val = _chunked_policy_gradient_step(
                policy, batch.rollouts, advantages, chunk_size=chunk_size
            )
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.model.parameters() if p.requires_grad],
                cfg.get("grad_clip", 1.0),
            )
            optim.step()
            # Sync the updated LoRA to vLLM so the NEXT rollout is on-policy.
            policy.sync_lora_to_vllm()

            if step % log_every == 0:
                n_flip = sum(
                    r.component_scores.r_outcome > 0 for r in batch.rollouts
                )
                log = {
                    "step": step,
                    "qid": qid,
                    "loss": loss_val,
                    "reward_mean": float(rewards.mean()),
                    "reward_max": float(rewards.max()),
                    "r_outcome_mean": float(np.mean([r.component_scores.r_outcome for r in batch.rollouts])),
                    "r_retrieval_mean": float(np.mean([r.component_scores.r_retrieval for r in batch.rollouts])),
                    "r_answer_div_mean": float(np.mean([r.component_scores.r_answer_div for r in batch.rollouts])),
                    "r_stealth_mean": float(np.mean([r.component_scores.r_stealth for r in batch.rollouts])),
                    "flips_in_group": int(n_flip),
                }
                logf.write(json.dumps(log) + "\n")
                logf.flush()
                tqdm.write(f"  step={step} loss={log['loss']:.3f} R={log['reward_mean']:.3f} flips={n_flip}/{group_size}")

            if step > 0 and step % ckpt_every == 0:
                ckpt = _save_checkpoint(out_dir, step, policy, optim, rng)
                tqdm.write(f"  [ckpt] saved {ckpt}")

    if stopped_early_at is not None:
        print(
            f"[train] stopped early at step {stopped_early_at} due to --max_hours; "
            f"resume with --resume_from {out_dir}/adapter_step_{stopped_early_at}"
        )
    else:
        final_ckpt = out_dir / "adapter_final"
        policy.save_adapter(str(final_ckpt))
        torch.save({
            "step": max_steps - 1,
            "optim": optim.state_dict(),
            "py_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
        }, final_ckpt / "trainer_state.pt")
        print(f"[train] done. final adapter -> {final_ckpt}")


if __name__ == "__main__":
    main()
