"""Stage B — HotFlip gradient-based discrete text optimization.

For each hub h, greedily flip tokens in a seed string to maximize
cos(encode(text), h). Uses MiniLM (all-MiniLM-L6-v2) directly via HF
transformers so we can backprop through the encoder.

Algorithm, per hub (HotFlip / Wallace-style):
  init: tokenize a seed string (optionally a BoN winner)
  for step in 1..n_iter:
    1. encode(ids) → pooled emb → cos with h → loss = -cos
    2. grad = ∂loss/∂input_embeds  at each (position, hidden_dim)
    3. for each (pos, vocab_token):
         Δloss ≈ grad[pos] · (vocab_embed[token] − vocab_embed[ids[pos]])
       (first-order approximation)
    4. take top-k candidate (pos, token) pairs by predicted Δloss
    5. evaluate each ACTUAL cos by re-encoding with the flipped sequence
    6. apply the single swap that most improves cos; stop if no swap helps

The top-k candidate re-evaluation step (5) is the key escape from the
first-order approximation's blind spots — gradients lie about large flips.

Fluency is NOT regularized in this first cut. Text may drift toward
adversarial-looking gibberish. Eyeball the outputs; if unreadable, add
an LM-PPL term in a follow-up.

Usage (on cluster):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m attacks.hubness.stage_b_grad \\
        --hubs results/stage_a/hubs_K30.pkl \\
        --n_iter 400 --top_k_cand 20 \\
        --out results/stage_b/grad_K30.json

    # to seed from BoN outputs (recommended):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m attacks.hubness.stage_b_grad \\
        --hubs results/stage_a/hubs_K30.pkl \\
        --seed_from results/stage_b/bon_K30_N64.json \\
        --n_iter 400 --out results/stage_b/grad_K30_bon_seed.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks.hubness.stage_b_common import (
    PoisonSessionRecord,
    format_round_text,
    load_hubs,
    read_poison_file,
    write_poison_file,
)


# Differentiable MiniLM wrapper — recreates what sentence-transformers
# does for all-MiniLM-L6-v2 (mean-pool + L2 normalize) but exposes
# gradients w.r.t. input embeddings.
def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _encode_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward MiniLM from pre-computed input embeddings. Returns
    L2-normalized mean-pooled sentence embedding.
    """
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    pooled = _mean_pool(out.last_hidden_state, attention_mask)
    return F.normalize(pooled, p=2, dim=1)


def _encode_from_ids(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = _mean_pool(out.last_hidden_state, attention_mask)
    return F.normalize(pooled, p=2, dim=1)


# MiniLM chunks separate U/A via the newline inside the formatted string.
# We treat a single tokenization of "User: X\nAssistant: Y" as the unit of
# optimization. To keep round structure stable, we protect the literal
# prefix tokens ("User:", "Assistant:") from being flipped — only content
# tokens are candidates.
def _build_ids_with_protected_mask(
    tokenizer, user_msg: str, assistant_msg: str, device: str,
):
    text = format_round_text(user_msg, assistant_msg)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    ids = enc["input_ids"][0]           # (T,)
    attn = enc["attention_mask"][0]     # (T,)

    # Protect the BERT special tokens ([CLS], [SEP], [PAD]) and the
    # "user", ":", "assistant" framing tokens that encode the role
    # structure. Tokens for "user" and "assistant" in bert-base's vocab
    # tend to be single-token; protect them wherever they appear.
    protect_strs = {"user", "assistant", "[CLS]", "[SEP]", "[PAD]", ":", "\n"}
    special_ids = set(tokenizer.all_special_ids)
    protected = torch.zeros_like(ids, dtype=torch.bool)
    for i, tid in enumerate(ids.tolist()):
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tid in special_ids:
            protected[i] = True
            continue
        tok_stripped = tok.replace("##", "").lower()
        if tok_stripped in protect_strs:
            protected[i] = True
    return ids, attn, protected


def _decode_round(tokenizer, ids: torch.Tensor) -> tuple[str, str]:
    """Decode the HotFlip-optimized ids back into (user_msg, assistant_msg).

    We tokenized the full format string "User: X\\nAssistant: Y". To split
    it back, find the first "assistant" token after a newline; everything
    between colon and newline after "User" is user_msg, everything after
    colon after "Assistant" is assistant_msg. If splitting fails, we fall
    back to using the whole decoded string as user_msg with a trivial
    assistant response — still gets indexed, just loses role fidelity.
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # Look for the pattern "user :" and "assistant :"
    low = text.lower()
    u_idx = low.find("user")
    a_idx = low.find("assistant", u_idx + 1 if u_idx >= 0 else 0)
    if u_idx < 0 or a_idx < 0:
        return text.strip(), "Got it."
    # Content between User and Assistant markers
    after_user = text[u_idx:]
    # first colon after "User"
    u_colon = after_user.find(":")
    # Split point at "assistant"
    rel_a = after_user.lower().find("assistant")
    if u_colon < 0 or rel_a < 0 or u_colon >= rel_a:
        return text.strip(), "Got it."
    user_msg = after_user[u_colon + 1 : rel_a].strip()
    rest = after_user[rel_a:]
    a_colon = rest.find(":")
    if a_colon < 0:
        return user_msg, rest.strip()
    assistant_msg = rest[a_colon + 1 :].strip()
    return user_msg, assistant_msg


@torch.no_grad()
def _cos_with_ids(model, ids: torch.Tensor, attn: torch.Tensor, h: torch.Tensor) -> float:
    vec = _encode_from_ids(model, ids.unsqueeze(0), attn.unsqueeze(0))
    return float((vec[0] * h).sum())


def _hotflip_one_hub(
    model,
    tokenizer,
    hub: torch.Tensor,
    seed_user: str,
    seed_asst: str,
    n_iter: int,
    top_k_cand: int,
    device: str,
    log_every: int,
) -> tuple[str, str, float, int]:
    """Optimize text toward hub. Returns (user_msg, assistant_msg, final_cos, steps_applied)."""
    ids, attn, protected = _build_ids_with_protected_mask(
        tokenizer, seed_user, seed_asst, device,
    )
    T = ids.shape[0]
    vocab_w = model.get_input_embeddings().weight  # (V, D)

    current_cos = _cos_with_ids(model, ids, attn, hub)
    applied = 0
    for step in range(n_iter):
        # Forward+backward on input embeddings.
        emb = model.get_input_embeddings()(ids.unsqueeze(0)).detach()
        emb.requires_grad_(True)
        sent_vec = _encode_from_embeds(model, emb, attn.unsqueeze(0))
        loss = -(sent_vec[0] * hub).sum()  # maximize cos
        model.zero_grad()
        loss.backward()
        grad = emb.grad.detach()[0]  # (T, D)

        # First-order Δloss estimate:
        # Δloss(pos, v) ≈ grad[pos] · (vocab_w[v] − vocab_w[ids[pos]])
        # We want the MOST NEGATIVE Δloss (i.e., improves cos the most).
        current_emb = vocab_w[ids]                          # (T, D)
        # (T, V) = grad @ vocab_w.T − (grad · current_emb).sum(-1)[:, None]
        gv = grad @ vocab_w.T                                # (T, V)
        g_cur = (grad * current_emb).sum(dim=1, keepdim=True)  # (T, 1)
        delta = gv - g_cur                                   # (T, V)

        # Mask out protected positions (no swaps there).
        delta[protected] = float("inf")  # large positive = not chosen

        # Mask out swapping to the same token at that position.
        delta.scatter_(1, ids.unsqueeze(1), float("inf"))

        # Pick top-k_cand most-negative deltas.
        flat = delta.flatten()
        cand_flat = torch.topk(flat, k=top_k_cand, largest=False).indices
        cand_pos = cand_flat // vocab_w.shape[0]
        cand_tok = cand_flat % vocab_w.shape[0]

        # Evaluate each candidate by actually re-encoding.
        best_cos = current_cos
        best_swap = None
        for i in range(top_k_cand):
            pos = int(cand_pos[i])
            new_tok = int(cand_tok[i])
            trial = ids.clone()
            trial[pos] = new_tok
            cos = _cos_with_ids(model, trial, attn, hub)
            if cos > best_cos + 1e-6:
                best_cos = cos
                best_swap = (pos, new_tok)

        if best_swap is None:
            break  # converged — no first-order candidate actually improves cos
        ids[best_swap[0]] = best_swap[1]
        current_cos = best_cos
        applied += 1
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step+1:4d}  cos={current_cos:.4f}  applied={applied}")

    user_msg, asst_msg = _decode_round(tokenizer, ids)
    return user_msg, asst_msg, current_cos, applied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hubs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed_from", default=None,
                   help="Optional poison file (retrieval/bon) to seed initial text per hub.")
    p.add_argument("--n_iter", type=int, default=400)
    p.add_argument("--top_k_cand", type=int, default=20,
                   help="Number of top-gradient candidates to actually re-encode per step.")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    hubs_blob = load_hubs(args.hubs)
    H_np = hubs_blob["hubs"]
    K = len(H_np)
    model_name = hubs_blob["model"]
    print(f"[stage_b/grad] K={K}  encoder={model_name}  device={args.device}")

    # Seed texts per hub.
    if args.seed_from:
        seed_payload = read_poison_file(args.seed_from)
        seed_by_hub = {s["hub_idx"]: (s["user_msg"], s["assistant_msg"]) for s in seed_payload["sessions"]}
        print(f"[stage_b/grad] seeding from {args.seed_from} "
              f"({len(seed_by_hub)} hubs; missing hubs will use default seed)")
    else:
        seed_by_hub = {}

    default_seed = (
        "I was thinking about my work schedule and preferences lately.",
        "That sounds reasonable — let me know if you want me to remember any details.",
    )

    # Load MiniLM via HF transformers (differentiable).
    from transformers import AutoModel, AutoTokenizer
    full_name = (
        model_name if model_name.startswith("sentence-transformers/")
        else f"sentence-transformers/{model_name}"
    )
    print(f"[stage_b/grad] loading {full_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(full_name)
    model = AutoModel.from_pretrained(full_name).to(args.device).eval()
    # We want grads through the model but not through its parameters.
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # Move hubs to torch.
    hubs_t = torch.tensor(H_np, device=args.device, dtype=model.dtype if hasattr(model, "dtype") else torch.float32)

    records: list[PoisonSessionRecord] = []
    t0_total = time.time()
    for k in range(K):
        seed_u, seed_a = seed_by_hub.get(k, default_seed)
        print(f"[stage_b/grad] hub {k+1}/{K}  seed_cos=... (computing)")
        t0 = time.time()
        user_msg, asst_msg, final_cos, applied = _hotflip_one_hub(
            model, tokenizer, hubs_t[k],
            seed_u, seed_a,
            n_iter=args.n_iter, top_k_cand=args.top_k_cand,
            device=args.device, log_every=args.log_every,
        )
        dt = time.time() - t0
        print(f"  hub {k+1}: final cos={final_cos:.4f}  steps_applied={applied}  ({dt:.1f}s)")
        records.append(PoisonSessionRecord(
            hub_idx=k,
            user_msg=user_msg,
            assistant_msg=asst_msg,
            cos_to_hub=float(final_cos),
            meta={
                "steps_applied": applied,
                "n_iter_budget": args.n_iter,
                "seed_user": seed_u,
                "seed_assistant": seed_a,
            },
        ))

    cos_values = np.array([r.cos_to_hub for r in records])
    print(f"[stage_b/grad] total {time.time()-t0_total:.1f}s  "
          f"cos to hub:  mean={cos_values.mean():.3f}  "
          f"min={cos_values.min():.3f}  max={cos_values.max():.3f}")

    write_poison_file(
        args.out, method="grad_hotflip", hubs_source=args.hubs,
        sessions=records,
        extra={
            "n_iter": args.n_iter,
            "top_k_cand": args.top_k_cand,
            "seed_from": args.seed_from,
            "mean_cos_to_hub": float(cos_values.mean()),
        },
    )
    print(f"[stage_b/grad] wrote {args.out}")


if __name__ == "__main__":
    main()
