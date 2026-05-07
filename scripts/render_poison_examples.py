"""Render representative poison chunks from Stage B poison-file JSONs to a
single markdown document.

For each poison file, flattens records → chunks (treating each round inside
a multi-round retrieval record as one chunk), sorts by cos-to-hub
(post-PI cos if available, else pre-PI), and emits the highest, median,
and lowest cos chunk per config. Each chunk is rendered as the EXACT text
RAGMemory.index() will encode and index — so the doc shows what the
defender's encoder sees, not a summary.

Usage:
    python3 scripts/render_poison_examples.py \\
        --poison_files results/stage_b/retrieval_K*.json \\
        --out          results/stage_b/POISON_EXAMPLES.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.hubness.stage_b_common import format_round_text


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten_chunks(poison: dict) -> list[dict]:
    """Flatten poison records into a list of chunks. Each chunk = one round.

    Returns dicts with: hub_idx, round_idx_within_hub, user_msg, assistant_msg,
    cos_pre_pi, cos_post_pi (None if PI not applied), cos_effective.
    """
    chunks: list[dict] = []
    for s in poison["sessions"]:
        if "rounds" in s:
            for ridx, r in enumerate(s["rounds"]):
                cos_pre = r.get("cos_to_hub")
                cos_post = r.get("cos_to_hub_after_pi")
                chunks.append({
                    "hub_idx": s["hub_idx"],
                    "round_idx_within_hub": ridx,
                    "user_msg": r["user_msg"],
                    "assistant_msg": r["assistant_msg"],
                    "cos_pre_pi": cos_pre,
                    "cos_post_pi": cos_post,
                    "cos_effective": cos_post if cos_post is not None else cos_pre,
                })
        else:
            cos_pre = s.get("cos_to_hub")
            chunks.append({
                "hub_idx": s["hub_idx"],
                "round_idx_within_hub": 0,
                "user_msg": s["user_msg"],
                "assistant_msg": s["assistant_msg"],
                "cos_pre_pi": cos_pre,
                "cos_post_pi": None,
                "cos_effective": cos_pre,
            })
    return chunks


def _pick_three(chunks: list[dict]) -> list[tuple[str, dict]]:
    """Pick (highest, median, lowest) by cos_effective. Returns [(label, chunk)]."""
    if not chunks:
        return []
    sorted_chunks = sorted(chunks, key=lambda c: c["cos_effective"] or 0.0, reverse=True)
    n = len(sorted_chunks)
    if n == 1:
        return [("only chunk", sorted_chunks[0])]
    if n == 2:
        return [("highest", sorted_chunks[0]), ("lowest", sorted_chunks[-1])]
    return [
        ("highest cos", sorted_chunks[0]),
        ("median cos",  sorted_chunks[n // 2]),
        ("lowest cos",  sorted_chunks[-1]),
    ]


def _render(poisons: list[tuple[str, str, dict]]) -> str:
    """poisons: list of (path, stem, poison_data). Returns markdown string."""
    out: list[str] = []
    out.append("# Poison Chunk Examples — Stage B Retrieval Hub Strategy")
    out.append("")
    out.append("For each `(K, M, PI)` config in the consolidated final sweep, "
               "this document shows three example chunks: highest, median, and "
               "lowest cos-to-hub. Each chunk is rendered exactly as "
               "`RAGMemory.index()` will format and encode it — `User: ...` "
               "followed by `Assistant: ...`, with PI-template dialogue appended "
               "after the natural assistant content for `*_PI` configs.")
    out.append("")
    out.append("Cos values reflect the chunk's post-encoding similarity to its "
               "assigned hub vector (cos = 1 ⇒ identical direction in embedding "
               "space). For PI configs, the displayed cos is post-PI (after "
               "appending the override template) — the relevant number for "
               "whether the chunk wins top-k retrieval at eval time.")
    out.append("")

    # ── Summary table ───────────────────────────────────────────────
    out.append("## Summary")
    out.append("")
    out.append("| Config | K | M | PI | total chunks | distinct rounds | mean cos (effective) |")
    out.append("|---|---:|---:|:--:|---:|---:|---:|")
    for path, stem, d in poisons:
        K = d.get("K", "?")
        M = d.get("M", 1)
        pi = d.get("pi_mode", "none")
        total = d.get("total_chunks_injected", K * M)
        distinct = d.get("distinct_rounds_selected", "—")
        mean_cos_effective = d.get("post_pi_mean_cos_to_hub", d.get("mean_cos_to_hub"))
        mean_cos_str = f"{mean_cos_effective:.3f}" if isinstance(mean_cos_effective, (int, float)) else "—"
        out.append(f"| `{stem}` | {K} | {M} | {pi} | {total} | {distinct} | {mean_cos_str} |")
    out.append("")

    # ── Per-config example chunks ──────────────────────────────────
    for path, stem, d in poisons:
        K = d.get("K", "?")
        M = d.get("M", 1)
        pi = d.get("pi_mode", "none")
        out.append(f"## `{stem}` — K={K}, M={M}, PI={pi}")
        out.append("")
        out.append(f"Source: `{path}`")
        out.append("")
        chunks = _flatten_chunks(d)
        if not chunks:
            out.append("_No chunks in this poison file._")
            out.append("")
            continue
        for label, c in _pick_three(chunks):
            cos_effective = c["cos_effective"]
            cos_pre = c["cos_pre_pi"]
            cos_post = c["cos_post_pi"]
            out.append(f"### {label} — hub_idx={c['hub_idx']}, "
                       f"round_within_hub={c['round_idx_within_hub']}")
            if cos_post is not None:
                out.append(f"cos to hub (pre-PI): {cos_pre:.3f}  →  "
                           f"cos to hub (post-PI): {cos_post:.3f}")
            else:
                out.append(f"cos to hub: {cos_effective:.3f}")
            out.append("")
            indexed_text = format_round_text(c["user_msg"], c["assistant_msg"])
            out.append("```")
            out.append(indexed_text)
            out.append("```")
            out.append("")
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poison_files", nargs="+", required=True,
                   help="One or more Stage B poison-file JSON paths.")
    p.add_argument("--out", required=True, help="Output markdown path.")
    args = p.parse_args()

    poisons: list[tuple[str, str, dict]] = []
    for pf in args.poison_files:
        d = _load(pf)
        stem = Path(pf).stem
        poisons.append((pf, stem, d))
    print(f"[render_poison_examples] loaded {len(poisons)} poison files")

    md = _render(poisons)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[render_poison_examples] wrote {out_path}  ({len(md)} chars, "
          f"{md.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
