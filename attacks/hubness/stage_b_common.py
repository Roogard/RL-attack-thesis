"""Shared utilities for Stage B text-realization scripts.

All three methods (retrieval, BoN, gradient-based) emit the same session
JSON format so `attacks/eval_hubs.py --poison_file <json>` can inject any
of them identically.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def load_hubs(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


@dataclass
class PoisonSessionRecord:
    hub_idx: int
    user_msg: str
    assistant_msg: str
    cos_to_hub: float
    meta: dict = field(default_factory=dict)


def write_poison_file(
    path: str,
    method: str,
    hubs_source: str,
    sessions: list[PoisonSessionRecord],
    extra: dict | None = None,
) -> None:
    payload = {
        "method": method,
        "hubs_source": hubs_source,
        "K": len(sessions),
        "sessions": [asdict(s) for s in sessions],
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_poison_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_round_text(user_msg: str, assistant_msg: str) -> str:
    """Exact formatting RAGMemory.index() applies to a 2-turn round.

    Stage B scripts use this to compute cos-to-hub of the chunk that
    will actually be indexed, so the reported cos matches what the
    encoder will produce when the session goes through mem.index().
    """
    return f"User: {user_msg.strip()}\nAssistant: {assistant_msg.strip()}"


def encode_many(encoder, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """L2-normalized sentence-transformer encoding, returns np.ndarray."""
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return l2_normalize(np.asarray(vecs, dtype=np.float32))


# ── Corpus utilities (shared by attacks/corpora/* loaders) ─────────────────

_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Whitespace + unicode normalization for dedup. Not for display."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return _WS_RE.sub(" ", s)


def round_hash(user_msg: str, assistant_msg: str) -> str:
    """Stable 16-char hash over a normalized (user, assistant) pair."""
    key = normalize_text(user_msg) + "\n|\n" + normalize_text(assistant_msg)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def write_corpus_jsonl(
    path: str,
    rounds: Iterable[tuple[str, str, dict]],
    dedup: bool = True,
) -> dict:
    """Stream (user, assistant, meta) triples to JSONL, optionally deduped.

    Returns summary dict with counts. Each line:
      {"user": str, "assistant": str, "meta": dict, "hash": str}
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_written = 0
    n_dup = 0
    n_empty = 0
    with open(path, "w", encoding="utf-8") as f:
        for user, asst, meta in rounds:
            if not user or not asst:
                n_empty += 1
                continue
            user = user.strip()
            asst = asst.strip()
            if not user or not asst:
                n_empty += 1
                continue
            h = round_hash(user, asst)
            if dedup:
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
            f.write(json.dumps(
                {"user": user, "assistant": asst, "meta": meta, "hash": h},
                ensure_ascii=False,
            ) + "\n")
            n_written += 1
    return {"written": n_written, "duplicates": n_dup, "empty": n_empty, "path": path}


def iter_corpus_jsonl(path: str) -> Iterator[dict]:
    """Yield each row (dict) from a corpus JSONL written by write_corpus_jsonl."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ── LongMemEval train-split round extraction (used by retrieval + nl_memory) ──


def iter_longmemeval_train_rounds(
    data_path: str, train_qids: list[str]
) -> Iterator[tuple[str, str, dict]]:
    """Walk every train-split question's haystack, yield (u, a, meta) rounds.

    Was inlined inside stage_b_retrieval._collect_corpus_rounds; lifted here
    so the corpora package can reuse it without duplicating logic.

    Dedup is the caller's responsibility (write_corpus_jsonl handles it
    when the rounds get streamed to disk).
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    by_qid = {q["question_id"]: q for q in data}
    for qid in train_qids:
        q = by_qid.get(qid)
        if not q:
            continue
        sessions = q["haystack_sessions"]
        dates = q["haystack_dates"]
        sids = q.get(
            "haystack_session_ids",
            [str(i) for i in range(len(sessions))],
        )
        for session, date, sid in zip(sessions, dates, sids):
            i = 0
            round_idx = 0
            while i < len(session):
                if (
                    i + 1 < len(session)
                    and session[i]["role"] == "user"
                    and session[i + 1]["role"] == "assistant"
                ):
                    u = session[i]["content"].strip()
                    a = session[i + 1]["content"].strip()
                    yield u, a, {
                        "source": "longmemeval_train",
                        "source_qid": qid,
                        "source_session_id": sid,
                        "source_date": date,
                        "round_index": round_idx,
                    }
                    i += 2
                    round_idx += 1
                else:
                    i += 1
