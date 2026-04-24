"""Shared utilities for Stage B text-realization scripts.

All three methods (retrieval, BoN, gradient-based) emit the same session
JSON format so `attacks/eval_hubs.py --poison_file <json>` can inject any
of them identically.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
