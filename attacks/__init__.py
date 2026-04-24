"""Common interface for memory-poisoning attacks.

PoisonSession is the unit of injection — text turns (realistic, the default)
or raw vectors (privileged-attacker upper bound, Stage A vector-mode only).
Attack is the protocol any concrete attack implements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class PoisonSession:
    """One attacker-authored session.

    Text mode (default — all realism-respecting attacks):
        turns = [{role: 'user'|'assistant', content: str}, ...]
        Passes through the defender's encoder + write guard.

    Vector mode (privileged upper bound only):
        vectors = (n_chunks, d) L2-normalized array, paired with `documents`
        (text returned by retrieve() when each vector wins). Bypasses
        encoder and guard via MemoryStore.index_raw_embeddings().
    """
    date: str
    session_id: str
    turns: Optional[list[dict]] = None
    vectors: Optional[np.ndarray] = None
    documents: Optional[list[str]] = None

    def __post_init__(self):
        if (self.turns is None) == (self.vectors is None):
            raise ValueError(
                "PoisonSession must have exactly one of (turns, vectors)"
            )
        if self.vectors is not None and self.documents is None:
            self.documents = ["[injected]" for _ in range(len(self.vectors))]


@dataclass
class WritePolicy:
    """How the defender ingests text into memory.

    Attacks use this to (a) reconstruct D_est from observed conversations
    and (b) score guard-pass during text realization. `encoder_model_name`
    must match the defender's encoder; `guard` is None when no write filter.
    """
    encoder_model_name: str = "all-MiniLM-L6-v2"
    guard: Optional[Callable[[str], bool]] = None  # True = accept


@runtime_checkable
class Attack(Protocol):
    def generate_poisons(
        self,
        observed_history: Optional[list[list[dict]]],
        K: int,
        write_policy: WritePolicy,
    ) -> list[PoisonSession]:
        """Produce poisoned sessions for injection.

        observed_history: list of sessions (each a list of {role, content}
            turns) the attacker has observed. None = cold-start (use default
            hubs / generic priors).
        K: total number of poisoned chunks across the returned sessions.
        write_policy: the defender's write policy.
        """
        ...
