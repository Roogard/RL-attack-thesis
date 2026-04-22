"""Per-epoch cache of clean-pipeline artifacts.

Clean retrieval, clean answer, clean-correctness, question embedding, and
clean-prediction embedding are identical for every rollout that uses the
same question. Build them once per epoch and reuse across G*B rollouts.

Usage:
    cache = CleanCache.build(questions)
    cache.save("results/caches/epoch_0.pkl")
    ...
    entry = cache[qid]
    entry.clean_correct, entry.q_embed, entry.clean_ctx  # ...
"""

from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from harness import ask_qwen_batch, judge_answer_local
from memory.rag import RAGMemory


@dataclass
class CacheEntry:
    qid: str
    question: str
    question_date: str
    question_type: str
    correct_answer: str
    clean_ctx: str
    clean_pred: str
    clean_correct: bool
    q_embed: np.ndarray           # (d,)
    clean_pred_embed: np.ndarray  # (d,) — used as R_KL surrogate baseline
    clean_top1_embed: np.ndarray  # (d,) — embedding of top-1 retrieved clean chunk


def _get_embedder(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _top1_chunk_text(ctx: str) -> str:
    """Return the first chunk from a RAG-formatted context block."""
    if not ctx:
        return ""
    # RAG joins chunks with "\n\n---\n\n"; the first one starts with
    # "[Session Date: ...]\n<round text>". We return it verbatim.
    return ctx.split("\n\n---\n\n", 1)[0]


class CleanCache:
    def __init__(self, entries: dict[str, CacheEntry]):
        self._entries: dict[str, CacheEntry] = entries

    # ── public API ────────────────────────────────────────────────────────

    def __getitem__(self, qid: str) -> CacheEntry:
        return self._entries[qid]

    def __contains__(self, qid: str) -> bool:
        return qid in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def qids(self) -> list[str]:
        return list(self._entries.keys())

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def filter_clean_correct(self) -> "CleanCache":
        """Return a new cache containing only questions the clean model got right.

        Training reward is only well-defined on these — poisoning a question
        the victim already fails is meaningless.
        """
        return CleanCache({k: v for k, v in self._entries.items() if v.clean_correct})

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._entries, f)

    @classmethod
    def load(cls, path: str) -> "CleanCache":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        questions: Iterable[dict],
        embed_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        answer_batch_size: int = 8,
    ) -> "CleanCache":
        """Build cache from scratch by running the clean pipeline on each question.

        Args:
            questions: iterable of LongMemEval question dicts.
            embed_model_name: MiniLM model matching RAGMemory's embedder.
            top_k: retrieval top-k (must match RAG default for consistency).
            answer_batch_size: batch size for clean ask_qwen_batch calls.
        """
        questions = list(questions)
        for q in questions:
            q.setdefault(
                "haystack_session_ids",
                [f"s{i}" for i in range(len(q["haystack_sessions"]))],
            )

        embedder = _get_embedder(embed_model_name)

        # 1) Clean retrieval contexts (per-question; RAG instances are cheap)
        clean_ctxs: list[str] = []
        for q in tqdm(questions, desc="caches: clean retrieve"):
            mem = RAGMemory(model_name=embed_model_name)
            mem.index(q["haystack_sessions"], q["haystack_dates"], q["haystack_session_ids"])
            ctx = mem.retrieve(q["question"], q["question_date"], top_k=top_k)
            clean_ctxs.append(ctx)
            mem.clear()

        # 2) Clean answers — batched for throughput
        clean_preds: list[str] = []
        for i in tqdm(range(0, len(questions), answer_batch_size),
                      desc="caches: clean answer"):
            batch = questions[i:i + answer_batch_size]
            preds = ask_qwen_batch(
                clean_ctxs[i:i + answer_batch_size],
                [q["question"] for q in batch],
                [q["question_date"] for q in batch],
            )
            clean_preds.extend(preds)

        # 3) Clean correctness via local judge
        clean_corrects: list[bool] = []
        for q, pred in tqdm(list(zip(questions, clean_preds)),
                            desc="caches: judge clean"):
            clean_corrects.append(
                judge_answer_local(
                    q["question"], q["answer"], pred, q["question_type"], q["question_id"]
                )
            )

        # 4) Embeddings: question, clean_pred, top-1 clean chunk
        q_texts = [q["question"] for q in questions]
        pred_texts = clean_preds
        top1_texts = [_top1_chunk_text(c) for c in clean_ctxs]

        print("[caches] embedding texts ...")
        q_embeds = embedder.encode(q_texts, normalize_embeddings=True, show_progress_bar=False)
        pred_embeds = embedder.encode(pred_texts, normalize_embeddings=True, show_progress_bar=False)
        top1_embeds = embedder.encode(top1_texts, normalize_embeddings=True, show_progress_bar=False)

        entries: dict[str, CacheEntry] = {}
        for i, q in enumerate(questions):
            entries[q["question_id"]] = CacheEntry(
                qid=q["question_id"],
                question=q["question"],
                question_date=q["question_date"],
                question_type=q["question_type"],
                correct_answer=q["answer"],
                clean_ctx=clean_ctxs[i],
                clean_pred=clean_preds[i],
                clean_correct=bool(clean_corrects[i]),
                q_embed=np.asarray(q_embeds[i], dtype=np.float32),
                clean_pred_embed=np.asarray(pred_embeds[i], dtype=np.float32),
                clean_top1_embed=np.asarray(top1_embeds[i], dtype=np.float32),
            )
        return cls(entries)
