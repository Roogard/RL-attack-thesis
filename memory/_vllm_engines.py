"""Lazy vLLM engine factories shared across memory modules and the harness.

Two engines per process:
- mem-agent (driaforall/mem-agent, Qwen3-4B): drives RLMemory's agentic loop
- answer model (Qwen2.5-3B with YaRN to 128K): generates final answers

Both can co-reside on one GPU at gpu_memory_utilization=0.4 each
(~32GB on a 40GB A100; tighten if your GPUs are smaller).

The answer engine uses hf_overrides to inject the YaRN rope_scaling so
full_history (~115K-token contexts) fits in 128K of position embeddings.
Prefix caching is on for the mem-agent because the agentic loop appends
to a growing prompt every turn — vLLM reuses the prefix KV cache.
"""

from __future__ import annotations

import threading
from typing import Any

_MEM_MODEL_ID = "driaforall/mem-agent"
_ANSWER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

_GPU_MEM_UTIL = 0.45

_mem_engine = None
_mem_tokenizer = None
_mem_lock = threading.Lock()

_answer_engine = None
_answer_tokenizer = None
_answer_lock = threading.Lock()


def _build_engine(
    model_id: str,
    max_model_len: int,
    enable_prefix_caching: bool,
    hf_overrides: dict | None = None,
):
    from vllm import LLM
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": "bfloat16",
        "gpu_memory_utilization": _GPU_MEM_UTIL,
        "enable_prefix_caching": enable_prefix_caching,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    if hf_overrides is not None:
        kwargs["hf_overrides"] = hf_overrides
    return LLM(**kwargs)


def get_mem_engine():
    """Return (engine, tokenizer) for the mem-agent (4B). Lazy + cached."""
    global _mem_engine, _mem_tokenizer
    if _mem_engine is None:
        with _mem_lock:
            if _mem_engine is None:
                from transformers import AutoTokenizer
                print(f"[vllm] Loading {_MEM_MODEL_ID} ...")
                _mem_engine = _build_engine(
                    _MEM_MODEL_ID,
                    max_model_len=16384,
                    enable_prefix_caching=True,
                )
                _mem_tokenizer = AutoTokenizer.from_pretrained(_MEM_MODEL_ID)
                print("[vllm] mem-agent ready.")
    return _mem_engine, _mem_tokenizer


def get_answer_engine():
    """Return (engine, tokenizer) for the answer model (3B + YaRN). Lazy + cached."""
    global _answer_engine, _answer_tokenizer
    if _answer_engine is None:
        with _answer_lock:
            if _answer_engine is None:
                from transformers import AutoTokenizer
                print(f"[vllm] Loading {_ANSWER_MODEL_ID} (YaRN 128K) ...")
                _answer_engine = _build_engine(
                    _ANSWER_MODEL_ID,
                    max_model_len=131072,
                    enable_prefix_caching=False,
                    hf_overrides={
                        "rope_scaling": {
                            "rope_type": "yarn",
                            "factor": 4.0,
                            "original_max_position_embeddings": 32768,
                        },
                        "max_position_embeddings": 131072,
                    },
                )
                _answer_tokenizer = AutoTokenizer.from_pretrained(_ANSWER_MODEL_ID)
                print("[vllm] answer model ready.")
    return _answer_engine, _answer_tokenizer


def generate_one(engine, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    """Run one greedy generation through a vLLM engine. Returns decoded text."""
    return generate_many(engine, tokenizer, [messages], max_new_tokens)[0]


def generate_many(
    engine,
    tokenizer,
    messages_batch: list[list[dict]],
    max_new_tokens: int,
) -> list[str]:
    """Run B greedy generations in one vLLM call. Returns texts in input order.

    This is the throughput lever — calling generate() with B prompts at once
    lets vLLM's continuous batching keep the GPU saturated during decode.
    """
    from vllm import SamplingParams
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = engine.generate(prompts, params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]
