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

import os
import threading
from typing import Any

_MEM_MODEL_ID = "driaforall/mem-agent"
_ANSWER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
_ATTACKER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Override with VLLM_GPU_MEM_UTIL=0.25 etc. when sharing a GPU with other
# jobs or when the training-time HF models (AttackerPolicy + 7B judge +
# activations) need more headroom on the same device.
_GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.40"))
# Separate knob for the attacker engine — it holds short prompts so it can
# run with a tighter KV cache budget than the answer engine.
_ATTACKER_GPU_MEM_UTIL = float(os.environ.get("VLLM_ATTACKER_GPU_MEM_UTIL", "0.15"))

_mem_engine = None
_mem_tokenizer = None
_mem_lock = threading.Lock()

_answer_engine = None
_answer_tokenizer = None
_answer_lock = threading.Lock()

_attacker_engine = None
_attacker_tokenizer = None
_attacker_lock = threading.Lock()


def _build_engine(
    model_id: str,
    max_model_len: int,
    enable_prefix_caching: bool,
    hf_overrides: dict | None = None,
    gpu_memory_utilization: float | None = None,
    enable_lora: bool = False,
    max_lora_rank: int = 16,
    max_loras: int = 1,
):
    from vllm import LLM
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": "bfloat16",
        "gpu_memory_utilization": (
            gpu_memory_utilization if gpu_memory_utilization is not None else _GPU_MEM_UTIL
        ),
        "enable_prefix_caching": enable_prefix_caching,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    if hf_overrides is not None:
        kwargs["hf_overrides"] = hf_overrides
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = max_lora_rank
        kwargs["max_loras"] = max_loras
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


def get_attacker_engine(max_lora_rank: int = 16):
    """Return (engine, tokenizer) for the attacker model (3B + LoRA). Lazy + cached.

    Separate engine from the answer model because we want LoRA support and
    independent sampling. The training loop saves the current LoRA adapter
    to disk after every optim.step() and passes a fresh LoRARequest (new
    int_id) so vLLM reloads the updated weights on the next rollout.
    """
    global _attacker_engine, _attacker_tokenizer
    if _attacker_engine is None:
        with _attacker_lock:
            if _attacker_engine is None:
                from transformers import AutoTokenizer
                print(
                    f"[vllm] Loading {_ATTACKER_MODEL_ID} "
                    f"(attacker, LoRA-enabled, util={_ATTACKER_GPU_MEM_UTIL}) ..."
                )
                # 16384 fits the 8-probe × 3-chunk memory readout (~5k tokens)
                # plus prior-session summaries and the generated response with
                # safe margin. Qwen2.5-3B supports 32k natively; we stop at
                # 16k so the KV cache budget stays generous enough for group
                # batching at util=0.15.
                _attacker_engine = _build_engine(
                    _ATTACKER_MODEL_ID,
                    max_model_len=16384,
                    enable_prefix_caching=True,
                    gpu_memory_utilization=_ATTACKER_GPU_MEM_UTIL,
                    enable_lora=True,
                    max_lora_rank=max_lora_rank,
                    max_loras=1,
                )
                _attacker_tokenizer = AutoTokenizer.from_pretrained(_ATTACKER_MODEL_ID)
                print("[vllm] attacker engine ready.")
    return _attacker_engine, _attacker_tokenizer


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
                    enable_prefix_caching=True,
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
