"""
RLMemory: MemoryStore implementation wrapping driaforall/mem-agent.

The mem-agent model (Qwen3-4B, GSPO-trained) manages an Obsidian-style
markdown file system. At index() time we drive the model's
think→python→execute→result agentic loop once per session to build/update
markdown files in a per-instance temp directory. At retrieve() time we
compile all markdown files into a context string without any LLM call.
"""

from __future__ import annotations

import builtins
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from memory.base import MemoryStore

# ---------------------------------------------------------------------------
# Module-level lazy singleton for the 4B model
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_MODEL_LOCK = threading.Lock()
_MODEL_ID = "driaforall/mem-agent"

SYSTEM_PROMPT = """You are an LLM agent with a self-managed, Obsidian-like memory system. \
You interact with memory using Python code blocks.

## CRITICAL: Response Format Rules

**EVERY response MUST follow this EXACT structure:**

1. **Always start with `<think>`** - Your reasoning about the query and what memory operations are needed
2. **Always follow with `<python>`** - Either:
   - Python code to interact with memory, OR
   - Empty tags `<python></python>` if no memory interaction needed
3. **Only provide `<reply>` if `<python>` is empty** - Your response to the user
4. **The `<python></python>` and `<reply></reply>` MUST be separate**

## Memory API

**CRITICAL: ALWAYS assign function results to variables or they will be LOST!**

```python
create_file(file_path: str, content: str = "") -> bool
update_file(file_path: str, old_content: str, new_content: str) -> Union[bool, str]
read_file(file_path: str) -> str
delete_file(file_path: str) -> bool
check_if_file_exists(file_path: str) -> bool
create_dir(dir_path: str) -> bool
list_files() -> str
check_if_dir_exists(dir_path: str) -> bool
```

## Memory Structure

- `user.md`: Personal information & attributes about the user
- `entities/`: One file per person/place/organization
- Facts: `- fact_name: fact_value`

## Important Operating Rules

1. ALWAYS check if `user.md` exists and read it before other operations
2. Save only persistent, reusable information (not temp facts, calculations)
3. No duplicates — check existing content before adding
4. ALWAYS capture return values: `result = create_file(...)` not `create_file(...)`
5. Wait for `<result>` blocks before proceeding
"""

_PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.DOTALL)


def _get_model():
    """Lazy-load the model and tokenizer exactly once (thread-safe)."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                print(f"[RLMemory] Loading {_MODEL_ID} ...")
                _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_ID)
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    _MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                _MODEL.eval()
                print("[RLMemory] Model loaded.")
    return _MODEL, _TOKENIZER


# ---------------------------------------------------------------------------
# File-operation functions (closures bound to a specific directory)
# ---------------------------------------------------------------------------

def _make_file_ops(base_dir: str) -> dict[str, Any]:
    """
    Return file-operation functions rooted at base_dir.
    These become the exec namespace — the model cannot touch anything outside base_dir.
    """
    base = Path(base_dir).resolve()

    def _safe(p: str) -> Path:
        resolved = (base / p).resolve()
        if not str(resolved).startswith(str(base)):
            raise PermissionError(f"Path outside memory dir: {p}")
        return resolved

    def create_file(file_path: str, content: str = "") -> bool:
        try:
            dst = _safe(file_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            return str(e)

    def update_file(file_path: str, old_content: str, new_content: str):
        try:
            dst = _safe(file_path)
            text = dst.read_text(encoding="utf-8")
            if old_content not in text:
                return f"String not found: {old_content!r}"
            dst.write_text(text.replace(old_content, new_content, 1), encoding="utf-8")
            return True
        except Exception as e:
            return str(e)

    def read_file(file_path: str) -> str:
        try:
            return _safe(file_path).read_text(encoding="utf-8")
        except Exception as e:
            return str(e)

    def delete_file(file_path: str) -> bool:
        try:
            _safe(file_path).unlink()
            return True
        except Exception as e:
            return str(e)

    def check_if_file_exists(file_path: str) -> bool:
        try:
            return _safe(file_path).exists()
        except Exception:
            return False

    def create_dir(dir_path: str) -> bool:
        try:
            _safe(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            return str(e)

    def check_if_dir_exists(dir_path: str) -> bool:
        try:
            return _safe(dir_path).is_dir()
        except Exception:
            return False

    def list_files() -> str:
        lines = []
        for root, dirs, files in os.walk(str(base)):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            level = len(Path(root).relative_to(base).parts)
            indent = "  " * level
            lines.append(f"{indent}{Path(root).name}/")
            for f in sorted(files):
                lines.append(f"{indent}  {f}")
        return "\n".join(lines)

    return {
        "create_file": create_file,
        "update_file": update_file,
        "read_file": read_file,
        "delete_file": delete_file,
        "check_if_file_exists": check_if_file_exists,
        "create_dir": create_dir,
        "check_if_dir_exists": check_if_dir_exists,
        "list_files": list_files,
    }


# ---------------------------------------------------------------------------
# Low-level inference helpers
# ---------------------------------------------------------------------------

def _exec_code(code: str, file_ops: dict[str, Any]) -> dict:
    """Execute model-generated code in an isolated namespace.

    Captures stdout so the agent's own print() observations are fed back
    into the next <result> block instead of leaking to our terminal.
    """
    import contextlib
    import io
    exec_globals: dict = {"__builtins__": builtins.__dict__}
    exec_globals.update(file_ops)
    exec_locals: dict = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<mem-agent>", "exec"), exec_globals, exec_locals)
    except Exception as e:
        exec_locals["_exec_error"] = str(e)
    captured = buf.getvalue()
    if captured:
        exec_locals["_stdout"] = captured
    return exec_locals


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    """Run one forward pass and return the decoded response string."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _run_agentic_loop(
    model,
    tokenizer,
    user_message: str,
    file_ops: dict[str, Any],
    max_tool_turns: int = 6,
) -> None:
    """
    Drive the mem-agent agentic loop for a single user message.

    Loop: generate → parse <python> → exec → append <result> → repeat.
    Terminates when <python> is empty or max_tool_turns is reached.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_tool_turns):
        response = _generate(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        m = _PYTHON_RE.search(response)
        code = m.group(1).strip() if m else None

        if code:
            local_vars = _exec_code(code, file_ops)
            messages.append({"role": "user", "content": f"<result>\n{local_vars}\n</result>"})
        else:
            break


# ---------------------------------------------------------------------------
# Session formatting
# ---------------------------------------------------------------------------

def _format_session_for_indexing(session: list[dict], date: str) -> str:
    """
    Convert a session and its date into a single user message for mem-agent.
    Sent as one message (not replayed turn-by-turn) to avoid 50x inference cost.
    """
    turns = "\n".join(
        f"{t['role'].capitalize()}: {t['content'].strip()}" for t in session
    )
    return (
        f"[Session Date: {date}]\n"
        "The following is a conversation that took place. "
        "Please update the memory files to capture any important personal "
        "information, preferences, relationships, goals, or recurring topics. "
        "Ignore trivial exchanges (greetings, one-off calculations, etc.).\n\n"
        f"{turns}"
    )


# ---------------------------------------------------------------------------
# MemoryStore implementation
# ---------------------------------------------------------------------------

class RLMemory(MemoryStore):
    """
    MemoryStore backed by driaforall/mem-agent (Qwen3-4B, GSPO-trained).

    index(): one agentic-loop inference call per session builds/updates
             markdown files in a per-instance temporary directory.

    retrieve(): compiles all markdown files into a context string (no LLM call).

    clear(): wipes and recreates the temp memory directory.
    """

    def __init__(self, max_tool_turns: int = 6):
        self.max_tool_turns = max_tool_turns
        self._tmp_root: str = tempfile.mkdtemp(prefix="rl_memory_")
        self._memory_dir: str = os.path.join(self._tmp_root, "memory")
        os.makedirs(self._memory_dir, exist_ok=True)

    def index(self, sessions, dates, session_ids):
        model, tokenizer = _get_model()
        file_ops = _make_file_ops(self._memory_dir)

        for session, date, sid in zip(sessions, dates, session_ids):
            user_msg = _format_session_for_indexing(session, date)
            _run_agentic_loop(
                model, tokenizer, user_msg, file_ops,
                max_tool_turns=self.max_tool_turns,
            )

    def retrieve(self, question, question_date, top_k=10):
        root = Path(self._memory_dir)
        if not root.exists():
            return ""

        md_files = sorted(root.rglob("*.md"))
        if not md_files:
            return ""

        chunks = []
        for md_path in md_files[:top_k]:
            rel = md_path.relative_to(root)
            try:
                content = md_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if content:
                chunks.append(f"[Memory: {rel}]\n{content}")

        return "\n\n---\n\n".join(chunks)

    def clear(self):
        shutil.rmtree(self._memory_dir, ignore_errors=True)
        os.makedirs(self._memory_dir, exist_ok=True)

    def __del__(self):
        try:
            if os.path.exists(self._tmp_root):
                shutil.rmtree(self._tmp_root, ignore_errors=True)
        except Exception:
            pass
