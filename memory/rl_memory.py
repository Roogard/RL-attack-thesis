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
from pathlib import Path
from typing import Any

from memory import _vllm_engines
from memory.base import MemoryStore

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


def _run_agentic_loop(
    engine,
    tokenizer,
    user_message: str,
    file_ops: dict[str, Any],
    max_tool_turns: int = 6,
) -> None:
    """
    Drive the mem-agent agentic loop for a single user message.

    Loop: generate → parse <python> → exec → append <result> → repeat.
    Terminates when <python> is empty or max_tool_turns is reached.
    vLLM's automatic prefix caching makes the growing prompt cheap.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_tool_turns):
        response = _vllm_engines.generate_one(
            engine, tokenizer, messages, max_new_tokens=512
        )
        messages.append({"role": "assistant", "content": response})

        m = _PYTHON_RE.search(response)
        code = m.group(1).strip() if m else None

        if code:
            local_vars = _exec_code(code, file_ops)
            messages.append({"role": "user", "content": f"<result>\n{local_vars}\n</result>"})
        else:
            break


def _run_agentic_loops_batched(
    engine,
    tokenizer,
    user_messages: list[str],
    file_ops_list: list[dict[str, Any]],
    max_tool_turns: int = 6,
) -> None:
    """
    Drive B independent agentic loops in lockstep for one session per question.

    Each call advances B questions by exactly one session-worth of agentic-loop
    work. At every step we batch all still-active conversations into a single
    vLLM.generate() call so the GPU sees parallel work instead of one
    sequence at a time.
    """
    n = len(user_messages)
    assert len(file_ops_list) == n

    messages_per_q: list[list[dict] | None] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_messages[i]},
        ]
        for i in range(n)
    ]
    turns: list[int] = [0] * n

    while True:
        active = [i for i in range(n) if messages_per_q[i] is not None]
        if not active:
            break

        responses = _vllm_engines.generate_many(
            engine,
            tokenizer,
            [messages_per_q[i] for i in active],
            max_new_tokens=512,
        )

        for j, i in enumerate(active):
            response = responses[j]
            messages_per_q[i].append({"role": "assistant", "content": response})
            turns[i] += 1

            m = _PYTHON_RE.search(response)
            code = m.group(1).strip() if m else None

            if code and turns[i] < max_tool_turns:
                local_vars = _exec_code(code, file_ops_list[i])
                messages_per_q[i].append(
                    {"role": "user", "content": f"<result>\n{local_vars}\n</result>"}
                )
            else:
                # No more tool calls (or hit turn cap) — this question's loop is done
                messages_per_q[i] = None


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
        engine, tokenizer = _vllm_engines.get_mem_engine()
        file_ops = _make_file_ops(self._memory_dir)

        for session, date, sid in zip(sessions, dates, session_ids):
            user_msg = _format_session_for_indexing(session, date)
            _run_agentic_loop(
                engine, tokenizer, user_msg, file_ops,
                max_tool_turns=self.max_tool_turns,
            )

    @classmethod
    def index_batch(cls, instances, questions):
        """Index B questions concurrently — the throughput-critical path.

        Walks every question's session list in lockstep. At each session
        position S, we collect the S'th session of every still-active
        question into one batched _run_agentic_loops_batched() call, which
        in turn batches every per-turn generate() across questions.

        This keeps vLLM's continuous batching saturated. With B=16 this
        roughly 4-5x's effective throughput vs B=1 on H100-class GPUs.
        """
        engine, tokenizer = _vllm_engines.get_mem_engine()
        n = len(instances)
        assert len(questions) == n

        file_ops_per_q = [_make_file_ops(inst._memory_dir) for inst in instances]
        max_sessions = max(len(q["haystack_sessions"]) for q in questions)
        max_tool_turns = instances[0].max_tool_turns

        for s_idx in range(max_sessions):
            # Collect the questions that still have a session at index s_idx
            active_indices = [
                i for i in range(n) if s_idx < len(questions[i]["haystack_sessions"])
            ]
            if not active_indices:
                break

            user_msgs = [
                _format_session_for_indexing(
                    questions[i]["haystack_sessions"][s_idx],
                    questions[i]["haystack_dates"][s_idx],
                )
                for i in active_indices
            ]
            file_ops = [file_ops_per_q[i] for i in active_indices]

            _run_agentic_loops_batched(
                engine, tokenizer, user_msgs, file_ops,
                max_tool_turns=max_tool_turns,
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
