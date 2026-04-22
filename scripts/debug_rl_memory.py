"""Stage 2 of the rl_memory failure investigation.

For each of N picked qids (default: the stage-2 candidates from
results/benchmark/rl_memory.failure_modes.json), runs the mem-agent
indexing pipeline SERIALLY (not via index_batch — keeps causality clear)
with per-session instrumentation. Dumps everything to disk so we can
diagnose what the agent actually wrote vs what it should have written.

Per question, writes results/benchmark/rl_memory_debug/<qid>/:
  - meta.json                — question, ground truth, n sessions, answer session ids
  - session_NNN.json         — for each session: turn-by-turn agent output,
                                exec results, prompt token count, post-session
                                memory_dir snapshot (file list + file contents)
  - final_memory_dir/        — copy of the final memory dir
  - retrieve_output.txt      — verbatim retrieve() output
  - flags.json               — summary booleans:
                                 * did agent ever produce a <python> block?
                                 * does any final memory file contain the GT answer string?
                                 * did retrieve() output contain the GT answer string?
                                 * did any prompt exceed 16384 tokens?
                                 * did the answer-bearing session(s) ever hit max_tool_turns?

Run on Microway (needs the mem-agent vLLM engine):
  python scripts/debug_rl_memory.py
  python scripts/debug_rl_memory.py --qids e47becba 1e043500
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import _vllm_engines
from memory.rl_memory import (
    RLMemory,
    SYSTEM_PROMPT,
    _PYTHON_RE,
    _exec_code,
    _format_session_for_indexing,
    _make_file_ops,
)


DEFAULT_DEBUG_DIR = "results/benchmark/rl_memory_debug"
DATASET_PATH = "LongMemEval/data/longmemeval_s_cleaned.json"
SUMMARY_PATH = "results/benchmark/rl_memory.failure_modes.json"


def _snapshot_memory_dir(memory_dir: str) -> dict[str, str]:
    """Return {relpath: content} for every file under memory_dir."""
    root = Path(memory_dir)
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                out[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                out[str(path.relative_to(root))] = f"<read error: {e}>"
    return out


def _count_prompt_tokens(tokenizer, messages: list[dict]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _instrumented_agentic_loop(
    engine,
    tokenizer,
    user_message: str,
    file_ops: dict[str, Any],
    max_tool_turns: int = 6,
) -> dict:
    """Single-session loop with per-turn capture.

    Mirrors memory.rl_memory._run_agentic_loop but records everything.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    record = {
        "max_tool_turns": max_tool_turns,
        "turns": [],
        "user_message_chars": len(user_message),
        "stopped_reason": None,
    }

    for turn_idx in range(max_tool_turns):
        prompt_tokens = _count_prompt_tokens(tokenizer, messages)
        response = _vllm_engines.generate_one(
            engine, tokenizer, messages, max_new_tokens=512
        )
        messages.append({"role": "assistant", "content": response})

        m = _PYTHON_RE.search(response)
        code = m.group(1).strip() if m else None

        turn_record: dict[str, Any] = {
            "turn": turn_idx,
            "prompt_tokens": prompt_tokens,
            "response": response,
            "had_python_block": code is not None,
            "code": code,
        }

        if code:
            local_vars = _exec_code(code, file_ops)
            turn_record["exec_result"] = {k: repr(v)[:500] for k, v in local_vars.items()}
            messages.append(
                {"role": "user", "content": f"<result>\n{local_vars}\n</result>"}
            )
            record["turns"].append(turn_record)
            if turn_idx == max_tool_turns - 1:
                record["stopped_reason"] = "hit_max_turns"
        else:
            turn_record["exec_result"] = None
            record["turns"].append(turn_record)
            record["stopped_reason"] = "no_python_block"
            break

    if record["stopped_reason"] is None:
        record["stopped_reason"] = "hit_max_turns"
    return record


def _gt_answer_string(oracle_entry: dict) -> str:
    """Cast answer (sometimes int, sometimes str) to lowercase string for substring match."""
    return str(oracle_entry.get("answer", "")).lower()


def _contains_answer(text: str, gt_lower: str) -> bool:
    if not gt_lower or not text:
        return False
    return gt_lower in text.lower()


def diagnose_one(
    qid: str,
    question: dict,
    oracle_entry: dict,
    debug_dir: str,
    max_tool_turns: int = 6,
) -> dict:
    """Run instrumented indexing for one question and dump everything."""
    out_dir = Path(debug_dir) / qid
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    engine, tokenizer = _vllm_engines.get_mem_engine()
    inst = RLMemory(max_tool_turns=max_tool_turns)
    inst.clear()
    file_ops = _make_file_ops(inst._memory_dir)

    answer_session_ids = set(oracle_entry.get("answer_session_ids", []))
    sessions = question["haystack_sessions"]
    dates = question["haystack_dates"]
    sids = question.get("haystack_session_ids", [str(i) for i in range(len(sessions))])

    flags = {
        "qid": qid,
        "question_type": oracle_entry.get("question_type"),
        "n_sessions_total": len(sessions),
        "answer_session_ids": list(answer_session_ids),
        "answer_session_indices": [i for i, s in enumerate(sids) if s in answer_session_ids],
        "ever_produced_python_block": False,
        "max_prompt_tokens_seen": 0,
        "n_sessions_overflowed_16k": 0,
        "answer_session_hit_max_turns": False,
        "answer_in_final_memory": False,
        "answer_in_retrieve_output": False,
        "n_final_md_files": 0,
        "n_files_returned_by_retrieve": 0,
    }

    meta = {
        "qid": qid,
        "question": question.get("question"),
        "question_date": question.get("question_date"),
        "question_type": oracle_entry.get("question_type"),
        "ground_truth_answer": oracle_entry.get("answer"),
        "answer_session_ids": list(answer_session_ids),
        "n_sessions": len(sessions),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for s_idx, (session, date, sid) in enumerate(zip(sessions, dates, sids)):
        is_answer_session = sid in answer_session_ids
        user_msg = _format_session_for_indexing(session, date)

        record = _instrumented_agentic_loop(
            engine, tokenizer, user_msg, file_ops, max_tool_turns=max_tool_turns
        )
        record.update({
            "session_index": s_idx,
            "session_id": sid,
            "session_date": date,
            "is_answer_session": is_answer_session,
            "n_turns_in_session": len(session),
            "post_session_memory": _snapshot_memory_dir(inst._memory_dir),
        })

        for t in record["turns"]:
            if t["had_python_block"]:
                flags["ever_produced_python_block"] = True
            flags["max_prompt_tokens_seen"] = max(
                flags["max_prompt_tokens_seen"], t["prompt_tokens"]
            )
            if t["prompt_tokens"] > 16384:
                flags["n_sessions_overflowed_16k"] += 1
                break

        if is_answer_session and record["stopped_reason"] == "hit_max_turns":
            flags["answer_session_hit_max_turns"] = True

        (out_dir / f"session_{s_idx:03d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    final_snapshot = _snapshot_memory_dir(inst._memory_dir)
    final_dir_copy = out_dir / "final_memory_dir"
    if Path(inst._memory_dir).exists():
        shutil.copytree(inst._memory_dir, final_dir_copy)

    flags["n_final_md_files"] = sum(1 for k in final_snapshot if k.endswith(".md"))
    gt_lower = _gt_answer_string(oracle_entry)
    flags["answer_in_final_memory"] = any(
        _contains_answer(content, gt_lower) for content in final_snapshot.values()
    )

    retrieve_output = inst.retrieve(question["question"], question["question_date"])
    (out_dir / "retrieve_output.txt").write_text(retrieve_output, encoding="utf-8")
    flags["answer_in_retrieve_output"] = _contains_answer(retrieve_output, gt_lower)
    flags["n_files_returned_by_retrieve"] = retrieve_output.count("[Memory:")

    (out_dir / "flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    inst.clear()
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qids", nargs="*", default=None,
        help="Question IDs to diagnose. Default: stage-2 candidates from failure_modes.json",
    )
    parser.add_argument("--debug-dir", default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--max-tool-turns", type=int, default=6)
    args = parser.parse_args()

    if args.qids:
        qids = args.qids
    else:
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)
        qids = [c["question_id"] for c in summary["stage2_candidates"]]
    print(f"[debug] qids to diagnose: {qids}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    by_qid = {q["question_id"]: q for q in data}

    with open("LongMemEval/data/longmemeval_oracle.json", encoding="utf-8") as f:
        oracle = {q["question_id"]: q for q in json.load(f)}

    os.makedirs(args.debug_dir, exist_ok=True)

    all_flags = []
    for qid in qids:
        if qid not in by_qid:
            print(f"[debug] WARNING: {qid} not in dataset, skipping")
            continue
        if qid not in oracle:
            print(f"[debug] WARNING: {qid} not in oracle, skipping")
            continue
        print(f"\n[debug] === diagnosing {qid} ===")
        flags = diagnose_one(
            qid, by_qid[qid], oracle[qid], args.debug_dir,
            max_tool_turns=args.max_tool_turns,
        )
        all_flags.append(flags)
        print(f"[debug] flags: {json.dumps({k: v for k, v in flags.items() if k != 'answer_session_ids'}, indent=2)}")

    summary_out = os.path.join(args.debug_dir, "_summary.json")
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(all_flags, f, indent=2)
    print(f"\n[debug] summary written to {summary_out}")


if __name__ == "__main__":
    main()
