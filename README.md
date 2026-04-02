# Thesis: Adversarial Memory Evaluation for LLM Agents

Evaluates and attacks memory systems in LLM-based conversational agents using the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark.

## First-time setup

```powershell
cd LongMemEval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-claude.txt
```

Download the benchmark data:
```powershell
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" -o data/longmemeval_s_cleaned.json
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" -o data/longmemeval_oracle.json
```

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Usage

Activate the venv, then run from the project root:
```powershell
.\LongMemEval\.venv\Scripts\Activate.ps1
```

Run all memory types:
```powershell
python main.py
```

Run a specific memory type:
```powershell
python main.py --memory rag
```

Quick test (limit questions, skip eval):
```powershell
python main.py --memory rag --limit 5 --skip-eval
```

Results are saved to `results/<timestamp>/` with per-type prediction files and a `comparison.json` summary.

## Memory types

| Name | Description |
|------|-------------|
| `full_history` | In-context baseline — full conversation history in prompt |
| `rag` | ChromaDB vector store with round-level chunking and `all-MiniLM-L6-v2` embeddings |

## Project structure

```
main.py            Orchestrator (argparse, runs memory types, triggers eval)
harness.py         LLM calls (Claude for answers, GPT-4o for judging)
eval.py            Scoring and comparison across memory types
memory/
  base.py          MemoryStore ABC (index / retrieve / clear)
  full_history.py  In-context baseline
  rag.py           ChromaDB RAG memory
direction.md       Research direction and threat model
```