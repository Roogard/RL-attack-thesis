# Thesis Project: Adversarial Memory Evaluation for LLM Agents

## Project Overview
This project evaluates and attacks memory systems in LLM-based conversational agents using the **LongMemEval** benchmark (ICLR 2025, arXiv:2410.10813).

## Research Goals
1. **Phase 1 — Baseline Agent**: Build a production-grade agent that runs on LongMemEval and achieves competitive performance.
2. **Phase 2 — Memory Variants**: Implement and compare multiple memory architectures on the benchmark.
3. **Phase 3 — Adversarial Attacks**: Design and evaluate adversarial policy attacks that exploit memory vulnerabilities.

## LongMemEval Benchmark
- **GitHub**: https://github.com/xiaowu0162/LongMemEval
- **Dataset**: HuggingFace `xiaowu0162/longmemeval-cleaned`
- **Tasks**: Information Extraction, Multi-Session Reasoning, Knowledge Updates, Temporal Reasoning, Abstention
- **Scale**: LongMemEval_S (~115K tokens), LongMemEval_M (~1.5M tokens)
- **Metric**: LLM-judged accuracy (GPT-4o evaluator, >97% agreement with humans)

## Memory Architectures to Compare
- **In-context**: Full history in context window (baseline)
- **RAG / Vector DB**: Dense embeddings + semantic retrieval
- **Key-Value Store**: Fact-extracted keys with augmented values
- **Temporal-aware RAG**: Time-filtered retrieval + query expansion
- **RL-managed memory** (primary novel contribution): Agent policy trained via RL to decide what to store/retrieve/forget — the core thesis contribution
- **Graph-based** (stretch): Knowledge graph for multi-hop reasoning

## Adversarial Attack Directions
- Distractor memory injection (irrelevant noise degrading retrieval)
- Temporal confusion attacks (contradictory or ambiguous timestamps)
- Multi-hop poisoning (corrupting intermediate facts in reasoning chains)
- Abstention bypass (crafting inputs that force hallucination instead of "I don't know")

## Three-Stage Memory Framework (from paper)
1. **Indexing**: Chunk history → extract key-value pairs (fact augmentation)
2. **Retrieval**: BM25 / dense retrieval + time-aware query expansion
3. **Reading**: Chain-of-Note structured prompting → final answer

## Key Results from Paper (Baselines to Beat)
- All SOTA models show ~30% accuracy drop vs oracle
- Round-level decomposition > session-level for retrieval
- Fact-augmented key expansion: +4% Recall@k
- Time-aware query expansion: +7–11% on temporal tasks
- Structured prompting (CoT/CoN): +10% reading accuracy

## Stack Decisions (TBD by user)
- Language: Python
- LLM API: OpenAI / Anthropic (Claude)
- Vector DB: (TBD — Chroma, Qdrant, or Pinecone)
- Embeddings: (TBD — OpenAI, Stella V5, or gte-Qwen2)
- Experiment tracking: (TBD — W&B or MLflow)

## Coding Conventions
- User writes the code; Claude assists with planning, debugging, and review
- Keep memory system implementations modular (swap-in/swap-out architecture)
- Each memory variant should expose the same interface for fair comparison
- Log retrieval traces for ablation analysis
