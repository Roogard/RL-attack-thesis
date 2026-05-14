# Thesis Project: Adversarial Memory Evaluation for LLM Agents

## Repo at a glance

This project evaluates and attacks memory systems in LLM-based conversational agents using the **LongMemEval** benchmark (ICLR 2025, arXiv:2410.10813).

The codebase has two threads:
- **Research engineering** (everything outside `paper/`) — agent harness, memory backends, attack pipelines, evaluation scripts, results dumps.
- **Thesis writeup** (everything inside `paper/`) — local clone of the Overleaf project. See [paper/CLAUDE.md](paper/CLAUDE.md) — it auto-loads when you work in `paper/`.

## Paper-writing workflow

If you are touching `paper/`, **load [paper/CLAUDE.md](paper/CLAUDE.md) first** — it has the notation, headline numbers, drafting loop, and frozen-content rules.

The paper is split across separate Claude chats by job:
1. **Setup chat** — built the environment (done).
2. **Outline chat** — refines [paper/OUTLINE.md](paper/OUTLINE.md).
3. **Per-section chats** — one chat per `§X.Y`, drafts into `paper/main.tex`.

See [paper/WORKFLOW.md](paper/WORKFLOW.md) for the three-chat structure and [paper/templates/section_chat_bootstrap.md](paper/templates/section_chat_bootstrap.md) for the per-section bootstrap prompt.

**Paper scope (advisor-aligned, locked):**
- ~12 pages. (`example_thesis/Thesis_Paper_Aidan.pdf` is a formatting reference only, not a length target.)
- **Pure attack paper.** The hubness attack is the sole contribution.
- **RAG vector-database memory only.** Other memory variants in the research codebase (full_history, key-value, RL-managed) are out of paper scope per advisor.
- Headline configuration: M=3+PI as the combined-attack centerpiece; K100_M5_PI as the high-budget extreme. Canonical numbers live in [paper/CLAUDE.md](paper/CLAUDE.md) and [ATTACK.md](ATTACK.md).

The CLAUDE.md sections below describe the research project (not the paper) — useful when working on code, experiments, or new attack variants.

---

## Research Goals (broader than the paper)

1. **Phase 1 — Baseline Agent**: Build a production-grade agent that runs on LongMemEval and achieves competitive performance. (Implemented.)
2. **Phase 2 — Memory Variants**: Implement and compare multiple memory architectures on the benchmark. (Partial; out of paper scope.)
3. **Phase 3 — Adversarial Attacks**: Design and evaluate adversarial policy attacks that exploit memory vulnerabilities. **This is what the paper is about.**

## LongMemEval Benchmark
- **GitHub**: https://github.com/xiaowu0162/LongMemEval
- **Dataset**: HuggingFace `xiaowu0162/longmemeval-cleaned`
- **Tasks**: Information Extraction, Multi-Session Reasoning, Knowledge Updates, Temporal Reasoning, Abstention
- **Scale**: LongMemEval_S (~115K tokens), LongMemEval_M (~1.5M tokens)
- **Metric**: LLM-judged accuracy (GPT-4o evaluator, >97% agreement with humans)

## Adversarial Attack Direction (paper contribution)

The attacker injects K synthetic "hub" vectors (k-means centroids of training-question embeddings) into the victim's retrieval index. Hubs win top-k slots from legitimate documents for arbitrary unseen queries. See [ATTACK.md](ATTACK.md) for the full engineering spec and [direction.md](direction.md) for the related-work differentiation.

**Two-stage attack decomposition:**
- **Stage A** — privileged vector-mode injection (raw embedding write). Used as the upper-bound mechanism analysis.
- **Stage B** — realistic text-mode injection (text indexed through the defender's encoder). Three realization methods: retrieval-from-corpus (headline), best-of-N generation (ablation), gradient/HotFlip (ablation).

**Spine of the paper: displacement vs. weaponization.** ~75% of the privileged-mode attack effect is geometric displacement (multi-chunk M); ~25% is payload weaponization (PI prompt-injection bait). Stage B M × PI factorial realizes the same decomposition under the realistic threat model.

## Stack (locked)

- **Language**: Python
- **LLM API**: OpenAI (GPT-4o for judging), local vLLM for reader
- **Reader model**: Qwen2.5-3B-Instruct (via vLLM)
- **Judge**: GPT-4o (final), Qwen2.5-7B (local iteration)
- **Vector DB**: Chroma
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2`
- **Benchmark split**: 80/10/10 stratified (398 train / 48 val / 52 test) from `configs/splits/rag_attack.json`

## Coding Conventions
- For **paper prose**, the user is lead author and edits in Overleaf; Claude drafts `.tex` per [paper/OUTLINE.md](paper/OUTLINE.md). See [paper/CLAUDE.md](paper/CLAUDE.md).
- For **code**, Claude writes implementations; user reviews and directs strategy.
- Memory backends share a common `MemoryStore` interface (`memory/base.py`).
- Log retrieval traces for ablation analysis.

## Three-Stage Memory Framework (LongMemEval paper)
1. **Indexing**: Chunk history → extract key-value pairs.
2. **Retrieval**: BM25 / dense retrieval + time-aware query expansion.
3. **Reading**: Chain-of-Note structured prompting → final answer.

## Key Results from LongMemEval paper (context, not paper targets)
- All SOTA models show ~30% accuracy drop vs oracle.
- Round-level decomposition > session-level for retrieval.
- Fact-augmented key expansion: +4% Recall@k.
- Time-aware query expansion: +7–11% on temporal tasks.
- Structured prompting (CoT/CoN): +10% reading accuracy.
