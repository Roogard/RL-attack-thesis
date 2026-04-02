# Research Direction: RL-Trained Adversarial Memory Poisoning

## Core Thesis

An RL-trained attacker can learn to generate poisoned conversation sessions that degrade an LLM agent's memory-dependent performance — without knowing what questions the agent will be asked. The attacker knows the domain (e.g., "this agent handles multi-session personal conversations") and can observe stored memories, but has no access to downstream evaluation queries.

## Threat Model

### Attacker Capabilities
- **Domain knowledge**: Knows the general purpose of the agent (personal assistant, medical, etc.)
- **Memory read access**: Can observe what is currently stored in the agent's memory (realistic — many agents surface memory in responses like "I remember you said X")
- **Session injection**: Can submit conversation sessions that pass through the agent's normal memory pipeline (the agent decides what to store, the attacker doesn't write directly)
- **Budget-constrained**: Limited to N poisoned sessions (e.g., 1–10) out of a larger history (e.g., 50 sessions)

### Attacker Does NOT Have
- Knowledge of what questions will be asked (the key difference from MINJA)
- Direct write access to the memory store (the key difference from pre-MINJA work)
- Knowledge of the memory system's implementation details

### Where This Sits

| | Knows future queries | Doesn't know future queries |
|---|---|---|
| **Has memory read access** | (overpowered) | **This work** |
| **No memory access** | MINJA | Blind attack (ablation) |

## Approach

### Training the Attacker
- **Policy**: A local LLM (small enough for university cluster) serves as the attacker
- **Input**: Current memory contents + domain context
- **Output**: Poisoned conversation sessions (user/assistant turns)
- **Training**: GRPO (Group Relative Policy Optimization), same family as Memory-R1
- **Reward signal**: Accuracy drop on LongMemEval questions after injecting the poisoned sessions
  - The attacker never sees the questions — they are only used to compute reward
  - Reward = (clean accuracy - poisoned accuracy), higher is better for the attacker

### Evaluation Pipeline
1. Take a LongMemEval question with its clean haystack (50 sessions)
2. Attacker generates N poisoned sessions given memory read access
3. Insert poisoned sessions into the haystack
4. Target agent's memory system indexes the combined haystack (poison may or may not be stored depending on memory type)
5. Agent answers LongMemEval questions using its memory
6. GPT-4o judges correctness
7. Measure accuracy drop vs. clean baseline

### Target Memory Architectures
Each architecture has different vulnerability profiles:
- **Full history (in-context)**: Poison always present, but diluted by volume
- **RAG / Vector DB**: Poison must be semantically similar to real answers to get retrieved
- **Key-Value Store**: Poison must produce extractable facts that contradict real ones
- **Temporal-aware RAG**: Poison must navigate time-based filtering
- **RL-managed memory** (stretch): Adversarial RL vs. RL — the attacker must fool a learned storage policy

## Differentiation from Existing Work

### vs. MINJA (2025)
- MINJA requires knowledge of victim queries (specific terms that trigger the attack)
- MINJA is a targeted backdoor; this is general degradation
- MINJA uses handcrafted injection techniques; this uses a learned RL policy

### vs. AgentPoison (NeurIPS 2024)
- AgentPoison poisons RAG knowledge bases with trigger-activated backdoors
- Requires the attacker to define a trigger; this work is trigger-free
- AgentPoison targets static KBs; this targets dynamic conversation-built memory

### vs. PoisonedRAG (USENIX Security 2025)
- PoisonedRAG injects documents directly into the knowledge store
- Skips the memory system's own filtering/indexing pipeline
- Targets static document retrieval, not agent memory

### vs. MemoryGraft (2025)
- MemoryGraft poisons experience retrieval via indirect injection (README files, etc.)
- Different attack vector (external content vs. conversation sessions)
- Not RL-based

### Novel Contributions
1. **RL-trained attacker** that learns poisoning strategies end-to-end (no handcrafted heuristics)
2. **Domain-aware but query-blind** threat model (realistic middle ground)
3. **Budget-constrained formulation** (N poisoned out of M total sessions)
4. **Cross-architecture vulnerability comparison** (same attacker vs. multiple memory types)
5. **Direct optimization on damage** (reward = accuracy drop, not a proxy like storage success)

## Generalization Strategy

The concern: results are tied to LongMemEval. Mitigations:

1. **Held-out task categories**: LongMemEval has 5 task types (information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention). Train on 3, evaluate on 2. Cross-task transfer = evidence of generality.
2. **Train/test question split**: Use half the questions for reward, hold out the rest. If accuracy drops on held-out questions, the agent learned general corruption.
3. **Qualitative analysis**: Show the generated poisoned sessions exhibit interpretable, generalizable strategies (contradicting facts, temporal confusion, diluting retrieval signal) rather than benchmark-specific tricks.
4. **Cross-architecture transfer**: If the same attacker degrades RAG, KV store, and full history, the strategies are architecture-general.

## Phased Execution

### Phase 1 — Baseline (current)
- Benchmark all memory types clean on LongMemEval
- Establish performance ceiling for each architecture
- File structure: `main.py`, `harness.py`, `eval.py`, `memory/`

### Phase 2 — RL Attacker (core contribution)
- Implement GRPO training loop
- Train attacker against one memory type (start with RAG — easiest to poison via retrieval)
- Show measurable accuracy degradation with increasing attacker budget (1, 3, 5, 10 sessions)
- Ablation: remove memory read access, measure how much attack weakens

### Phase 3 — Cross-Architecture Vulnerability
- Test trained attacker against all memory types
- Retrain attacker per-architecture if needed
- Produce vulnerability profile: which memory types are most/least susceptible?
- Held-out task category experiments for generalization evidence

### Phase 4 — Stretch Goals (if time permits)
- Attack RL-managed memory (adversarial RL vs. RL)
- Meta-learning across multiple benchmarks (requires finding/creating additional memory benchmarks)
- Defense analysis: test against A-MemGuard and show RL attacker bypasses it

## Compute Plan

- **University supercomputer** for GRPO training (local models for both attacker and target agent)
- **Local models**: Attacker policy + target agent (small enough for cluster GPUs)
- **GPT-4o**: Final evaluation judging only (can also explore local judge as cost reduction)
- **LongMemEval_S** (~115K tokens): Use the smaller split for training iterations, full set for final eval

## Key Papers

- Memory-R1 (2025): RL policy for memory management — arxiv.org/abs/2508.19828
- MINJA (2025): Query-only memory injection — arxiv.org/abs/2503.03704
- AgentPoison (NeurIPS 2024): Backdoor attacks on agent memory/RAG
- PoisonedRAG (USENIX Security 2025): Knowledge base poisoning
- MemoryGraft (2025): Indirect injection via experience poisoning
- A-MemGuard (2025): Defense framework for agent memory — arxiv.org/abs/2510.02373
- LongMemEval (ICLR 2025): Memory evaluation benchmark — arxiv.org/abs/2410.10813
