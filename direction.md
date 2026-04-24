# Research Direction: Adversarial Hubness Attacks on Memory-Augmented LLMs

> **Pivot note (April 2026):** the project pivoted from an RL-trained text-generation attacker (still under [attack/](attack/)) to a geometric hubness attack ([attacks/](attacks/)). The threat model and phasing below are updated; the legacy RL framing is preserved in the "Previous direction" section at the bottom for context. The current engineering spec is [ATTACK.md](ATTACK.md).

## Core Thesis (revised)

A query-blind attacker with memory read access can inject a small number of synthetic hub vectors into a memory-augmented LLM's retrieval index such that the vectors are retrieved for arbitrary unseen queries. When the contaminated context reaches the answer model, the memory system's **confident-answer rate** drops sharply — the attacked agent refuses to answer or hallucinates at much higher rates than the clean baseline. The attack targets the utility of the memory, not only the accuracy of any single answer: a memory system that abstains on half of user questions is functionally broken even if its raw accuracy partially survives via coerced correct-refusals on unanswerable questions.

## Primary metric: confident-answer rate

Traditional RAG attack literature reports accuracy drops. LongMemEval (our benchmark) mixes two question classes:
- **Recall** questions: "what did I say about X?" — abstention is a wrong answer.
- **Abstention** questions (`_abs` suffix): "did we ever discuss Y?" where Y was not discussed — abstention is a correct answer.

An attack that pushes the model toward abstention gets paradoxical credit on the abstention bucket, so *overall* accuracy drop is a weak signal. The attacker's real figure of merit is **confident-answer rate** = `1 − P(model abstains)`. This captures both attack success modes the attacker cares about — induced abstention and induced hallucination — as "memory did not give a confident correct answer." Recall-only accuracy drop is reported alongside as a secondary sanity check.

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

### Temporal-Leakage Caveat
Each attacker-generated session is indexed with a `question_date` timestamp, which the attacker can observe (it is part of the conversation's metadata and visible through memory read-access). This means "query-blind" holds for *semantic* content but leaks along the *temporal* axis — an attacker can implicitly target "things that happened on/near date X" without seeing the query itself. Frame this in the thesis as: temporal context is part of domain knowledge, not query knowledge. The `*_abs` abstention and temporal-reasoning task types are where this caveat matters most and should be reported separately in ablations.

### Where This Sits

| | Knows future queries | Doesn't know future queries |
|---|---|---|
| **Has memory read access** | (overpowered) | **This work** |
| **No memory access** | MINJA | Blind attack (ablation) |

## Approach (revised)

### Stage A — Hub vector generation (done)
- Compute `H = k-means(Q_train, K)` — centroids of the training-split question embeddings, L2-normalized.
- Inject `H` into the victim's retrieval index via `MemoryStore.index_raw_embeddings()` (bypasses the encoder — this is the privileged vector-mode upper bound).
- Per-hub "document" payloads vary: placeholder (`[injected]`), contradictory (fake corrections), or prompt-injection (abstention-bait templates).
- Diagnostics: displacement rate (fraction of queries where a hub beats the legitimate top-1), hub share in top-k, decoupling cosine between payload-text encoding and hub vector.

### Stage B — Text realization (planned, not built)
- Drop the vector-mode privilege. The attacker can only write text; its vector is whatever the defender's encoder produces.
- Goal: generate natural-looking conversation turns that (a) encode near a target hub AND (b) signal contextual emptiness / memory uncertainty, without obvious prompt-injection markers.
- Likely approach: BoN search from a small attacker LLM scored by (cosine-to-hub) + (fluency) + (abstention-proxy). RL is an option if BoN undershoots.
- Stage B's achievable attack is strictly ≤ Stage A's ceiling. Part of the thesis contribution is characterizing that gap.

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

### Novel Contributions (revised for hubs)
1. **Geometric attack on memory retrieval** via adversarial hubs (k-means centroids of the training-query distribution) — no per-query knowledge, no trained generator in Stage A.
2. **Domain-aware but query-blind** threat model — the attacker has memory read access but never sees downstream queries.
3. **Budget-constrained formulation** (K injected hub chunks out of ~250 haystack chunks).
4. **Confident-answer rate as the primary attack metric** — sidesteps the abstention-bucket confound that hides the attack's effect on the user's memory system utility.
5. **Characterization of the Stage A / Stage B gap** — decoupling cosine quantifies how much of the Stage A ceiling is due to the vector-mode privilege and cannot be reached by a realistic text-mode attack.

## Generalization Strategy

The concern: results are tied to LongMemEval. Mitigations:

1. **Held-out task categories**: LongMemEval has 5 task types (information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention). Train on 3, evaluate on 2. Cross-task transfer = evidence of generality.
2. **Train/test question split**: Use half the questions for reward, hold out the rest. If accuracy drops on held-out questions, the agent learned general corruption.
3. **Qualitative analysis**: Show the generated poisoned sessions exhibit interpretable, generalizable strategies (contradicting facts, temporal confusion, diluting retrieval signal) rather than benchmark-specific tricks.
4. **Cross-architecture transfer**: If the same attacker degrades RAG, KV store, and full history, the strategies are architecture-general.

## Phased Execution (revised)

### Phase 1 — Baseline (done)
- Benchmark all memory types clean on LongMemEval.
- File structure: `main.py`, `harness.py`, `eval.py`, `memory/`.

### Phase 2 — Stage A hubness attack (current; Part 2c in progress)
- k-means hub generation on the training-split question embeddings.
- Vector-mode injection via `MemoryStore.index_raw_embeddings()`.
- End-to-end evaluation on val (n=48) and test (n=52) splits with local 7B judge + GPT-4o.
- Primary metric: confident-answer rate drop. Secondary: recall-only accuracy drop, hub share, abstention delta, decoupling cosine.
- Best val result: K=30 prompt_injection → −39.6pp confident-answer rate drop (memory refuses 52% of the time vs 12.5% clean).

### Phase 3 — Stage B text realization (planned)
- Attacker can only write text. Text goes through the defender's encoder; its vector is the encoding. No vector-mode privilege.
- Objective: produce natural-looking chat turns that encode near a target hub AND signal contextual uncertainty.
- Approach (first cut): BoN sampling from a small attacker LLM, scored by cosine-to-hub + fluency + abstention-proxy. RL as a fallback if BoN plateaus.
- Evaluation on the same val/test splits with the same metrics, so the Stage A/B gap is directly readable.

### Phase 4 — Cross-architecture transfer (stretch)
- Re-run the Stage A hub attack against full_history and rl_memory (non-RAG architectures) by attacking whatever they use for addressability (or showing they are naturally robust because there is no vector index to inject into).
- Vulnerability profile across memory architectures.

### Phase 5 — Defense characterization (stretch)
- Test Stage A and (if built) Stage B against A-MemGuard and any published defenses.
- Claim we can make: "defenses trained on content-injection attacks do not generalize to geometric-displacement attacks" — if it holds.

### Legacy: Phase 2 (REINFORCE, parked)
The earlier RL-trained attacker under [attack/](attack/) is not the active track. It may reappear in the ablation section of the thesis as "learned text-generation attacker vs geometric hub attacker," but we are not investing further engineering cycles in it.

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
