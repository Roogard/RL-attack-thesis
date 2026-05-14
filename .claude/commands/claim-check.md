---
description: Run claim-traceability + latex-style review on §$1 of paper/main.tex. Returns pass/fail checklist with file:line citations. Usage: /claim-check 5.5
argument-hint: <X.Y> (e.g., 5.5, 3.4, 2.3) — or "all" to audit the whole paper
---

Run the full review pass on **§$1** of `paper/main.tex`.

In parallel (single message, two Agent tool uses), dispatch:

1. **`claim-traceability-reviewer`** — for §$1: every numeric claim must trace to `results/`, `ATTACK.md`, or `paper/CLAUDE.md` headline-numbers table; every `\cite{key}` must resolve in `refs.bib`; every cited code path must exist (`Glob` check).

2. **`latex-style-reviewer`** — for §$1: notation discipline (K/M/PI), spine presence (if §1/§4/§5.4/§5.5/§6), no BadVision template detritus, unresolved `% TODO` markers, page-budget estimate, frozen-content diff (abstract), LaTeX hygiene.

After both agents return, summarize:
- Total issues across both reviews.
- Which issues block §$1 from being marked `landed`.
- Suggested fixes (with `file:line` references).

If both reviews pass: tell me §$1 is ready for `landed` state transition.
