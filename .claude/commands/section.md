---
description: Bootstrap a per-section drafting chat for thesis §$1. Reads OUTLINE.md + paper/CLAUDE.md + anchor files, then dispatches the paper-section-drafter agent. Usage: /section 5.2
argument-hint: <X.Y> (e.g., 5.2, 4.4, 2.3)
---

Drafting **§$1** of the thesis paper.

Bootstrap protocol — execute in order before proposing any prose:

1. Read `paper/CLAUDE.md` (conventions, notation, spine, headline numbers).
2. Read `paper/OUTLINE.md` (section state, page budget for §$1, anchor files, per-section drafting loop, files-and-artifacts table).
3. Read `paper/main.tex` around the §$1 scaffold to see current `\subsection{...}` structure and any `% TODO` markers.
4. Read this section's anchor files (from OUTLINE.md "Files & artifacts" table).
5. If §$1 is in §3, §5, or §6: also read `ATTACK.md` (esp. Part 2c table and Final consolidated sweep — these are the canonical numbers).
6. If §$1 is in §2: also read `direction.md` (related-work prose source).

After reading, before drafting:
- Confirm any cluster-side results files are present locally (§5.3 needs `results/stage_a/eval_test_conf.json`; §5.5 needs `results/stage_b/eval_test_final_sweep.{json,csv}`). If missing, surface as a blocker — do not fabricate numbers.
- Confirm notation discipline (K, M, PI per CLAUDE.md).
- Confirm the displacement-vs-weaponization spine threads through §$1 if it's one of §1, §4, §5.4, §5.5, §6.
- Confirm page budget for §$1 from OUTLINE.md.

Then dispatch the `paper-section-drafter` agent on §$1. The agent will:
- Outline the section in 3-6 sentences first.
- Write a single `Edit` to `paper/main.tex` replacing the empty subsection scaffolds with prose.
- Mark judgment calls with `% TODO`.
- Drop `% source: <path>` comments next to non-obvious numbers.

After drafting:
- Update the OUTLINE.md status table: §$1 → `drafting`. Append a 3-line change summary at the bottom of OUTLINE.md.
- Run the `claim-traceability-reviewer` agent on the drafted section.
- Run the `latex-style-reviewer` agent on the drafted section.
- Surface findings to me. Wait for me to review in Overleaf before marking §$1 landed.
