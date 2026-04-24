"""Text-realizable adversarial-hubness attack on agentic memory.

Stage A: hub vector generation against (Q_est, D_est) — see stage_a_hubs.py
Stage B: text realization (BoN / Evo / RL / GCG) — see stage_b_*.py
Stage C: write-guard evasion — integrated into Stage B objectives via guard.py
"""
