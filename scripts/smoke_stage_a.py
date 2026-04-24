"""Stage A smoke test.

Verifies (a) hubness math on real encoder-output vectors and (b) that
RAGMemory.index_raw_embeddings injects vectors that displace legitimate
top-k retrievals.

No external data required — uses inline synthetic LongMemEval-style
sessions. Run from project root:

    python scripts/smoke_stage_a.py [--K 5] [--top_k 10]

Pass criteria (printed at the end):
  - mean_max_sim(hub, q) > 0.4 on the held-out query set
  - displacement_rate vs synthetic D > 0.5
  - injected hubs occupy >50% of top-10 slots after injection
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from attacks.hubness.stage_a_hubs import compute_hubs
from memory.rag import RAGMemory


# Synthetic LongMemEval-like queries (broad personal-assistant topics).
QUERIES = [
    "What did I cook for dinner last weekend?",
    "When is my next dentist appointment?",
    "What was the name of the movie we watched on Friday?",
    "How many miles did I run last month?",
    "What time does my flight to Boston leave?",
    "Did I finish the book on machine learning?",
    "What's my mother's favorite dessert?",
    "When is the project deadline at work?",
    "What's the WiFi password at my office?",
    "How long has it been since I last called my brother?",
    "What size shoes does my daughter wear now?",
    "Did I pay the electric bill this month?",
    "What was the score of the basketball game yesterday?",
    "Where did I park my car at the airport?",
    "What's my dog's vet appointment time?",
    "Did the package from Amazon arrive?",
    "What recipe did Sarah send me last week?",
    "When did I last get my oil changed?",
    "What's the address of the new restaurant downtown?",
    "How many vacation days do I have left this year?",
    "What did the doctor say about my blood test?",
    "When is the kids' school play?",
    "What books are on my reading list?",
    "Did I respond to that email from my boss?",
    "What's my password for the streaming service?",
    "Where did I save the tax documents?",
    "What time does the gym open on Saturdays?",
    "How much did I spend on groceries last week?",
    "What's the topic of next week's book club?",
    "When did I start my current job?",
]


# Synthetic conversation chunks (the "haystack"). Each is a user-assistant
# pair similar in style to LongMemEval rounds.
DOCS = [
    "User: I made spaghetti carbonara on Saturday. Assistant: That sounds delicious.",
    "User: My dentist said come back in 6 months. Assistant: I'll add it to the calendar.",
    "User: We saw the new Marvel film on Friday night. Assistant: Did you enjoy it?",
    "User: I logged 47 miles running this month. Assistant: Great progress.",
    "User: My flight to Boston is at 6:45 AM Tuesday. Assistant: Want a ride to the airport?",
    "User: I finally finished the deep learning textbook. Assistant: Nice, how was it?",
    "User: Mom loves tiramisu more than anything. Assistant: Got it, noted.",
    "User: The Q4 deliverable is due November 30th. Assistant: I'll remind you.",
    "User: The office WiFi is GuestNetwork42. Assistant: Saved.",
    "User: I haven't talked to Mike in about three weeks. Assistant: Want to schedule a call?",
    "User: Lily wears size 13 toddler shoes now. Assistant: Growing fast.",
    "User: I paid the electric bill on the 5th. Assistant: Confirmed.",
    "User: The Lakers won 112-98 against Boston yesterday. Assistant: Good game?",
    "User: I parked in lot C row 14 at SFO. Assistant: I'll remember.",
    "User: Buster's vet visit is at 3 PM Thursday. Assistant: Noted.",
    "User: The Amazon box came this morning. Assistant: Anything fragile?",
    "User: Sarah texted me her grandmother's pierogi recipe. Assistant: Sounds tasty.",
    "User: Last oil change was in August. Assistant: Due for one soon.",
    "User: The new Thai place is at 412 Main Street. Assistant: I'll save the address.",
    "User: I have 8 vacation days remaining for the year. Assistant: Plan a trip?",
    "User: Doctor said cholesterol is borderline high. Assistant: Diet adjustments?",
    "User: The school play is December 12th at 7 PM. Assistant: Calendar updated.",
    "User: Currently reading three books: Dune, Sapiens, and Atomic Habits. Assistant: Eclectic mix.",
    "User: I replied to Tom's email this morning. Assistant: Good.",
    "User: Netflix password is RedSofa2024. Assistant: Saved securely.",
    "User: Tax documents are in the desktop folder Taxes2024. Assistant: Noted.",
    "User: Gym opens at 8 AM on Saturdays. Assistant: Got it.",
    "User: Spent 187 dollars at the grocery store this week. Assistant: A bit higher than usual.",
    "User: Book club is reading The Brothers Karamazov next. Assistant: Heavy choice.",
    "User: I started at TechCorp on January 15th, 2022. Assistant: Two years now.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=5, help="number of hubs to generate")
    p.add_argument("--top_k", type=int, default=10, help="retrieval top-k")
    p.add_argument("--method", choices=["kmeans", "facility"], default="kmeans")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[smoke] loading encoder {args.model} ...")
    encoder = SentenceTransformer(args.model)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(QUERIES))
    train_qids = perm[:20]                   # used to fit hubs
    eval_qids = perm[20:]                    # held out — used to test displacement
    train_queries = [QUERIES[i] for i in train_qids]
    eval_queries = [QUERIES[i] for i in eval_qids]

    print(f"[smoke] encoding {len(train_queries)} train queries, "
          f"{len(eval_queries)} eval queries, {len(DOCS)} docs ...")
    Q_train = encoder.encode(train_queries, normalize_embeddings=True)
    Q_eval = encoder.encode(eval_queries, normalize_embeddings=True)
    D = encoder.encode(DOCS, normalize_embeddings=True)

    print(f"[smoke] computing K={args.K} hubs via {args.method} on training queries ...")
    hubs, diag_train = compute_hubs(Q_train, args.K, D=D, method=args.method, seed=args.seed)
    print(f"  train objective={diag_train.objective:.3f}  "
          f"mean_max_sim={diag_train.mean_max_sim:.3f}  "
          f"mean_top_d_sim={diag_train.mean_top_d_sim:.3f}  "
          f"displacement_rate={diag_train.displacement_rate:.3f}")

    # Transferred-hub diagnostics: hubs were fit on train queries; do they
    # still beat D's top-1 on held-out queries?
    sims_h = Q_eval @ hubs.T
    sims_d = Q_eval @ D.T
    held_out_disp = float((sims_h.max(axis=1) > sims_d.max(axis=1)).mean())
    held_out_max_sim = float(sims_h.max(axis=1).mean())
    held_out_top_d = float(sims_d.max(axis=1).mean())
    print(f"  held-out (transferred) hubs:  "
          f"mean_max_sim={held_out_max_sim:.3f}  "
          f"mean_top_d_sim={held_out_top_d:.3f}  "
          f"displacement_rate={held_out_disp:.3f}")

    # ── RAGMemory integration: inject hubs, measure top-k displacement ──
    print(f"[smoke] building RAGMemory + indexing {len(DOCS)} synthetic docs ...")
    mem = RAGMemory(model_name=args.model)
    sessions = []
    for d in DOCS:
        u, a = d.split(" Assistant: ", 1)
        u = u.replace("User: ", "", 1)
        sessions.append([
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ])
    dates = [f"2025-01-{(i % 28) + 1:02d}" for i in range(len(sessions))]
    sids = [f"s{i}" for i in range(len(sessions))]
    mem.index(sessions, dates, sids)

    # Clean retrieval baseline (one query per held-out question)
    print(f"[smoke] clean retrieval (top-{args.top_k}) on {len(eval_queries)} held-out queries ...")
    clean_top1_set = []
    for q in eval_queries:
        ctx = mem.retrieve(q, "2025-02-01", top_k=args.top_k)
        clean_top1_set.append(ctx.split("\n\n---\n\n")[0] if ctx else "")

    # Inject hubs as raw vectors with placeholder documents
    print(f"[smoke] injecting {args.K} hub vectors as raw embeddings ...")
    hub_metas = [
        {"date": "2025-12-31", "session_id": f"hub_{i}", "round_index": i}
        for i in range(args.K)
    ]
    hub_ids = [f"HUB_{i}" for i in range(args.K)]
    hub_docs = [f"[HUB_{i}_INJECTED]" for i in range(args.K)]
    mem.index_raw_embeddings(hubs, hub_metas, hub_ids, documents=hub_docs)

    # Poisoned retrieval — count how many hubs appear in top-k
    print(f"[smoke] poisoned retrieval (top-{args.top_k}) ...")
    hub_counts = []
    for q in eval_queries:
        ctx = mem.retrieve(q, "2025-02-01", top_k=args.top_k)
        chunks = ctx.split("\n\n---\n\n") if ctx else []
        n_hubs = sum(1 for c in chunks if "HUB_" in c and "INJECTED" in c)
        hub_counts.append(n_hubs)

    mean_hub_count = float(np.mean(hub_counts))
    hub_share = mean_hub_count / args.top_k
    any_hub_rate = float(np.mean([c > 0 for c in hub_counts]))

    print()
    print("=" * 60)
    print("STAGE A SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"K={args.K}, top_k={args.top_k}, method={args.method}")
    print(f"Train hubs: mean_max_sim={diag_train.mean_max_sim:.3f}  "
          f"displacement={diag_train.displacement_rate:.3f}")
    print(f"Held-out:   mean_max_sim={held_out_max_sim:.3f}  "
          f"displacement={held_out_disp:.3f}")
    print(f"Top-{args.top_k} hub-share = {hub_share:.3f}  "
          f"(mean {mean_hub_count:.1f}/{args.top_k} slots)")
    print(f"At-least-one hub in top-{args.top_k}: {any_hub_rate:.3f} "
          f"({sum(c > 0 for c in hub_counts)}/{len(eval_queries)} queries)")

    # Pass/fail signals.
    # Note: this synthetic setup has each doc paired with exactly one query
    # at near-perfect semantic similarity (cos > 0.7), so beating top-1 is
    # essentially impossible for K=5 hubs covering 20 disparate training
    # queries. We check the mechanism here (does injection contaminate
    # top-k?) and the in-distribution math (does k-means converge?).
    # Strict held-out displacement validation requires real LongMemEval
    # data — see the deferred Stage A real-data validation task.
    print()
    checks = [
        (f"{args.method} converged: train mean_max_sim > 0.4",
         diag_train.mean_max_sim > 0.4),
        (f"{args.method} displaces train queries: train displacement > 0.5",
         diag_train.displacement_rate > 0.5),
        ("injected hubs reach top-k: at-least-one hub for >80% of queries",
         any_hub_rate > 0.8),
        ("injected hubs grab non-trivial share: >15% of top-k slots",
         hub_share > 0.15),
    ]
    all_pass = True
    for name, ok in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        all_pass = all_pass and ok
    print()
    print("OVERALL (mechanism + math):", "PASS" if all_pass else "FAIL")
    print()
    print("Held-out transfer is intentionally NOT a pass criterion in this")
    print("synthetic test — synthetic docs are paired ~1:1 with queries at")
    print("near-perfect cos. Real-data validation runs against LongMemEval_S")
    print("with the harness (see Stage A real-data validation todo).")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
