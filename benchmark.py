"""
ChainLink Accuracy Benchmark
=============================
Tests chain reasoning against a realistic 50-memory dataset
with 10 queries, each having known expected chain connections.

Measures:
- Chain Recall: % of expected chain connections actually found
- Chain Precision: % of flagged chains that are real chains
- Vector-only Recall: What plain vector search would find (baseline)
- False Negative analysis: Which chains were missed and why
"""

import json
import time
import os
import sys

# --- THE MEMORY DATASET (50 realistic user memories) ---
MEMORIES = [
    # Health cluster
    "I have a severe shellfish allergy — anaphylaxis risk",                    # 0
    "My doctor prescribed an EpiPen I carry everywhere",                       # 1
    "I take 20mg of Lisinopril daily for blood pressure",                      # 2
    "I'm lactose intolerant but can handle aged cheeses",                      # 3
    "I run 5K three times a week, usually in the morning",                     # 4
    "My resting heart rate is around 52 bpm",                                  # 5

    # Food / cooking cluster
    "I love Thai food, especially green curry and pad thai",                   # 6
    "Thai green curry paste typically contains shrimp paste (kapi)",           # 7
    "I've been learning to make homemade pasta on weekends",                   # 8
    "My favorite restaurant is Siam Garden on 5th street",                     # 9
    "I'm trying to eat more plant-based meals during the week",               # 10
    "Parmesan cheese is naturally lactose-free due to aging",                  # 11

    # Travel cluster
    "I'm flying to Tokyo next month for a two-week trip",                      # 12
    "Japan is 14 hours ahead of US Pacific time",                              # 13
    "I get bad jet lag going eastbound — takes me 3-4 days to adjust",        # 14
    "I always book aisle seats on flights over 4 hours",                       # 15
    "My passport expires in September 2026",                                    # 16
    "Japan requires 6 months passport validity for entry",                     # 17
    "I have TSA PreCheck — my Known Traveler Number is 12345678",             # 18
    "Japanese convenience stores have amazing onigiri and bento boxes",        # 19

    # Work cluster
    "I'm a senior software engineer at Acme Corp",                             # 20
    "Our team standup is at 9am Pacific every weekday",                        # 21
    "I'm leading the migration from MongoDB to PostgreSQL",                    # 22
    "The database migration deadline is April 15th",                           # 23
    "My manager Sarah wants weekly progress reports on Fridays",              # 24
    "I prefer async communication over meetings",                              # 25
    "Our production database has 2.3TB of data",                               # 26

    # Family / social cluster
    "My daughter Emma turns 7 in April",                                        # 27
    "Emma is obsessed with dinosaurs, especially T-Rex",                       # 28
    "My wife's birthday is March 28th",                                         # 29
    "We adopted a golden retriever named Max last year",                       # 30
    "Max has a chicken allergy — he eats grain-free food",                     # 31
    "My parents live in Portland and visit every Thanksgiving",                # 32

    # Finance cluster
    "I max out my 401k contribution every year",                                # 33
    "I have about $15k in an emergency fund at Ally Bank",                     # 34
    "I'm saving for a down payment on a house — goal is $80k",               # 35
    "My monthly mortgage budget is around $3,200",                              # 36
    "I have $42k in student loans at 5.2% interest",                           # 37

    # Hobbies / misc cluster
    "I play guitar and recently joined a jazz band",                           # 38
    "I'm reading 'Thinking Fast and Slow' by Kahneman",                       # 39
    "I built a home lab with a Raspberry Pi cluster",                          # 40
    "I use Obsidian for all my personal notes and journaling",                # 41
    "I'm training for a half marathon in June",                                # 42
    "My PR for a half marathon is 1:47:22",                                     # 43

    # Tech preferences
    "I use VS Code with the Vim extension for all coding",                     # 44
    "My preferred stack is Python + FastAPI + PostgreSQL",                      # 45
    "I run Arch Linux on my personal laptop",                                   # 46
    "I have a Claude Code subscription and use it daily",                      # 47

    # Random but connectable
    "Glucosamine supplements can come from shellfish shells",                  # 48
    "My gym buddy recommended glucosamine for my knee pain",                  # 49
]

# --- TEST QUERIES WITH EXPECTED CONNECTIONS ---
# Each query has:
#   - expected_chains: memories that SHOULD be found via chain reasoning (not obvious vector matches)
#   - expected_direct: memories that vector search should find easily
#   - description: what chain the system should trace

TEST_QUERIES = [
    {
        "query": "What should I know before ordering dinner at a Thai restaurant?",
        "expected_chains": [0, 1, 7],  # shellfish allergy, EpiPen, shrimp paste in Thai food
        "expected_direct": [6, 9],      # loves Thai, Siam Garden
        "description": "Thai food → shrimp paste → shellfish allergy → EpiPen",
    },
    {
        "query": "Help me prepare for my Japan trip",
        "expected_chains": [14, 16, 17, 21],  # jet lag, passport expiry, passport validity req, standup time (timezone)
        "expected_direct": [12, 13, 15, 18, 19],  # Tokyo trip, timezone, aisle seat, TSA, convenience stores
        "description": "Japan trip → passport validity → passport expiry; timezone → jet lag; timezone → meeting conflicts",
    },
    {
        "query": "Birthday gift ideas for my wife",
        "expected_chains": [8, 38, 39],  # pasta making (shared activity), guitar/jazz, book she might like
        "expected_direct": [29],           # wife's birthday
        "description": "Wife's birthday → interests/hobbies that could inspire gift ideas",
    },
    {
        "query": "Can I take glucosamine supplements?",
        "expected_chains": [0, 48],  # shellfish allergy + glucosamine from shellfish
        "expected_direct": [49],      # gym buddy recommended glucosamine
        "description": "Glucosamine → shellfish shells → shellfish allergy (DANGER chain)",
    },
    {
        "query": "Plan Emma's birthday party",
        "expected_chains": [28, 30, 31],  # dinosaurs for theme, dog (consider pet-friendly), dog food allergy
        "expected_direct": [27],            # Emma turns 7
        "description": "Emma birthday → dinosaur theme; party → dog considerations",
    },
    {
        "query": "How should I handle time zones during my trip for work meetings?",
        "expected_chains": [12, 14, 21],  # Tokyo trip, jet lag, standup at 9am
        "expected_direct": [13],            # 14 hours ahead
        "description": "Time zones → Tokyo trip → 14hr diff → 9am standup = 11pm local → jet lag compounds it",
    },
    {
        "query": "What cheese can I eat with my pasta?",
        "expected_chains": [3, 11],  # lactose intolerant, parmesan is lactose-free
        "expected_direct": [8],       # homemade pasta
        "description": "Cheese → lactose intolerance → aged cheese OK → parmesan specifically",
    },
    {
        "query": "Should I pay off debt or save for the house?",
        "expected_chains": [33, 34, 37],  # 401k, emergency fund, student loans
        "expected_direct": [35, 36],       # down payment goal, mortgage budget
        "description": "House savings → student loan debt tradeoff → 401k contributions → emergency fund adequacy",
    },
    {
        "query": "Improve my running performance",
        "expected_chains": [4, 5, 2],  # 5K routine, resting HR, blood pressure meds (affects HR)
        "expected_direct": [42, 43],    # half marathon training, PR time
        "description": "Running → current routine → resting HR → BP meds affecting performance",
    },
    {
        "query": "Set up a development environment for the database migration",
        "expected_chains": [22, 23, 26, 45],  # MongoDB→PG migration, deadline, data size, preferred stack
        "expected_direct": [44, 46],            # VS Code, Arch Linux
        "description": "Dev env → migration project → PostgreSQL stack → deadline pressure → data volume considerations",
    },
]


def run_benchmark():
    from chainlink_memory import ChainLink

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    print("=" * 70)
    print("CHAINLINK ACCURACY BENCHMARK")
    print(f"Memories: {len(MEMORIES)} | Queries: {len(TEST_QUERIES)}")
    print("=" * 70)

    # Initialize and load memories
    cl = ChainLink(api_key=api_key)
    print("\nLoading memories...")
    t0 = time.time()
    cl.add_many(MEMORIES)
    load_time = time.time() - t0
    print(f"  Loaded {cl.count()} memories in {load_time:.1f}s")

    # Run each query
    total_chain_expected = 0
    total_chain_found = 0
    total_chain_flagged = 0
    total_chain_correct = 0
    total_direct_expected = 0
    total_direct_found = 0
    query_times = []

    results_detail = []

    for i, test in enumerate(TEST_QUERIES):
        print(f"\n{'─' * 70}")
        print(f"QUERY {i+1}: \"{test['query']}\"")
        print(f"  Expected chain: {test['description']}")

        t0 = time.time()
        results = cl.query(test["query"], top_k=10)
        elapsed = time.time() - t0
        query_times.append(elapsed)

        found_texts = {r.text: r for r in results}
        found_indices = set()
        for r in results:
            for j, m in enumerate(MEMORIES):
                if r.text == m:
                    found_indices.add(j)

        # Check chain recall
        chain_expected = set(test["expected_chains"])
        chain_found = chain_expected & found_indices
        chain_missed = chain_expected - found_indices

        # Check which found results are flagged as chains
        chain_flagged = set()
        for r in results:
            if r.is_chain:
                for j, m in enumerate(MEMORIES):
                    if r.text == m:
                        chain_flagged.add(j)

        # Chain precision: of flagged chains, how many are in expected_chains?
        chain_correct = chain_flagged & chain_expected

        # Direct recall
        direct_expected = set(test["expected_direct"])
        direct_found = direct_expected & found_indices

        total_chain_expected += len(chain_expected)
        total_chain_found += len(chain_found)
        total_chain_flagged += len(chain_flagged)
        total_chain_correct += len(chain_correct)
        total_direct_expected += len(direct_expected)
        total_direct_found += len(direct_found)

        chain_recall = len(chain_found) / len(chain_expected) * 100 if chain_expected else 0
        direct_recall = len(direct_found) / len(direct_expected) * 100 if direct_expected else 0

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Results returned: {len(results)}")
        print(f"  Chain recall: {len(chain_found)}/{len(chain_expected)} ({chain_recall:.0f}%)")
        print(f"  Direct recall: {len(direct_found)}/{len(direct_expected)} ({direct_recall:.0f}%)")

        if chain_missed:
            print(f"  MISSED chains:")
            for idx in chain_missed:
                print(f"    [{idx}] {MEMORIES[idx][:70]}")

        print(f"  Results:")
        for r in results:
            tag = "[CHAIN]" if r.is_chain else "[VEC]  "
            idx = "?"
            for j, m in enumerate(MEMORIES):
                if r.text == m:
                    idx = j
                    break
            in_expected = "✓" if idx in chain_expected else ("·" if idx in direct_expected else " ")
            print(f"    {tag} [{r.score:.3f}] ({in_expected}) [{idx}] {r.text[:65]}")

        results_detail.append({
            "query": test["query"],
            "chain_recall": chain_recall,
            "direct_recall": direct_recall,
            "chain_found": sorted(chain_found),
            "chain_missed": sorted(chain_missed),
            "time_s": elapsed,
        })

    # --- SUMMARY ---
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    overall_chain_recall = total_chain_found / total_chain_expected * 100 if total_chain_expected else 0
    overall_direct_recall = total_direct_found / total_direct_expected * 100 if total_direct_expected else 0
    overall_chain_precision = total_chain_correct / total_chain_flagged * 100 if total_chain_flagged else 0
    avg_time = sum(query_times) / len(query_times)

    print(f"\n  CHAIN RECALL:    {total_chain_found}/{total_chain_expected} = {overall_chain_recall:.1f}%")
    print(f"    (Expected indirect connections that were actually returned)")
    print(f"\n  CHAIN PRECISION: {total_chain_correct}/{total_chain_flagged} = {overall_chain_precision:.1f}%")
    print(f"    (Of results flagged is_chain, how many were expected chains)")
    print(f"\n  DIRECT RECALL:   {total_direct_found}/{total_direct_expected} = {overall_direct_recall:.1f}%")
    print(f"    (Vector-obvious results that were returned)")
    print(f"\n  AVG QUERY TIME:  {avg_time:.1f}s")
    print(f"  TOTAL TIME:      {sum(query_times):.1f}s for {len(TEST_QUERIES)} queries")
    print(f"  MEMORY COUNT:    {len(MEMORIES)}")

    # Per-query breakdown
    print(f"\n  Per-query chain recall:")
    for i, rd in enumerate(results_detail):
        bar = "█" * int(rd["chain_recall"] / 5) + "░" * (20 - int(rd["chain_recall"] / 5))
        print(f"    Q{i+1:2d}: {bar} {rd['chain_recall']:5.1f}%  ({rd['time_s']:.1f}s)  {TEST_QUERIES[i]['query'][:45]}")

    # Failure analysis
    all_missed = []
    for i, rd in enumerate(results_detail):
        for idx in rd["chain_missed"]:
            all_missed.append((i+1, idx, MEMORIES[idx]))

    if all_missed:
        print(f"\n  MISSED CONNECTIONS ({len(all_missed)} total):")
        for qnum, idx, text in all_missed:
            print(f"    Q{qnum}: [{idx}] {text[:65]}")

    print(f"\n{'=' * 70}")
    grade = "A+" if overall_chain_recall >= 90 else "A" if overall_chain_recall >= 80 else "B" if overall_chain_recall >= 70 else "C" if overall_chain_recall >= 60 else "D" if overall_chain_recall >= 50 else "F"
    print(f"  GRADE: {grade}  (Chain Recall {overall_chain_recall:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_benchmark()
