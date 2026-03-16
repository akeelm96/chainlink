"""
ChainLink Usage Tracking Tests
===============================
Tests for rate limiting, query counting, instance tracking, and paid tiers.
"""

import os
import sys
import tempfile
import time

# Ensure chainlink_memory is importable
sys.path.insert(0, os.path.dirname(__file__))

from chainlink_memory.usage import (
    UsageTracker,
    FREE_QUERIES_PER_INSTANCE,
    FREE_MAX_INSTANCES,
    FREE_TOTAL_QUERIES,
    PAID_PACK_SIZE,
    PAID_PACK_PRICE_CENTS,
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1


def fresh_tracker():
    """Create a tracker with a temp database."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return UsageTracker(db_path=f.name), f.name


print("=" * 60)
print("CHAINLINK USAGE TRACKING TESTS")
print("=" * 60)

# --- Test 1: Constants ---
print("\n--- Test 1: Constants ---")
test("Free queries per instance = 200", FREE_QUERIES_PER_INSTANCE == 200)
test("Free max instances = 5", FREE_MAX_INSTANCES == 5)
test("Free total = 1000", FREE_TOTAL_QUERIES == 1000)
test("Paid pack size = 500", PAID_PACK_SIZE == 500)
test("Paid pack price = $2.00", PAID_PACK_PRICE_CENTS == 200)

# --- Test 2: Register and get account ---
print("\n--- Test 2: Account Registration ---")
tracker, db = fresh_tracker()
account = tracker.register_key("hash_abc", name="test_app")
test("Account created", account is not None)
test("Plan is free", account["plan"] == "free")
test("Free used = 0", account["free_tier"]["used"] == 0)
test("Free remaining = 1000", account["free_tier"]["remaining"] == 1000)
test("Paid balance = 0", account["paid_tier"]["balance"] == 0)
test("Instance count = 0", account["instance_count"] == 0)
os.unlink(db)

# --- Test 3: First query creates instance ---
print("\n--- Test 3: First Query Allowance ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_001")
allowed, tier, msg = tracker.check_allowance("hash_001", "my_app")
test("First query allowed", allowed)
test("Tier is free", tier == "free")
account = tracker.get_account("hash_001")
test("Instance created", "my_app" in account["instances"])
os.unlink(db)

# --- Test 4: Record and count queries ---
print("\n--- Test 4: Query Recording ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_002")
for i in range(10):
    allowed, tier, _ = tracker.check_allowance("hash_002", "app1")
    tracker.record_query("hash_002", "app1", tier, f"query_{i}", 50, 100.0)

account = tracker.get_account("hash_002")
test("10 free used", account["free_tier"]["used"] == 10)
test("990 free remaining", account["free_tier"]["remaining"] == 990)
test("Instance shows 10 used", account["instances"]["app1"]["queries_used"] == 10)
test("Total queries = 10", account["total_queries"] == 10)
os.unlink(db)

# --- Test 5: Per-instance limit (200) ---
print("\n--- Test 5: Per-Instance Limit ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_003")

# Burn through 200 queries on one instance
for i in range(200):
    allowed, tier, _ = tracker.check_allowance("hash_003", "app1")
    assert allowed, f"Query {i} should be allowed"
    tracker.record_query("hash_003", "app1", tier)

# 201st should be blocked
allowed, tier, msg = tracker.check_allowance("hash_003", "app1")
test("201st query blocked", not allowed)
test("Message mentions 200", "200" in msg)

# But a different instance should work
allowed2, tier2, _ = tracker.check_allowance("hash_003", "app2")
test("Different instance still works", allowed2)
os.unlink(db)

# --- Test 6: Instance limit (5) ---
print("\n--- Test 6: Instance Limit ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_004")

# Create 5 instances
for i in range(5):
    allowed, _, _ = tracker.check_allowance("hash_004", f"instance_{i}")
    test(f"Instance {i} allowed", allowed)

# 6th instance should be blocked (no paid balance)
allowed, _, msg = tracker.check_allowance("hash_004", "instance_6")
test("6th instance blocked", not allowed)
test("Message mentions instance limit", "instance" in msg.lower() or "5" in msg)
os.unlink(db)

# --- Test 7: Total free limit (1000) ---
print("\n--- Test 7: Total Free Limit ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_005")

# Use 200 queries across 5 instances = 1000 total
for inst in range(5):
    for q in range(200):
        allowed, tier, _ = tracker.check_allowance("hash_005", f"inst_{inst}")
        tracker.record_query("hash_005", f"inst_{inst}", tier)

account = tracker.get_account("hash_005")
test("1000 free used", account["free_tier"]["used"] == 1000)
test("0 free remaining", account["free_tier"]["remaining"] == 0)

# Next query should fail
allowed, _, msg = tracker.check_allowance("hash_005", "inst_0")
test("1001st query blocked", not allowed)
test("Message mentions purchase", "purchase" in msg.lower())
os.unlink(db)

# --- Test 8: Purchase query packs ---
print("\n--- Test 8: Purchasing Queries ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_006")

# Exhaust free tier
for inst in range(5):
    for q in range(200):
        allowed, tier, _ = tracker.check_allowance("hash_006", f"i_{inst}")
        tracker.record_query("hash_006", f"i_{inst}", tier)

# Should be blocked
allowed, _, _ = tracker.check_allowance("hash_006", "i_0")
test("Blocked before purchase", not allowed)

# Buy 1 pack ($2 for 500)
result = tracker.add_paid_queries("hash_006", packs=1, stripe_payment_id="pi_test123")
test("Plan upgraded to paid", result["plan"] == "paid")
test("Paid balance = 500", result["paid_tier"]["balance"] == 500)

# Now queries should work on paid tier
allowed, tier, msg = tracker.check_allowance("hash_006", "i_0")
test("Query allowed after purchase", allowed)
test("Billed to paid tier", tier == "paid")

# Record and check deduction
tracker.record_query("hash_006", "i_0", "paid")
account = tracker.get_account("hash_006")
test("Paid balance decremented to 499", account["paid_tier"]["balance"] == 499)
test("Total paid used = 1", account["paid_tier"]["total_used"] == 1)
os.unlink(db)

# --- Test 9: Paid users can exceed instance limit ---
print("\n--- Test 9: Paid Bypasses Instance Limit ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_007")

# Create 5 instances
for i in range(5):
    tracker.check_allowance("hash_007", f"inst_{i}")

# Buy queries
tracker.add_paid_queries("hash_007", packs=1)

# 6th instance should now work because paid
allowed, tier, _ = tracker.check_allowance("hash_007", "inst_new_6")
test("6th instance allowed for paid user", allowed)
test("Uses paid tier", tier == "paid")
os.unlink(db)

# --- Test 10: Usage stats ---
print("\n--- Test 10: Usage Stats ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_008")
for i in range(5):
    tracker.check_allowance("hash_008", "stats_app")
    tracker.record_query("hash_008", "stats_app", "free", f"test query {i}", 20, 150.0)

stats = tracker.get_usage_stats("hash_008")
test("Recent queries returned", len(stats["recent_queries"]) == 5)
test("Free tier counted", stats["totals_by_tier"].get("free") == 5)
os.unlink(db)

# --- Test 11: Unknown account ---
print("\n--- Test 11: Unknown Account ---")
tracker, db = fresh_tracker()
allowed, _, msg = tracker.check_allowance("nonexistent_hash", "app")
test("Unknown account rejected", not allowed)
test("Message says register", "register" in msg.lower() or "not found" in msg.lower())
os.unlink(db)

# --- Test 12: Multiple packs ---
print("\n--- Test 12: Multiple Pack Purchase ---")
tracker, db = fresh_tracker()
tracker.register_key("hash_009")
result = tracker.add_paid_queries("hash_009", packs=3)
test("3 packs = 1500 queries", result["paid_tier"]["balance"] == 1500)
os.unlink(db)

# --- Summary ---
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{failed} TESTS FAILED")
print("=" * 60)
