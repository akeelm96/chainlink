"""
Test: 200 query per-instance limit stress test.
Simulates a real user hitting the free tier wall on a single instance.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

from chainlink_memory.usage import UsageTracker, FREE_QUERIES_PER_INSTANCE

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


print("=" * 60)
print("200-QUERY INSTANCE LIMIT STRESS TEST")
print("=" * 60)

# Fresh DB
tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
tracker = UsageTracker(db_path=tmp.name)
tracker.register_key("stress_key", name="stress_test")

# --- Fire 200 queries and verify each is allowed ---
print(f"\nFiring {FREE_QUERIES_PER_INSTANCE} queries on instance 'my_app'...")
blocked_at = None
t0 = time.time()

for i in range(250):  # Try 250, expect block at 201
    allowed, tier, msg = tracker.check_allowance("stress_key", "my_app")

    if not allowed:
        blocked_at = i
        break

    tracker.record_query(
        key_hash="stress_key",
        instance_id="my_app",
        tier=tier,
        query_preview=f"query number {i+1}",
        n_memories=50,
        latency_ms=100.0,
    )

elapsed = time.time() - t0

print(f"  Completed in {elapsed:.2f}s")
print(f"  Blocked at query #{blocked_at + 1 if blocked_at else 'never'}")

test("Blocked at exactly query 201", blocked_at == 200)

# --- Verify account state ---
account = tracker.get_account("stress_key")
test("200 free queries used", account["free_tier"]["used"] == 200)
test("800 free remaining (global)", account["free_tier"]["remaining"] == 800)
test("Instance shows 200 used", account["instances"]["my_app"]["queries_used"] == 200)
test("Instance shows 0 remaining", account["instances"]["my_app"]["remaining"] == 0)

# --- Verify the block message ---
allowed, tier, msg = tracker.check_allowance("stress_key", "my_app")
test("Still blocked on retry", not allowed)
test("Message mentions 200", "200" in msg)
test("Message mentions purchase", "purchase" in msg.lower())
print(f"  Block message: {msg}")

# --- Verify other instances still work ---
print("\nTesting cross-instance availability...")
allowed2, tier2, _ = tracker.check_allowance("stress_key", "second_app")
test("Second instance still allowed", allowed2)
test("Second instance on free tier", tier2 == "free")

# Fire 200 on second instance too
for i in range(200):
    allowed, tier, _ = tracker.check_allowance("stress_key", "second_app")
    tracker.record_query("stress_key", "second_app", tier)

allowed3, _, msg3 = tracker.check_allowance("stress_key", "second_app")
test("Second instance also blocked at 200", not allowed3)

account = tracker.get_account("stress_key")
test("400 total free used", account["free_tier"]["used"] == 400)
test("600 free remaining", account["free_tier"]["remaining"] == 600)

# --- Buy queries and verify unblock ---
print("\nPurchasing 1 query pack ($2/500)...")
result = tracker.add_paid_queries("stress_key", packs=1, stripe_payment_id="pi_stress_test")
test("Purchase succeeded", result["paid_tier"]["balance"] == 500)

# Now the blocked instance should work on paid tier
allowed4, tier4, msg4 = tracker.check_allowance("stress_key", "my_app")
test("Instance unblocked after purchase", allowed4)
test("Bills to paid tier", tier4 == "paid")

# Fire 10 paid queries
for i in range(10):
    allowed, tier, _ = tracker.check_allowance("stress_key", "my_app")
    tracker.record_query("stress_key", "my_app", tier)

account = tracker.get_account("stress_key")
test("Paid balance = 490", account["paid_tier"]["balance"] == 490)
test("Paid used = 10", account["paid_tier"]["total_used"] == 10)
test("Instance total = 210", account["instances"]["my_app"]["queries_used"] == 210)

# --- Usage stats ---
print("\nChecking usage log...")
stats = tracker.get_usage_stats("stress_key", last_n=5)
test("Recent queries logged", len(stats["recent_queries"]) == 5)
test("Free tier counted", stats["totals_by_tier"].get("free", 0) == 400)
test("Paid tier counted", stats["totals_by_tier"].get("paid", 0) == 10)

# Cleanup
try:
    os.unlink(tmp.name)
except (PermissionError, OSError):
    pass

# --- Summary ---
print(f"\n{'=' * 60}")
total = passed + failed
print(f"RESULTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{failed} TESTS FAILED")
print("=" * 60)
