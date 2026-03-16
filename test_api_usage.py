"""
Test API endpoints for usage tracking and rate limiting.
Uses httpx ASGITransport to test without starting a server.
"""

import os
import sys
import json
import tempfile
import hashlib

sys.path.insert(0, os.path.dirname(__file__))

# Set up temp DB before importing app
tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp_db.close()
os.environ["CHAINLINK_DB_PATH"] = tmp_db.name
os.environ["CHAINLINK_ADMIN_SECRET"] = "test_admin_secret"

import httpx
import asyncio
from api import app, hash_key, load_api_keys, save_api_keys, API_KEYS_FILE, get_tracker

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


async def run_tests():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        print("\n--- API Test 1: Create Key (Admin) ---")
        r = await client.post(
            "/admin/keys?name=test_user",
            headers={"Authorization": "Bearer test_admin_secret"}
        )
        test("Key created (200)", r.status_code == 200)
        data = r.json()
        api_key = data["api_key"]
        test("Key starts with cl_", api_key.startswith("cl_"))
        test("Free queries shown", data["free_queries"] == 1000)
        test("Instances allowed shown", data["instances_allowed"] == 5)

        print("\n--- API Test 2: Check Usage (Empty) ---")
        r = await client.get(
            "/v1/usage",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        test("Usage endpoint works", r.status_code == 200)
        usage = r.json()
        test("Plan is free", usage["plan"] == "free")
        test("0 queries used", usage["free_queries_used"] == 0)
        test("1000 remaining", usage["free_queries_remaining"] == 1000)

        print("\n--- API Test 3: Purchase Queries ---")
        r = await client.post(
            "/v1/purchase",
            json={"packs": 2, "stripe_payment_id": "pi_test"},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        test("Purchase works", r.status_code == 200)
        pdata = r.json()
        test("1000 queries added", pdata["queries_added"] == 1000)
        test("$4.00 charged", pdata["amount_charged"] == "$4.00")
        test("Balance = 1000", pdata["new_paid_balance"] == 1000)

        print("\n--- API Test 4: Usage After Purchase ---")
        r = await client.get(
            "/v1/usage",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        usage = r.json()
        test("Plan upgraded to paid", usage["plan"] == "paid")
        test("Paid balance = 1000", usage["paid_balance"] == 1000)

        print("\n--- API Test 5: Rate Limit (No Auth) ---")
        r = await client.post("/v1/connections", json={
            "query": "test",
            "memories": ["a", "b"],
        })
        test("No auth = 401", r.status_code == 401)

        print("\n--- API Test 6: Rate Limit (Bad Key) ---")
        r = await client.post("/v1/connections", json={
            "query": "test",
            "memories": ["a", "b"],
        }, headers={"Authorization": "Bearer cl_fake_key"})
        test("Bad key = 401", r.status_code == 401)

        print("\n--- API Test 7: Admin Usage View ---")
        r = await client.get(
            "/admin/usage",
            headers={"Authorization": "Bearer test_admin_secret"}
        )
        test("Admin usage works", r.status_code == 200)
        admin = r.json()
        test("Shows accounts", len(admin["accounts"]) >= 1)

        print("\n--- API Test 8: Health Check ---")
        r = await client.get("/v1/health")
        test("Health ok", r.status_code == 200)
        test("Version 0.2.0", r.json()["version"] == "0.2.0")

    # Cleanup
    try:
        os.unlink(tmp_db.name)
    except (PermissionError, OSError):
        pass


print("=" * 60)
print("CHAINLINK API USAGE INTEGRATION TESTS")
print("=" * 60)

asyncio.run(run_tests())

print(f"\n{'=' * 60}")
total = passed + failed
print(f"RESULTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{failed} TESTS FAILED")
print("=" * 60)
