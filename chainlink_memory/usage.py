"""
ChainLink Usage Tracking & Rate Limiting
=========================================
SQLite-backed query tracking with per-instance limits and paid tiers.

Free tier: 1000 queries (200 per instance, max 5 instances)
Paid tier: Purchase query packs ($2/500 queries)
"""

import sqlite3
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple


# --- Defaults ---
FREE_QUERIES_PER_INSTANCE = 200
FREE_MAX_INSTANCES = 5
FREE_TOTAL_QUERIES = FREE_QUERIES_PER_INSTANCE * FREE_MAX_INSTANCES  # 1000
PAID_PACK_SIZE = 500
PAID_PACK_PRICE_CENTS = 200  # $2.00


class UsageTracker:
    """
    Tracks query usage per API key with instance-level limits.

    Each API key gets:
    - Up to 5 instances (apps/projects)
    - 200 free queries per instance
    - 1000 total free queries
    - After free tier: must purchase query packs

    Thread-safe via SQLite WAL mode + connection per thread.
    """

    def __init__(self, db_path: str = "chainlink_usage.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
                key_hash TEXT PRIMARY KEY,
                name TEXT DEFAULT '',
                plan TEXT DEFAULT 'free',
                total_free_used INTEGER DEFAULT 0,
                paid_balance INTEGER DEFAULT 0,
                total_paid_used INTEGER DEFAULT 0,
                created_at REAL,
                last_query_at REAL
            );

            CREATE TABLE IF NOT EXISTS instances (
                key_hash TEXT,
                instance_id TEXT,
                queries_used INTEGER DEFAULT 0,
                created_at REAL,
                last_query_at REAL,
                PRIMARY KEY (key_hash, instance_id),
                FOREIGN KEY (key_hash) REFERENCES accounts(key_hash)
            );

            CREATE TABLE IF NOT EXISTS purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT,
                amount_cents INTEGER,
                queries_added INTEGER,
                stripe_payment_id TEXT,
                created_at REAL,
                FOREIGN KEY (key_hash) REFERENCES accounts(key_hash)
            );

            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT,
                instance_id TEXT,
                query_preview TEXT,
                n_memories INTEGER,
                latency_ms REAL,
                tier TEXT,
                created_at REAL
            );
        """)
        conn.commit()

    def register_key(self, key_hash: str, name: str = "") -> Dict:
        """Register a new API key. Returns account info."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR IGNORE INTO accounts
               (key_hash, name, plan, total_free_used, paid_balance, total_paid_used, created_at)
               VALUES (?, ?, 'free', 0, 0, 0, ?)""",
            (key_hash, name, time.time())
        )
        conn.commit()
        return self.get_account(key_hash)

    def get_account(self, key_hash: str) -> Optional[Dict]:
        """Get account info including usage."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM accounts WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if not row:
            return None

        instances = conn.execute(
            "SELECT instance_id, queries_used FROM instances WHERE key_hash = ?",
            (key_hash,)
        ).fetchall()

        free_remaining = max(0, FREE_TOTAL_QUERIES - row["total_free_used"])

        return {
            "key_hash": row["key_hash"],
            "name": row["name"],
            "plan": row["plan"],
            "free_tier": {
                "used": row["total_free_used"],
                "limit": FREE_TOTAL_QUERIES,
                "remaining": free_remaining,
            },
            "paid_tier": {
                "balance": row["paid_balance"],
                "total_purchased": row["total_paid_used"] + row["paid_balance"],
                "total_used": row["total_paid_used"],
            },
            "instances": {
                r["instance_id"]: {
                    "queries_used": r["queries_used"],
                    "limit": FREE_QUERIES_PER_INSTANCE,
                    "remaining": max(0, FREE_QUERIES_PER_INSTANCE - r["queries_used"]),
                }
                for r in instances
            },
            "instance_count": len(instances),
            "max_instances": FREE_MAX_INSTANCES,
            "total_queries": row["total_free_used"] + row["total_paid_used"],
        }

    def check_allowance(self, key_hash: str, instance_id: str = "default") -> Tuple[bool, str, str]:
        """
        Check if a query is allowed. Returns (allowed, tier, message).

        tier is "free" or "paid" — tells you which balance to deduct from.
        """
        conn = self._get_conn()

        # Ensure account exists
        account = conn.execute(
            "SELECT * FROM accounts WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if not account:
            return False, "", "Account not found. Register an API key first."

        # Check instance
        instance = conn.execute(
            "SELECT * FROM instances WHERE key_hash = ? AND instance_id = ?",
            (key_hash, instance_id)
        ).fetchone()

        # New instance?
        if not instance:
            instance_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM instances WHERE key_hash = ?",
                (key_hash,)
            ).fetchone()["cnt"]

            # Check if still in free tier and instance limit reached
            if account["total_free_used"] < FREE_TOTAL_QUERIES:
                if instance_count >= FREE_MAX_INSTANCES:
                    # Can still use paid balance even if instance limit hit
                    if account["paid_balance"] <= 0:
                        return False, "", (
                            f"Free tier instance limit reached ({FREE_MAX_INSTANCES} instances). "
                            f"Purchase a query pack to add more instances."
                        )

            # Create the instance
            conn.execute(
                "INSERT INTO instances (key_hash, instance_id, queries_used, created_at) VALUES (?, ?, 0, ?)",
                (key_hash, instance_id, time.time())
            )
            conn.commit()
            instance = conn.execute(
                "SELECT * FROM instances WHERE key_hash = ? AND instance_id = ?",
                (key_hash, instance_id)
            ).fetchone()

        # Determine which tier to bill
        free_used = account["total_free_used"]
        instance_used = instance["queries_used"]
        paid_balance = account["paid_balance"]

        # Count instances to check if this one is beyond the free cap
        instance_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM instances WHERE key_hash = ?",
            (key_hash,)
        ).fetchone()["cnt"]
        is_extra_instance = instance_count > FREE_MAX_INSTANCES

        # Still has free queries?
        if free_used < FREE_TOTAL_QUERIES and not is_extra_instance:
            # Check per-instance limit
            if instance_used >= FREE_QUERIES_PER_INSTANCE:
                # This instance exhausted, but maybe paid balance?
                if paid_balance > 0:
                    return True, "paid", "Instance free limit reached. Using paid balance."
                return False, "", (
                    f"Instance '{instance_id}' has used all {FREE_QUERIES_PER_INSTANCE} free queries. "
                    f"Purchase a query pack to continue."
                )
            return True, "free", "OK"

        # Free tier exhausted — check paid balance
        if paid_balance > 0:
            return True, "paid", "Using paid balance."

        return False, "", (
            f"Free tier exhausted ({FREE_TOTAL_QUERIES} queries). "
            f"Purchase a query pack to continue. "
            f"$2.00 for {PAID_PACK_SIZE} queries at /v1/purchase."
        )

    def record_query(self, key_hash: str, instance_id: str, tier: str,
                     query_preview: str = "", n_memories: int = 0, latency_ms: float = 0):
        """Record a query and deduct from the appropriate balance."""
        conn = self._get_conn()
        now = time.time()

        if tier == "free":
            conn.execute(
                "UPDATE accounts SET total_free_used = total_free_used + 1, last_query_at = ? WHERE key_hash = ?",
                (now, key_hash)
            )
        elif tier == "paid":
            conn.execute(
                """UPDATE accounts SET
                   paid_balance = paid_balance - 1,
                   total_paid_used = total_paid_used + 1,
                   last_query_at = ?
                   WHERE key_hash = ?""",
                (now, key_hash)
            )

        # Update instance counter
        conn.execute(
            "UPDATE instances SET queries_used = queries_used + 1, last_query_at = ? WHERE key_hash = ? AND instance_id = ?",
            (now, key_hash, instance_id)
        )

        # Log
        conn.execute(
            """INSERT INTO query_log (key_hash, instance_id, query_preview, n_memories, latency_ms, tier, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (key_hash, instance_id, query_preview[:50], n_memories, latency_ms, tier, now)
        )
        conn.commit()

    def add_paid_queries(self, key_hash: str, packs: int = 1,
                         stripe_payment_id: str = "") -> Dict:
        """
        Add purchased query packs to an account.
        Returns updated account info.
        """
        conn = self._get_conn()
        queries_to_add = packs * PAID_PACK_SIZE
        amount_cents = packs * PAID_PACK_PRICE_CENTS

        conn.execute(
            "UPDATE accounts SET paid_balance = paid_balance + ?, plan = 'paid' WHERE key_hash = ?",
            (queries_to_add, key_hash)
        )
        conn.execute(
            """INSERT INTO purchases (key_hash, amount_cents, queries_added, stripe_payment_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (key_hash, amount_cents, queries_to_add, stripe_payment_id, time.time())
        )
        conn.commit()

        # Lift instance limit for paid users
        return self.get_account(key_hash)

    def get_usage_stats(self, key_hash: str, last_n: int = 20) -> Dict:
        """Get usage stats and recent query log."""
        conn = self._get_conn()

        recent = conn.execute(
            """SELECT instance_id, query_preview, n_memories, latency_ms, tier, created_at
               FROM query_log WHERE key_hash = ? ORDER BY created_at DESC LIMIT ?""",
            (key_hash, last_n)
        ).fetchall()

        totals = conn.execute(
            """SELECT tier, COUNT(*) as cnt FROM query_log
               WHERE key_hash = ? GROUP BY tier""",
            (key_hash,)
        ).fetchall()

        return {
            "recent_queries": [dict(r) for r in recent],
            "totals_by_tier": {r["tier"]: r["cnt"] for r in totals},
        }
