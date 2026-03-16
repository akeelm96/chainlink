"""
ChainLink API
=============
One endpoint. Send memories + query, get chain connections back.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/v1/connections \
      -H "Authorization: Bearer cl_test_key" \
      -H "Content-Type: application/json" \
      -d '{"query": "anything to worry about?", "memories": ["...", "..."]}'
"""

import os
import time
import uuid
import hashlib
import json
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from chainlink_memory.engine import ChainEngine
from chainlink_memory.usage import (
    UsageTracker, FREE_TOTAL_QUERIES, FREE_QUERIES_PER_INSTANCE,
    FREE_MAX_INSTANCES, PAID_PACK_SIZE, PAID_PACK_PRICE_CENTS
)

# --- App ---
app = FastAPI(
    title="ChainLink",
    description="Find implicit connections between memories that vector search misses.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_KEYS_FILE = Path("api_keys.json")
DB_PATH = os.environ.get("CHAINLINK_DB_PATH", "chainlink_usage.db")

# --- Engine (lazy init) ---
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = ChainEngine(anthropic_api_key=ANTHROPIC_API_KEY)
    return _engine

# --- Usage Tracker (lazy init) ---
_tracker = None

def get_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker(db_path=DB_PATH)
    return _tracker


# --- API Key Management ---
def load_api_keys() -> dict:
    """Load API keys from file. Format: {key_hash: {name, created, active}}"""
    if API_KEYS_FILE.exists():
        return json.loads(API_KEYS_FILE.read_text())
    return {}

def save_api_keys(keys: dict):
    API_KEYS_FILE.write_text(json.dumps(keys, indent=2, default=str))

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def validate_api_key(authorization: str) -> Optional[str]:
    """Validate Bearer token. Returns key hash or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:].strip()
    h = hash_key(token)
    keys = load_api_keys()
    if h in keys and keys[h].get("active", True):
        return h
    return None


# --- Request / Response Models ---
class ConnectionsRequest(BaseModel):
    query: str = Field(..., description="What to search for connections about")
    memories: List[str] = Field(..., description="List of memory texts to search through")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    include_reasons: bool = Field(True, description="Include LLM chain reasoning explanations")
    instance_id: str = Field("default", description="Instance/app identifier for usage tracking")

class Connection(BaseModel):
    text: str
    score: float
    chain_reason: str = ""
    source: str = ""
    is_chain: bool = False

class ConnectionsResponse(BaseModel):
    connections: List[Connection]
    chains_found: int
    total_candidates: int
    query: str
    latency_ms: float
    approx_tokens: int

class UsageResponse(BaseModel):
    plan: str
    free_queries_used: int
    free_queries_limit: int
    free_queries_remaining: int
    paid_balance: int
    total_queries: int
    instances: dict
    instance_count: int
    max_instances: int

class PurchaseRequest(BaseModel):
    packs: int = Field(1, ge=1, le=100, description="Number of query packs to purchase ($2/500 queries)")
    stripe_payment_id: str = Field("", description="Stripe payment intent ID for verification")


# --- Endpoints ---

@app.post("/v1/connections", response_model=ConnectionsResponse)
async def find_connections(
    req: ConnectionsRequest,
    authorization: str = Header(None)
):
    """
    Find implicit chain connections between memories.

    Send a query and a list of memories. Returns ranked results
    including indirect connections that vector search would miss.

    Rate limits: 200 queries per instance, 1000 total free. Then $2/500 queries.
    """
    # Auth
    key_hash = validate_api_key(authorization)
    if key_hash is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use 'Authorization: Bearer <key>'"
        )

    # Check usage allowance
    tracker = get_tracker()
    allowed, tier, message = tracker.check_allowance(key_hash, req.instance_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": message,
                "upgrade_url": "/v1/purchase",
            }
        )

    # Validate
    if not req.memories:
        raise HTTPException(status_code=400, detail="memories list cannot be empty")
    if len(req.memories) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 memories per request")
    if len(req.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")

    # Run engine
    engine = get_engine()
    result = engine.find_connections(
        query=req.query,
        memories=req.memories,
        top_k=req.top_k,
    )

    # Strip reasons if not requested
    if not req.include_reasons:
        for c in result["connections"]:
            c["chain_reason"] = ""

    # Record usage
    tracker.record_query(
        key_hash=key_hash,
        instance_id=req.instance_id,
        tier=tier,
        query_preview=req.query,
        n_memories=len(req.memories),
        latency_ms=result["latency_ms"],
    )

    return result


@app.get("/v1/usage", response_model=UsageResponse)
async def get_usage(authorization: str = Header(None)):
    """Check your current usage and remaining queries."""
    key_hash = validate_api_key(authorization)
    if key_hash is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    tracker = get_tracker()
    account = tracker.get_account(key_hash)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")

    return UsageResponse(
        plan=account["plan"],
        free_queries_used=account["free_tier"]["used"],
        free_queries_limit=account["free_tier"]["limit"],
        free_queries_remaining=account["free_tier"]["remaining"],
        paid_balance=account["paid_tier"]["balance"],
        total_queries=account["total_queries"],
        instances=account["instances"],
        instance_count=account["instance_count"],
        max_instances=account["max_instances"],
    )


@app.post("/v1/purchase")
async def purchase_queries(
    req: PurchaseRequest,
    authorization: str = Header(None)
):
    """
    Purchase query packs. $2 per 500 queries.

    In production, this validates a Stripe payment intent.
    For now, accepts purchases directly (integrate Stripe before launch).
    """
    key_hash = validate_api_key(authorization)
    if key_hash is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    tracker = get_tracker()
    account = tracker.add_paid_queries(
        key_hash=key_hash,
        packs=req.packs,
        stripe_payment_id=req.stripe_payment_id,
    )

    total_cost = req.packs * PAID_PACK_PRICE_CENTS / 100
    total_queries = req.packs * PAID_PACK_SIZE

    return {
        "message": f"Added {total_queries} queries for ${total_cost:.2f}",
        "queries_added": total_queries,
        "amount_charged": f"${total_cost:.2f}",
        "new_paid_balance": account["paid_tier"]["balance"],
        "plan": account["plan"],
    }


@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


# --- Admin: Key Management ---

ADMIN_SECRET = os.environ.get("CHAINLINK_ADMIN_SECRET", "admin_change_me")

@app.post("/admin/keys")
async def create_api_key(
    name: str = "default",
    authorization: str = Header(None)
):
    """Create a new API key. Requires admin auth."""
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=403, detail="Admin access required")

    raw_key = f"cl_{uuid.uuid4().hex}"
    h = hash_key(raw_key)

    keys = load_api_keys()
    keys[h] = {
        "name": name,
        "created": datetime.utcnow().isoformat(),
        "active": True,
    }
    save_api_keys(keys)

    # Register in usage tracker
    tracker = get_tracker()
    tracker.register_key(h, name=name)

    return {
        "api_key": raw_key,
        "name": name,
        "free_queries": FREE_TOTAL_QUERIES,
        "instances_allowed": FREE_MAX_INSTANCES,
        "queries_per_instance": FREE_QUERIES_PER_INSTANCE,
        "message": "Save this key — it won't be shown again."
    }


@app.get("/admin/usage")
async def admin_usage(authorization: str = Header(None)):
    """View all usage stats. Requires admin auth."""
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=403, detail="Admin access required")

    keys = load_api_keys()
    tracker = get_tracker()

    accounts = []
    for h, info in keys.items():
        account = tracker.get_account(h)
        if account:
            accounts.append({
                "name": info.get("name", ""),
                "key_prefix": h[:12],
                "plan": account["plan"],
                "free_used": account["free_tier"]["used"],
                "paid_balance": account["paid_tier"]["balance"],
                "total_queries": account["total_queries"],
                "instances": account["instance_count"],
            })

    return {
        "total_keys": len(keys),
        "accounts": accounts,
    }


# --- Landing Page ---
@app.get("/", response_class=HTMLResponse)
async def landing():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChainLink — Find connections your vector search misses</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0a; color: #e0e0e0; line-height: 1.6; }
  .hero { max-width: 720px; margin: 0 auto; padding: 80px 24px; }
  h1 { font-size: 2.4rem; font-weight: 700; color: #fff; margin-bottom: 12px; }
  .tagline { font-size: 1.2rem; color: #888; margin-bottom: 48px; }
  .highlight { color: #4ade80; }
  .stat { display: inline-block; background: #1a1a1a; border: 1px solid #333;
          border-radius: 8px; padding: 16px 24px; margin: 8px 8px 8px 0; }
  .stat-num { font-size: 1.8rem; font-weight: 700; color: #4ade80; }
  .stat-label { font-size: 0.85rem; color: #888; }
  .code-block { background: #111; border: 1px solid #333; border-radius: 8px;
                padding: 24px; margin: 32px 0; overflow-x: auto; font-size: 0.9rem; }
  code { font-family: 'SF Mono', 'Fira Code', monospace; color: #e0e0e0; }
  .kw { color: #c084fc; }
  .str { color: #4ade80; }
  .cmt { color: #666; }
  .section { margin: 48px 0; }
  h2 { font-size: 1.4rem; color: #fff; margin-bottom: 16px; }
  p { color: #aaa; margin-bottom: 16px; }
  .how-step { display: flex; gap: 16px; margin: 16px 0; align-items: flex-start; }
  .step-num { background: #4ade80; color: #000; font-weight: 700; border-radius: 50%;
              min-width: 28px; height: 28px; display: flex; align-items: center;
              justify-content: center; font-size: 0.85rem; margin-top: 2px; }
  .step-text { color: #ccc; }
  .cta { display: inline-block; background: #4ade80; color: #000; font-weight: 600;
         padding: 12px 32px; border-radius: 6px; text-decoration: none;
         margin-top: 24px; font-size: 1rem; }
  .cta:hover { background: #22c55e; }
  .footer { margin-top: 80px; padding-top: 24px; border-top: 1px solid #222;
            color: #555; font-size: 0.85rem; }
  .pricing-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 24px 0; }
  .price-card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 24px; }
  .price-card.featured { border-color: #4ade80; }
  .price-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .price-amount { font-size: 2rem; font-weight: 700; color: #fff; margin: 8px 0; }
  .price-detail { font-size: 0.9rem; color: #aaa; }
</style>
</head>
<body>
<div class="hero">
  <h1>Chain<span class="highlight">Link</span></h1>
  <p class="tagline">Find connections your vector search misses.<br>
     One API call. Multi-hop chain reasoning for AI agents.</p>

  <div>
    <div class="stat"><div class="stat-num">93.3%</div><div class="stat-label">chain recall</div></div>
    <div class="stat"><div class="stat-num">68.8%</div><div class="stat-label">vector search alone</div></div>
    <div class="stat"><div class="stat-num">~$0.001</div><div class="stat-label">per query cost</div></div>
  </div>

  <div class="section">
    <h2>The problem</h2>
    <p>Your agent stores hundreds of memories. A user asks about Friday's dinner.
       Vector search finds "Thai restaurant on Friday" — but misses that
       "Thai curries use shrimp paste" connects to "severe shellfish allergy."
       That's the chain that matters. That's what ChainLink finds.</p>
  </div>

  <div class="section">
    <h2>How it works</h2>
    <div class="how-step"><div class="step-num">1</div><div class="step-text">You send memories + a query</div></div>
    <div class="how-step"><div class="step-num">2</div><div class="step-text">We find vector-similar candidates, then expand to their neighbors</div></div>
    <div class="how-step"><div class="step-num">3</div><div class="step-text">An LLM reasons about multi-hop chains between candidates</div></div>
    <div class="how-step"><div class="step-num">4</div><div class="step-text">You get ranked results with chain explanations</div></div>
  </div>

  <div class="code-block"><code><span class="cmt"># pip install chainlink-memory</span>
<span class="kw">from</span> chainlink_memory <span class="kw">import</span> ChainLink

memory = ChainLink()
memory.add(<span class="str">"User loves Thai food"</span>)
memory.add(<span class="str">"Thai curries often contain shrimp paste"</span>)
memory.add(<span class="str">"User has a severe shellfish allergy"</span>)

results = memory.query(<span class="str">"plan Friday dinner"</span>)
<span class="cmt"># Finds: Thai food → shrimp paste → shellfish allergy</span>
<span class="cmt"># Vector search alone would miss this chain</span></code></div>

  <div class="section">
    <h2>Pricing</h2>
    <div class="pricing-grid">
      <div class="price-card">
        <div class="price-label">Free tier</div>
        <div class="price-amount">$0</div>
        <div class="price-detail">1,000 queries to start<br>200 per app instance<br>Up to 5 instances</div>
      </div>
      <div class="price-card featured">
        <div class="price-label">Pay as you go</div>
        <div class="price-amount">$2<span style="font-size:1rem;color:#888">/500 queries</span></div>
        <div class="price-detail">$0.004 per query<br>Unlimited instances<br>No subscription needed</div>
      </div>
    </div>
  </div>

  <a class="cta" href="/docs">API Docs</a>

  <div class="footer">
    <p>ChainLink — built by Akeel Mohammed</p>
    <p><a href="/docs" style="color: #4ade80;">API Docs</a> &nbsp;|&nbsp;
       <a href="/v1/health" style="color: #4ade80;">Status</a> &nbsp;|&nbsp;
       <a href="mailto:akeel.m96@gmail.com" style="color: #4ade80;">Contact</a></p>
  </div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
