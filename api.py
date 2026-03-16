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

# --- App ---
app = FastAPI(
    title="ChainLink",
    description="Find implicit connections between memories that vector search misses.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
USAGE_LOG = Path("usage_log.jsonl")
API_KEYS_FILE = Path("api_keys.json")

# --- Engine (lazy init) ---
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = ChainEngine(anthropic_api_key=ANTHROPIC_API_KEY)
    return _engine


# --- API Key Management ---
def load_api_keys() -> dict:
    """Load API keys from file. Format: {key_hash: {name, created, queries, active}}"""
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

def log_usage(key_hash: str, query: str, n_memories: int,
              n_connections: int, latency_ms: float, approx_tokens: int):
    """Append usage to log file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "key_hash": key_hash[:12],
        "query_preview": query[:50],
        "n_memories": n_memories,
        "n_connections": n_connections,
        "latency_ms": latency_ms,
        "approx_tokens": approx_tokens,
    }
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Update query count for this key
    keys = load_api_keys()
    if key_hash in keys:
        keys[key_hash]["queries"] = keys[key_hash].get("queries", 0) + 1
        keys[key_hash]["last_used"] = datetime.utcnow().isoformat()
        save_api_keys(keys)


# --- Request / Response Models ---
class ConnectionsRequest(BaseModel):
    query: str = Field(..., description="What to search for connections about")
    memories: List[str] = Field(..., description="List of memory texts to search through")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    include_reasons: bool = Field(True, description="Include LLM chain reasoning explanations")

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
    """
    # Auth
    key_hash = validate_api_key(authorization)
    if key_hash is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use 'Authorization: Bearer <key>'"
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

    # Log usage
    log_usage(
        key_hash=key_hash,
        query=req.query,
        n_memories=len(req.memories),
        n_connections=result["chains_found"],
        latency_ms=result["latency_ms"],
        approx_tokens=result["approx_tokens"],
    )

    return result


@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


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
        "queries": 0,
        "active": True,
    }
    save_api_keys(keys)

    return {
        "api_key": raw_key,
        "name": name,
        "message": "Save this key — it won't be shown again."
    }


@app.get("/admin/usage")
async def get_usage(authorization: str = Header(None)):
    """View usage stats. Requires admin auth."""
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=403, detail="Admin access required")

    keys = load_api_keys()
    total_queries = sum(k.get("queries", 0) for k in keys.values())

    recent = []
    if USAGE_LOG.exists():
        lines = USAGE_LOG.read_text().strip().split("\n")
        for line in lines[-20:]:  # last 20
            try:
                recent.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return {
        "total_keys": len(keys),
        "total_queries": total_queries,
        "keys": {h[:12]: {"name": v["name"], "queries": v.get("queries", 0),
                          "active": v.get("active", True)}
                 for h, v in keys.items()},
        "recent_usage": recent,
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
</style>
</head>
<body>
<div class="hero">
  <h1>Chain<span class="highlight">Link</span></h1>
  <p class="tagline">Find connections your vector search misses.<br>
     One API call. Multi-hop chain reasoning for AI agents.</p>

  <div>
    <div class="stat"><div class="stat-num">96.9%</div><div class="stat-label">chain recall</div></div>
    <div class="stat"><div class="stat-num">68.8%</div><div class="stat-label">vector search alone</div></div>
    <div class="stat"><div class="stat-num">~$0.001</div><div class="stat-label">per query</div></div>
  </div>

  <div class="section">
    <h2>The problem</h2>
    <p>Your agent stores hundreds of memories. A user asks about Friday's dinner.
       Vector search finds "Thai restaurant on Friday" and "shellfish allergy" —
       but misses "Thai curries use shrimp paste." That's the connection that matters.
       That's what ChainLink finds.</p>
  </div>

  <div class="section">
    <h2>How it works</h2>
    <div class="how-step"><div class="step-num">1</div><div class="step-text">You send memories + a query</div></div>
    <div class="how-step"><div class="step-num">2</div><div class="step-text">We find vector-similar candidates, then expand to their neighbors</div></div>
    <div class="how-step"><div class="step-num">3</div><div class="step-text">An LLM reasons about multi-hop chains between candidates</div></div>
    <div class="how-step"><div class="step-num">4</div><div class="step-text">You get ranked results with chain explanations</div></div>
  </div>

  <div class="code-block"><code><span class="cmt"># One API call</span>
<span class="kw">import</span> requests

result = requests.post(<span class="str">"https://api.chainlink.dev/v1/connections"</span>, json={
    <span class="str">"query"</span>: <span class="str">"anything to worry about for Friday?"</span>,
    <span class="str">"memories"</span>: [
        <span class="str">"Thai restaurant dinner on Friday"</span>,
        <span class="str">"Thai curries use shrimp paste"</span>,
        <span class="str">"I have a severe shellfish allergy"</span>,
        <span class="str">"Water cooler delivery on Monday"</span>,
        <span class="cmt">// ... hundreds more</span>
    ]
}, headers={<span class="str">"Authorization"</span>: <span class="str">"Bearer cl_your_key"</span>})

<span class="cmt"># Returns the chain vector search would miss:</span>
<span class="cmt"># [Thai dinner] → [shrimp paste] → [shellfish allergy]</span></code></div>

  <div class="section">
    <h2>Pricing</h2>
    <p>Pay per query. No subscriptions, no minimums.</p>
    <div class="stat"><div class="stat-num">1,000</div><div class="stat-label">free queries to start</div></div>
    <div class="stat"><div class="stat-num">$0.002</div><div class="stat-label">per query after</div></div>
  </div>

  <a class="cta" href="mailto:akeel.m96@gmail.com?subject=ChainLink API Access">Get API Access</a>

  <div class="footer">
    <p>ChainLink — built by Akeel Mohammed</p>
    <p><a href="/docs" style="color: #4ade80;">API Docs</a> &nbsp;|&nbsp;
       <a href="/v1/health" style="color: #4ade80;">Status</a></p>
  </div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
