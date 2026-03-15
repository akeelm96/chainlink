# ChainLink

**Find connections your vector search misses.**

One API call. Multi-hop chain reasoning for AI agents.

## The Problem

Your agent stores hundreds of memories. A user asks about Friday's dinner. Vector search finds "Thai restaurant on Friday" and "shellfish allergy" — but misses "Thai curries use shrimp paste." That's the connection that matters. That's what ChainLink finds.

## How It Works

1. You send memories + a query
2. We find vector-similar candidates, then expand to their neighbors
3. An LLM reasons about multi-hop chains between candidates
4. You get ranked results with chain explanations

## Quick Start

```bash
# Clone and run
git clone https://github.com/akeelm96/chainlink.git
cd chainlink
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."
export CHAINLINK_ADMIN_SECRET="your_admin_secret"

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Usage

```python
import requests

result = requests.post("http://localhost:8000/v1/connections", json={
    "query": "anything to worry about for Friday?",
    "memories": [
        "Thai restaurant dinner on Friday",
        "Thai curries use shrimp paste",
        "I have a severe shellfish allergy",
        "Water cooler delivery on Monday",
    ]
}, headers={"Authorization": "Bearer cl_your_key"})

# Returns the chain vector search would miss:
# [Thai dinner] → [shrimp paste] → [shellfish allergy]
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/connections` | Find chain connections |
| GET | `/v1/health` | Health check |
| POST | `/admin/keys` | Create API key (admin) |
| GET | `/admin/usage` | View usage stats (admin) |
| GET | `/` | Landing page |
| GET | `/docs` | Auto-generated API docs |

## Docker

```bash
docker build -t chainlink .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="sk-ant-..." chainlink
```

## Performance

| Method | Chain Recall |
|--------|-------------|
| Vector search alone | 68.8% |
| ChainLink (vector + expansion + LLM) | 96.9% |

## License

MIT

---

Built by Akeel Mohammed
