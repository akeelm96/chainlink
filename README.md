# ChainLink

**Chain-aware memory for AI agents.** Find connections your vector search misses.

`pip install chainlink-memory`

## Quick Start

```python
from chainlink import ChainLink

memory = ChainLink()

# Store memories
memory.add("User loves Thai food")
memory.add("User has a severe shellfish allergy")
memory.add("Thai curries often contain shrimp paste")

# Query — finds chains vector search misses
results = memory.query("plan Friday dinner")
for r in results:
    print(f"[{r.score}] {r.text}")
    if r.is_chain:
        print(f"  Chain: {r.reason}")
```

**Output:**
```
[0.95] User loves Thai food
[0.91] Thai curries often contain shrimp paste
  Chain: Thai food contains shrimp paste, dangerous for shellfish allergy
[0.88] User has a severe shellfish allergy
  Chain: Shellfish allergy is directly relevant to Thai food dinner plan
```

Vector search alone would miss the shrimp paste connection. ChainLink catches it.

## Why ChainLink?

Your AI agent stores memories. When a user asks a question, you search for relevant ones. But vector search only finds **directly** similar memories. It misses **indirect** connections — the ones that require reasoning across multiple hops.

ChainLink adds neighborhood expansion and LLM chain reasoning on top of vector search. The result: 96.9% chain recall vs 68.8% for vector search alone.

## Installation

```bash
pip install chainlink-memory
```

Requires an Anthropic API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## API Reference

### `ChainLink(api_key=None, persist_path=None, model="claude-haiku-4-5-20251001")`

Create a memory store with chain reasoning.

- `api_key`: Anthropic API key (or set `ANTHROPIC_API_KEY` env var)
- `persist_path`: Save memories to disk as JSON (optional)
- `model`: LLM model for chain reasoning

### `memory.add(text, metadata=None) -> int`

Store a memory. Returns the memory ID.

```python
memory.add("User prefers window seats")
memory.add("Meeting at 3pm", metadata={"type": "calendar"})
```

### `memory.add_many(texts) -> List[int]`

Store multiple memories at once.

```python
memory.add_many(["fact 1", "fact 2", "fact 3"])
```

### `memory.query(query, top_k=5) -> List[QueryResult]`

Find relevant memories including chain connections.

```python
results = memory.query("anything to worry about?")
for r in results:
    r.text       # the memory text
    r.score      # relevance score 0-1
    r.reason     # chain reasoning explanation
    r.is_chain   # True if found via chain (not direct vector match)
    r.source     # "vector" or "chain"
```

### `memory.remove(id)`, `memory.clear()`, `memory.count()`, `memory.get_all()`

Manage stored memories.

## Persistence

Save memories across sessions:

```python
memory = ChainLink(persist_path="./memories.json")
memory.add("remembers across restarts")
```

## How It Works

1. **Embed** all memories using sentence-transformers
2. **Vector search** finds directly similar candidates
3. **Neighborhood expansion** finds memories similar to those candidates (the key insight)
4. **LLM chain reasoning** scores all candidates for indirect relevance
5. **Returns** ranked results with explanations

The neighborhood expansion step is what makes it work. When you search for "Friday dinner", vector search finds "Thai restaurant on Friday". Neighborhood expansion then finds "shrimp paste" because it's close to "Thai restaurant" in embedding space — even though "shrimp paste" is invisible to the original query.

## Self-Hosted API Server

ChainLink also includes a FastAPI server if you want to run it as a service:

```bash
pip install chainlink-memory[server]
export ANTHROPIC_API_KEY="sk-ant-..."
export CHAINLINK_ADMIN_SECRET="your_secret"
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints: `POST /v1/connections`, `GET /v1/health`, `POST /admin/keys`, `GET /admin/usage`

## Docker

```bash
docker build -t chainlink .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="sk-ant-..." chainlink
```

## Performance

| Method | Chain Recall |
|--------|-------------|
| Vector search alone | 68.8% |
| + Neighborhood expansion | 77.2% |
| + LLM chain reasoning | 96.9% |

Cost: ~$0.001 per query using Claude Haiku.

## Use Cases

- **Personal AI assistants** — catch safety-critical connections (allergies, medications, schedules)
- **Customer support agents** — link customer history to current issues
- **Knowledge management** — surface hidden connections in notes and documents
- **Due diligence tools** — find non-obvious risk connections across data sources

## License

MIT

---

Built by Akeel Mohammed — akeel.m96@gmail.com
