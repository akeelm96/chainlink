# ChainLink Memory

**Chain-aware memory for AI agents.** Find connections your vector search misses.

## Install

```bash
pip install chainlink-memory
```

## Quick Start

```python
from chainlink_memory import ChainLink

memory = ChainLink()

memory.add("User loves Thai food")
memory.add("User has a severe shellfish allergy")
memory.add("Thai curries often contain shrimp paste")

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

ChainLink adds neighborhood expansion and LLM chain reasoning on top of vector search. The result: **96.9% chain recall** vs 68.8% for vector search alone.

## Requirements

- Python 3.9+
- Anthropic API key (for chain reasoning via Claude Haiku)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## API

### `ChainLink(api_key=None, persist_path=None, model="claude-haiku-4-5-20251001")`

Create a memory store with chain reasoning.

### `memory.add(text, metadata=None) -> int`

Store a memory. Returns the memory ID.

### `memory.query(query, top_k=5) -> List[QueryResult]`

Find relevant memories including chain connections. Each result has `text`, `score`, `reason`, `is_chain`, and `source`.

### `memory.add_many(texts)`, `memory.remove(id)`, `memory.clear()`, `memory.count()`, `memory.get_all()`

Manage stored memories.

## Persistence

```python
memory = ChainLink(persist_path="./memories.json")
memory.add("remembers across restarts")
```

## MCP Server

ChainLink ships as an MCP server for Claude Code, Cursor, Windsurf, and other AI tools.

**Install:**
```bash
pip install chainlink-memory[mcp]
```

**Claude Code** — add to `~/.claude.json` or project `.mcp.json`:
```json
{
    "mcpServers": {
        "chainlink-memory": {
            "command": "chainlink-mcp",
            "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}
        }
    }
}
```

**Cursor** — add to `.cursor/mcp.json`:
```json
{
    "mcpServers": {
        "chainlink-memory": {
            "command": "chainlink-mcp",
            "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}
        }
    }
}
```

Once configured, AI assistants get these tools:

- `store_memory` — Save a fact, preference, or context
- `query_memory` — Search with chain reasoning
- `store_memories` — Batch store
- `list_memories` — View all memories
- `remove_memory` — Delete by ID
- `memory_stats` — Check status

Memories persist to `~/.chainlink/memories.json` by default.

## How It Works

1. **Embed** all memories using sentence-transformers
2. **Vector search** finds directly similar candidates
3. **Neighborhood expansion** finds memories similar to those candidates
4. **LLM chain reasoning** scores all candidates for indirect relevance
5. **Returns** ranked results with explanations

The neighborhood expansion step is the key. When you search for "Friday dinner", vector search finds "Thai restaurant on Friday". Neighborhood expansion then finds "shrimp paste" because it's close to "Thai restaurant" in embedding space — even though "shrimp paste" is invisible to the original query.

## Performance

| Method | Chain Recall |
|--------|-------------|
| Vector search alone | 68.8% |
| + Neighborhood expansion | 77.2% |
| + LLM chain reasoning | **96.9%** |

Cost: ~$0.001 per query using Claude Haiku.

## License

MIT

---

Built by Akeel Mohammed — akeel.m96@gmail.com
