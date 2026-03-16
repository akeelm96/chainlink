"""
ChainLink SDK
=============
The simplest way to add chain-aware memory to any AI agent.

Usage:
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
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class Memory:
    """A stored memory with metadata."""
    text: str
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)
    id: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class QueryResult:
    """A single result from a chain query."""
    text: str
    score: float
    reason: str = ""
    source: str = ""  # "vector" or "chain"
    is_chain: bool = False


class ChainLink:
    """
    Chain-aware memory for AI agents.

    Finds implicit connections between memories that vector search misses.
    96.9% chain recall vs 68.8% for pure vector search.

    Args:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        persist_path: Path to save memories to disk (optional)
        model: LLM model for chain reasoning (default: claude-haiku-4-5-20251001)

    Example:
        memory = ChainLink()
        memory.add("User loves Thai food")
        memory.add("User has a severe shellfish allergy")
        memory.add("Thai curries often contain shrimp paste")
        results = memory.query("plan Friday dinner")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        persist_path: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        from chainlink_memory.engine import ChainEngine
        self._engine = ChainEngine(
            anthropic_api_key=api_key,
            model=model,
        )
        self._memories: List[Memory] = []
        self._next_id = 1
        self._persist_path = Path(persist_path) if persist_path else None

        # Load from disk if path exists
        if self._persist_path and self._persist_path.exists():
            self._load()

    def add(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Store a memory. Returns the memory ID.

        Args:
            text: The memory text to store
            metadata: Optional key-value metadata

        Returns:
            Memory ID (int)
        """
        mem = Memory(
            text=text,
            metadata=metadata or {},
            id=self._next_id,
        )
        self._memories.append(mem)
        self._next_id += 1

        if self._persist_path:
            self._save()

        return mem.id

    def add_many(self, texts: List[str]) -> List[int]:
        """Store multiple memories at once. Returns list of memory IDs."""
        return [self.add(text) for text in texts]

    def query(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """
        Find relevant memories including chain connections.

        Args:
            query: What to search for
            top_k: Number of results to return (default 5)

        Returns:
            List of QueryResult with text, score, reason, and is_chain flag
        """
        if not self._memories:
            return []

        texts = [m.text for m in self._memories]
        result = self._engine.find_connections(
            query=query,
            memories=texts,
            top_k=top_k,
        )

        return [
            QueryResult(
                text=c["text"],
                score=c["score"],
                reason=c.get("chain_reason", ""),
                source="chain" if c.get("is_chain") else "vector",
                is_chain=c.get("is_chain", False),
            )
            for c in result["connections"]
        ]

    def search(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Alias for query()."""
        return self.query(query, top_k=top_k)

    def get_all(self) -> List[Memory]:
        """Return all stored memories."""
        return list(self._memories)

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self._memories)

    def remove(self, memory_id: int) -> bool:
        """Remove a memory by ID. Returns True if found and removed."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != memory_id]
        removed = len(self._memories) < before

        if removed and self._persist_path:
            self._save()

        return removed

    def clear(self):
        """Remove all memories."""
        self._memories = []
        self._next_id = 1

        if self._persist_path:
            self._save()

    def _save(self):
        """Save memories to disk as JSON."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self._memories
            ],
        }
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        """Load memories from disk."""
        try:
            data = json.loads(self._persist_path.read_text())
            self._next_id = data.get("next_id", 1)
            self._memories = [
                Memory(
                    id=m["id"],
                    text=m["text"],
                    timestamp=m.get("timestamp", 0),
                    metadata=m.get("metadata", {}),
                )
                for m in data.get("memories", [])
            ]
        except (json.JSONDecodeError, KeyError):
            self._memories = []
            self._next_id = 1

    def __len__(self):
        return len(self._memories)

    def __repr__(self):
        return f"ChainLink(memories={len(self._memories)})"
