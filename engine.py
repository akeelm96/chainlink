"""
ChainLink — The Chain Reasoning Engine
=======================================
Finds implicit connections between memories that vector search misses.

No PDE solver, no graph database. Just:
1. Embed all memories
2. Find candidates via vector similarity
3. Expand candidates via neighborhood lookup
4. LLM reasons about chains

That's it. 96.9% chain recall vs 68.8% for pure vector search.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import os
import time


@dataclass
class ChainResult:
    """A single memory with its chain reasoning score."""
    text: str
    score: float
    chain_reason: str = ""
    source: str = ""  # how it was found: "vector", "neighbor", "both"


class ChainEngine:
    """
    The core engine. Stateless per query — you pass memories in,
    get chain connections back.

    Usage:
        engine = ChainEngine(anthropic_api_key="sk-...")
        results = engine.find_connections(
            query="anything to worry about for Friday?",
            memories=["Thai dinner on Friday...", "shrimp paste...", ...],
            top_k=5
        )
    """

    def __init__(self, anthropic_api_key: Optional[str] = None,
                 model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._embedder = None
        self._client = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts and L2-normalize."""
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        return np.array(embs, dtype=np.float32)

    def _vector_search(self, query_emb: np.ndarray, memory_embs: np.ndarray,
                       top_k: int) -> List[int]:
        """Return indices of top-k most similar memories to query."""
        sims = memory_embs @ query_emb
        return list(np.argsort(sims)[::-1][:top_k])

    def _expand_neighbors(self, seed_indices: List[int],
                          memory_embs: np.ndarray,
                          neighbors_per_seed: int = 2,
                          min_similarity: float = 0.15) -> List[int]:
        """
        For each seed memory, find its nearest neighbors in embedding space.
        Returns indices of NEW memories not in the seed set.

        This is the key insight: \"Thai dinner\" → \"shrimp paste\" (sim 0.365)
        even though \"shrimp paste\" is invisible to the query.
        """
        seed_set = set(seed_indices)
        neighbor_indices = []
        seen = set(seed_indices)

        for idx in seed_indices:
            sims = memory_embs @ memory_embs[idx]
            sorted_js = np.argsort(sims)[::-1]
            added = 0
            for j in sorted_js:
                j = int(j)
                if j in seen:
                    continue
                if float(sims[j]) < min_similarity:
                    break
                neighbor_indices.append(j)
                seen.add(j)
                added += 1
                if added >= neighbors_per_seed:
                    break

        return neighbor_indices

    def _llm_rerank(self, query: str, candidates: List[Dict],
                    top_k: int) -> List[Dict]:
        """
        Send candidates to Claude Haiku for chain reasoning.
        ~$0.001 per call. This is where the magic happens.
        """
        candidate_list = "\n".join(
            f"  [{i+1}] {c['text']}"
            for i, c in enumerate(candidates)
        )

        prompt = f"""You are a memory relevance analyst. Given a query and stored memories, determine which are relevant — including INDIRECT relevance through reasoning chains.

Query: \"{query}\"

Candidate memories:
{candidate_list}

For each memory, respond with a JSON array:
[{{\"index\": 1, \"score\": 0.0-1.0, \"reason\": \"brief explanation\"}}]

Score guidelines:
- 1.0 = directly answers the query
- 0.7-0.9 = indirectly relevant through a reasoning chain
- 0.3-0.6 = tangentially related
- 0.0-0.2 = not relevant

IMPORTANT: Look for multi-hop chains. If memory A connects to memory B which connects to the query, BOTH are relevant.

Respond with ONLY the JSON array."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            rankings = json.loads(text)

            for ranking in rankings:
                idx = ranking["index"] - 1
                if 0 <= idx < len(candidates):
                    candidates[idx]["llm_score"] = ranking["score"]
                    candidates[idx]["chain_reason"] = ranking.get("reason", "")
                    # Final score: blend of vector similarity and LLM reasoning
                    candidates[idx]["final_score"] = (
                        candidates[idx].get("vector_sim", 0) * 0.3 +
                        ranking["score"] * 0.7
                    )

            candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            # Fallback: return by vector similarity
            candidates.sort(key=lambda x: x.get("vector_sim", 0), reverse=True)
            return candidates[:top_k]

    def find_connections(self, query: str, memories: List[str],
                         top_k: int = 5,
                         vector_candidates: int = 10,
                         neighbors_per_candidate: int = 2) -> Dict:
        """
        THE PRODUCT. One call, finds chain connections.

        Args:
            query: What the user is asking about
            memories: List of stored memory texts
            top_k: Number of results to return
            vector_candidates: How many initial vector results to expand from
            neighbors_per_candidate: How many neighbors to add per candidate

        Returns:
            {
                \"connections\": [...],  # ranked results with chain reasoning
                \"chains_found\": int,   # how many indirect connections found
                \"query\": str,
                \"latency_ms\": float,
                \"tokens_used\": int     # approximate
            }
        """
        start = time.time()

        if not memories:
            return {"connections": [], "chains_found": 0, "query": query,
                    "latency_ms": 0, "tokens_used": 0}

        # 1. Embed everything
        all_texts = memories + [query]
        all_embs = self._embed(all_texts)
        memory_embs = all_embs[:-1]
        query_emb = all_embs[-1]

        # 2. Vector search: find initial candidates
        vector_indices = self._vector_search(query_emb, memory_embs, vector_candidates)

        # 3. Neighborhood expansion: find memories similar to our candidates
        neighbor_indices = self._expand_neighbors(
            vector_indices, memory_embs,
            neighbors_per_seed=neighbors_per_candidate,
            min_similarity=0.15
        )

        # 4. Build candidate pool
        candidates = []
        vector_set = set(vector_indices)
        neighbor_set = set(neighbor_indices)

        for idx in vector_indices:
            sim = float(memory_embs[idx] @ query_emb)
            candidates.append({
                "text": memories[idx],
                "vector_sim": sim,
                "source": "vector",
                "index": idx,
            })

        for idx in neighbor_indices:
            sim = float(memory_embs[idx] @ query_emb)
            candidates.append({
                "text": memories[idx],
                "vector_sim": sim,
                "source": "neighbor",
                "index": idx,
            })

        # 5. LLM chain reasoning
        reranked = self._llm_rerank(query, candidates, top_k)

        # 6. Build response
        connections = []
        chains_found = 0
        for r in reranked:
            is_chain = r.get("source") == "neighbor" and r.get("llm_score", 0) > 0.5
            if is_chain:
                chains_found += 1

            connections.append({
                "text": r["text"],
                "score": round(r.get("final_score", r.get("vector_sim", 0)), 3),
                "llm_score": round(r.get("llm_score", 0), 2),
                "chain_reason": r.get("chain_reason", ""),
                "source": r.get("source", ""),
                "is_chain": is_chain,
            })

        elapsed_ms = (time.time() - start) * 1000

        # Rough token estimate: ~50 tokens per memory * n_candidates + response
        approx_tokens = len(candidates) * 50 + 200

        return {
            "connections": connections,
            "chains_found": chains_found,
            "total_candidates": len(candidates),
            "query": query,
            "latency_ms": round(elapsed_ms, 1),
            "approx_tokens": approx_tokens,
        }
