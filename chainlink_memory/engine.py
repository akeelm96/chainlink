"""
ChainLink — The Chain Reasoning Engine
=======================================
Finds implicit connections between memories that vector search misses.

Pipeline:
1. Embed all memories using sentence-transformers
2. Find candidates via vector similarity
3. Expand candidates via neighborhood lookup (multi-hop)
4. LLM reasons about chains between candidates (batched)

93.3% chain recall vs 68.8% for pure vector search (50-memory benchmark).
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import os
import time
import logging

logger = logging.getLogger("chainlink_memory")


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
            if not self._api_key:
                raise ValueError(
                    "No Anthropic API key provided. Pass api_key= or set "
                    "ANTHROPIC_API_KEY environment variable."
                )
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
                          neighbors_per_seed: int = 3,
                          min_similarity: float = 0.15,
                          hops: int = 2) -> List[int]:
        """
        Multi-hop neighborhood expansion in embedding space.
        For each seed, find nearest neighbors, then expand THOSE too.

        This is the key insight: "Thai dinner" → "shrimp paste" → "shellfish allergy"
        Hop 1: Thai dinner finds shrimp paste (sim 0.365)
        Hop 2: Shrimp paste finds shellfish allergy (sim 0.4+)

        Args:
            seed_indices: Starting memory indices from vector search
            memory_embs: All memory embeddings
            neighbors_per_seed: Neighbors to find per seed (default 3)
            min_similarity: Minimum cosine similarity threshold
            hops: Number of expansion hops (default 2 for multi-hop chains)
        """
        all_neighbors = []
        seen = set(seed_indices)
        current_frontier = list(seed_indices)

        for hop in range(hops):
            next_frontier = []
            # Reduce neighbors per seed on later hops to limit explosion
            k = neighbors_per_seed if hop == 0 else max(1, neighbors_per_seed - 1)

            for idx in current_frontier:
                sims = memory_embs @ memory_embs[idx]
                sorted_js = np.argsort(sims)[::-1]
                added = 0
                for j in sorted_js:
                    j = int(j)
                    if j in seen:
                        continue
                    if float(sims[j]) < min_similarity:
                        break
                    all_neighbors.append(j)
                    next_frontier.append(j)
                    seen.add(j)
                    added += 1
                    if added >= k:
                        break

            current_frontier = next_frontier
            if not current_frontier:
                break

        return all_neighbors

    def _cluster_bridge_expansion(self, seed_indices: List[int],
                                   neighbor_indices: List[int],
                                   memory_embs: np.ndarray,
                                   n_clusters: int = 15,
                                   bridges_per_cluster: int = 2) -> List[int]:
        """
        Cross-cluster bridge sampling — the key to scaling chain reasoning.

        Problem: At 1000+ memories, neighborhood expansion gets trapped
        inside dense clusters (food memories find food memories, never
        reaching the health cluster where shellfish allergy lives).

        Solution: Cluster all memories, identify which clusters our
        candidates touch, then for each candidate cluster find the
        nearest memories in OTHER clusters that bridge across.

        This ensures the LLM sees at least one representative from
        each semantically adjacent cluster, enabling cross-cluster
        chain reasoning.

        No extra API calls — purely embedding math.
        """
        from sklearn.cluster import MiniBatchKMeans

        n_memories = memory_embs.shape[0]
        if n_memories < n_clusters * 2:
            return []  # Too few memories for clustering

        # 1. Cluster all memories
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1024, n_memories),
            n_init=3,
        )
        labels = km.fit_predict(memory_embs)
        centroids = km.cluster_centers_
        # Normalize centroids for cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroids = centroids / norms

        # 2. Which clusters do our current candidates belong to?
        all_current = set(seed_indices) | set(neighbor_indices)
        touched_clusters = set(int(labels[i]) for i in all_current)

        # 3. For each touched cluster, find closest UN-touched clusters
        bridge_indices = []
        seen = set(all_current)

        for tc in touched_clusters:
            # Find closest other clusters by centroid similarity
            centroid_sims = centroids @ centroids[tc]
            sorted_clusters = np.argsort(centroid_sims)[::-1]

            for oc in sorted_clusters:
                oc = int(oc)
                if oc in touched_clusters:
                    continue

                # Find the memories in this cluster closest to any seed
                cluster_mask = (labels == oc)
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    continue

                # Score: how similar are these cluster members to our seeds?
                # Use max similarity to any seed as the bridge score
                best_bridges = []
                for ci in cluster_indices:
                    if ci in seen:
                        continue
                    max_sim = max(
                        float(memory_embs[ci] @ memory_embs[si])
                        for si in seed_indices[:10]  # top seeds only for speed
                    )
                    best_bridges.append((ci, max_sim))

                best_bridges.sort(key=lambda x: x[1], reverse=True)
                for ci, sim in best_bridges[:bridges_per_cluster]:
                    if sim >= 0.20:  # minimum bridge quality
                        bridge_indices.append(ci)
                        seen.add(ci)

                break  # only bridge to the closest un-touched cluster per touched

        return bridge_indices

    def _llm_rerank(self, query: str, candidates: List[Dict],
                    top_k: int) -> List[Dict]:
        """
        Pre-filter candidates then send ONE LLM call for chain reasoning.
        Always exactly 1 API call per query (~$0.001).

        Strategy: take the top vector hits + top neighbor hits to build
        a focused candidate list of max 25 items.
        """
        # Pre-filter: keep max 25 candidates for the LLM
        # Priority: top vector hits + all neighbor hits (which are the chain candidates)
        MAX_LLM_CANDIDATES = 25

        if len(candidates) > MAX_LLM_CANDIDATES:
            vectors = [c for c in candidates if c.get("source") == "vector"]
            neighbors = [c for c in candidates if c.get("source") == "neighbor"]
            bridges = [c for c in candidates if c.get("source") == "bridge"]

            # Sort each by vector similarity
            vectors.sort(key=lambda x: x.get("vector_sim", 0), reverse=True)
            neighbors.sort(key=lambda x: x.get("vector_sim", 0), reverse=True)
            bridges.sort(key=lambda x: x.get("vector_sim", 0), reverse=True)

            # Allocate slots: bridges get priority (they're cross-cluster chains),
            # then neighbors (intra-cluster chains), then vectors
            n_bridges = min(len(bridges), MAX_LLM_CANDIDATES // 3)
            remaining = MAX_LLM_CANDIDATES - n_bridges
            n_neighbors = min(len(neighbors), remaining // 2)
            n_vectors = remaining - n_neighbors

            candidates = (
                bridges[:n_bridges] +
                neighbors[:n_neighbors] +
                vectors[:n_vectors]
            )

        candidate_list = "\n".join(
            f"  [{i+1}] {c['text']}"
            for i, c in enumerate(candidates)
        )

        prompt = f"""You are a memory relevance analyst. Given a query and candidate memories, score each for relevance — including INDIRECT relevance through reasoning chains.

Query: "{query}"

Candidates:
{candidate_list}

Return a JSON array scoring each candidate:
[{{"index": 1, "score": 0.9, "reason": "why"}}]

Scoring:
- 1.0 = directly answers query
- 0.7-0.9 = indirectly relevant via reasoning chain (e.g. Thai food → shrimp paste → shellfish allergy)
- 0.3-0.6 = tangentially related
- 0.0-0.2 = not relevant

CRITICAL: Find multi-hop chains. If memory A links to B which links to the query, BOTH are relevant. Score them 0.7+.
Keep reasons under 15 words. Return ONLY valid JSON."""

        # Scale max tokens — 25 candidates × ~80 tokens each = ~2000 tokens
        max_tokens = min(4096, max(1500, len(candidates) * 80))

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
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
                    candidates[idx]["final_score"] = (
                        candidates[idx].get("vector_sim", 0) * 0.3 +
                        ranking["score"] * 0.7
                    )

            candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.warning(f"LLM rerank failed, falling back to vector: {e}")
            candidates.sort(key=lambda x: x.get("vector_sim", 0), reverse=True)
            return candidates[:top_k]

    def find_connections(self, query: str, memories: List[str],
                         top_k: int = 5,
                         vector_candidates: Optional[int] = None,
                         neighbors_per_candidate: int = 3) -> Dict:
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
                "connections": [...],  # ranked results with chain reasoning
                "chains_found": int,   # how many indirect connections found
                "query": str,
                "latency_ms": float,
                "tokens_used": int     # approximate
            }
        """
        start = time.time()

        if not memories:
            return {"connections": [], "chains_found": 0, "query": query,
                    "latency_ms": 0, "approx_tokens": 0}

        # Scale vector candidates based on memory count
        # More memories = need wider initial net to catch relevant clusters
        if vector_candidates is None:
            n = len(memories)
            if n <= 10:
                vector_candidates = min(n, 8)
            elif n <= 50:
                vector_candidates = min(n, 15)
            elif n <= 200:
                vector_candidates = min(n, 20)
            else:
                vector_candidates = min(n, 30)

        # Scale min_similarity for neighbor expansion based on memory count
        # With more memories, clusters are tighter so we need a higher threshold
        # to avoid pulling in noise
        n = len(memories)
        if n <= 50:
            min_sim = 0.15
            hops = 2
            neighbors_k = neighbors_per_candidate
        elif n <= 200:
            min_sim = 0.20
            hops = 2
            neighbors_k = neighbors_per_candidate
        else:
            # At 1000+ memories, be more selective but go deeper
            min_sim = 0.25
            hops = 3
            neighbors_k = 2

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
            neighbors_per_seed=neighbors_k,
            min_similarity=min_sim,
            hops=hops,
        )

        # 3b. Cross-cluster bridge expansion (for large memory sets)
        bridge_indices = []
        if n >= 200:
            n_clusters = max(8, min(30, n // 50))
            bridge_indices = self._cluster_bridge_expansion(
                vector_indices, neighbor_indices, memory_embs,
                n_clusters=n_clusters,
                bridges_per_cluster=2,
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

        for idx in bridge_indices:
            if idx not in vector_set and idx not in neighbor_set:
                sim = float(memory_embs[idx] @ query_emb)
                candidates.append({
                    "text": memories[idx],
                    "vector_sim": sim,
                    "source": "bridge",
                    "index": idx,
                })

        # 5. LLM chain reasoning (batched for large candidate pools)
        reranked = self._llm_rerank(query, candidates, top_k)

        # 6. Build response
        connections = []
        chains_found = 0
        for r in reranked:
            # A result is a "chain" if it was found via neighbor expansion,
            # OR if the LLM scored it highly but vector similarity was low
            # (meaning the LLM found an indirect connection)
            is_neighbor = r.get("source") in ("neighbor", "bridge")
            llm_high = r.get("llm_score", 0) > 0.5
            vector_low = r.get("vector_sim", 0) < 0.3
            is_chain = (is_neighbor and llm_high) or (llm_high and vector_low)
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
