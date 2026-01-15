"""
Runs hybrid recall and reranking to return the most relevant chunks.
"""

from typing import List, Dict, Any

from agent.retrieval.hybrid_search import HybridSearch
from agent.retrieval.reranker import Reranker
from agent.config import CONFIG


class RetrievalAgent:
    def __init__(self) -> None:
        self.recall = HybridSearch()
        self.reranker = Reranker()

        cfg = CONFIG.get("retrieval", {})
        self.top_k = cfg.get("top_k", 5)
        self.threshold = cfg.get("reranker_threshold", 0.5)

    def search(
        self,
        query: str,
        user_id: str,
        project_id: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank document chunks for a given query.
        """
        query = (query or "").strip()
        if len(query) < 2:
            return []

        # Hybrid recall
        try:
            candidates = self.recall.run(query, user_id, project_id)
        except Exception:
            return []

        if not candidates:
            return []

        # Cross‑encoder reranking
        try:
            ranked = self.reranker.rerank(query, candidates)
        except Exception:
            ranked = candidates

        k = top_k or self.top_k

        # Apply score threshold
        filtered = [
            item for item in ranked
            if item.get("score", 0.0) >= self.threshold
        ]

        if not filtered:
            return ranked[:k]

        return filtered[:k]
