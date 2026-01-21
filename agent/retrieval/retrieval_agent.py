"""
Runs hybrid recall and reranking to return the most relevant chunks.
"""

from typing import List, Dict, Any
import logging

from agent.retrieval.hybrid_search import HybridSearch
from agent.retrieval.reranker import Reranker
from agent.config import CONFIG

logger = logging.getLogger(__name__)


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

        # Validate tenant inputs early
        if not isinstance(user_id, str) or not user_id.strip():
            logger.warning("RetrievalAgent.search called with invalid user_id")
            return []
        if not isinstance(project_id, str) or not project_id.strip():
            logger.warning("RetrievalAgent.search called with invalid project_id")
            return []

        # Hybrid recall
        try:
            candidates = self.recall.run(query, user_id, project_id)
        except Exception as exc:
            logger.exception("Hybrid recall failed: %s", exc)
            return []

        if not candidates:
            return []

        # Cross‑encoder reranking
        try:
            ranked = self.reranker.rerank(query, candidates)
        except Exception as exc:
            logger.exception("Reranker failed, returning unranked candidates: %s", exc)
            ranked = candidates

        # Determine k safely
        try:
            k = int(top_k) if top_k is not None else int(self.top_k)
            if k <= 0:
                k = int(self.top_k)
        except Exception:
            k = int(self.top_k)

        # Ensure threshold is numeric
        try:
            threshold = float(self.threshold)
        except Exception:
            threshold = 0.5

        # Apply score threshold
        filtered = [
            item for item in ranked
            if item.get("score", 0.0) >= threshold
        ]

        # If threshold filters everything out, return top-k from ranked list
        if not filtered:
            result = ranked[:k]
        else:
            result = filtered[:k]

        logger.debug(
            "RetrievalAgent.search results: query=%s user=%s project=%s candidates=%d returned=%d",
            query, user_id, project_id, len(candidates), len(result)
        )

        return result
