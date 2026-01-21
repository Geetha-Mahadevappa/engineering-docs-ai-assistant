"""
Reranker: assigns a relevance score to each candidate snippet.

This component takes the broad set of results from hybrid recall
and evaluates how well each snippet answers the user's query.
A cross‑encoder reads the query and the text together, which
produces a more accurate relevance score than cosine similarity
or keyword scoring.
"""

from typing import List, Dict, Any
import logging
from sentence_transformers import CrossEncoder
from agent.config import CONFIG

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        """
        Load the cross‑encoder model used for scoring.
        The model compares (query, text) pairs directly and
        returns a single relevance score for each pair.
        """
        model_name = CONFIG.get("reranker", {}).get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score each candidate snippet and return them sorted
        from most relevant to least relevant.

        Final sort order:
        1. cross‑encoder score (primary)
        2. rrf_score from hybrid recall (secondary)
        """
        if not candidates:
            return []

        try:
            # Prepare(query, text) pairs
            pairs = [[query, c.get("text", "")] for c in candidates]
            scores = self.model.predict(pairs)
            if len(scores) != len(candidates):
                # log and fallback
                logger.warning("Reranker returned %d scores for %d candidates", len(scores), len(candidates))
                return candidates

            for i, score in enumerate(scores):
                candidates[i]["score"] = float(score)

            # stable sort: primary score, secondary rrf_score, tertiary doc_id
            return sorted(
                candidates,
                key=lambda x: (
                    x.get("score", 0.0),
                    x.get("rrf_score", 0.0),
                    x.get("doc_id", "")
                ),
                reverse=True,
            )

        except Exception as exc:
            logger.exception("Reranker failed: %s", exc)
            return candidates
