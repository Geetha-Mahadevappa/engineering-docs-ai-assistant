"""
Hybrid search: stage‑1 recall using dense vectors (Qdrant) and
keyword search (Elasticsearch).

This component focuses on broad recall. It gathers potentially
relevant snippets from both engines and returns a merged,
deduplicated list. Final ranking is handled separately.
"""

from typing import List, Dict, Any, Optional
import time
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from agent.memory.elasticsearch_service import ElasticsearchService
from agent.config import CONFIG


class HybridSearch:
    def __init__(self) -> None:
        self.qdrant = QdrantClient(url=CONFIG["qdrant"]["url"])
        self.collection = CONFIG["qdrant"]["collection"]
        self.es = ElasticsearchService()
        self.embedding_url = CONFIG["embedding_service"]["url"]
        self.limit = CONFIG["retrieval"]["hybrid_limit"]

    def run(self, query: str, user_id: str, project_id: str) -> List[Dict[str, Any]]:
        """
        Perform hybrid recall:
        - Dense semantic search (Qdrant)
        - Keyword BM25 search (Elasticsearch)

        Returns a merged, deduplicated list of candidate snippets.
        """
        query = (query or "").strip()
        if len(query) < 2:
            return []

        # Dense embedding
        query_vector = self._safe_embed_query(query)

        # Dense recall (if embedding succeeded)
        if query_vector is not None:
            security_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="project_id", match=MatchValue(value=project_id)),
                ]
            )
            vector_hits = self._safe_qdrant_search(query_vector, security_filter)
        else:
            vector_hits = []

        # Keyword recall
        keyword_hits = self._safe_keyword_search(query, user_id, project_id)

        return self._merge(vector_hits, keyword_hits)

    def _safe_embed_query(self, query: str) -> Optional[List[float]]:
        """
        Call the embedding service with basic retry logic.
        Returns None if all attempts fail.
        """
        payload = {"texts": [f"query: {query}"]}

        for _ in range(2):
            try:
                response = requests.post(self.embedding_url, json=payload, timeout=5)
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings")
                if not embeddings or not isinstance(embeddings, list):
                    return None
                return embeddings[0]
            except Exception:
                time.sleep(0.2)

        return None

    def _safe_qdrant_search(self, query_vector: List[float], security_filter: Filter):
        try:
            return self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=security_filter,
                limit=self.limit,
            )
        except Exception:
            return []

    def _safe_keyword_search(self, query: str, user_id: str, project_id: str):
        try:
            return self.es.search(
                query=query,
                user_id=user_id,
                project_id=project_id,
                limit=self.limit,
            )
        except Exception:
            return []

    def _merge(self, vector_hits, keyword_hits) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from Qdrant and Elasticsearch.
        Compute an RRF score for each snippet based on its rank in
        each engine. Deduplication is based on the text content.
        """
        unique: Dict[str, Dict[str, Any]] = {}
        k = 60  # RRF constant

        # Assign ranks for Qdrant hits
        for rank, hit in enumerate(vector_hits):
            payload = getattr(hit, "payload", {}) or {}
            text = payload.get("text")
            if not text:
                continue

            rrf_score = 1 / (k + rank + 1)

            unique[text] = {
                "text": text,
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("batch_start", 0),
                "rrf_score": rrf_score,
            }

        # Assign ranks for Elasticsearch hits
        for rank, hit in enumerate(keyword_hits):
            text = hit.get("text")
            if not text:
                continue

            rrf_score = 1 / (k + rank + 1)

            if text in unique:
                # Add to existing RRF score
                unique[text]["rrf_score"] += rrf_score
            else:
                unique[text] = {
                    "text": text,
                    "doc_id": hit.get("doc_id"),
                    "chunk_id": hit.get("chunk_id", 0),
                    "rrf_score": rrf_score,
                }

        return list(unique.values())


